import json
import os
import random
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import AdamW
from langdetect import detect


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FrameworkConfig:
    model_name: str = "google/gemma-3-12b-it"
    # TIP: For maximum speed on DeepSpeed ZeRO-3, set load_in_4bit=False and use bf16.
    # For ZeRO-2, load_in_4bit=True (QLoRA) works great.
    load_in_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )

    data_path: str = "data/gams_ft_dataset.json"
    max_seq_len: int = 2048
    semantic_model_name: str = "all-MiniLM-L6-v2"
    min_cosine_similarity: float = 0.35

    judge_criteria: list = field(default_factory=lambda: [
        "grammar", "semantics", "flow", "completeness", "clarity",
    ])

    dpo_beta: float = 0.1
    thinking_bonus: float = 0.3
    min_think_tokens: int = 20
    learning_rate: float = 5e-5
    num_epochs: int = 1

    # Batch sizes are now PER GPU
    inference_batch_size: int = 16
    batch_size: int = 4
    grad_accumulation_steps: int = 8

    warmup_steps: int = 50
    cosine_cycle_steps: int = 200
    sft_loss_weight: float = 0.3
    output_dir: str = "./checkpoints"
    ref_update_interval: int = 100

    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


THINK_START = "<think>"
THINK_END = "</think>"


# ---------------------------------------------------------------------------
# Prompts & Utilities
# ---------------------------------------------------------------------------
def build_judge_system_prompt(criteria: list[str]) -> str:
    def _desc(name):
        return {
            "grammar": "grammatical and linguistic correctness",
            "semantics": "semantic accuracy and factual correctness",
            "flow": "readability, coherence, and logical flow",
            "completeness": "how thoroughly the answer covers the topic",
            "clarity": "simplicity and clarity of explanation",
        }.get(name, name)

    criteria_lines = "\n".join(f"{i + 1}. {c:<14} – {_desc(c)}" for i, c in enumerate(criteria))
    return f"""\
You are a strict but fair language judge. You will be given a QUESTION, ANSWER 1, and ANSWER 2.
Score BOTH ANSWERS on these {len(criteria)} criteria (score 1–5, where 5 is best):
{}

Return ONLY a valid JSON object with exactly this structure:
{{
  "ANSWER 1": {{"grammar": <1-5>, ...}},
  "ANSWER 2": {{"grammar": <1-5>, ...}}
}}
Do NOT add any explanation, markdown, or extra text."""


def build_dpo_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    prompt = "You are a helpful assistant. " \
             "CRITICAL: Always respond in the SAME LANGUAGE as the user's question. " \
             f"You MUST reason step by step inside {} and {} tags first. " \
             f"You MUST include the closing {} tag when you are done reasoning! " \
             "Then provide your final answer. \nQuestion:\n"
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": prompt + question}],
        tokenize=False, add_generation_prompt=True
    )


@contextmanager
def left_padding(tokenizer: AutoTokenizer):
    original = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = original


def strip_thinking(text: str) -> str:
    if THINK_END in text: return text.split(THINK_END)[-1].strip()
    return text.replace(THINK_START, "").strip()


def has_thinking(text: str, tokenizer, min_tokens: int) -> bool:
    if THINK_END in text:
        think_content = text.split(THINK_START)[-1].split(THINK_END)[0].strip()
    elif THINK_START in text:
        think_content = text.split(THINK_START)[-1].strip()
    else:
        return False
    return bool(think_content) and len(tokenizer.encode(think_content, add_special_tokens=False)) >= min_tokens


def aggregate_score(scores: dict, cfg: FrameworkConfig) -> float:
    vals = [scores.get(k, 3) for k in cfg.judge_criteria]
    return (sum(vals) / len(vals) - 1) / 4.0


# ---------------------------------------------------------------------------
# Dataset & Loading
# ---------------------------------------------------------------------------
class RLHFDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        convs = self.data[idx].get("conversations", [])
        q = convs[0]["value"] if len(convs) >= 1 else ""
        g = convs[1]["value"] if len(convs) >= 2 else ""
        return {"question": q, "gold": g}


def load_model_and_tokenizer(cfg: FrameworkConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    ) if cfg.load_in_4bit else None

    # NO device_map="auto" -> DeepSpeed/Accelerate handles distribution automatically
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not cfg.load_in_4bit else None,
        attn_implementation="flash_attention_2"
    )

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_target_modules, bias="none"
    )
    model = get_peft_model(model, lora_config, adapter_name="default")
    model.add_adapter("ref", lora_config)

    for name, param in model.named_parameters():
        if "ref" in name: param.requires_grad_(False)

    return model, tokenizer


def sync_ref_model(model, accelerator) -> None:
    unwrapped = accelerator.unwrap_model(model)
    with torch.no_grad():
        state_dict = dict(unwrapped.named_parameters())
        for name, param in state_dict.items():
            if "default" in name:
                ref_name = name.replace("default", "ref")
                if ref_name in state_dict:
                    state_dict[ref_name].data.copy_(param.data)
    if accelerator.is_main_process:
        print(f"[Ref] Synced adapter parameter tensors to reference adapter.")


# ---------------------------------------------------------------------------
# Inference (Generation & Judging)
# ---------------------------------------------------------------------------
def generate_answers_batch(model, tokenizer, questions, cfg, accelerator):
    prompts = [build_dpo_prompt(q, tokenizer) for q in questions]
    with left_padding(tokenizer):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False,
                           max_length=cfg.max_seq_len).to(accelerator.device)

    unwrapped_model = accelerator.unwrap_model(model)
    with torch.inference_mode():
        output_ids = unwrapped_model.generate(
            **inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )

    results = []
    for i in range(len(questions)):
        new_ids = output_ids[i, inputs["input_ids"].shape[1]:]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    return results


def judge_answers_batch(model, tokenizer, questions, generated_answers, gold_answers, cfg, judge_system_prompt,
                        accelerator):
    orderings = []
    for gen, gold in zip(generated_answers, gold_answers):
        if random.randint(0, 1) == 0:
            orderings.append((gen, gold, "ANSWER 1", "ANSWER 2"))
        else:
            orderings.append((gold, gen, "ANSWER 2", "ANSWER 1"))

    prompts = []
    for q, (a1, a2, _, _) in zip(questions, orderings):
        user_content = f"QUESTION:\n{}\n\nANSWER 1:\n{}\n\nANSWER 2:\n{}\n"
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": judge_system_prompt}, {"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True
        ))

    with left_padding(tokenizer):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False,
                           max_length=cfg.max_seq_len).to(accelerator.device)

    unwrapped_model = accelerator.unwrap_model(model)
    with torch.inference_mode():
        output_ids = unwrapped_model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True,
                                              pad_token_id=tokenizer.pad_token_id)

    results = []
    for i, (_, _, generated_key, gs_key) in enumerate(orderings):
        raw = tokenizer.decode(output_ids[i, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            scores = json.loads(cleaned)
            scores_generated = scores[generated_key]
            scores_gs = scores[gs_key]
            for k in cfg.judge_criteria:
                scores_generated[k] = int(max(1, min(5, scores_generated.get(k, 1))))
                scores_gs[k] = int(max(1, min(5, scores_gs.get(k, 5))))
            results.append((scores_generated, scores_gs))
        except:
            results.append(({k: 1 for k in cfg.judge_criteria}, {k: 5 for k in cfg.judge_criteria}))
    return results


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def batched_log_probs(model, tokenizer, prompts, completions, max_len, accelerator, no_grad=False):
    encoded_full, encoded_prompt = [], []
    for p, c in zip(prompts, completions):
        encoded_full.append(
            tokenizer(p + c, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False))
        encoded_prompt.append(
            tokenizer(p, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False))

    batch_size = len(prompts)
    max_full_len = max(e["input_ids"].shape[1] for e in encoded_full)

    padded_ids = torch.full((batch_size, max_full_len), tokenizer.pad_token_id, dtype=torch.long,
                            device=accelerator.device)
    padded_attn = torch.zeros((batch_size, max_full_len), dtype=torch.long, device=accelerator.device)

    for i, enc in enumerate(encoded_full):
        seq_len = enc["input_ids"].shape[1]
        padded_ids[i, :seq_len] = enc["input_ids"].squeeze(0)
        padded_attn[i, :seq_len] = enc["attention_mask"].squeeze(0)

    ctx = torch.inference_mode() if no_grad else torch.enable_grad()
    with ctx:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=padded_ids, attention_mask=padded_attn).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = padded_ids[:, 1:].contiguous().unsqueeze(-1)

    token_logits = shift_logits.gather(2, shift_labels).squeeze(-1)
    log_z = torch.logsumexp(shift_logits, dim=-1)
    token_lp = token_logits - log_z

    results = []
    for i, enc_p in enumerate(encoded_prompt):
        prompt_len = enc_p["input_ids"].shape[1]
        seq_len = encoded_full[i]["input_ids"].shape[1]

        mask = torch.zeros(seq_len - 1, dtype=torch.bool, device=accelerator.device)
        mask[prompt_len - 1:seq_len - 1] = True

        valid_lp = token_lp[i, :seq_len - 1]
        masked = valid_lp * mask.float()
        results.append(masked.sum() / mask.sum().float().clamp(min=1))

    return results


def batched_sft_loss(model, tokenizer, prompts, completions, max_len, accelerator):
    encoded_full, prompt_lengths = [], []
    for p, c in zip(prompts, completions):
        encoded_full.append(
            tokenizer(p + c, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False))
        prompt_lengths.append(
            tokenizer(p, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False)[
                "input_ids"].shape[1])

    batch_size = len(prompts)
    max_full_len = max(e["input_ids"].shape[1] for e in encoded_full)

    padded_ids = torch.full((batch_size, max_full_len), tokenizer.pad_token_id, dtype=torch.long,
                            device=accelerator.device)
    padded_attn = torch.zeros((batch_size, max_full_len), dtype=torch.long, device=accelerator.device)
    labels = torch.full((batch_size, max_full_len), -100, dtype=torch.long, device=accelerator.device)

    for i, (enc, p_len) in enumerate(zip(encoded_full, prompt_lengths)):
        seq_len = enc["input_ids"].shape[1]
        padded_ids[i, :seq_len] = enc["input_ids"].squeeze(0)
        padded_attn[i, :seq_len] = enc["attention_mask"].squeeze(0)
        labels[i, p_len - 1:seq_len - 1] = enc["input_ids"].squeeze(0)[p_len:seq_len]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=padded_ids, attention_mask=padded_attn, labels=labels).loss
    return loss


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main(cfg: FrameworkConfig):
    # Accelerator initializes distributed training (DeepSpeed ZeRO, DDP, etc.)
    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accumulation_steps)
    set_seed(42)

    accelerator.print("[Init] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Load Semantic Guardrail on local device
    guardrail_encoder = SentenceTransformer(cfg.semantic_model_name, device=accelerator.device)
    judge_system_prompt = build_judge_system_prompt(cfg.judge_criteria)

    dataset = RLHFDataset(cfg.data_path)
    dataloader = DataLoader(dataset, batch_size=cfg.inference_batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=len(dataloader) * cfg.num_epochs, num_cycles=cfg.num_epochs
    )

    # Accelerator wraps objects to handle batch sharding and gradient syncs automatically
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0

    for epoch in range(cfg.num_epochs):
        accelerator.print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{cfg.num_epochs}\n{'=' * 60}")

        for batch_idx, batch in enumerate(dataloader):
            questions = batch["question"]
            golds = batch["gold"]

            # ---------------- 1. INFERENCE PHASE (Generation) ----------------
            accelerator.unwrap_model(model).set_adapter("default")
            model.eval()

            generated_raw = generate_answers_batch(model, tokenizer, questions, cfg, accelerator)
            clean_gens = [strip_thinking(g) for g in generated_raw]

            # ---------------- 2. GUARDRAIL PHASE ----------------
            gen_embs = guardrail_encoder.encode(clean_gens, convert_to_tensor=True)
            gold_embs = guardrail_encoder.encode(golds, convert_to_tensor=True)
            cosine_scores = F.cosine_similarity(gen_embs, gold_embs)

            passing_idx = []
            for i in range(len(questions)):
                sim = cosine_scores[i].item()
                try:
                    same_lang = detect(clean_gens[i]) == detect(golds[i])
                except:
                    same_lang = False
                if sim >= cfg.min_cosine_similarity and same_lang:
                    passing_idx.append(i)

            if not passing_idx:
                continue

            pass_questions = [questions[i] for i in passing_idx]
            pass_clean_gens = [clean_gens[i] for i in passing_idx]
            pass_golds = [golds[i] for i in passing_idx]

            # ---------------- 3. JUDGING PHASE ----------------
            judge_results = judge_answers_batch(
                model, tokenizer, pass_questions, pass_clean_gens, pass_golds, cfg, judge_system_prompt, accelerator
            )

            valid_examples = []
            for rank, orig_idx in enumerate(passing_idx):
                scores_generated, scores_gs = judge_results[rank]
                base_weight_gen = aggregate_score(scores_generated, cfg)
                base_weight_gs = aggregate_score(scores_gs, cfg)

                score_delta = abs(base_weight_gen - base_weight_gs)
                think_bonus = cfg.thinking_bonus if has_thinking(generated_raw[orig_idx], tokenizer,
                                                                 cfg.min_think_tokens) else 0.0
                reward_weight = min(1.0, max(max(base_weight_gen, base_weight_gs), score_delta + think_bonus))

                if base_weight_gen >= base_weight_gs:
                    chosen, rejected = generated_raw[orig_idx], golds[orig_idx]
                else:
                    chosen, rejected = golds[orig_idx], generated_raw[orig_idx]

                sft_user = f"QUESTION:\n{questions[orig_idx]}\n\nANSWER 1:\n{pass_clean_gens[rank]}\n\nANSWER 2:\n{golds[orig_idx]}"
                judge_prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": judge_system_prompt}, {"role": "user", "content": sft_user}],
                    tokenize=False, add_generation_prompt=True
                )
                judge_response = json.dumps({"ANSWER 1": scores_generated, "ANSWER 2": scores_gs}, ensure_ascii=False)

                valid_examples.append({
                    "q": questions[orig_idx], "chosen": chosen, "rejected": rejected,
                    "weight": reward_weight, "j_prompt": judge_prompt, "j_resp": judge_response
                })

            # ---------------- 4. TRAINING PHASE ----------------
            sub_batches = [valid_examples[i: i + cfg.batch_size] for i in range(0, len(valid_examples), cfg.batch_size)]

            # Accelerate natively handles gradient checkpointing if configured
            model.train()

            for sub_batch in sub_batches:
                # 'accumulate' gracefully abstracts away grad accumulation math and DeepSpeed boundaries
                with accelerator.accumulate(model):
                    prompts = [build_dpo_prompt(ex["q"], tokenizer) for ex in sub_batch]
                    chosen = [ex["chosen"] for ex in sub_batch]
                    rejected = [ex["rejected"] for ex in sub_batch]
                    weights = torch.tensor([ex["weight"] for ex in sub_batch], device=accelerator.device)
                    j_prompts = [ex["j_prompt"] for ex in sub_batch]
                    j_resps = [ex["j_resp"] for ex in sub_batch]

                    # --- Ref pass (no_grad) ---
                    accelerator.unwrap_model(model).set_adapter("ref")
                    model.eval()
                    ref_lps = batched_log_probs(model, tokenizer, prompts * 2, chosen + rejected, cfg.max_seq_len,
                                                accelerator, no_grad=True)
                    # Detach removes the graph so memory is instantly freed
                    ref_chosen = torch.stack([t.detach() for t in ref_lps[:len(chosen)]])
                    ref_rejected = torch.stack([t.detach() for t in ref_lps[len(chosen):]])

                    # --- Policy pass (grad) ---
                    accelerator.unwrap_model(model).set_adapter("default")
                    model.train()
                    pol_lps = batched_log_probs(model, tokenizer, prompts * 2, chosen + rejected, cfg.max_seq_len,
                                                accelerator, no_grad=False)
                    pol_chosen = torch.stack(pol_lps[:len(chosen)])
                    pol_rejected = torch.stack(pol_lps[len(chosen):])

                    # --- DPO Loss ---
                    chosen_ratio = pol_chosen - ref_chosen
                    rejected_ratio = pol_rejected - ref_rejected
                    margin = cfg.dpo_beta * (chosen_ratio - rejected_ratio) * weights
                    loss_dpo = -F.logsigmoid(margin).mean()

                    # --- SFT Loss ---
                    loss_sft = batched_sft_loss(model, tokenizer, j_prompts, j_resps, cfg.max_seq_len, accelerator)

                    # --- Total Loss & Backward ---
                    loss = loss_dpo + cfg.sft_loss_weight * loss_sft
                    accelerator.backward(loss)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Using accelerator.sync_gradients tells us if a full gradient accumulation boundary was crossed
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.print(f"Step {} | Loss: {loss.item():.4f}")

                if global_step % cfg.ref_update_interval == 0:
                    sync_ref_model(model, accelerator)

        # ---------------- SAVE CHECKPOINT ----------------
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt = f"{cfg.output_dir}/epoch_{epoch + 1}"
            accelerator.unwrap_model(model).save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            accelerator.print(f"[Checkpoint] Saved to {}")


if __name__ == "__main__":
    cfg = FrameworkConfig()
    main(cfg)