import json
import os
import random
import re
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

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
    # Model
    model_name: str = "google/gemma-3-12b-it"
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )

    # Data
    data_path: str = "data/gams_ft_dataset.json"
    max_seq_len: int = 2048

    # Semantic guardrail
    semantic_model_name: str = "all-MiniLM-L6-v2"
    min_cosine_similarity: float = 0.35

    # Judge criteria — drives the dynamic system prompt
    judge_criteria: list = field(default_factory=lambda: [
        "grammar",
        "semantics",
        "flow",
        "completeness",
        "clarity",
    ])

    # Training
    dpo_beta: float = 0.1
    learning_rate: float = 5e-5
    num_epochs: int = 1
    inference_batch_size: int = 4   # large batch for generation + judging
    batch_size: int = 4              # smaller sub-batch for SFT + DPO training
    grad_accumulation_steps: int = 1
    warmup_steps: int = 50
    cosine_cycle_steps: int = 200
    sft_loss_weight: float = 0.4
    output_dir: str = "./checkpoints"

    ref_update_interval: int = 100

    # Generation
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Dynamic judge system prompt
# ---------------------------------------------------------------------------
def build_judge_system_prompt(criteria: list[str]) -> str:
    criteria_lines = "\n".join(
        f"{i + 1}. {c:<14} – {_criterion_description(c)}"
        for i, c in enumerate(criteria)
    )
    criteria_keys = ", ".join(f'"{c}": <1-5>' for c in criteria)
    return f"""\
You are a strict but fair language judge. You will be given:
- A QUESTION
- ANSWER 1
- ANSWER 2

Your task is to score BOTH ANSWERS on these {len(criteria)} criteria (score 1–5, where 5 is best):
{criteria_lines}

Return ONLY a valid JSON object with exactly this structure:
{{
  "ANSWER 1": {{{criteria_keys}}},
  "ANSWER 2": {{{criteria_keys}}}
}}

Do NOT add any explanation, markdown, or extra text."""


def _criterion_description(name: str) -> str:
    return {
        "grammar": "grammatical and linguistic correctness",
        "semantics": "semantic accuracy and factual correctness",
        "flow": "readability, coherence, and logical flow",
        "completeness": "how thoroughly the answer covers the topic",
        "clarity": "simplicity and clarity of explanation",
    }.get(name, name)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _build_bnb_config(cfg: FrameworkConfig) -> Optional[BitsAndBytesConfig]:
    if not cfg.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _build_lora_config(cfg: FrameworkConfig) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )


def load_model_and_tokenizer(cfg: FrameworkConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=_build_bnb_config(cfg),
        device_map="auto",
        torch_dtype=torch.bfloat16 if not cfg.load_in_4bit else None,
        attn_implementation="flash_attention_2"
    )

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = _build_lora_config(cfg)
    model = get_peft_model(model, lora_config, adapter_name="default")
    model.add_adapter("ref", lora_config)

    for name, param in model.named_parameters():
        if "ref" in name:
            param.requires_grad_(False)

    model.print_trainable_parameters()
    return model, tokenizer


def sync_ref_model(model) -> None:
    copied = 0
    with torch.no_grad():
        state_dict = dict(model.named_parameters())
        for name, param in state_dict.items():
            if "default" in name:
                ref_name = name.replace("default", "ref")
                if ref_name in state_dict:
                    state_dict[ref_name].data.copy_(param.data)
                    copied += 1
    print(f"[Ref] Synced adapter parameter tensors to reference adapter.")


# ---------------------------------------------------------------------------
# Padding-side context manager
# ---------------------------------------------------------------------------

@contextmanager
def left_padding(tokenizer: AutoTokenizer):
    original = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = original


# ---------------------------------------------------------------------------
# VRAM cleanup helper
# ---------------------------------------------------------------------------
def release_vram(label: str = "") -> None:
    """Synchronise CUDA, clear the cache and run garbage collection."""
    import gc
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    tag = f" [{label}]" if label else ""
    print(f"[VRAM]{tag} Cache cleared.")


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------
def generate_answers_batch(
        model,
        tokenizer,
        questions: list[str],
        cfg: FrameworkConfig,
        system_prompt: Optional[str] = None,
) -> list[str]:

    prompt = "Always respond in a grammatically correct manner using correct noun-adjective gender/number agreement, even if the question contains typos or other grammatically incorrect words." \
             "\nCRITICAL: Always respond in the SAME LANGUAGE as the user's question. " \
             "\nProvide your answer to the question below. \nQuestion:\n"

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": ""},
             {"role": "user", "content": prompt + q}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]

    print(prompts)

    with left_padding(tokenizer):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=cfg.max_seq_len,
        ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Free the input tensors before decoding
    del inputs
    release_vram("post-generation inputs")

    results = []
    for i in range(len(questions)):
        new_ids = output_ids[i, input_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        print(text)
        results.append(text)

    del output_ids
    release_vram("post-generation outputs")
    return results


# ---------------------------------------------------------------------------
# Batched judging
# ---------------------------------------------------------------------------
def judge_answers_batch(
        model,
        tokenizer,
        questions: list[str],
        generated_answers: list[str],
        gold_answers: list[str],
        filtered_idx: list[int],
        cfg: FrameworkConfig,
        judge_system_prompt: str,
) -> list[tuple[dict, dict]]:
    orderings = []
    for gen, gold in zip(generated_answers, gold_answers):
        if random.randint(0, 1) == 0:
            orderings.append((gen, gold, "ANSWER 1", "ANSWER 2"))
        else:
            orderings.append((gold, gen, "ANSWER 2", "ANSWER 1"))

    prompts = []
    for q, (a1, a2, _, _) in zip(questions, orderings):
        user_content = (
            f"QUESTION:\n{q}\n\n"
            f"ANSWER 1:\n{a1}\n\n"
            f"ANSWER 2:\n{a2}"
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": judge_system_prompt},
                 {"role": "user", "content": user_content}],
                tokenize=False, add_generation_prompt=True,
            )
        )

    with left_padding(tokenizer):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=cfg.max_seq_len,
        ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = output_ids[:, input_length:]

    del inputs
    release_vram("post-judge inputs")

    results = []
    for i, (_, _, generated_key, gs_key) in enumerate(orderings):
        new_ids = generated_tokens[i]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if i in filtered_idx:
            print(f"Filtered out (using 1 vs 5) for idx:", i)
            results.append(
                ({k: 1 for k in cfg.judge_criteria},
                 {k: 5 for k in cfg.judge_criteria})
            )
        else:
            try:
                print('----------------------------JUDGE------------------------------')
                print(raw)
                print('----------------------------JUDGE------------------------------')
                cleaned = raw.replace("```json", "").replace("```", "").strip()
                scores = json.loads(cleaned)
                scores_generated = scores[generated_key]
                scores_gs = scores[gs_key]
                for k in cfg.judge_criteria:
                    scores_generated[k] = int(max(1, min(5, scores_generated.get(k, 1))))
                    scores_gs[k] = int(max(1, min(5, scores_gs.get(k, 5))))
                results.append((scores_generated, scores_gs))
            except (json.JSONDecodeError, KeyError, ValueError):
                print(f"[Judge ] Parse failed — using fallback (1 vs 5).")
                results.append(
                    ({k: 1 for k in cfg.judge_criteria},
                     {k: 5 for k in cfg.judge_criteria})
                )

    del output_ids
    release_vram("post-judge outputs")
    return results


def aggregate_score(scores: dict, cfg: FrameworkConfig) -> float:
    vals = [scores.get(k, 3) for k in cfg.judge_criteria]
    return (sum(vals) / len(vals) - 1) / 4.0


# ---------------------------------------------------------------------------
# Semantic guardrail (vectorised)
# ---------------------------------------------------------------------------
class SemanticGuardrail:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)

    def filter_batch(
            self,
            generated_texts: list[str],
            gold_texts: list[str],
            threshold: float,
    ) -> list[bool]:
        n = len(generated_texts)
        texts = generated_texts + gold_texts
        embs = self.encoder.encode(texts, convert_to_numpy=True)
        gen_embs = embs[:n]
        gold_embs = embs[n:]

        keep = []
        for i in range(n):
            sim = float(cosine_similarity([gen_embs[i]], [gold_embs[i]])[0][0])
            print(f"[Semantic] cosine = {sim:.4f}")
            try:
                same_lang = detect(generated_texts[i]) == detect(gold_texts[i])
            except:
                same_lang = False
            print("Same language in gs and generated", same_lang)
            keep.append(sim >= threshold and same_lang)
        return keep


# ---------------------------------------------------------------------------
# Training example dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    question: str
    dpo_prompt: str
    chosen_completion: str
    rejected_completion: str
    reward_weight: float
    judge_prompt: str
    judge_response: str



# ---------------------------------------------------------------------------
# DPO prompt builder
# ---------------------------------------------------------------------------
def build_dpo_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": ""
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Inference phase: generate + guardrail + judge for a large inference batch
# Returns a flat list of TrainingExample (or None for discarded items).
# The model must already be set to "default" adapter and eval() before calling.
# ---------------------------------------------------------------------------
def run_inference_phase(
        items: list[dict],
        model,
        tokenizer,
        guardrail: SemanticGuardrail,
        cfg: FrameworkConfig,
        judge_system_prompt: str,
) -> list[Optional[TrainingExample]]:
    """
    Phase 1 – Generation (inference batch = cfg.inference_batch_size)
    Phase 2 – Semantic guardrail (CPU, no GPU memory impact)
    Phase 3 – Judging (inference batch = cfg.inference_batch_size)

    VRAM is released after generation output tensors are decoded and after
    judge output tensors are decoded.
    """
    questions, golds = [], []
    for item in items:
        convs = item.get("conversations", [])
        questions.append(convs[0]["value"] if len(convs) >= 1 else "")
        golds.append(convs[1]["value"] if len(convs) >= 2 else "")

    # ── Phase 1: generation ──────────────────────────────────────────────
    print(f"[Inference] Generating answers for {len(questions)} questions …")
    model.set_adapter("default")
    model.eval()

    generated_raw = generate_answers_batch(model, tokenizer, questions, cfg)

    keep_mask = guardrail.filter_batch(generated_raw, golds, cfg.min_cosine_similarity)

    filtered_idx = [i for i, keep in enumerate(keep_mask) if not keep]
    all_idx = [i for i, keep in enumerate(keep_mask)]
    for i, keep in enumerate(keep_mask):
        if not keep:
            print(f"[Guardrail] item {i} DISCARDED — too far from gold or wrong language.")


    pass_questions  = [q for q in questions]
    pass_clean_gens = [c for c in generated_raw]
    pass_golds      = [g for g in golds]

    # ── Phase 3: judging ─────────────────────────────────────────────────
    print(f"[Inference] Judging {len(pass_questions)} surviving answers …")
    # Still uses "default" adapter / eval mode from generation phase
    judge_results = judge_answers_batch(
        model, tokenizer,
        pass_questions, pass_clean_gens, pass_golds, filtered_idx,
        cfg, judge_system_prompt,
    )
    # judge_answers_batch already calls release_vram internally

    # ── Assemble TrainingExamples ────────────────────────────────────────
    output: list[Optional[TrainingExample]] = [None] * len(items)

    for rank, orig_idx in enumerate(all_idx):
        scores_generated, scores_gs = judge_results[rank]
        base_weight_gen = aggregate_score(scores_generated, cfg)
        base_weight_gs  = aggregate_score(scores_gs,        cfg)

        score_delta  = abs(base_weight_gen - base_weight_gs)
        score_chosen  = max(base_weight_gen, base_weight_gs)
        reward_weight = min(1.0, max(score_chosen, score_delta))

        print(
            f"[Reward] score_gen={base_weight_gen:.3f}  "
            f"score_gs={base_weight_gs:.3f}  "
            f"delta={score_delta:.3f} "
            f"reward_weight={reward_weight:.3f}"
        )

        dpo_prompt = build_dpo_prompt(questions[orig_idx], tokenizer)

        if base_weight_gen >= base_weight_gs:
            chosen_completion   = generated_raw[orig_idx]
            rejected_completion = golds[orig_idx]
        else:
            chosen_completion   = golds[orig_idx]
            rejected_completion = generated_raw[orig_idx]

        sft_user = (
            f"QUESTION:\n{questions[orig_idx]}\n\n"
            f"ANSWER 1:\n{pass_clean_gens[rank]}\n\n"
            f"ANSWER 2:\n{golds[orig_idx]}"
        )
        judge_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": judge_system_prompt},
             {"role": "user", "content": sft_user}],
            tokenize=False, add_generation_prompt=True,
        )
        judge_response = json.dumps(
            {"ANSWER 1": scores_generated, "ANSWER 2": scores_gs},
            ensure_ascii=False,
        )

        output[orig_idx] = TrainingExample(
            question=questions[orig_idx],
            dpo_prompt=dpo_prompt,
            chosen_completion=chosen_completion,
            rejected_completion=rejected_completion,
            reward_weight=reward_weight,
            judge_prompt=judge_prompt,
            judge_response=judge_response,
        )

    return output


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def batched_log_probs(
        model,
        tokenizer,
        prompts: list[str],
        completions: list[str],
        max_len: int,
        no_grad: bool = False,
) -> list[torch.Tensor]:
    assert len(prompts) == len(completions)

    encoded_full   = []
    encoded_prompt = []
    for p, c in zip(prompts, completions):
        enc_f = tokenizer(
            p + c, return_tensors="pt", truncation=True, max_length=max_len,
            return_token_type_ids=True, add_special_tokens=False
        )
        enc_p = tokenizer(
            p, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False
        )
        encoded_full.append(enc_f)
        encoded_prompt.append(enc_p)

    max_full_len = max(e["input_ids"].shape[1] for e in encoded_full)
    pad_id       = tokenizer.pad_token_id
    batch_size   = len(prompts)

    padded_ids  = torch.full((batch_size, max_full_len), pad_id, dtype=torch.long)
    padded_attn = torch.zeros((batch_size, max_full_len), dtype=torch.long)
    padded_ttids = torch.zeros((batch_size, max_full_len), dtype=torch.long)

    for i, enc in enumerate(encoded_full):
        seq_len = enc["input_ids"].shape[1]
        padded_ids[i,   :seq_len] = enc["input_ids"].squeeze(0)
        padded_attn[i,  :seq_len] = enc["attention_mask"].squeeze(0)
        padded_ttids[i, :seq_len] = enc["token_type_ids"].squeeze(0)

    padded_ids   = padded_ids.to(model.device)
    padded_attn  = padded_attn.to(model.device)
    padded_ttids = padded_ttids.to(model.device)

    ctx = torch.inference_mode() if no_grad else torch.enable_grad()
    with ctx:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs  = model(
                input_ids=padded_ids,
                attention_mask=padded_attn,
                token_type_ids=padded_ttids,
            )
            logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = padded_ids[:, 1:].contiguous().unsqueeze(-1)

    del outputs, logits, padded_ids, padded_attn, padded_ttids

    token_logits = shift_logits.gather(2, shift_labels).squeeze(-1)
    log_z        = torch.logsumexp(shift_logits, dim=-1)
    token_lp     = token_logits - log_z

    del shift_logits, log_z, token_logits, shift_labels

    results = []
    for i, enc_p in enumerate(encoded_prompt):
        # The prompt length tells us where the completion starts
        prompt_len = enc_p["input_ids"].shape[1]
        # The full length (prompt + completion)
        seq_len = encoded_full[i]["input_ids"].shape[1]
        # Shifted labels mean the prediction for the first completion token
        # is at index [prompt_len - 1] in the shift_logits/token_lp tensor.
        # We want to sum from the prompt end to the end of the sequence.

        # Correct indexing:
        # prompt_len - 1 is the log-prob of the FIRST token of the completion
        # seq_len - 1 is the log-prob of the LAST token of the completion
        completion_lp = token_lp[i, prompt_len - 1: seq_len - 1]

        # DO NOT divide by length. Use the raw sum.
        results.append(completion_lp.sum().clone())

    del token_lp
    return results


def dpo_loss_weighted_batch(
        ref_lp_chosen:   list[torch.Tensor],
        ref_lp_rejected: list[torch.Tensor],
        pol_lp_chosen:   list[torch.Tensor],
        pol_lp_rejected: list[torch.Tensor],
        beta: float,
        reward_weights: list[float],
) -> torch.Tensor:
    losses = []
    for rc, rr, pc, pr, w in zip(
            ref_lp_chosen,   ref_lp_rejected,
            pol_lp_chosen,   pol_lp_rejected,
            reward_weights,
    ):
        chosen_ratio  = pc - rc
        rejected_ratio = pr - rr
        margin = beta * (chosen_ratio - rejected_ratio) * w
        losses.append(-F.logsigmoid(margin))
    return torch.stack(losses).mean()


def batched_sft_loss(
        model,
        tokenizer,
        prompts: list[str],
        completions: list[str],
        max_len: int,
) -> torch.Tensor:
    assert len(prompts) == len(completions)
    batch_size = len(prompts)

    encoded_full   = []
    prompt_lengths = []
    for p, c in zip(prompts, completions):
        enc_f = tokenizer(
            p + c, return_tensors="pt", truncation=True, max_length=max_len,
            return_token_type_ids=True, add_special_tokens=False
        )
        enc_p = tokenizer(
            p, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False
        )
        encoded_full.append(enc_f)
        prompt_lengths.append(enc_p["input_ids"].shape[1])

    max_full_len = max(e["input_ids"].shape[1] for e in encoded_full)
    pad_id = tokenizer.pad_token_id

    padded_ids   = torch.full((batch_size, max_full_len), pad_id, dtype=torch.long)
    padded_attn  = torch.zeros((batch_size, max_full_len), dtype=torch.long)
    padded_ttids = torch.zeros((batch_size, max_full_len), dtype=torch.long)
    labels       = torch.full((batch_size, max_full_len), -100, dtype=torch.long)

    for i, (enc, p_len) in enumerate(zip(encoded_full, prompt_lengths)):
        seq_len = enc["input_ids"].shape[1]
        padded_ids[i,   :seq_len] = enc["input_ids"].squeeze(0)
        padded_attn[i,  :seq_len] = enc["attention_mask"].squeeze(0)
        padded_ttids[i, :seq_len] = enc["token_type_ids"].squeeze(0)
        labels[i, p_len:seq_len]  = enc["input_ids"].squeeze(0)[p_len:]

    padded_ids   = padded_ids.to(model.device)
    padded_attn  = padded_attn.to(model.device)
    padded_ttids = padded_ttids.to(model.device)
    labels       = labels.to(model.device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(
            input_ids=padded_ids,
            attention_mask=padded_attn,
            token_type_ids=padded_ttids,
            labels=labels,
        ).loss

    del padded_ids, padded_attn, padded_ttids, labels
    return loss


# ---------------------------------------------------------------------------
# Training phase: consume a list of TrainingExamples in sub-batches
# Returns (global_step, accum_count, running_loss) updated values.
# ---------------------------------------------------------------------------
def run_training_phase(
        valid_examples: list[TrainingExample],
        model,
        tokenizer,
        cfg: FrameworkConfig,
        optimizer,
        scheduler,
        global_step: int,
        accum_count: int,
        running_loss: float,
) -> tuple[int, int, float]:
    """
    Splits *valid_examples* (the output of one inference batch) into
    sub-batches of size cfg.batch_size and runs SFT + DPO on each one,
    releasing VRAM between sub-batches.
    """
    sub_batches = [
        valid_examples[i: i + cfg.batch_size]
        for i in range(0, len(valid_examples), cfg.batch_size)
    ]

    # Enable gradient checkpointing for training, will be disabled after the loop
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    for sb_idx, sub_batch in enumerate(sub_batches):
        n = len(sub_batch)
        print(f"  [Train sub-batch {sb_idx + 1}/{len(sub_batches)}] {n} examples")

        dpo_prompts          = [ex.dpo_prompt          for ex in sub_batch]
        chosen_completions   = [ex.chosen_completion   for ex in sub_batch]
        rejected_completions = [ex.rejected_completion for ex in sub_batch]
        reward_weights       = [ex.reward_weight       for ex in sub_batch]
        judge_prompts        = [ex.judge_prompt        for ex in sub_batch]
        judge_responses      = [ex.judge_response      for ex in sub_batch]

        # ── Ref log-probs (no grad, ref adapter) ──────────────────────
        model.set_adapter("ref")
        model.eval()
        ref_lps = batched_log_probs(
            model, tokenizer,
            dpo_prompts + dpo_prompts,
            chosen_completions + rejected_completions,
            cfg.max_seq_len, no_grad=True,
        )
        # Detach explicitly: even though no_grad=True, severing the tensor
        # from any autograd state ensures nothing from the ref forward pass
        # lingers in memory during the policy forward pass.
        ref_lp_chosen   = [t.detach() for t in ref_lps[:n]]
        ref_lp_rejected = [t.detach() for t in ref_lps[n:]]
        del ref_lps

        release_vram(f"sub-batch {sb_idx + 1} ref logprobs")

        # ── Policy log-probs + SFT (with grad, default adapter) ───────
        model.set_adapter("default")
        model.train()

        pol_lps = batched_log_probs(
            model, tokenizer,
            dpo_prompts + dpo_prompts,
            chosen_completions + rejected_completions,
            cfg.max_seq_len, no_grad=False,
        )
        pol_lp_chosen   = pol_lps[:n]
        pol_lp_rejected = pol_lps[n:]
        del pol_lps

        loss_dpo = dpo_loss_weighted_batch(
            ref_lp_chosen,   ref_lp_rejected,
            pol_lp_chosen,   pol_lp_rejected,
            beta=cfg.dpo_beta,
            reward_weights=reward_weights,
        )

        loss_sft = batched_sft_loss(
            model, tokenizer,
            dpo_prompts, chosen_completions, cfg.max_seq_len,
        )

        print('--------------Chosen completion-------------------------')
        print("prompts", dpo_prompts)
        print('\n\n\ Completions', chosen_completions)

        loss = (1 - cfg.sft_loss_weight) * loss_dpo + cfg.sft_loss_weight * loss_sft
        (loss / cfg.grad_accumulation_steps).backward()

        running_loss += loss.item()
        accum_count  += n

        # Free intermediate tensors now that gradients are accumulated
        del ref_lp_chosen, ref_lp_rejected
        del pol_lp_chosen, pol_lp_rejected
        #del loss_dpo, loss_sft, loss
        del loss_dpo, loss
        release_vram(f"sub-batch {sb_idx + 1} post-backward")

        # ── Optimizer step when accumulation threshold is reached ──────
        if accum_count >= cfg.grad_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accum_count  = 0

            avg          = running_loss / cfg.grad_accumulation_steps
            running_loss = 0.0
            print(
                f"  Step {global_step:4d} | loss={avg:.4f}"
            )

            if cfg.ref_update_interval > 0 and global_step % cfg.ref_update_interval == 0:
                sync_ref_model(model)

            release_vram(f"step {global_step} post-optimizer")

    # Disable gradient checkpointing so the next inference phase is clean
    model.gradient_checkpointing_disable()
    release_vram("end of training phase")

    return global_step, accum_count, running_loss


# ---------------------------------------------------------------------------
# Online training loop
# ---------------------------------------------------------------------------
def train_online(
        model,
        tokenizer,
        raw_data: list[dict],
        guardrail: SemanticGuardrail,
        cfg: FrameworkConfig,
) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    # NOTE: gradient checkpointing is enabled only inside run_training_phase
    # and disabled again before returning, so inference always runs without it.
    model.gradient_checkpointing_disable()

    judge_system_prompt = build_judge_system_prompt(cfg.judge_criteria)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    placeholder_total = max(1, cfg.cosine_cycle_steps * cfg.num_epochs)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=placeholder_total,
        num_cycles=cfg.num_epochs,
    )

    global_step  = 0
    running_loss = 0.0
    accum_count  = 0

    for epoch in range(cfg.num_epochs):
        print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{cfg.num_epochs}\n{'=' * 60}")

        epoch_data = random.sample(raw_data, len(raw_data))
        skipped    = 0
        optimizer.zero_grad()

        # Iterate over large INFERENCE batches
        for inf_start in range(0, len(epoch_data), cfg.inference_batch_size):
            inf_items = epoch_data[inf_start: inf_start + cfg.inference_batch_size]
            print(
                f"\n[Epoch {epoch + 1} | Inference items "
                f"{inf_start + 1}–{inf_start + len(inf_items)}/{len(epoch_data)}]"
            )

            # ── INFERENCE PHASE (generation + guardrail + judging) ─────
            examples = run_inference_phase(
                inf_items, model, tokenizer, guardrail, cfg, judge_system_prompt
            )

            valid_examples = [ex for ex in examples if ex is not None]
            skipped       += sum(1 for ex in examples if ex is None)

            if not valid_examples:
                print("[Batch] All items discarded by guardrail — skipping.")
                continue

            # VRAM is already clean after inference phase; now switch to training
            release_vram("between inference and training phases")

            # ── TRAINING PHASE (sub-batches of cfg.batch_size) ────────
            print(
                f"[Train] {len(valid_examples)} valid examples → "
                f"{(len(valid_examples) + cfg.batch_size - 1) // cfg.batch_size} sub-batches "
                f"of up to {cfg.batch_size}"
            )
            global_step, accum_count, running_loss = run_training_phase(
                valid_examples, model, tokenizer, cfg,
                optimizer, scheduler,
                global_step, accum_count, running_loss,
            )

        # ── Flush any remaining gradient accumulation at end of epoch ─
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accum_count  = 0
            print(f"  Step {global_step:4d} | flushed partial accumulation batch")
            release_vram("end-of-epoch flush")

        print(f"\n[Epoch {epoch + 1}] Skipped {skipped}/{len(epoch_data)} items (guardrail).")

        ckpt = f"{cfg.output_dir}/epoch_{epoch + 1}"
        model.set_adapter("default")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"[Checkpoint] Saved to {ckpt}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg: FrameworkConfig):
    print("[Init] Loading model with dual-adapters (Policy + Ref) and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    print("[Init] Loading semantic guardrail...")
    guardrail = SemanticGuardrail(cfg.semantic_model_name)

    print(f"[Data] Reading {cfg.data_path}...")
    raw_data = load_jsonl(cfg.data_path)
    print(f"[Data] {len(raw_data)} raw examples loaded.")

    train_online(model, tokenizer, raw_data, guardrail, cfg)
    print("\n[Done] Training complete.")


if __name__ == "__main__":
    cfg = FrameworkConfig(
        model_name="google/gemma-3-12b-it",
        data_path="data/nemotron_sft_all_final_98k.json",
        num_epochs=1,
        load_in_4bit=True,
        inference_batch_size=8,   # generate + judge 16 at a time
        batch_size=2,              # train 4 at a time
        output_dir="./checkpoints",
        ref_update_interval=100,
    )
    main(cfg)