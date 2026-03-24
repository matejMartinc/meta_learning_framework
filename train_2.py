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
    thinking_bonus: float = 0.3
    min_think_tokens: int = 20
    learning_rate: float = 5e-5
    num_epochs: int = 1
    batch_size: int = 2
    grad_accumulation_steps: int = 8
    warmup_steps: int = 50
    cosine_cycle_steps: int = 200
    sft_loss_weight: float = 0.3
    output_dir: str = "./checkpoints"

    ref_update_interval: int = 100

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

THINK_START = "<think>"
THINK_END = "</think>"
SPECIAL_TOKENS = [THINK_START, THINK_END]


def add_special_tokens(tokenizer: AutoTokenizer) -> None:
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in SPECIAL_TOKENS if t not in existing]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"[Tokenizer] Added special tokens: {new_tokens}")
    else:
        print("[Tokenizer] Special tokens already present.")


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
    add_special_tokens(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=_build_bnb_config(cfg),
        device_map="auto",
        torch_dtype=torch.bfloat16 if not cfg.load_in_4bit else None,
        attn_implementation="flash_attention_2"
    )
    model.resize_token_embeddings(len(tokenizer))

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
# Batched generation
# ---------------------------------------------------------------------------
def generate_answers_batch(
        model,
        tokenizer,
        questions: list[str],
        cfg: FrameworkConfig,
        system_prompt: Optional[str] = None,
) -> list[str]:
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. "
            "CRITICAL: Always respond in the SAME LANGUAGE as the user's question. "
            f"You MUST reason step by step inside {THINK_START} and {THINK_END} tags first. "
            f"You MUST include the closing {THINK_END} tag when you are done reasoning! "
            "Then provide your final answer."
        )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        ) + f"{THINK_START}\n"
        for q in questions
    ]

    with left_padding(tokenizer):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=cfg.max_seq_len,
        ).to(model.device)

    prompt_lengths = inputs["attention_mask"].sum(dim=1)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i, prompt_len in enumerate(prompt_lengths.tolist()):
        new_ids = output_ids[i, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=False).strip()

        # Strip generation artifacts like eos/pad tokens so they don't break logic
        if tokenizer.eos_token:
            text = text.replace(tokenizer.eos_token, "")
        if tokenizer.pad_token:
            text = text.replace(tokenizer.pad_token, "")

        text = f"{THINK_START}\n" + text.strip()
        results.append(text)
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
        print('\n*******************************\n')
        print(user_content)
        print('------------------------------------')
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

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i, (_, _, generated_key, gs_key) in enumerate(orderings):
        new_ids = output_ids[i, inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        try:
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
            keep.append(sim >= threshold)
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
# Think-tag helpers (FIXED)
# ---------------------------------------------------------------------------
def strip_thinking(text: str) -> str:
    """Safely strip thinking tags. If </think> is missing, returns the text without returning empty strings."""
    if THINK_END in text:
        # Properly closed: take everything after </think>
        return text.split(THINK_END)[-1].strip()

    # Not closed: just remove the <think> literal so we don't evaluate empty strings vs the gold standard
    return text.replace(THINK_START, "").strip()


def has_thinking(text: str, tokenizer, min_tokens: int) -> bool:
    """Count tokens even if the model failed to close the tag."""
    if THINK_END in text:
        think_content = text.split(THINK_START)[-1].split(THINK_END)[0].strip()
    elif THINK_START in text:
        think_content = text.split(THINK_START)[-1].strip()
    else:
        return False

    if not think_content:
        return False

    return len(tokenizer.encode(think_content, add_special_tokens=False)) >= min_tokens


# ---------------------------------------------------------------------------
# DPO prompt builder
# ---------------------------------------------------------------------------
def build_dpo_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "CRITICAL: Always respond in the SAME LANGUAGE as the user's question. "
                f"You MUST reason step by step inside {THINK_START} and {THINK_END} tags first. "
                f"You MUST include the closing {THINK_END} tag when you are done reasoning! "
                "Then provide your final answer."
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Batched online example builder
# ---------------------------------------------------------------------------
def build_online_batch(
        items: list[dict],
        model,
        tokenizer,
        guardrail: SemanticGuardrail,
        cfg: FrameworkConfig,
        judge_system_prompt: str,
) -> list[Optional[TrainingExample]]:
    questions, golds = [], []
    for item in items:
        convs = item.get("conversations", [])
        questions.append(convs[0]["value"] if len(convs) >= 1 else "")
        golds.append(convs[1]["value"] if len(convs) >= 2 else "")

    generated_raw = generate_answers_batch(model, tokenizer, questions, cfg)

    for i, g in enumerate(zip(questions, generated_raw)):
        print(f"[Question {i}] {g[0]}")
        print(f"[Gen {i}] {g[1]}")

    # With the new strip_thinking, this clean_gens array will accurately contain text, not empty strings!
    clean_gens = [strip_thinking(g) for g in generated_raw]
    keep_mask = guardrail.filter_batch(clean_gens, golds, cfg.min_cosine_similarity)

    passing_idx = [i for i, keep in enumerate(keep_mask) if keep]
    for i, keep in enumerate(keep_mask):
        if not keep:
            print(f"[Guardrail ] DISCARDED — semantically too far from gold.")

    if not passing_idx:
        return [None] * len(items)

    pass_questions = [questions[i] for i in passing_idx]
    pass_clean_gens = [clean_gens[i] for i in passing_idx]
    pass_golds = [golds[i] for i in passing_idx]

    judge_results = judge_answers_batch(
        model, tokenizer,
        pass_questions, pass_clean_gens, pass_golds,
        cfg, judge_system_prompt,
    )

    output: list[Optional[TrainingExample]] = [None] * len(items)

    for rank, orig_idx in enumerate(passing_idx):
        scores_generated, scores_gs = judge_results[rank]
        base_weight_gen = aggregate_score(scores_generated, cfg)
        base_weight_gs = aggregate_score(scores_gs, cfg)

        score_delta = abs(base_weight_gen - base_weight_gs)
        think_bonus = (
            cfg.thinking_bonus
            if has_thinking(generated_raw[orig_idx], tokenizer, cfg.min_think_tokens)
            else 0.0
        )
        score_chosen = max(base_weight_gen, base_weight_gs)
        reward_weight = min(1.0, max(score_chosen, score_delta + think_bonus))

        dpo_prompt = build_dpo_prompt(questions[orig_idx], tokenizer)

        if base_weight_gen >= base_weight_gs:
            chosen_completion = generated_raw[orig_idx]
            rejected_completion = golds[orig_idx]
        else:
            chosen_completion = golds[orig_idx]
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

    encoded_full = []
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
    pad_id = tokenizer.pad_token_id
    batch_size = len(prompts)

    padded_ids = torch.full((batch_size, max_full_len), pad_id, dtype=torch.long)
    padded_attn = torch.zeros((batch_size, max_full_len), dtype=torch.long)
    padded_ttids = torch.zeros((batch_size, max_full_len), dtype=torch.long)

    for i, enc in enumerate(encoded_full):
        seq_len = enc["input_ids"].shape[1]
        padded_ids[i, :seq_len] = enc["input_ids"].squeeze(0)
        padded_attn[i, :seq_len] = enc["attention_mask"].squeeze(0)
        padded_ttids[i, :seq_len] = enc["token_type_ids"].squeeze(0)

    padded_ids = padded_ids.to(model.device)
    padded_attn = padded_attn.to(model.device)
    padded_ttids = padded_ttids.to(model.device)

    ctx = torch.inference_mode() if no_grad else torch.enable_grad()
    with ctx:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=padded_ids,
                attention_mask=padded_attn,
                token_type_ids=padded_ttids,
            )
            logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = padded_ids[:, 1:].contiguous().unsqueeze(-1)

    del outputs, logits

    token_logits = shift_logits.gather(2, shift_labels).squeeze(-1)
    log_z = torch.logsumexp(shift_logits, dim=-1)
    token_lp = token_logits - log_z

    del shift_logits, log_z

    results = []
    for i, enc_p in enumerate(encoded_prompt):
        prompt_len = enc_p["input_ids"].shape[1]
        seq_len = encoded_full[i]["input_ids"].shape[1]

        mask = torch.zeros(seq_len - 1, dtype=torch.bool, device=model.device)
        mask[prompt_len:] = True
        valid_lp = token_lp[i, :seq_len - 1]
        masked = valid_lp * mask.float()
        results.append(masked.sum() / mask.sum().float().clamp(min=1))

    return results


def dpo_loss_weighted_batch(
        ref_lp_chosen: list[torch.Tensor],
        ref_lp_rejected: list[torch.Tensor],
        pol_lp_chosen: list[torch.Tensor],
        pol_lp_rejected: list[torch.Tensor],
        beta: float,
        reward_weights: list[float],
) -> torch.Tensor:
    losses = []
    for rc, rr, pc, pr, w in zip(
            ref_lp_chosen, ref_lp_rejected,
            pol_lp_chosen, pol_lp_rejected,
            reward_weights,
    ):
        chosen_ratio = pc - rc
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

    encoded_full = []
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

    padded_ids = torch.full((batch_size, max_full_len), pad_id, dtype=torch.long)
    padded_attn = torch.zeros((batch_size, max_full_len), dtype=torch.long)
    padded_ttids = torch.zeros((batch_size, max_full_len), dtype=torch.long)
    labels = torch.full((batch_size, max_full_len), -100, dtype=torch.long)

    for i, (enc, p_len) in enumerate(zip(encoded_full, prompt_lengths)):
        seq_len = enc["input_ids"].shape[1]
        padded_ids[i, :seq_len] = enc["input_ids"].squeeze(0)
        padded_attn[i, :seq_len] = enc["attention_mask"].squeeze(0)
        padded_ttids[i, :seq_len] = enc["token_type_ids"].squeeze(0)
        labels[i, p_len:seq_len] = enc["input_ids"].squeeze(0)[p_len:]

    padded_ids = padded_ids.to(model.device)
    padded_attn = padded_attn.to(model.device)
    padded_ttids = padded_ttids.to(model.device)
    labels = labels.to(model.device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(
            input_ids=padded_ids,
            attention_mask=padded_attn,
            token_type_ids=padded_ttids,
            labels=labels,
        ).loss

    return loss


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
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    judge_system_prompt = build_judge_system_prompt(cfg.judge_criteria)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    placeholder_total = max(1, cfg.cosine_cycle_steps * cfg.num_epochs)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=placeholder_total,
        num_cycles=cfg.num_epochs,
    )

    global_step = 0
    running_loss = 0.0
    accum_count = 0
    for epoch in range(cfg.num_epochs):
        print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{cfg.num_epochs}\n{'=' * 60}")

        epoch_data = random.sample(raw_data, len(raw_data))
        skipped = 0
        optimizer.zero_grad()

        for batch_start in range(0, len(epoch_data), cfg.batch_size):
            batch_items = epoch_data[batch_start: batch_start + cfg.batch_size]
            print(f"\n[Epoch {epoch + 1} | Items {batch_start + 1}–{batch_start + len(batch_items)}/{len(epoch_data)}]")

            model.set_adapter("default")
            model.eval()
            examples = build_online_batch(
                batch_items, model, tokenizer, guardrail, cfg, judge_system_prompt
            )

            valid_examples = [ex for ex in examples if ex is not None]
            skipped += sum(1 for ex in examples if ex is None)

            if not valid_examples:
                print("[Batch] All items discarded by guardrail — skipping.")
                continue

            dpo_prompts = [ex.dpo_prompt for ex in valid_examples]
            chosen_completions = [ex.chosen_completion for ex in valid_examples]
            rejected_completions = [ex.rejected_completion for ex in valid_examples]
            reward_weights = [ex.reward_weight for ex in valid_examples]
            n = len(valid_examples)

            model.set_adapter("ref")
            model.eval()
            ref_lps = batched_log_probs(
                model, tokenizer,
                dpo_prompts + dpo_prompts,
                chosen_completions + rejected_completions,
                cfg.max_seq_len, no_grad=True,
            )
            ref_lp_chosen = ref_lps[:n]
            ref_lp_rejected = ref_lps[n:]

            model.set_adapter("default")
            model.train()
            pol_lps = batched_log_probs(
                model, tokenizer,
                dpo_prompts + dpo_prompts,
                chosen_completions + rejected_completions,
                cfg.max_seq_len, no_grad=False,
            )
            pol_lp_chosen = pol_lps[:n]
            pol_lp_rejected = pol_lps[n:]

            loss_dpo = dpo_loss_weighted_batch(
                ref_lp_chosen, ref_lp_rejected,
                pol_lp_chosen, pol_lp_rejected,
                beta=cfg.dpo_beta,
                reward_weights=reward_weights,
            )

            judge_prompts = [ex.judge_prompt for ex in valid_examples]
            judge_responses = [ex.judge_response for ex in valid_examples]

            loss_sft = batched_sft_loss(
                model, tokenizer,
                judge_prompts, judge_responses, cfg.max_seq_len,
            )

            loss = loss_dpo + cfg.sft_loss_weight * loss_sft

            effective_batch = len(valid_examples)
            (loss / cfg.grad_accumulation_steps).backward()

            running_loss += loss.item()
            accum_count += effective_batch

            if accum_count >= cfg.grad_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count = 0

                avg = running_loss / cfg.grad_accumulation_steps
                running_loss = 0.0
                print(
                    f"  Step {global_step:4d} | loss={avg:.4f}  dpo={loss_dpo.item():.4f}  "
                    f"sft={loss_sft.item():.4f}"
                )

                if cfg.ref_update_interval > 0 and global_step % cfg.ref_update_interval == 0:
                    sync_ref_model(model)

                if global_step % 100 == 0:
                    torch.cuda.empty_cache()

        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accum_count = 0
            print(f"  Step {global_step:4d} | flushed partial accumulation batch")

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
        batch_size=2,
        output_dir="./checkpoints",
        ref_update_interval=100,
    )
    main(cfg)