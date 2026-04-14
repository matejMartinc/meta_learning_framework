import json
import os
import random
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Any
from PIL import Image

from accelerate import Accelerator
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import AdamW
from langdetect import detect
from peft import LoraConfig, get_peft_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FrameworkConfig:
    # Model
    model_name: str = "google/gemma-3-12b-it"

    # Data
    data_path: str = "data/gams_ft_dataset.json"
    max_seq_len: int = (2 * 1024) + 256

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
    inference_batch_size: int = 32  # large batch for generation
    max_judge_batch_size: int = 8  # max batch size for judging
    batch_size: int = 4  # smaller sub-batch for SFT + DPO training
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
    A QUESTION
    ANSWER 1
    ANSWER 2

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
def load_models_and_processor(cfg: FrameworkConfig, accelerator: Accelerator):
    accelerator.print("[Init] Loading processor...")
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    accelerator.print("[Init] Loading base model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # 1. Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    accelerator.print("[Init] Injecting Policy & Reference Adapters...")
    # 2. Wrap model and create the first adapter ("policy")
    model = get_peft_model(base_model, lora_config, adapter_name="policy")

    # 3. Add the second adapter ("reference")
    model.add_adapter("reference", lora_config)

    # 4. Set requires_grad rules
    for name, param in model.named_parameters():
        if "reference" in name:
            param.requires_grad_(False)  # Ref adapter is frozen during training
        elif "policy" in name:
            param.requires_grad_(True)  # Policy adapter is trained
        else:
            param.requires_grad_(False)  # Base model is frozen

    return model, processor




# ---------------------------------------------------------------------------
# VRAM cleanup helper
# ---------------------------------------------------------------------------
def release_vram(accelerator: Accelerator, label: str = "") -> None:
    """Synchronise CUDA, clear the cache and run garbage collection."""
    import gc
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    # accelerator.print(f"[VRAM] [{label}] Cache cleared.") # Un-comment to trace memory


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------
def generate_answers_batch(
        model,
        processor,
        questions: list[dict],
        cfg: FrameworkConfig,
        accelerator: Accelerator,
) -> list[str]:
    prompt_sys = "Always respond in a grammatically correct manner using correct noun-adjective gender/number agreement, even if the question contains typos or other grammatically incorrect words.\nCRITICAL: Always respond in the SAME LANGUAGE as the user's question.\nProvide your answer to the question below. \nQuestion:\n"

    batch_messages = []
    for q in questions:
        clean_text = q["text"].replace("<image>\n", "").replace("<image>", "")
        if q["image"] is not None:
            msg = [
                {"role": "user",
                 "content": [{"type": "image", "image": q["image"]}, {"type": "text", "text": prompt_sys + clean_text}]}
            ]
        else:
            msg = [
                {"role": "user",
                 "content": [{"type": "text", "text": prompt_sys + clean_text}]}
            ]
        batch_messages.append(msg)

    # Set padding side to left for generation
    processor.tokenizer.padding_side = "left"

    # Process the batch
    inputs = processor.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,  # Required for batches
        return_dict=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(accelerator.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad(), torch.inference_mode():
        output_ids = accelerator.unwrap_model(model).generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    del inputs
    release_vram(accelerator, "post-generation inputs")

    results = []
    for i in range(len(questions)):
        new_ids = output_ids[i, input_len:]
        text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        results.append(text)

    del output_ids
    release_vram(accelerator, "post-generation outputs")
    return results


# ---------------------------------------------------------------------------
# Batched judging
# ---------------------------------------------------------------------------
def judge_answers_batch(
        model,
        processor,
        questions: list[dict],
        generated_answers: list[str],
        gold_answers: list[str],
        filtered_idx: list[int],
        cfg: FrameworkConfig,
        judge_system_prompt: str,
        accelerator: Accelerator,
) -> list[tuple[dict, dict]]:
    num_examples = len(questions)
    results = [None] * num_examples
    filtered_set = set(filtered_idx)

    # 1. Immediately handle auto-scored examples
    for idx in filtered_idx:
        results[idx] = (
            {k: 1 for k in cfg.judge_criteria},
            {k: 5 for k in cfg.judge_criteria}
        )

    # 2. Identify indices that actually need the LLM
    indices_to_judge = [i for i in range(num_examples) if i not in filtered_set]

    if not indices_to_judge:
        return results

    # 3. Process only the required indices in chunks
    for start_ptr in range(0, len(indices_to_judge), cfg.max_judge_batch_size):
        end_ptr = min(start_ptr + cfg.max_judge_batch_size, len(indices_to_judge))
        current_batch_indices = indices_to_judge[start_ptr:end_ptr]

        chunk_questions = [questions[i] for i in current_batch_indices]
        chunk_gen = [generated_answers[i] for i in current_batch_indices]
        chunk_gold = [gold_answers[i] for i in current_batch_indices]

        chunk_orderings = []
        for gen, gold in zip(chunk_gen, chunk_gold):
            if random.randint(0, 1) == 0:
                chunk_orderings.append((gen, gold, "ANSWER 1", "ANSWER 2"))
            else:
                chunk_orderings.append((gold, gen, "ANSWER 2", "ANSWER 1"))

        batch_messages = []
        for q_dict, (a1, a2, _, _) in zip(chunk_questions, chunk_orderings):
            clean_text = q_dict["text"].replace("<image>\n", "").replace("<image>", "")
            user_content = f"QUESTION:\n{clean_text}\n\nANSWER 1:\n{a1}\n\nANSWER 2:\n{a2}"
            if q_dict["image"] is not None:
                msg = [
                    {"role": "user",
                     "content": [{"type": "image", "image": q_dict["image"]},
                                 {"type": "text", "text": judge_system_prompt + user_content}]}
                ]
            else:
                msg = [
                    {"role": "user",
                     "content": [{"type": "text", "text": judge_system_prompt + user_content}]}
                ]
            batch_messages.append(msg)
            #accelerator.print("---------------------Generated text:\n----------------------")
            #accelerator.print(judge_system_prompt + user_content)
            #accelerator.print("------------------------------------------------------------")

        # Set padding side to left for generation
        processor.tokenizer.padding_side = "left"

        # Process the batch
        inputs = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,  # Required for batches
            truncation=True,
            max_length=cfg.max_seq_len,
            return_dict=True,
            return_tensors="pt"
        ).to(accelerator.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = accelerator.unwrap_model(model).generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        generated_tokens = output_ids[:, input_length:]
        del inputs
        release_vram(accelerator, "post-judge inputs chunk")

        for i, global_idx in enumerate(current_batch_indices):
            _, _, generated_key, gs_key = chunk_orderings[i]
            raw = processor.tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip()

            try:
                #accelerator.print('----------------------------JUDGE------------------------------')
                #accelerator.print(raw)
                #accelerator.print('----------------------------JUDGE------------------------------')
                cleaned = raw.replace("```json", "").replace("```", "").strip()
                scores = json.loads(cleaned)
                scores_generated = scores[generated_key]
                scores_gs = scores[gs_key]

                for k in cfg.judge_criteria:
                    scores_generated[k] = int(max(1, min(5, scores_generated.get(k, 1))))
                    scores_gs[k] = int(max(1, min(5, scores_gs.get(k, 5))))

                results[global_idx] = (scores_generated, scores_gs)
            except (json.JSONDecodeError, KeyError, ValueError):
                accelerator.print(f"[Judge ] Parse failed for idx {global_idx} — using fallback.")
                results[global_idx] = (
                    {k: 1 for k in cfg.judge_criteria},
                    {k: 5 for k in cfg.judge_criteria}
                )

        del output_ids
        release_vram(accelerator, "post-judge chunk complete")

    return results


def aggregate_score(scores: dict, cfg: FrameworkConfig) -> float:
    vals = [scores.get(k, 3) for k in cfg.judge_criteria]
    return (sum(vals) / len(vals) - 1) / 4.0


# ---------------------------------------------------------------------------
# Semantic guardrail (vectorised, text only)
# ---------------------------------------------------------------------------
class SemanticGuardrail:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.encoder = SentenceTransformer(model_name, device=device)

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
            try:
                same_lang = detect(generated_texts[i]) == detect(gold_texts[i])
            except:
                same_lang = False
            keep.append(sim >= threshold and same_lang)
        return keep


# ---------------------------------------------------------------------------
# Training example dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainingExample:
    question_text: str
    question_image: Optional[Any]
    dpo_prompt: str
    chosen_completion: str
    rejected_completion: str
    reward_weight: float
    judge_prompt: str
    judge_response: str


# ---------------------------------------------------------------------------
# DPO prompt builder
# ---------------------------------------------------------------------------
def build_dpo_prompt(q_dict: dict, processor: AutoProcessor) -> str:
    content_list = []
    if q_dict["image"] is not None:
        content_list.append({"type": "image"})

    clean_text = q_dict["text"].replace("<image>\n", "").replace("<image>", "")
    content_list.append({"type": "text", "text": clean_text})

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": content_list},
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Inference phase
# ---------------------------------------------------------------------------
def run_inference_phase(
        items: list[dict],
        model,
        processor,
        guardrail: SemanticGuardrail,
        cfg: FrameworkConfig,
        judge_system_prompt: str,
        accelerator: Accelerator,
) -> list[Optional[TrainingExample]]:
    base_data_dir = os.path.dirname(cfg.data_path)
    questions, golds = [], []
    for item in items:
        raw_convs = item.get("conversations", [])
        if len(raw_convs) > 0 and isinstance(raw_convs[0], list):
            convs = raw_convs[0]
        else:
            convs = raw_convs
        q_text = convs[0]["value"] if len(convs) >= 1 else ""
        g_text = convs[1]["value"] if len(convs) >= 2 else ""

        if "image" in item:
            img_path = os.path.join(base_data_dir, item.get("image"))
            try:
                img_obj = Image.open(img_path).convert("RGB")
            except Exception as e:
                accelerator.print(f"[Warning] Failed to open image {img_path}")
        else:
            img_obj = None
        questions.append({"text": q_text, "image": img_obj})
        golds.append(g_text)

    # ── Phase 1: generation ──────────────────────────────────────────────
    accelerator.print(f"[Inference] Generating answers for {len(questions)} questions …")
    model.eval()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.set_adapter("policy")

    generated_raw = generate_answers_batch(model, processor, questions, cfg, accelerator)
    #accelerator.print("---------------------Questions:\n----------------------")
    #accelerator.print(questions)
    #accelerator.print("---------------------Generated text:\n----------------------")
    #accelerator.print(generated_raw)
    #accelerator.print("---------------------GS text:\n----------------------")
    #accelerator.print(golds)
    #accelerator.print("------------------------------------------------------------")


    # ── Phase 2: semantic guardrail ──────────────────────────────────────
    keep_mask = guardrail.filter_batch(generated_raw, golds, cfg.min_cosine_similarity)

    filtered_idx = [i for i, keep in enumerate(keep_mask) if not keep]
    all_idx = [i for i, keep in enumerate(keep_mask)]
    for i, keep in enumerate(keep_mask):
        if not keep:
            accelerator.print(f"[Guardrail] item {i} DISCARDED — too far from gold or wrong language.")

    pass_questions = [q for q in questions]
    pass_clean_gens = [c for c in generated_raw]
    pass_golds = [g for g in golds]

    # ── Phase 3: judging ─────────────────────────────────────────────────
    accelerator.print(f"[Inference] Judging {len(pass_questions) - len(filtered_idx)} surviving answers …")

    judge_results = judge_answers_batch(
        model, processor,
        pass_questions, pass_clean_gens, pass_golds, filtered_idx,
        cfg, judge_system_prompt, accelerator
    )

    # ── Assemble TrainingExamples ────────────────────────────────────────
    output: list[Optional[TrainingExample]] = [None] * len(items)

    for rank, orig_idx in enumerate(all_idx):
        scores_generated, scores_gs = judge_results[rank]
        base_weight_gen = aggregate_score(scores_generated, cfg)
        base_weight_gs = aggregate_score(scores_gs, cfg)

        score_delta = abs(base_weight_gen - base_weight_gs)
        score_chosen = max(base_weight_gen, base_weight_gs)
        reward_weight = min(1.0, max(score_chosen, score_delta))

        dpo_prompt = build_dpo_prompt(questions[orig_idx], processor)

        if base_weight_gen > base_weight_gs:
            chosen_completion = generated_raw[orig_idx]
            rejected_completion = golds[orig_idx]
        else:
            chosen_completion = golds[orig_idx]
            rejected_completion = generated_raw[orig_idx]

        clean_text = questions[orig_idx]["text"].replace("<image>\n", "").replace("<image>", "")
        sft_user = (
            f"QUESTION:\n{clean_text}\n\n"
            f"ANSWER 1:\n{pass_clean_gens[rank]}\n\n"
            f"ANSWER 2:\n{golds[orig_idx]}"
        )
        content_list = []
        if questions[orig_idx]["image"] is not None:
            content_list.append({"type": "image"})
        content_list.append({"type": "text", "text": sft_user})

        judge_prompt = processor.apply_chat_template(
            [{"role": "system", "content": judge_system_prompt},
             {"role": "user", "content": content_list}],
            tokenize=False, add_generation_prompt=True,
        )

        judge_response = json.dumps(
            {"ANSWER 1": scores_generated, "ANSWER 2": scores_gs},
            ensure_ascii=False,
        )

        output[orig_idx] = TrainingExample(
            question_text=questions[orig_idx]["text"],
            question_image=questions[orig_idx]["image"],
            dpo_prompt=dpo_prompt,
            chosen_completion=chosen_completion,
            rejected_completion=rejected_completion,
            reward_weight=reward_weight,
            judge_prompt=judge_prompt,
            judge_response=judge_response,
        )

    return output


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


def compute_logprobs_and_sft(
        model,
        processor,
        prompts: list[str],
        completions: list[str],
        images_list: list[Optional[Any]],
        max_len: int,
        compute_sft: bool = False,
        no_grad: bool = False,
):
    # Determine if there are ANY images in this training batch
    has_any_image = any(img is not None for img in images_list)

    # Pad images so the length matches the text length exactly
    batch_imgs = [[img] if img is not None else [] for img in images_list]

    # Right padding naturally isolates targets at the end
    original_pad_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "right"

    enc_f_batch = processor(
        text=[p + c for p, c in zip(prompts, completions)],
        images=batch_imgs if has_any_image else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False
    ).to(model.device)

    # Find where completion starts inside the right-padded sequences
    prompt_lengths = []
    seq_lengths = []
    for i, (p, img) in enumerate(zip(prompts, images_list)):
        enc_p = processor(
            text=p,
            images=[img] if img is not None else None,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            add_special_tokens=False
        )
        prompt_lengths.append(enc_p["input_ids"].shape[1])
        seq_lengths.append(enc_f_batch["attention_mask"][i].sum().item())

    processor.tokenizer.padding_side = original_pad_side

    ctx = torch.inference_mode() if no_grad else torch.enable_grad()
    with ctx, torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(**enc_f_batch)

    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = enc_f_batch["input_ids"][:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    ce_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    ce_loss = ce_loss.view(len(prompts), -1)
    token_lp = -ce_loss

    results_lp = []
    sft_loss_total = 0.0
    sft_tokens = 0

    for i, p_len in enumerate(prompt_lengths):
        s_len = seq_lengths[i]
        completion_logprobs = token_lp[i, p_len - 1: s_len - 1]
        results_lp.append(completion_logprobs.sum())

        if compute_sft:
            sft_loss_total -= completion_logprobs.sum()
            sft_tokens += completion_logprobs.numel()

    sft_loss = (sft_loss_total / sft_tokens) if (compute_sft and sft_tokens > 0) else None

    if no_grad:
        results_lp = [lp.detach() for lp in results_lp]

    return results_lp, sft_loss


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------
def run_training_phase(
        valid_examples: list[TrainingExample],
        model,
        processor,
        cfg: FrameworkConfig,
        optimizer,
        scheduler,
        global_step: int,
        accelerator: Accelerator,
) -> int:
    sub_batches = [
        valid_examples[i: i + cfg.batch_size]
        for i in range(0, len(valid_examples), cfg.batch_size)
    ]

    accelerator.unwrap_model(model).gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    for sb_idx, sub_batch in enumerate(sub_batches):
        accelerator.print(f"  [Train sub-batch {sb_idx + 1}/{len(sub_batches)}] {len(sub_batch)} examples")

        dpo_prompts = [ex.dpo_prompt for ex in sub_batch]
        chosen_completions = [ex.chosen_completion for ex in sub_batch]
        rejected_completions = [ex.rejected_completion for ex in sub_batch]
        reward_weights = [ex.reward_weight for ex in sub_batch]
        images_list = [ex.question_image for ex in sub_batch]

        # ── Ref log-probs (no grad, ref model) ──────────────────────
        # Ensure we are working with the raw model interface for PEFT adapter switching
        unwrapped_model = accelerator.unwrap_model(model)

        # ── Ref log-probs (no grad, reference adapter) ──────────────────────
        model.eval()
        unwrapped_model.set_adapter("reference")  # Switch to Reference

        with torch.no_grad():
            ref_lp_chosen, _ = compute_logprobs_and_sft(
                model, processor, dpo_prompts, chosen_completions, images_list, cfg.max_seq_len, compute_sft=False,
                no_grad=True
            )
            ref_lp_rejected, _ = compute_logprobs_and_sft(
                model, processor, dpo_prompts, rejected_completions, images_list, cfg.max_seq_len, compute_sft=False,
                no_grad=True
            )
        release_vram(accelerator, f"sub-batch {sb_idx + 1} ref logprobs")

        # ── Policy log-probs + SFT (with grad, policy adapter) ───────
        model.train()
        unwrapped_model.set_adapter("policy")  # Switch back to Policy

        with accelerator.accumulate(model):
            pol_lp_chosen, loss_sft = compute_logprobs_and_sft(
                model, processor, dpo_prompts, chosen_completions, images_list, cfg.max_seq_len, compute_sft=True,
                no_grad=False
            )
            pol_lp_rejected, _ = compute_logprobs_and_sft(
                model, processor, dpo_prompts, rejected_completions, images_list, cfg.max_seq_len, compute_sft=False,
                no_grad=False
            )

            loss_dpo = dpo_loss_weighted_batch(
                ref_lp_chosen, ref_lp_rejected,
                pol_lp_chosen, pol_lp_rejected,
                beta=cfg.dpo_beta,
                reward_weights=reward_weights,
            )

            loss = (1 - cfg.sft_loss_weight) * loss_dpo + cfg.sft_loss_weight * loss_sft

            # Accelerate natively handles scaling loss by accumulation steps
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Only log properly when gradients actually stepped
        if accelerator.sync_gradients:
            global_step += 1
            accelerator.print(f"  Step {global_step:4d} | loss={loss.item():.4f}")

            # ── ONLINE SYNC: Copy Policy weights to Reference weights ──
            if cfg.ref_update_interval > 0 and global_step % cfg.ref_update_interval == 0:
                accelerator.wait_for_everyone()

                with torch.no_grad():
                    state_dict = unwrapped_model.state_dict()
                    for name, param in unwrapped_model.named_parameters():
                        if "reference" in name:
                            # Find the matching policy parameter name
                            policy_name = name.replace("reference", "policy")
                            if policy_name in state_dict:
                                # Copy data directly (this is completely safe under DeepSpeed ZeRO
                                # because local shard shapes match perfectly across adapters)
                                param.data.copy_(state_dict[policy_name].data)

                accelerator.print(f"[Ref] Synced policy adapter to reference adapter across GPUs.")

        del ref_lp_chosen, ref_lp_rejected
        del pol_lp_chosen, pol_lp_rejected
        del loss_dpo, loss
        release_vram(accelerator, f"sub-batch {sb_idx + 1} post-backward")

    accelerator.unwrap_model(model).gradient_checkpointing_disable()
    release_vram(accelerator, "end of training phase")

    return global_step


# ---------------------------------------------------------------------------
# Online training loop
# ---------------------------------------------------------------------------
def train_online(
        model,
        processor,
        raw_data: list[dict],
        guardrail: SemanticGuardrail,
        cfg: FrameworkConfig,
        accelerator: Accelerator,
        save_every_k_steps: int = 5000,  # New parameter
) -> None:
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)

    accelerator.unwrap_model(model).gradient_checkpointing_disable()
    judge_system_prompt = build_judge_system_prompt(cfg.judge_criteria)

    min_len = (len(raw_data) // accelerator.num_processes) * accelerator.num_processes
    raw_data = raw_data[:min_len]

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    placeholder_total = max(1, cfg.cosine_cycle_steps * cfg.num_epochs)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=placeholder_total,
        num_cycles=cfg.num_epochs,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    epoch_data = raw_data[accelerator.process_index:: accelerator.num_processes]

    global_step = 0

    for epoch in range(cfg.num_epochs):
        accelerator.print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{cfg.num_epochs}\n{'=' * 60}")

        epoch_data = random.sample(epoch_data, len(epoch_data))
        skipped = 0
        optimizer.zero_grad()

        for inf_start in range(0, len(epoch_data), cfg.inference_batch_size):
            inf_items = epoch_data[inf_start: inf_start + cfg.inference_batch_size]
            accelerator.print(
                f"\n[GPU {accelerator.process_index} Epoch {epoch + 1} | Inference items "
                f"{inf_start + 1}–{inf_start + len(inf_items)}/{len(epoch_data)}]"
            )

            # ── INFERENCE PHASE ─────
            examples = run_inference_phase(
                inf_items, model, processor, guardrail, cfg, judge_system_prompt, accelerator
            )

            valid_examples = [ex for ex in examples if ex is not None]
            skipped += sum(1 for ex in examples if ex is None)

            if not valid_examples:
                accelerator.print("[Batch] All items discarded by guardrail — skipping.")
                continue

            release_vram(accelerator, "between inference and training phases")

            # ── TRAINING PHASE ────────
            accelerator.print(
                f"[Train] {len(valid_examples)} valid examples → "
                f"{(len(valid_examples) + cfg.batch_size - 1) // cfg.batch_size} sub-batches "
                f"of up to {cfg.batch_size}"
            )

            # We track the step count before and after to see if we crossed a 'k' threshold
            previous_step = global_step

            global_step = run_training_phase(
                valid_examples, model, processor, cfg,
                optimizer, scheduler, global_step, accelerator
            )

            # ── STEP-BASED CHECKPOINT ──
            # Check if we passed a multiple of k during this training phase
            if ((global_step * cfg.batch_size) // save_every_k_steps) > ((previous_step * cfg.batch_size) // save_every_k_steps):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt = f"{cfg.output_dir}/step_{global_step}"
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                    accelerator.print(f"\n[Checkpoint] Step-based save at step {global_step} to {ckpt}")

        accelerator.print(
            f"\n[Epoch {epoch + 1}] GPU {accelerator.process_index} Skipped {skipped}/{len(epoch_data)} items.")

        # ── EPOCH-BASED CHECKPOINT ──
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt = f"{cfg.output_dir}/epoch_{epoch + 1}"
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
            accelerator.print(f"[Checkpoint] End of epoch save to {ckpt}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
     with open(path, "r", encoding="utf-8") as f:
         return [json.loads(line) for line in f]

#def load_json(path: str) -> list[dict]:
#     with open(path, encoding="utf-8") as f:
#         return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg: FrameworkConfig):
    accelerator = Accelerator()

    accelerator.print("[Init] Loading model and processor...")
    model, processor = load_models_and_processor(cfg, accelerator)

    accelerator.print("[Init] Loading semantic guardrail...")
    guardrail = SemanticGuardrail(cfg.semantic_model_name, device="cpu")

    accelerator.print(f"[Data] Reading {cfg.data_path}...")
    raw_data = load_jsonl(cfg.data_path)
    accelerator.print(f"[Data] {len(raw_data)} raw examples loaded.")

    train_online(model, processor, raw_data, guardrail, cfg, accelerator)
    accelerator.print("\n[Done] Training complete.")


if __name__ == "__main__":
    cfg = FrameworkConfig(
        model_name="google/gemma-3-12b-it",
        #data_path="data/nemotron_sft_all_final_5k_sample.jsonl",
        data_path = "data/train_gams_nemotron.jsonl",
        num_epochs=1,
        inference_batch_size=128,  # generate batch
        max_judge_batch_size=32,  # judge batch at a time
        batch_size=8,  # train 2 at a time (per-GPU)
        output_dir="./checkpoints_meta_learning",
        ref_update_interval=100,
    )
    main(cfg)

# accelerate launch --config_file accelerate_config.yaml train_deepspeed.py
