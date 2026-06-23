import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import random
import dataclasses
import torch
import torch.nn.functional as F
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

import base64
from io import BytesIO
import concurrent.futures
from openai import OpenAI
import re


vllm_client = OpenAI(
    api_key="EMPTY",
    base_url=os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")
)
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
    semantic_model_name: str = "intfloat/multilingual-e5-large"
    min_cosine_similarity: float = 0.70  # e5-large requires a higher threshold (e.g., 0.70 - 0.85)

    # Judge criteria — drives the dynamic system prompt
    judge_criteria: list = field(default_factory=lambda: [
        "grammar",
        "semantics",
        "flow",
        "completeness",
        "clarity",
    ])

    # Training
    orpo_beta: float = 0.2
    learning_rate: float = 2e-5
    num_epochs: int = 1
    inference_batch_size: int = 32  # large batch for generation
    max_judge_batch_size: int = 8  # max batch size for judging
    batch_size: int = 2  # smaller sub-batch for SFT + DPO training
    grad_accumulation_steps: int = 2
    warmup_steps: int = 200
    sft_loss_weight: float = 0.4
    output_dir: str = "./checkpoints"

    ref_update_interval: int = 200

    # Generation
    max_new_tokens: int = 1540
    temperature: float = 0.3
    top_p: float = 0.95

    # WandB
    wandb_project: str = "gemma-online-training"

def is_slovenian(text: str) -> bool:
    if not text or not isinstance(text, str) or not text.strip():
        return False
    try:
        return detect(text) == 'sl'
    except:
        return False

def build_general_judge_system_prompt(criteria: list[str]) -> str:
    criteria_keys = ", ".join(f'"{c}": X' for c in criteria)
    return f"""\
You are an expert, ruthlessly strict proofreader and linguist. Your job is to catch subtle AI generation or translation errors that most people miss.

WARNING: Do not just read for general meaning. You MUST read for exact grammatical correctness, natural flow, and morphological accuracy in the language of the prompt.

You will be given:
A QUESTION
ANSWER 1
ANSWER 2

You must evaluate both answers.

Before scoring, you must complete an analysis checklist.

CHECKLIST OF COMMON AI ERRORS TO HUNT FOR:
1. Grammar/Syntax: Look closely at verb conjugations, noun cases/plurals, and article/particle usage.
2. Literal/Clunky Translations: Phrases that sound like they were machine-translated and do not sound native to the language.
3. Hallucinated Vocabulary: Words that do not exist or are completely misused in the given context.
4. Coherence/Flow: Stilted, robotic sentences that lack natural cadence.

STEP 1: GRAMMAR AND SYNTAX EXTRACTION
For EACH answer, explicitly list at least 2-3 suspicious or grammatically incorrect phrases. If you think the text is perfect, look harder. Break down why the grammar is wrong or why it sounds unnatural.

STEP 2: JUSTIFICATION
Write a harsh critique of the answers based on your Step 1 extraction. Penalize errors heavily. 1 error = max score of 4. 3+ errors = max score of 2.

STEP 3: JSON SCORES
Score 1-5 for: grammar, semantics, flow, completeness, clarity.
Structure your final JSON exactly like this inside markdown fences:
```json
{{
    "ANSWER 1": {{{criteria_keys}}},
    "ANSWER 2": {{{criteria_keys}}}
}}
```\n"""
# ---------------------------------------------------------------------------
# Dynamic judge system prompt
# ---------------------------------------------------------------------------
def build_judge_system_prompt(criteria: list[str]) -> str:
    criteria_keys = ", ".join(f'"{c}": X' for c in criteria)
    return f"""\
You are an expert, ruthlessly strict proofreader and linguist for the Slovenian language. Your job is to catch subtle AI translation errors that most people miss.

WARNING: Do not read for general meaning. You MUST read for exact morphological correctness (sklanjatev, spreganje, ujemanje v spolu in številu). 

You will be given:
A QUESTION
ANSWER 1
ANSWER 2

You must evaluate both answers.

Before scoring, you must complete an analysis checklist.

CHECKLIST OF COMMON AI ERRORS IN SLOVENIAN TO HUNT FOR:
1. Case/Gender/Number Mismatch: Look closely at adjectives and nouns. (e.g., "pomitega krožnike" is wrong, it must be "pomite krožnike").
2. Plural vs. Singular: Check words that are 'plurale tantum' (e.g., "prsi" is plural, not singular).
3. Croatian/Serbian Interference: AI often uses "celere" instead of "zelena", "šargarepa" instead of "korenje", or "juha" structure errors.
4. Literal English Calques (Anglicisms): "Nizek na ogljikove hidrate" (Low in carbs) is terrible Slovenian. It should be "z malo ogljikovimi hidrati". 
5. Hallucinated Vocabulary: Words that sound Slovenian but don't exist in this context (e.g., "rocajte" is wrong, "samodržavja" is wrong).

STEP 1: GRAMMAR AND SYNTAX EXTRACTION
For EACH answer, explicitly list at least 3 suspicious or grammatically incorrect phrases. If you think the text is perfect, look harder. Break down why the case (sklad) or gender (spol) is wrong, or why it sounds like an English translation.

STEP 2: JUSTIFICATION
Write a harsh critique of the answers based on your Step 1 extraction. Penalize errors heavily. 1 error = max score of 4. 3+ errors = max score of 2.

STEP 3: JSON SCORES
Score 1-5 for: grammar, semantics, flow, completeness, clarity.
Structure your final JSON exactly like this inside markdown fences:
{{
    "ANSWER 1": {{{criteria_keys}}},
    "ANSWER 2": {{{criteria_keys}}}
}}\n"""

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
        attn_implementation="flash_attention_2",
        device_map={"": accelerator.local_process_index}
    )
    base_model.enable_input_require_grads()

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
    accelerator.print("[Init] Syncing reference adapter to initial policy weights...")
    state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if "reference" in name:
            policy_name = name.replace("reference", "policy")
            if policy_name in state_dict:
                param.data.copy_(state_dict[policy_name].data)

    max_layer = 0
    model.layer_grad_scales = {}

    # 3. Set requires_grad rules
    for name, param in model.named_parameters():
        if "reference" in name:
            param.requires_grad_(False)
        elif "policy" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    model.max_layer_idx = max_layer
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


def encode_image_base64(img) -> str:
    buffered = BytesIO()
    # Convert RGBA to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_answers_batch(
        model,
        processor,
        questions: list[dict],
        cfg: FrameworkConfig,
        accelerator: Accelerator,
) -> list[str]:
    # -----------------------------------------------------------------------
    # 1. INSTANT LORA SYNC (Save to RAM Disk)
    # -----------------------------------------------------------------------
    policy_lora_path = "/dev/shm/policy_lora"
    reference_lora_path = "/dev/shm/reference_lora"

    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)

    # Only the main process needs to save the adapter
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        # Ensure we are saving the actively training policy adapter
        unwrapped_model.save_pretrained(policy_lora_path, selected_adapters=["policy"], state_dict=state_dict)
        unwrapped_model.save_pretrained(reference_lora_path, selected_adapters=["reference"], state_dict=state_dict)

    # Force all GPUs to wait until the main process finishes writing to RAM disk
    accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # 2. BATCHED VLLM GENERATION
    # -----------------------------------------------------------------------
    prompt_sys = "Provide your answer to the question below. CRITICAL: Always respond in the SAME LANGUAGE as the user's question. \nQuestion:\n"

    def fetch_vllm(q: dict) -> str:
        clean_text = q["text"].replace("<image>\n", "").replace("<image>", "")

        # Build standard OpenAI-compatible message array
        content = [{"type": "text", "text": prompt_sys + clean_text}]
        if q["image"] is not None:
            base64_image = encode_image_base64(q["image"])
            content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # By passing the absolute path to the model parameter,
        # vLLM automatically hot-loads the updated LoRA weights for this specific request!
        response = vllm_client.chat.completions.create(
            model="google/gemma-3-12b-it",  # <-- always the base model name
            messages=[{"role": "user", "content": content}],
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            extra_body={
                "lora_request": {
                    "lora_name": "policy",
                    "lora_int_id": 1,
                    "lora_local_path": "/dev/shm/policy_lora"
                }
            }
        )
        return response.choices[0].message.content.strip()

    accelerator.print(f"[vLLM] Sending {len(questions)} requests to vLLM server...")

    # Blast requests concurrently. vLLM will batch them automatically on its end.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(questions)) as executor:
        results = list(executor.map(fetch_vllm, questions))

    return results



def judge_answers_batch(
        model,  # Kept for API compatibility with run_inference_phase
        processor,  # Kept for API compatibility
        questions: list[dict],
        generated_answers: list[str],
        gold_answers: list[str],
        filtered_idx: set[int],
        non_sl_idx: set[int],
        cfg: FrameworkConfig,
        sl_judge_system_prompt: str,
        general_judge_system_prompt: str,
        accelerator: Accelerator,
) -> list[tuple[dict, dict]]:
    num_examples = len(questions)
    results = [None] * num_examples


    indices_to_judge = list(range(num_examples))

    # 3. Define the worker function for a single judgment
    def fetch_and_parse(idx: int):
        q_dict = questions[idx]
        gen_ans = generated_answers[idx]
        gold_ans = gold_answers[idx]

        # Select the correct prompt based on language
        current_prompt = general_judge_system_prompt if idx in non_sl_idx else sl_judge_system_prompt

        # Randomize order to prevent LLM position bias
        if random.randint(0, 1) == 0:
            a1, a2, gen_key, gs_key = gen_ans, gold_ans, "ANSWER 1", "ANSWER 2"
        else:
            a2, a1, gen_key, gs_key = gen_ans, gold_ans, "ANSWER 2", "ANSWER 1"

        clean_text = q_dict["text"].replace("<image>\n", "").replace("<image>", "")
        user_content = f"QUESTION:\n{clean_text}\n\nANSWER 1:\n{a1}\n\nANSWER 2:\n{a2}"

        # Construct OpenAI API-style messages
        content = [{"type": "text", "text": current_prompt + user_content}]
        if q_dict["image"] is not None:
            base64_image = encode_image_base64(q_dict["image"])
            content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # Step A: Call vLLM API
        try:
            response = vllm_client.chat.completions.create(
                model="google/gemma-3-12b-it",  # <-- always the base model name
                messages=[{"role": "user", "content": content}],
                max_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                extra_body={
                    "lora_request": {
                        "lora_name": "reference",
                        "lora_int_id": 2,
                        "lora_local_path": "/dev/shm/reference_lora"
                    }
                }
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            return idx, False, str(e), None

        # Step B: Parse JSON Output
        try:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
            if match:
                json_string = match.group(1)
            else:
                start_idx = raw.find('{')
                end_idx = raw.rfind('}')
                json_string = raw[start_idx:end_idx + 1]
            scores = json.loads(json_string)
            if idx in filtered_idx:
                scores_generated = {k: 1 for k in cfg.judge_criteria}
            else:
                scores_generated = scores[gen_key]
            scores_gs = scores[gs_key]

            for k in cfg.judge_criteria:
                scores_generated[k] = int(max(1, min(5, scores_generated.get(k, 1))))
                scores_gs[k] = int(max(1, min(5, scores_gs.get(k, 5))))

            return idx, True, scores_generated, scores_gs

        except Exception as e:
            return idx, False, f"Parse Error: {e} | Raw string: {raw}", None

    accelerator.print(f"[Judge] Sending {len(indices_to_judge)} requests to vLLM server...")

    # 4. Blast all required requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices_to_judge)) as executor:
        futures_results = list(executor.map(fetch_and_parse, indices_to_judge))

    # 5. Unpack results and handle any failures using your original fallback logic
    for res in futures_results:
        idx, success, data1, data2 = res
        if success:
            results[idx] = (data1, data2)
        else:
            accelerator.print(f"[Judge] Failed for idx {idx} — using fallback. Error: {data1}")
            results[idx] = (
                {k: 1 for k in cfg.judge_criteria},
                {k: 5 for k in cfg.judge_criteria}
            )

    return results

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

def aggregate_score(scores: dict, cfg: FrameworkConfig) -> float:
    # 1. STRICT VETO: If any single criterion is <= 2, the entire answer fails.
    for k in cfg.judge_criteria:
        if scores.get(k, 3) <= 2:
            return 0.1

    # 2. EQUAL AGGREGATION: If it survives the veto, average all scores equally.
    vals = [scores.get(k, 3) for k in cfg.judge_criteria]
    mean_score = sum(vals) / len(vals)

    # Normalize from a 1-5 scale to a 0.0-1.0 scale
    return (mean_score - 1) / 4.0

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
    sft_weight: float
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
        accelerator: Accelerator,
        global_step: int,
) -> list[Optional[TrainingExample]]:
    base_data_dir = os.path.dirname(cfg.data_path)
    sl_judge_system_prompt = build_judge_system_prompt(cfg.judge_criteria)
    general_judge_system_prompt = build_general_judge_system_prompt(cfg.judge_criteria)
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
    non_sl_idx = set()
    for i, (q_dict, g_text) in enumerate(zip(questions, golds)):
        if not is_slovenian(q_dict["text"]) and not is_slovenian(g_text):
            non_sl_idx.add(i)
    # ── Phase 2: semantic guardrail ──────────────────────────────────────
    keep_mask = guardrail.filter_batch(generated_raw, golds, cfg.min_cosine_similarity)

    filtered_idx = set([i for i, keep in enumerate(keep_mask) if not keep])

    for i in filtered_idx:
        accelerator.print(f"[Guardrail] item {i} DISCARDED — too far from gold. (Will still judge GS)")

    accelerator.print(
        f"[Lang Check] Found {len(non_sl_idx)} non-Slovenian items. Using general judge prompt for these.")

    pass_questions = [q for q in questions]
    pass_clean_gens = [c for c in generated_raw]
    pass_golds = [g for g in golds]

    # ── Phase 3: judging ─────────────────────────────────────────────────
    accelerator.print(f"[Inference] Judging {len(pass_questions) - len(filtered_idx)} surviving answers …")

    # Pass everything to the judge, with the indices routing the prompts
    judge_results = judge_answers_batch(
        model, processor,
        pass_questions, pass_clean_gens, pass_golds,
        filtered_idx=filtered_idx,
        non_sl_idx=non_sl_idx,
        cfg=cfg,
        sl_judge_system_prompt=sl_judge_system_prompt,
        general_judge_system_prompt=general_judge_system_prompt,
        accelerator=accelerator
    )

    # ── Assemble TrainingExamples ────────────────────────────────────────
    output: list[Optional[TrainingExample]] = [None] * len(items)

    # Define the log file path specific to this GPU to prevent write collisions
    log_file_path = os.path.join(cfg.output_dir, f"generation_judge_logs_gpu{accelerator.process_index}.jsonl")

    for rank, orig_idx in enumerate(range(len(questions))):
        scores_generated, scores_gs = judge_results[rank]
        item_id = items[orig_idx].get("id", f"item_{orig_idx}")
        log_record = {
            "id": item_id,
            "step": global_step,
            "question": questions[orig_idx]["text"],
            "gs_answer": golds[orig_idx],
            "generated_answer": pass_clean_gens[rank],
            "gs_score": scores_gs,
            "generated_score": scores_generated
        }

        # Append as a single line
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        is_non_sl = orig_idx in non_sl_idx

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
        if is_non_sl:
            judge_system_prompt = general_judge_system_prompt
        else:
            judge_system_prompt = sl_judge_system_prompt
        judge_prompt = processor.apply_chat_template(
            [{"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": content_list}],
            tokenize = False, add_generation_prompt = True,)

        judge_response = json.dumps(
            {"ANSWER 1": scores_generated,
             "ANSWER 2": scores_gs},
            ensure_ascii = False,)

        base_weight_gen = aggregate_score(scores_generated, cfg)
        base_weight_gs = aggregate_score(scores_gs, cfg)

        score_delta = abs(base_weight_gen - base_weight_gs)

        if base_weight_gen > base_weight_gs:
            chosen_completion = pass_clean_gens[rank]
            rejected_completion = pass_golds[rank]
            chosen_scores = scores_generated
        else:
            chosen_completion = pass_golds[rank]
            rejected_completion = pass_clean_gens[rank]
            chosen_scores = scores_gs

        chosen_grammar = chosen_scores.get("grammar", 1)
        chosen_semantics = chosen_scores.get("semantics", 1)

        # 1. DPO Weight: If both chosen and rejected are terribly flawed, ignore it.
        if chosen_grammar <= 2 and chosen_semantics <= 2:
            reward_weight = 0.0
        else:
            reward_weight = 1.0 if score_delta > 0.1 else 0.5

        # 2. SFT Weight: STRICT grammar filter.
        # If the chosen answer has bad grammar (< 4), DO NOT do SFT on it.
        if chosen_grammar >= 4 and chosen_semantics >= 4:
            sft_weight = min(1.0, score_delta)
        else:
            sft_weight = 0.0
        dpo_prompt = build_dpo_prompt(questions[orig_idx], processor)

        # We always populate the example (even if weights are 0.0) to prevent Distributed Deadlocks
        output[orig_idx] = TrainingExample(
            question_text=questions[orig_idx]["text"],
            question_image=questions[orig_idx]["image"],
            dpo_prompt=dpo_prompt,
            chosen_completion=chosen_completion,
            rejected_completion=rejected_completion,
            reward_weight=reward_weight,
            sft_weight=sft_weight,
            judge_prompt=judge_prompt,
            judge_response = judge_response,
        )

    return output


def orpo_loss_weighted_batch(
        pol_lp_chosen: list[torch.Tensor],
        pol_lp_rejected: list[torch.Tensor],
        reward_weights: list[float],
) -> torch.Tensor:
    losses = []
    for pc, pr, w in zip(pol_lp_chosen, pol_lp_rejected, reward_weights):
        # 1. Cast to float32 to prevent bfloat16 underflow/overflow.
        # In bfloat16, exp(-1e-5) rounds to exactly 1.0, causing log1p to return -inf.
        pc = pc.float()
        pr = pr.float()

        # 2. Clamp to -1e-3 to prevent the derivative from exploding.
        # At -1e-5, the gradient multiplier is 10^5, which causes weights to explode.
        pc = torch.clamp(pc, max=-1e-3)
        pr = torch.clamp(pr, max=-1e-3)

        # Chosen log odds
        log_odds_chosen = pc - torch.log1p(-torch.exp(pc))
        # Rejected log odds
        log_odds_rejected = pr - torch.log1p(-torch.exp(pr))

        # Log Odds Ratio
        log_odds_ratio = log_odds_chosen - log_odds_rejected

        # ORPO preference loss
        loss = -F.logsigmoid(log_odds_ratio)
        losses.append(loss * w)

    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def compute_logprobs_and_sft(
        model,
        processor,
        prompts: list[str],
        chosen_completions: list[str],
        rejected_completions: list[str],
        images_list: list[Optional[Any]],
        max_len: int,
        compute_sft: bool = False,
        no_grad: bool = False,
):
    """
    Compute log-probs for chosen AND rejected in a single forward pass each,
    but sequentially (not batched together) to keep peak memory flat.
    Chosen and rejected share the same prompt, so we encode them separately
    but avoid redundant prompt re-encoding.
    """
    batch_size = len(prompts)
    has_any_image = any(img is not None for img in images_list)
    batch_imgs = [[img] if img is not None else [] for img in images_list]

    original_pad_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "right"

    enc_p_batched = processor(
        text=prompts,
        images=batch_imgs if has_any_image else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )
    p_lens = [int(mask.sum().item()) for mask in enc_p_batched["attention_mask"]]
    del enc_p_batched


    def _single_forward(completions, is_chosen):
        # Dynamically fetch the model's correct EOS token (e.g., "<end_of_turn>" for Gemma)
        eos = processor.tokenizer.eos_token
        formatted_texts = [p + c + eos for p, c in zip(prompts, completions)]

        enc = processor(
            text=formatted_texts,
            images=batch_imgs if has_any_image else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        ).to(model.device)

        s_lens = [int(mask.sum().item()) for mask in enc["attention_mask"]]

        ctx = torch.inference_mode() if no_grad else torch.enable_grad()
        with ctx, torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**enc)

        # Shift once, then slice per-example to avoid holding full logits
        # We process token-by-token in the vocab dimension only for the
        # completion slice, so peak memory stays bounded.
        shift_logits = outputs.logits[:, :-1]
        shift_labels = enc["input_ids"][:, 1:]
        del outputs, enc  # free ASAP before per-example loop

        lp_list, sft_list = [], []
        for i, p_len in enumerate(p_lens):
            s_len = s_lens[i]
            # Explicitly append float32 zeros if completion is empty
            if s_len <= p_len:
                lp_list.append(torch.tensor(0.0, device=shift_logits.device, dtype=torch.float32))
                if compute_sft and is_chosen:
                    sft_list.append(torch.tensor(0.0, device=shift_logits.device, dtype=torch.float32))
                continue

            # Only slice the completion window
            ex_logits = shift_logits[i, p_len - 1: s_len - 1]  # [T_comp, V]
            ex_labels = shift_labels[i, p_len - 1: s_len - 1]  # [T_comp]

            # Explicitly cast ex_logits to float32 to prevent internal bfloat16 overflow
            # during log-sum-exp over Gemma's massive 256,000 token vocabulary
            ex_ce = F.cross_entropy(ex_logits.float(), ex_labels, reduction="none")

            mean_logprob = (-ex_ce).mean()
            lp_list.append(mean_logprob)
            if compute_sft and is_chosen:
                sft_list.append(ex_ce.mean())

        del shift_logits, shift_labels

        if no_grad:
            lp_list = [lp.detach() for lp in lp_list]

        sft_tensor = torch.stack(sft_list) if (compute_sft and is_chosen) else None
        return lp_list, sft_tensor

    # ── Run chosen, then rejected sequentially ────────────────────────────
    lp_chosen, sft_tensor = _single_forward(chosen_completions, is_chosen=True)
    lp_rejected, _ = _single_forward(rejected_completions, is_chosen=False)

    processor.tokenizer.padding_side = original_pad_side
    return lp_chosen, lp_rejected, sft_tensor


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
        valid_examples[i : i + cfg.batch_size]
        for i in range(0, len(valid_examples), cfg.batch_size)
    ]

    accelerator.unwrap_model(model).gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    accumulated_loss_orpo = 0.0
    accumulated_loss_sft = 0.0
    accumulated_rw = 0.0
    accumulated_steps = 0

    model.train()
    accelerator.unwrap_model(model).set_adapter("policy")

    for sb_idx, sub_batch in enumerate(sub_batches):
        accelerator.print(
            f"  [Train sub-batch {sb_idx + 1}/{len(sub_batches)}] {len(sub_batch)} examples"
        )
        unwrapped_model = accelerator.unwrap_model(model)
        dpo_prompts = [ex.dpo_prompt for ex in sub_batch]
        chosen_completions = [ex.chosen_completion for ex in sub_batch]
        rejected_completions = [ex.rejected_completion for ex in sub_batch]
        reward_weights = [ex.reward_weight for ex in sub_batch]
        sft_weights = [ex.sft_weight for ex in sub_batch]
        images_list = [ex.question_image for ex in sub_batch]

        # ── 1. Policy log-probs + SFT (NO REFERENCE PASS NEEDED!) ────────
        with accelerator.accumulate(model):
            pol_lp_chosen, pol_lp_rejected, sft_losses_per_ex = (
                compute_logprobs_and_sft(
                    model, processor,
                    dpo_prompts, chosen_completions, rejected_completions,
                    images_list, cfg.max_seq_len,
                    compute_sft=True, no_grad=False,
                )
            )

            # Calculate ORPO Preference Loss (lambda is often 0.1)
            loss_orpo_pref = orpo_loss_weighted_batch(
                pol_lp_chosen, pol_lp_rejected, reward_weights
            )

            # Calculate SFT Loss
            sft_w_tensor = torch.tensor(sft_weights, device=sft_losses_per_ex.device, dtype=sft_losses_per_ex.dtype)
            loss_sft = (sft_losses_per_ex * sft_w_tensor).mean()

            # Final ORPO combined loss (SFT is usually weighted higher in ORPO)
            loss = loss_sft + (cfg.orpo_beta * loss_orpo_pref)

            accelerator.backward(loss)

            accumulated_loss_orpo += loss_orpo_pref.item()
            accumulated_loss_sft += loss_sft.item()
            accumulated_rw += sum(reward_weights) / len(reward_weights)
            accumulated_steps += 1

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            scheduler.step()
            avg_orpo = accumulated_loss_orpo / accumulated_steps
            avg_sft = accumulated_loss_sft / accumulated_steps
            avg_rw = accumulated_rw / accumulated_steps

            accelerator.log({
                "train/loss_total": loss.item(),
                "train/loss_orpo_pref": avg_orpo,
                "train/loss_sft": avg_sft,
                "train/reward_weight": avg_rw,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/global_step": global_step,
            }, step=global_step)

            accumulated_loss_orpo = 0.0
            accumulated_loss_sft = 0.0
            accumulated_rw = 0.0
            accumulated_steps = 0

            accelerator.print(
                f"  Step {global_step:4d} | loss={loss.item():.4f} | LR={scheduler.get_last_lr()[0]:.8f}"
                f"| orpo={avg_orpo:.4f} | sft={avg_sft:.4f} | rw={avg_rw:.4f}"
            )

            # ── Online ref sync ──────────────────────────────────────────
            if cfg.ref_update_interval > 0 and global_step % cfg.ref_update_interval == 0:
                accelerator.wait_for_everyone()
                with torch.no_grad():
                    state_dict = unwrapped_model.state_dict()
                    for name, param in unwrapped_model.named_parameters():
                        if "reference" in name:
                            policy_name = name.replace("reference", "policy")
                            if policy_name in state_dict:
                                param.data.copy_(state_dict[policy_name].data)
                accelerator.print(
                    f"[Ref] Synced policy → reference at step {global_step}."
                )

        del pol_lp_chosen, pol_lp_rejected, loss_orpo_pref, loss_sft, loss

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

    min_len = (len(raw_data) // accelerator.num_processes) * accelerator.num_processes
    raw_data = raw_data[:min_len]

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    items_per_gpu = len(raw_data) // accelerator.num_processes
    max_steps_per_epoch = items_per_gpu // cfg.batch_size
    total_expected_steps = max_steps_per_epoch * cfg.num_epochs
    total_expected_steps = total_expected_steps // cfg.grad_accumulation_steps

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        # Multiply by num_processes to offset the internal division
        num_warmup_steps=cfg.warmup_steps * accelerator.num_processes,
        num_training_steps=max(1, total_expected_steps * accelerator.num_processes),
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
                inf_items, model, processor, guardrail, cfg, accelerator, global_step
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
                state_dict = accelerator.get_state_dict(model)
                if accelerator.is_main_process:
                    ckpt = f"{cfg.output_dir}/step_{global_step}"
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.set_adapter("policy")
                    unwrapped_model.save_pretrained(ckpt, selected_adapters=["policy"], state_dict=state_dict)
                    processor.save_pretrained(ckpt)
                    accelerator.print(f"\n[Checkpoint] Step-based save at step {global_step} to {ckpt}")

        accelerator.print(
            f"\n[Epoch {epoch + 1}] GPU {accelerator.process_index} Skipped {skipped}/{len(epoch_data)} items.")

        # ── EPOCH-BASED CHECKPOINT ──
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            ckpt = f"{cfg.output_dir}/epoch_{epoch + 1}"
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.set_adapter("policy")
            unwrapped_model.save_pretrained(ckpt, selected_adapters=["policy"], state_dict=state_dict)
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
    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=cfg.grad_accumulation_steps)

    # Initialize trackers if on main process
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.wandb_project,
            config=dataclasses.asdict(cfg)  # Easily log all parameters
        )

    accelerator.print("[Init] Loading model and processor...")
    model, processor = load_models_and_processor(cfg, accelerator)

    accelerator.print("[Init] Loading semantic guardrail...")
    guardrail = SemanticGuardrail(cfg.semantic_model_name, device="cpu")

    accelerator.print(f"[Data] Reading {cfg.data_path}...")
    raw_data = load_jsonl(cfg.data_path)
    accelerator.print(f"[Data] {len(raw_data)} raw examples loaded.")

    train_online(model, processor, raw_data, guardrail, cfg, accelerator)

    # <--- End Tracking nicely
    if accelerator.is_main_process:
        accelerator.end_training()
    accelerator.print("\n[Done] Training complete.")


if __name__ == "__main__":
    cfg = FrameworkConfig(
        model_name="google/gemma-3-12b-it",
        #data_path="data/nemotron_sft_all_final_5k_sample.jsonl",
        data_path = "data/train_gams_nemotron.jsonl",
        num_epochs=1,
        inference_batch_size=96,  # generate batch
        max_judge_batch_size=48,  # judge batch at a time
        batch_size=8,  # train 8 at a time (per-GPU)
        output_dir="./checkpoints_meta_learning",
        ref_update_interval=200,
        wandb_project="gemma-online-dpo-sft"
    )
    main(cfg)

# accelerate launch --config_file accelerate_config.yaml train_deepspeed.py
