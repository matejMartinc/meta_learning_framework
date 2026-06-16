import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

import json
import random
from dataclasses import dataclass, field
from typing import Optional, Any
from PIL import Image

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
    base_url=os.environ.get("VLLM_API_URL", "http://0.0.0.0:8003/v1")
)
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FrameworkConfig:
    # Model
    model_name: str = "gemma-3-12b-it"

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
    batch_size: int = 2  # smaller sub-batch for SFT + DPO training
    grad_accumulation_steps: int = 2
    warmup_steps: int = 200
    sft_loss_weight: float = 0.4
    output_dir: str = "./checkpoints"

    ref_update_interval: int = 200

    # Generation
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9

    # WandB
    wandb_project: str = "gemma-online-training"


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





def encode_image_base64(img) -> str:
    buffered = BytesIO()
    # Convert RGBA to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_answers_batch(
        questions: list[dict],
        cfg: FrameworkConfig,
) -> list[str]:
    # -----------------------------------------------------------------------
    # 1. INSTANT LORA SYNC (Save to RAM Disk)
    # -----------------------------------------------------------------------
    # 2. BATCHED VLLM GENERATION
    # -----------------------------------------------------------------------
    prompt_sys = "Provide your answer to the question below. CRITICAL: Always respond in the SAME LANGUAGE as the user's question. \nQuestion:\n"

    def fetch_vllm(q: dict) -> str:
        clean_text = q["text"].replace("<image>\n", "").replace("<image>", "")

        # Build standard OpenAI-compatible message array
        content = [{"type": "text", "text": prompt_sys + clean_text}]
        print(content[0]["text"])
        if q["image"] is not None:
            base64_image = encode_image_base64(q["image"])
            content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # By passing the absolute path to the model parameter,
        # vLLM automatically hot-loads the updated LoRA weights for this specific request!
        response = vllm_client.chat.completions.create(
            model="gemma-3-12b-it",  # <-- always the base model name
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

    # Blast requests concurrently. vLLM will batch them automatically on its end.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(questions)) as executor:
        results = list(executor.map(fetch_vllm, questions))

    return results



def judge_answers_batch(
        questions: list[dict],
        generated_answers: list[str],
        gold_answers: list[str],
        filtered_idx: list[int],
        cfg: FrameworkConfig,
        judge_system_prompt: str,
) -> list[tuple[dict, dict]]:
    num_examples = len(questions)
    results = [None] * num_examples
    filtered_set = set(filtered_idx)

    # 1. Immediately handle auto-scored (discarded) examples
    for idx in filtered_idx:
        results[idx] = (
            {k: 1 for k in cfg.judge_criteria},
            {k: 5 for k in cfg.judge_criteria}
        )

    # 2. Identify indices that actually need the LLM
    indices_to_judge = [i for i in range(num_examples)]

    # 3. Define the worker function for a single judgment
    def fetch_and_parse(idx: int):
        q_dict = questions[idx]
        gen_ans = generated_answers[idx]
        gold_ans = gold_answers[idx]

        # Randomize order to prevent LLM position bias
        if random.randint(0, 1) == 0:
            a1, a2, gen_key, gs_key = gen_ans, gold_ans, "ANSWER 1", "ANSWER 2"
        else:
            a2, a1, gen_key, gs_key = gen_ans, gold_ans, "ANSWER 2", "ANSWER 1"

        clean_text = q_dict["text"].replace("<image>\n", "").replace("<image>", "")
        user_content = f"QUESTION:\n{clean_text}\n\nANSWER 1:\n{a1}\n\nANSWER 2:\n{a2}"
        print(user_content)

        # Construct OpenAI API-style messages
        content = [{"type": "text", "text": judge_system_prompt + user_content}]
        if q_dict["image"] is not None:
            base64_image = encode_image_base64(q_dict["image"])
            content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # Step A: Call vLLM API
        try:
            response = vllm_client.chat.completions.create(
                model="gemma-3-12b-it",  # <-- always the base model name
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
            print("JUDGE WHOLE ANSWER:")
            print(raw)
            print("JUDGE PARSED ANSWER:")
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
            print("GS SCORES:", scores_gs)
            print("GENERATED SCORES:", scores_generated)

            for k in cfg.judge_criteria:
                scores_generated[k] = int(max(1, min(5, scores_generated.get(k, 1))))
                scores_gs[k] = int(max(1, min(5, scores_gs.get(k, 5))))

            return idx, True, scores_generated, scores_gs

        except Exception as e:
            return idx, False, f"Parse Error: {e} | Raw string: {raw}", None


    # 4. Blast all required requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices_to_judge)) as executor:
        futures_results = list(executor.map(fetch_and_parse, indices_to_judge))

    # 5. Unpack results and handle any failures using your original fallback logic
    for res in futures_results:
        idx, success, data1, data2 = res
        if success:
            results[idx] = (data1, data2)
        else:
            print(f"[Judge] Failed for idx {idx} — using fallback. Error: {data1}")
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
    vals = [scores.get(k, 3) for k in cfg.judge_criteria]
    return (sum(vals) / len(vals) - 1) / 4.0
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
    layer_mask_type: str
    judge_prompt: str
    judge_response: str


# ---------------------------------------------------------------------------
# Inference phase
# ---------------------------------------------------------------------------
def run_inference_phase(
        items: list[dict],
        guardrail: SemanticGuardrail,
        cfg: FrameworkConfig,
        judge_system_prompt: str,
        global_step: int,
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
                print(f"[Warning] Failed to open image {img_path}")
        else:
            img_obj = None
        questions.append({"text": q_text, "image": img_obj})
        golds.append(g_text)

    # ── Phase 1: generation ──────────────────────────────────────────────
    print(f"[Inference] Generating answers for {len(questions)} questions …")

    generated_raw = generate_answers_batch(questions, cfg)
    print(generated_raw)

    # ── Phase 2: semantic guardrail ──────────────────────────────────────
    keep_mask = guardrail.filter_batch(generated_raw, golds, cfg.min_cosine_similarity)

    filtered_idx = [i for i, keep in enumerate(keep_mask) if not keep]
    all_idx = [i for i, keep in enumerate(keep_mask)]
    for i, keep in enumerate(keep_mask):
        if not keep:
            print(f"[Guardrail] item {i} DISCARDED — too far from gold or wrong language.")

    pass_questions = [q for q in questions]
    pass_clean_gens = [c for c in generated_raw]
    pass_golds = [g for g in golds]

    # ── Phase 3: judging ─────────────────────────────────────────────────
    print(f"[Inference] Judging {len(pass_questions) - len(filtered_idx)} surviving answers …")

    judge_results = judge_answers_batch(
        pass_questions, pass_clean_gens, pass_golds, filtered_idx, cfg, judge_system_prompt
    )


    # Define the log file path specific to this GPU to prevent write collisions
    log_file_path = os.path.join(cfg.output_dir, f"generation_judge_logs.jsonl")

    for rank, orig_idx in enumerate(all_idx):
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
        # -------------------------------------------------------------------

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

        chosen_grammar = chosen_scores.get("grammar", 1)
        chosen_semantics = chosen_scores.get("semantics", 1)

        is_filtered = orig_idx in filtered_idx



# ---------------------------------------------------------------------------
# Online training loop
# ---------------------------------------------------------------------------
def train_online(
        raw_data: list[dict],
        guardrail: SemanticGuardrail,
        cfg: FrameworkConfig,
        save_every_k_steps: int = 5000,  # New parameter
) -> None:


    judge_system_prompt = build_judge_system_prompt(cfg.judge_criteria)

    global_step = 0

    for epoch in range(cfg.num_epochs):

        epoch_data = random.sample(raw_data, len(raw_data))
        skipped = 0

        for inf_start in range(0, len(epoch_data), cfg.inference_batch_size):
            inf_items = epoch_data[inf_start: inf_start + cfg.inference_batch_size]

            # ── INFERENCE PHASE ─────
            run_inference_phase(
                inf_items, guardrail, cfg, judge_system_prompt, global_step
            )


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

    print("[Init] Loading semantic guardrail...")
    guardrail = SemanticGuardrail(cfg.semantic_model_name, device="cpu")

    print(f"[Data] Reading {cfg.data_path}...")
    raw_data = load_jsonl(cfg.data_path)
    print(f"[Data] {len(raw_data)} raw examples loaded.")

    train_online(raw_data, guardrail, cfg)

if __name__ == "__main__":
    cfg = FrameworkConfig(
        model_name="gemma-3-12b-it",
        #data_path="data/nemotron_sft_all_final_5k_sample.jsonl",
        data_path = "data/train_gams_nemotron.jsonl",
        num_epochs=1,
        inference_batch_size=1,  # generate batch
        max_judge_batch_size=1,  # judge batch at a time
        batch_size=1,  # train 8 at a time (per-GPU)
        output_dir="./checkpoints_meta_learning",
        ref_update_interval=200,
        wandb_project="gemma-online-dpo-sft"
    )
    main(cfg)


