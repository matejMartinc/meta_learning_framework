
import os
import json
import random
from typing import List, Dict

# Import vLLM and Transformers
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- Configuration ---
MODEL_RESULT_FILES = [
    "results/gemma-3-12b-it_epoch_1_debugged_meta_learning_predictions.jsonl",
    "results/gemma-3-12b-it_sft_predictions.jsonl",
    "results/gemma-3-12b-it_base_predictions.jsonl"
]

OUTPUT_FILE = "LLM_as_a_judge_scores_gemma_4_judge.jsonl"

# Hugging Face Model ID
JUDGE_MODEL_ID = "google/gemma-4-31B-it"

# How many prompts to pass to vLLM at once (to save progress incrementally)
CHUNK_SIZE = 4

def load_jsonl_data(file_paths: List[str]) -> Dict[str, Dict]:
    """ Loads multiple JSONL files and groups predictions by 'id'. """
    all_data = {}

    for file_path in file_paths:
        model_tag = os.path.basename(file_path).replace(".jsonl", "")
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                item_id = item['id'] + '_' + str(idx)

                prompt_val = item['conversations'][0]['value']
                gold_val = item['conversations'][1]['value']
                prediction = item['prediction']

                if item_id not in all_data:
                    all_data[item_id] = {
                        "prompt": prompt_val,
                        "gold": gold_val,
                        "predictions": {}
                    }

                all_data[item_id]["predictions"][model_tag] = prediction

    return all_data

def create_evaluation_prompt(original_prompt, gold_standard, shuffled_responses):
    """ Creates the text prompt with populated variables. """

    # FIXED: Added the variables into the f-string correctly
    prompt = f"""You are an expert evaluator of large language models. Your task is to act as a judge and evaluate the following model-generated responses based on a given prompt and a gold standard response.

**Prompt:**
{original_prompt}

**Gold Standard Response:**
{gold_standard}

**Model-Generated Responses:**
"""
    # shuffled_responses is a list of tuples: (display_name, text)
    for display_name, answer in shuffled_responses:
        prompt += f"- {display_name}: {answer}\n"

    prompt += """
**Criteria (Score 1-5):**
1. "grammar": Grammatical and linguistic correctness.
2. "semantics": Accuracy in meaning compared to the gold standard.
3. "flow": Readability and natural coherence.
4. "completeness": Coverage of the original prompt's intent.
5. "factuality": Absence of hallucinations.

**Output Format:**
Return ONLY a JSON object. Do not include markdown formatting like ```json.
{
  "evaluations": [
    {
      "model": "Model X",
      "scores": {"grammar": 5, "semantics": 5, "flow": 5, "completeness": 5, "factuality": 5},
      "justification": "..."
    }
  ],
  "best_model": "Model X",
  "overall_justification": "..."
}
"""
    return prompt

def main():
    # 1. Load and group data
    print("Loading data...")
    data_by_id = load_jsonl_data(MODEL_RESULT_FILES)

    # 2. Initialize vLLM Model & Tokenizer
    print(f"Loading {JUDGE_MODEL_ID} in quantized format via vLLM...")

    # Note: If you downloaded a pre-quantized AWQ model (e.g., someone/gemma-4-31B-it-AWQ),
    # change `quantization="bitsandbytes"` to `quantization="awq"` and remove `load_format="bitsandbytes"`.
    llm = LLM(
        model=JUDGE_MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        max_model_len=8192,         # Adjust based on your VRAM and prompt length
        tensor_parallel_size=1      # Increase if you are spreading the model across multiple GPUs
    )

    # Set temperature to 0.0 for deterministic, reliable JSON outputs
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1500
    )

    # Load the tokenizer to apply the correct chat template for Gemma Instruction model
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)

    # 3. Prepare all prompts in memory
    print("Preparing prompts...")
    all_prompts = []
    all_metadata = []

    for item_id, content in data_by_id.items():
        models_to_eval = list(content['predictions'].items())
        random.shuffle(models_to_eval)

        mapping = {}
        display_responses = []
        for i, (actual_name, pred_text) in enumerate(models_to_eval):
            display_name = f"Model {i + 1}"
            mapping[display_name] = actual_name
            display_responses.append((display_name, pred_text))

        # Create raw string prompt
        raw_prompt = create_evaluation_prompt(
            content['prompt'],
            content['gold'],
            display_responses
        )

        # Apply Gemma's Instruct Chat Template
        messages = [{"role": "user", "content": raw_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        all_prompts.append(formatted_prompt)
        all_metadata.append({
            "id": item_id,
            "model_mapping": mapping
        })

    # 4. Generate and save in chunks (Continuous Batching)
    print(f"Starting batched generation for {len(all_prompts)} total items...")

    # Using 'a' (append) so that if we restart the script, we can keep previous records.
    # Note: If you stop/restart, you might want to add logic to skip already processed IDs.
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_write:

        # Process in chunks to flush to disk incrementally
        for i in range(0, len(all_prompts), CHUNK_SIZE):
            chunk_prompts = all_prompts[i : i + CHUNK_SIZE]
            chunk_metadata = all_metadata[i : i + CHUNK_SIZE]

            print(f"Processing chunk {i} to {i + len(chunk_prompts)}...")

            # vLLM automatically handles the batching logic inside this call
            outputs = llm.generate(chunk_prompts, sampling_params)

            for output, meta in zip(outputs, chunk_metadata):
                generated_text = output.outputs[0].text.strip()

                # Clean up markdown if the model hallucinated ```json anyway
                if "```json" in generated_text:
                    generated_text = generated_text.split("```json")[1].split("```")[0].strip()
                elif "```" in generated_text:
                    generated_text = generated_text.split("```")[1].strip()

                try:
                    parsed_json = json.loads(generated_text)
                    parsed_json["id"] = meta["id"]
                    parsed_json["model_mapping"] = meta["model_mapping"]

                    f_write.write(json.dumps(parsed_json, ensure_ascii=False) + '\n')
                except json.JSONDecodeError:
                    print(f"\n[Error] Failed to parse JSON for ID: {meta['id']}")
                    print(f"Raw Output: {generated_text}\n")

            f_write.flush()  # Save progress for this chunk

    print("Evaluation Complete!")

if __name__ == "__main__":
    main()