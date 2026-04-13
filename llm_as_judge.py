import os
import json
import random
import google.generativeai as genai
from typing import List, Dict

# --- Configuration ---
GOOGLE_API_KEY = "API-KEY"
genai.configure(api_key=GOOGLE_API_KEY)

# List of your model result files (JSONL format)
MODEL_RESULT_FILES = [
    "results/gemma-3-12b-it_meta_improved_predictions.jsonl",
    "results/gemma-3-12b-it_sft_predictions.jsonl",
    #"results/gemma-3-12b-it_base_predictions.jsonl"
]

OUTPUT_FILE = "LLM_as_a_judge_scores.jsonl"


def load_jsonl_data(file_paths: List[str]) -> Dict[str, Dict]:
    """
    Loads multiple JSONL files and groups predictions by 'id'.
    """
    all_data = {}

    for file_path in file_paths:
        model_tag = os.path.basename(file_path).replace(".jsonl", "")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                item_id = item['id']

                # Extract prompt and gold from the conversations list
                # Assuming index 0 is human and index 1 is gpt gold
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
    """
    Creates the prompt for the Gemini API with shuffled models.
    """
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
    data_by_id = load_jsonl_data(MODEL_RESULT_FILES)

    # 2. Initialize Gemini
    # Note: Ensure 'gemini-1.5-pro' or 'gemini-2.0-flash' is used as per current availability
    model = genai.GenerativeModel('gemini-2.5-pro')

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_write:
        for item_id, content in data_by_id.items():
            # Create a list of (actual_name, prediction)
            models_to_eval = list(content['predictions'].items())

            # Shuffle the order
            random.shuffle(models_to_eval)

            # Create a mapping for this specific run: e.g., "Model 1" -> "gemma-3-12b"
            # This allows the LLM to see generic names while we keep the key
            mapping = {}
            display_responses = []
            for i, (actual_name, pred_text) in enumerate(models_to_eval):
                display_name = f"Model {i + 1}"
                mapping[display_name] = actual_name
                display_responses.append((display_name, pred_text))

            print(f"Evaluating ID: {item_id}...")

            eval_prompt = create_evaluation_prompt(
                content['prompt'],
                content['gold'],
                display_responses
            )

            try:
                response = model.generate_content(eval_prompt)

                # Clean up response
                raw_text = response.text.strip()
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()

                parsed_json = json.loads(raw_text)

                # Add metadata back so you know which model was which
                parsed_json["id"] = item_id
                parsed_json["model_mapping"] = mapping
                print(parsed_json)

                f_write.write(json.dumps(parsed_json, ensure_ascii=False) + '\n')
                f_write.flush()  # Save progress frequently

            except Exception as e:
                print(f"Error evaluating {item_id}: {e}")


if __name__ == "__main__":
    main()