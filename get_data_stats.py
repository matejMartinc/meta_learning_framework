import json
from transformers import AutoTokenizer


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def analyze_dataset_length(path: str, model_id: str = "google/gemma-3-12b-it", threshold: int = 512):
    print(f"Loading tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = load_jsonl(path)
    total_samples = len(dataset)
    shorter_than_threshold = 0

    print(f"Processing {total_samples} samples...")

    for entry in dataset:
        # Navigate the conversations list to find the GPT response
        # Assuming the 'gpt' value is always the last 'value' in the conversations
        gpt_response = ""
        for turn in entry.get("conversations", []):
            if turn.get("from") == "human":
                gpt_response = turn.get("value", "")
                break  # Assuming one GPT response per ID for this check

        if gpt_response:
            # Tokenize and check length
            tokens = tokenizer.encode(gpt_response, add_special_tokens=False)
            if len(tokens) < threshold:
                shorter_than_threshold += 1

    percentage = (shorter_than_threshold / total_samples) * 100 if total_samples > 0 else 0

    print("\n--- Analysis Results ---")
    print(f"Total samples: {total_samples}")
    print(f"Samples shorter than {threshold} tokens: {shorter_than_threshold}")
    print(f"Percentage: {percentage:.2f}%")


if __name__ == "__main__":
    # Replace 'your_dataset.jsonl' with your actual file path
    analyze_dataset_length("data/nemotron_sft_all_final_5k_sample.jsonl")