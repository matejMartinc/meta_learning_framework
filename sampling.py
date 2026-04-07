import random
import json

data_path = 'data/gams_ft_dataset.json'
output_path = 'data/gams_ft_dataset_1k_sample.jsonl'

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# Load the data
raw_data = load_jsonl(data_path)

# Sample the data
epoch_data = random.sample(raw_data, 1000)

# Save the sampled data to the new file
save_jsonl(epoch_data, output_path)
print(f"Successfully saved 1000 samples to {output_path}")