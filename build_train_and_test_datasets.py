import random
import json

data_path_1 = 'data/gams_ft_dataset.json'
data_path_2 = 'data/nemotron_sft_all_final_98k.json'
test_path = 'data/gams_ft_dataset_1k_sample.jsonl'
output_train_path = 'data/train_gams_nemotron.jsonl'

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# Load the data
test_data = load_jsonl(test_path)
test_ids = set([x['id'] for x in test_data])
print("Num test:", len(test_ids))
gams_data = load_jsonl(data_path_1)
print("Num original gams", len(gams_data))
filtered_gams = [x for x in gams_data if x['id'] not in test_ids]
print("Num filtered gams", len(filtered_gams))

nemotron_data = load_jsonl(data_path_2)
train_data = nemotron_data + filtered_gams
random.shuffle(train_data)

# Save the sampled data to the new file
save_jsonl(train_data, output_train_path)
print(f"Successfully saved to {output_train_path}")