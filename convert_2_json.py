import json
import os
import random
from collections import defaultdict
import pandas as pd


def convert_format():
    output_json = open('datasets/all/llava_v1_5_mix665k_line_format.json', 'w', encoding='utf8')
    with open('datasets/all/llava_v1_5_mix665k.json', 'r') as file:
        input_data = json.load(file)
        for example in input_data:
            json.dump(example, output_json)
    output_json.close()

def convert_gams_ft_dataset():
    datasets = ['datasets/all/training_gams.jsonl', 'datasets/all/validation_gams.jsonl']
    all_data = []
    for f_path in datasets:
        with open(f_path, 'r') as file:
            l_list = file.readlines()
            for line in l_list:
                original = json.loads(line.strip())
                # Convert to desired format
                converted = {
                    "id": original["example_id"],
                    "conversations": [
                        {"from": "human", "value": original["prompt"]},
                        {"from": "gpt", "value": original["response"]}
                    ]
                }
                all_data.append(converted)


    output_json = open('datasets/all/gams_ft_dataset.json', 'w', encoding='utf8')
    for example in all_data:
        json_line = json.dumps(example, ensure_ascii=False)  # Convert dictionary to a JSON string
        output_json.write(json_line + '\n')
    output_json.close()