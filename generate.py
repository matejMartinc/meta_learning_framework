import json
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# 1. Configuration
base_model_id = "google/gemma-3-12b-it"
adapter_path = "checkpoints_meta_learning/epoch_1_debugged_meta_learning"
input_file = "data/gams_ft_dataset_1k_sample.jsonl"
output_file = "gemma-3-12b-it_epoch_1_debugged_meta_learning_predictions.jsonl"

# 2. Load Processor and Model (Match your training script!)
processor = AutoProcessor.from_pretrained(base_model_id)
processor.tokenizer.padding_side = 'left'

base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 3. Load the LoRA Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()


def generate_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    print(prompt)

    # 1. Build the formatted string using the chat template
    formatted_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False  # Explicitly tell it to just return the string
    )

    # 2. Pass the string into the processor to get the tensors
    inputs = processor(
        text=formatted_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    # Decode only the new tokens (skip the prompt)
    prompt_length = inputs["input_ids"].shape[-1]
    response = processor.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    print(response)
    print('-------------------------------------')
    return response.strip()


# 4. Process Dataset
results = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # Extract the human prompt from the first turn
        human_prompt = data['conversations'][0]['value']

        print(f"Processing ID: {data['id']}...")
        prediction = generate_response(human_prompt)

        # Append prediction to the original data structure
        data['prediction'] = prediction
        results.append(data)

# 5. Save Results
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in results:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Done! Results saved to {output_file}")