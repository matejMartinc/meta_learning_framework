import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configuration
base_model_id = "google/gemma-3-12b-it"  # Adjust if using 2b or 27b
adapter_path = "checkpoints_meta_learning/epoch_1"
input_file = "data/gams_ft_dataset_1k_sample.jsonl"
#output_file = "gemma-3-12b-it_meta_improved_predictions.jsonl"
output_file = "gemma-3-12b-it_base.jsonl"

# 2. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.padding_side = 'left'  # Better for batch inference

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 3. Load the LoRA Adapter
#model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()


def generate_response(prompt):
    # Format according to Gemma's chat template
    messages = [{"role": "user", "content": prompt}]
    print(prompt)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            do_sample=False,  # Set to True for creative tasks
        )

    # Decode only the new tokens (skip the prompt)
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
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