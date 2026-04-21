import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.linalg import eigvalsh
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import json
import pandas as pd


def load_json(path: str) -> list[dict]:
     with open(path, encoding="utf-8") as f:
         return json.load(f)


def get_laplacian_features_for_layer(layer_attn_tensor, k=5):
    """
    Processes a SINGLE layer tensor.
    layer_attn_tensor shape: (batch, num_heads, seq_len, seq_len)
    """
    # Remove batch dim -> (num_heads, seq_len, seq_len)
    attn = layer_attn_tensor[0].to(torch.float32)
    num_heads, seq_len, _ = attn.shape
    device = attn.device

    # 1. Weighted Out-Degree
    nom = attn.sum(dim=-1)
    denom = torch.arange(1, seq_len + 1, device=device).flip(dims=[0])
    weighted_degree = nom / denom

    # 2. Laplacian Diagonal
    diag_a = torch.diagonal(attn, dim1=-2, dim2=-1)
    lap_diag = weighted_degree - diag_a

    # 3. Top-K
    top_vals, _ = torch.topk(lap_diag, k=min(k, seq_len), dim=-1)

    # Handle padding if seq_len < k
    if seq_len < k:
        padding = torch.zeros((num_heads, k - seq_len), device=device)
        top_vals = torch.cat([top_vals, padding], dim=-1)

    # 4. Log-Det
    log_det = lap_diag.add(1e-6).log().mean(dim=-1, keepdim=True)

    # Combine and flatten
    return torch.cat([top_vals, log_det], dim=-1).flatten()

# LLM-AS-A-JUDGE (Real Gemma Implementation)
def judge_hallucination(question, response, reference):
    """
    Uses Gemma-3-12b as a judge to compare generated text against gold standard.
    """
    prompt = f"""Evaluate if the 'Generated Answer' is a hallucination based on the 'Reference Answer'.
Question: {question}
Reference Answer: {reference}
Generated Answer: {response}

Is the Generated Answer factually inconsistent with the Reference Answer or the Question? 
Respond with exactly one word: 'YES' (for hallucination) or 'NO' (for factual)."""

    judge_messages = [{"role": "user", "content": prompt}]
    judge_inputs = tokenizer.apply_chat_template(
        judge_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(judge_inputs, max_new_tokens=2)
    input_len = judge_inputs.shape[1]
    judge_output = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).upper()

    return 1 if "YES" in judge_output else 0



# 1. SETUP MODEL & DATA
model_id = "google/gemma-3-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_attentions=True
)

#dataset = load_dataset("truthful_qa", "generation", split="validation[:50]")
dataset_train = load_json('data/llama-2-70b-chat/triviaqa_train_tp1.0_10responses_with_em_labels.json')[:10000]
dataset_test = load_json('data/llama-2-70b-chat/triviaqa_dev_tp1.0_10responses_with_em_labels.json')[:2000]

collected_data = []


print("Generating responses and extracting features...")
for item in dataset_train:

    messages = [
        {"role": "user", "content": item['question']}
    ]
    # format_as_generation_prompt adds the assistant prefix so the model knows to start
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    print("Question:", " ".join(item['question'].split()))
    if isinstance(item["answer_ground_truth"], list):
        gs = [x for x in item["answer_ground_truth"] if len(x.strip()) > 0][0]
    elif isinstance(item["answer_ground_truth"], str):
        gs = item["answer_ground_truth"].strip()
    else:
        gs = "T don't know"
    print("GS:", gs)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_attentions=True
        )

    final_step_attentions = outputs.attentions[-1]
    all_layer_features = []

    for layer_idx in range(len(final_step_attentions)):
        # final_step_attentions[layer_idx] IS the tensor (1, heads, seq, seq)
        layer_tensor = final_step_attentions[layer_idx]

        # Call the updated single-layer function
        layer_evs = get_laplacian_features_for_layer(layer_tensor, k=5)
        all_layer_features.append(layer_evs)

    # Flatten into the 9,600-dim vector
    features = torch.cat(all_layer_features).detach().cpu().numpy()
    input_len = input_ids.shape[1]
    answer = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)
    print("Answer:", " ".join(answer.split()))
    print('-----------------------------------------------')
    # Store initial data
    collected_data.append({
        "question": " ".join(item['question'].split()),
        "gold_standard": " ".join(gs.split()),
        "generated_response": " ".join(answer.split()),
        "features": features.tolist()  # Convert numpy to list for JSON compatibility
    })


print("Judging responses for ground truth labels...")
for record in collected_data:
    label = judge_hallucination(record['question'], record["generated_response"], record["gold_standard"])
    record["label"] = label

# --- SAVING DATA BEFORE TRAINING ---
# Save as JSON (best for nested lists/features)
with open("results/hallucination_data.json", "w") as f:
    json.dump(collected_data, f, indent=4)

# 4. TRAIN & TEST CLASSIFIER
# Save as CSV (best for quick viewing, flattening features)
df = pd.DataFrame(collected_data)
df.to_csv("results/hallucination_data.csv", index=False)
print(f"Successfully saved {len(collected_data)} records to JSON and CSV.")

# 4. ENHANCED TRAIN & TEST
X = np.array([d["features"] for d in collected_data])
y = np.array([d["label"] for d in collected_data])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline: Scale -> Reduce Dim -> Classify
# n_components=100 captures the most important spectral variances
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('clf', LogisticRegression(class_weight='balanced')) # 'balanced' helps if hallucinations are rare
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print(classification_report(y_test, preds))