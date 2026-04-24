import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import json
import pandas as pd
import ast


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_laplacian_features_for_layer(layer_attn_tensor, k=5):
    """
    Processes a SINGLE layer tensor.
    layer_attn_tensor shape: (batch, num_heads, seq_len, seq_len)
    """
    attn = layer_attn_tensor[0].to(torch.float32)
    num_heads, seq_len, _ = attn.shape
    device = attn.device

    nom = attn.sum(dim=-1)
    denom = torch.arange(1, seq_len + 1, device=device).flip(dims=[0])
    weighted_degree = nom / denom

    diag_a = torch.diagonal(attn, dim1=-2, dim2=-1)
    lap_diag = weighted_degree - diag_a

    top_vals, _ = torch.topk(lap_diag, k=min(k, seq_len), dim=-1)

    if seq_len < k:
        padding = torch.zeros((num_heads, k - seq_len), device=device)
        top_vals = torch.cat([top_vals, padding], dim=-1)

    log_det = lap_diag.add(1e-6).log().mean(dim=-1, keepdim=True)

    # Return directly to CPU to avoid hoarding GPU memory
    result = torch.cat([top_vals, log_det], dim=-1).flatten().detach().cpu()

    # Cleanup local GPU variables
    del attn, nom, denom, diag_a, lap_diag, top_vals, log_det

    return result


def get_grounding_features_from_sequence(model, full_sequence_cpu, input_len, pad_len=50):
    answer_ids = full_sequence_cpu[input_len:]
    grounding_scores = []

    with torch.enable_grad():
        for i in range(len(answer_ids)):
            token_id = answer_ids[i].item()

            prefix_ids = full_sequence_cpu[:input_len + i].unsqueeze(0).to(model.device)

            # CRITICAL FIX: Extract embeddings, detach completely, clone, and require grad
            # This makes inputs_embeds a true "leaf node" isolated from prior graph history
            raw_embeds = model.get_input_embeddings()(prefix_ids)
            inputs_embeds = raw_embeds.detach().clone().requires_grad_(True)

            outputs = model(inputs_embeds=inputs_embeds)
            logits = outputs.logits[0, -1, :]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target_log_prob = log_probs[token_id]

            # Since model weights are frozen, this only computes gradients back to inputs_embeds
            model.zero_grad()
            target_log_prob.backward()

            prompt_grads = inputs_embeds.grad[0, :input_len, :]
            grounding_score = prompt_grads.norm().item()
            grounding_scores.append(grounding_score)

            # Memory cleanup
            del prefix_ids, raw_embeds, inputs_embeds, outputs, logits, log_probs, target_log_prob, prompt_grads

    if not grounding_scores:
        return [0.0] * (7 + pad_len)

    scores = np.array(grounding_scores)

    stats_features = [
        float(np.mean(scores)), float(np.var(scores)), float(np.max(scores)),
        float(np.min(scores)), float(np.median(scores)), float(np.percentile(scores, 25)),
        float(np.percentile(scores, 75))
    ]

    seq_features = grounding_scores.copy()
    if len(seq_features) >= pad_len:
        seq_features = seq_features[:pad_len]
    else:
        seq_features.extend([0.0] * (pad_len - len(seq_features)))

    return stats_features + seq_features

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

    # Memory cleanup for the judge
    del judge_inputs, output
    torch.cuda.empty_cache()

    return 1 if "YES" in judge_output else 0




# 1. SETUP MODEL & DATA
model_id = "google/gemma-3-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_attentions=True
).eval()


for param in model.parameters():
    param.requires_grad = False

#dataset = load_dataset("truthful_qa", "generation", split="validation[:50]")
dataset_train = load_json('data/llama-2-70b-chat/triviaqa_train_tp1.0_10responses_with_em_labels.json')[:10000]
dataset_test = load_json('data/llama-2-70b-chat/triviaqa_dev_tp1.0_10responses_with_em_labels.json')[:2000]

collected_data = []

print("Generating responses and extracting features...")
for item in dataset_train:
    messages = [
        {"role": "user", "content": item['question']}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    input_len = input_ids.shape[1]

    print("Question:", " ".join(item['question'].split()))
    if isinstance(item["answer_ground_truth"], list):
        gs = [x for x in item["answer_ground_truth"] if len(x.strip()) > 0][0]
    elif isinstance(item["answer_ground_truth"], str):
        gs = item["answer_ground_truth"].strip()
    else:
        gs = "I don't know"
    print("GS:", gs)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_attentions=True
        )

        # Decode answer before clearing outputs
    answer = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)

    # Secure the full sequence to CPU for grounding extraction later
    full_sequence_cpu = outputs.sequences[0].detach().cpu()

    # --- FEATURE 1: Attention Matrix Features ---
    final_step_attentions = outputs.attentions[-1]
    all_layer_features = []

    for layer_idx in range(len(final_step_attentions)):
        layer_tensor = final_step_attentions[layer_idx]

        # This function now natively returns a CPU tensor
        layer_evs = get_laplacian_features_for_layer(layer_tensor, k=5)
        all_layer_features.append(layer_evs)

        # Immediately delete the specific layer tensor from GPU
        del layer_tensor

    attn_features = torch.cat(all_layer_features).numpy()

    # --- CRITICAL: CLEAR HEAVY GENERATION ARTIFACTS BEFORE GRADIENTS ---
    del outputs
    del final_step_attentions
    del input_ids
    torch.cuda.empty_cache()  # Flush VRAM

    # --- FEATURE 2: Grounding Score Features ---
    # We pass the full_sequence_cpu, so it doesn't take up persistent GPU space
    grounding_features = get_grounding_features_from_sequence(model, full_sequence_cpu, input_len, pad_len=50)
    grounding_features = np.array(grounding_features)

    # Flush VRAM again after gradients
    torch.cuda.empty_cache()

    combined_features = np.concatenate([attn_features, grounding_features])

    print("Answer:", " ".join(answer.split()))
    print('-----------------------------------------------')

    collected_data.append({
        "question": " ".join(item['question'].split()),
        "gold_standard": " ".join(gs.split()),
        "generated_response": " ".join(answer.split()),
        "features": combined_features.tolist()
    })

print("Judging responses for ground truth labels...")
for record in collected_data:
    label = judge_hallucination(record['question'], record["generated_response"], record["gold_standard"])
    record["label"] = label

with open("results/hallucination_data.json", "w") as f:
    json.dump(collected_data, f, indent=4)

df = pd.DataFrame(collected_data)
df.to_csv("results/hallucination_data.csv", index=False)
print(f"Successfully saved {len(collected_data)} records to JSON and CSV.")

# 4. ENHANCED TRAIN & TEST
collected_data_csv = pd.read_csv("results/hallucination_data.csv", encoding='utf8')
collected_data_csv['features'] = collected_data_csv['features'].apply(ast.literal_eval)

X = np.array(collected_data_csv["features"].tolist())
y = np.array(collected_data_csv["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('clf', LogisticRegression(class_weight='balanced'))
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print(classification_report(y_test, preds))