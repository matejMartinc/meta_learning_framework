import torch
import numpy as np
import json
import pandas as pd
import ast
import gc

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- vLLM Imports ---
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    # 1. SETUP DATA
    dataset_train = load_json('data/llama-2-70b-chat/triviaqa_train_tp1.0_10responses_with_em_labels.json')[:10000]
    model_id = "google/gemma-3-12b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # =====================================================================
    # PHASE 1: vLLM for Fast Generation & Logprob Uncertainty Features
    # =====================================================================
    print("Loading vLLM for rapid generation...")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048  # Limits the KV cache allocation requirement
    )

    gen_prompts = []
    for item in dataset_train:
        messages = [{"role": "user", "content": item['question']}]
        gen_prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

    # Enable logprobs=5 to extract entropy and margins!
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0, logprobs=5)
    print("Batch generating responses and calculating Logprobs...")
    gen_outputs = llm.generate(gen_prompts, sampling_params)

    collected_data = []
    for item, output in zip(dataset_train, gen_outputs):
        if isinstance(item["answer_ground_truth"], list):
            gs = [x for x in item["answer_ground_truth"] if len(x.strip()) > 0][0]
        elif isinstance(item["answer_ground_truth"], str):
            gs = item["answer_ground_truth"].strip()
        else:
            gs = "I don't know"

        # Calculate Logprob Uncertainty Features on the fly
        token_entropies = []
        token_margins = []

        if output.outputs[0].logprobs is not None:
            for token_step in output.outputs[0].logprobs:
                # Get probabilities from logprobs for the top 5 choices
                probs = np.exp([v.logprob for v in token_step.values()])

                # Entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                token_entropies.append(entropy)

                # Margin (Difference between choice 1 and 2)
                sorted_probs = sorted(probs, reverse=True)
                margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
                token_margins.append(margin)

        # If no tokens generated, pad with zeros
        if not token_entropies:
            token_entropies = [0.0]
            token_margins = [0.0]

        logprob_features = [
            float(np.mean(token_entropies)), float(np.max(token_entropies)),
            float(np.min(token_entropies)), float(np.var(token_entropies)),
            float(np.mean(token_margins)), float(np.max(token_margins)),
            float(np.min(token_margins)), float(np.var(token_margins))
        ]

        collected_data.append({
            "question": " ".join(item['question'].split()),
            "gold_standard": " ".join(gs.split()),
            "generated_response": " ".join(output.outputs[0].text.split()),
            "prompt_token_ids": output.prompt_token_ids,
            "gen_token_ids": list(output.outputs[0].token_ids),
            "logprob_features": logprob_features
        })

    # 1B. Batch Judge Hallucinations
    judge_prompts = []
    for record in collected_data:
        prompt = f"""Evaluate if the 'Generated Answer' is a hallucination based on the 'Reference Answer'.
    Question: {record['question']}
    Reference Answer: {record['gold_standard']}
    Generated Answer: {record['generated_response']}
    
    Is the Generated Answer factually inconsistent with the Reference Answer or the Question? 
    Respond with exactly one word: 'YES' (for hallucination) or 'NO' (for factual)."""

        messages = [{"role": "user", "content": prompt}]
        judge_prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

    judge_sampling_params = SamplingParams(max_tokens=2, temperature=0.0)
    print("Batch judging responses...")
    judge_outputs = llm.generate(judge_prompts, judge_sampling_params)

    for record, output in zip(collected_data, judge_outputs):
        judge_text = output.outputs[0].text.upper()
        record["label"] = 1 if "YES" in judge_text else 0

    # 1C. DESTROY vLLM TO FREE MEMORY
    print("Destroying vLLM to free GPU memory...")
    del llm
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()

    # =====================================================================
    # PHASE 2: Hugging Face PyTorch for Hidden States (Lightning Fast)
    # =====================================================================
    print("Loading HuggingFace model for Hidden State extraction...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,  # Changed from torch_dtype to dtype
        device_map="auto",
        output_attentions=False
    ).eval()

    print("Extracting Hidden State features...")
    for i, record in enumerate(collected_data):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(collected_data)}")

        input_len = len(record["prompt_token_ids"])
        full_sequence_cpu = torch.tensor(record["prompt_token_ids"] + record["gen_token_ids"], dtype=torch.long)

        # Perform ONE single forward pass per sequence
        with torch.no_grad():
            # Move output_hidden_states=True here!
            outputs = model(
                full_sequence_cpu.unsqueeze(0).to(model.device),
                output_hidden_states=True
            )

            # Get the hidden states from the last layer (Shape: 1, seq_len, hidden_dim)
            last_layer_hidden_states = outputs.hidden_states[-1][0]

            # Isolate the hidden states corresponding only to the generated answer
            answer_hidden_states = last_layer_hidden_states[input_len:]

            if answer_hidden_states.shape[0] > 0:
                # Pool the hidden states AND cast to float32 before converting to numpy
                mean_state = answer_hidden_states.mean(dim=0).float().cpu().numpy()
                max_state = answer_hidden_states.max(dim=0).values.float().cpu().numpy()
            else:
                # Fallback if no answer was generated
                hidden_dim = last_layer_hidden_states.shape[-1]
                mean_state = np.zeros(hidden_dim, dtype=np.float32)
                max_state = np.zeros(hidden_dim, dtype=np.float32)

        # Combine Hidden States (Semantic knowledge) + Logprobs (Uncertainty)
        combined_features = np.concatenate([mean_state, max_state, record["logprob_features"]])
        record["features"] = combined_features.tolist()

        # Cleanup memory
        del record["prompt_token_ids"]
        del record["gen_token_ids"]
        del record["logprob_features"]
        del outputs

    # Flush memory explicitly after the loop
    torch.cuda.empty_cache()

    # Save Data
    with open("results/hallucination_data_fast.json", "w") as f:
        json.dump(collected_data, f, indent=4)
    df = pd.DataFrame(collected_data)
    df.to_csv("results/hallucination_data_fast.csv", index=False)
    print(f"Successfully saved {len(collected_data)} records.")

    # =====================================================================
    # PHASE 3: Feature Classification (Scikit-Learn)
    # =====================================================================
    collected_data_csv = pd.read_csv("results/hallucination_data_fast.csv", encoding='utf8')
    collected_data_csv['features'] = collected_data_csv['features'].apply(ast.literal_eval)

    X = np.array(collected_data_csv["features"].tolist())
    y = np.array(collected_data_csv["label"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_total_features = X_train.shape[1]
    num_logprob_features = 8  # We generated exactly 8 logprob statistical features
    num_hidden_features = num_total_features - num_logprob_features

    print(f"Training on {num_hidden_features} Hidden State features and {num_logprob_features} Logprob features.")

    preprocessor = ColumnTransformer(
        transformers=[
            # Apply PCA ONLY to the heavy, high-dimensional Hidden States (Mean & Max combined)
            ('hidden_pca', PCA(n_components=100), slice(0, num_hidden_features)),

            # Let the 8 Logprob statistical features pass through untouched
            ('logprob_pass', 'passthrough', slice(num_hidden_features, num_total_features))
        ]
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print(classification_report(y_test, preds))
if __name__ == "__main__":
    main()