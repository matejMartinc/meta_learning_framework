import os
import torch
import numpy as np
import json
import pandas as pd
import ast
import gc
import itertools
import re
import random

from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- vLLM Imports ---
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from threadpoolctl import threadpool_limits
import torch.nn.functional as F


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Fast string overlap function for self-consistency
def calculate_jaccard(text1: str, text2: str) -> float:
    set1 = set(re.findall(r'\w+', text1.lower()))
    set2 = set(re.findall(r'\w+', text2.lower()))
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def main():
    # 1. SETUP DATA
    os.makedirs("results", exist_ok=True)
    random.seed(42)  # For reproducibility when selecting 50 random samples
    dataset_train_raw = load_json('data/llama-2-70b-chat/triviaqa_train_tp1.0_10responses_with_em_labels.json')[:5000]

    train_data = []
    for item in dataset_train_raw:
        if isinstance(item["answer_ground_truth"], list):
            gs = [x for x in item["answer_ground_truth"] if len(x.strip()) > 0][0]
        elif isinstance(item["answer_ground_truth"], str):
            gs = item["answer_ground_truth"].strip()
        else:
            gs = "I don't know"

        train_data.append({
            "source": "train",
            "question": " ".join(item['question'].split()),
            "gold_standard": " ".join(gs.split()),
            "context": "",  # Not needed/used for original data
            "language": "en_train",
        })

    print("Loading HuggingFace evaluation dataset (multi-wiki-qa-synthetic-hallucinations)...")
    hf_dataset_id = "alexandrainst/multi-wiki-qa-synthetic-hallucinations"
    configs = []
    api = HfApi()
    info = api.dataset_info(hf_dataset_id)
    card_data = getattr(info, 'cardData', None) or getattr(info, 'card_data', {})
    if 'configs' in card_data:
        configs = [c['config_name'] for c in card_data['configs'] if 'config_name' in c]

    eval_data = []
    print(f"Discovered {len(configs)} languages. Sampling max 50 'hallucination==False' examples each...")
    for idx, config in enumerate(configs):
        if idx % 10 == 0:
            print(f"Downloading & Sampling from language {idx}/{len(configs)}: {config}")

        try:
            ds = load_dataset(
                "parquet",
                data_files=f"hf://datasets/{hf_dataset_id}/{config}/*.parquet",
                split="train"
            )

            # Filter for examples where dataset marks hallucination as False
            ds_filtered = ds.filter(lambda x: x["hallucination"] == False)
            if len(ds_filtered) == 0:
                print(config, len(ds_filtered))
            num_examples = min(500, len(ds_filtered))
            if num_examples == 0:
                continue

            # Random selection
            indices = random.sample(range(len(ds_filtered)), num_examples)
            sampled = ds_filtered.select(indices)

            for item in sampled:
                # Extract first element from context list as requested
                ctx = item["context"][0] if isinstance(item["context"], list) and len(item["context"]) > 0 else ""
                choice = random.choice(["eval", "train"])
                eval_data.append({
                    "source": choice,
                    "question": " ".join(str(item["question"]).split()),
                    "gold_standard": " ".join(str(item["answer"]).split()),
                    "context": " ".join(ctx.split()),
                    "language": config
                })
        except Exception as e:
            print(f"Warning: Could not process config {config}: {e}")

    print(f"Collected {len(eval_data)} evaluation examples.")

    # COMBINE ALL DATA for a single processing pipeline
    all_data = train_data + eval_data

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
    for item in all_data:
        messages = [{"role": "user", "content": item['question']}]
        gen_prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

    # 1A. Deterministic generation for Main Features (Temperature 0.0)
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0, logprobs=5)
    print("Batch generating primary responses and calculating Logprobs...")
    gen_outputs = llm.generate(gen_prompts, sampling_params)

    # 1B. Stochastic generation for Self-Consistency (Temperature 0.7, n=5)
    consistency_params = SamplingParams(max_tokens=50, temperature=0.7, n=5)
    print("Batch generating 5 stochastic samples per prompt for Self-Consistency...")
    consistency_outputs = llm.generate(gen_prompts, consistency_params)

    collected_data = []
    for item, output, cons_output in zip(all_data, gen_outputs, consistency_outputs):
        #CALCULATE SELF-CONSISTENCY SCORE ---
        # Get the 5 generated strings
        cons_texts = [o.text for o in cons_output.outputs]
        # Create all unique pairs (10 pairs for 5 items)
        pairs = list(itertools.combinations(cons_texts, 2))

        if pairs:
            scores = [calculate_jaccard(p[0], p[1]) for p in pairs]
            consistency_score = float(np.mean(scores))
        else:
            consistency_score = 1.0

        # Calculate Logprob Uncertainty Features on the fly
        token_entropies = []
        token_margins = []
        token_logprobs = []

        if output.outputs[0].logprobs is not None:
            for token_step in output.outputs[0].logprobs:
                # Get probabilities from logprobs for the top 5 choices
                probs = np.exp([v.logprob for v in token_step.values()])

                # Because temperature=0, the chosen token is the max logprob
                max_logprob = max([v.logprob for v in token_step.values()])
                token_logprobs.append(max_logprob)

                # Entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                token_entropies.append(entropy)

                # Margin
                sorted_probs = sorted(probs, reverse=True)
                margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
                token_margins.append(margin)

        seq_length = len(token_entropies)
        if seq_length > 0:
            # 1. Perplexity
            seq_logprob_sum = sum(token_logprobs)
            perplexity = float(np.exp(-seq_logprob_sum / seq_length))

            # 2. Entropy Drift (Second half mean - First half mean)
            midpoint = seq_length // 2
            if midpoint > 0:
                entropy_first_half = np.mean(token_entropies[:midpoint])
                entropy_second_half = np.mean(token_entropies[midpoint:])
                entropy_drift = float(entropy_second_half - entropy_first_half)
            else:
                entropy_drift = 0.0
        else:
            token_entropies = [0.0]
            token_margins = [0.0]
            seq_length = 0
            perplexity = 0.0
            entropy_drift = 0.0

        # Append new features to your list
        logprob_features = [
            float(np.mean(token_entropies)), float(np.max(token_entropies)),
            float(np.min(token_entropies)), float(np.var(token_entropies)),
            float(np.mean(token_margins)), float(np.max(token_margins)),
            float(np.min(token_margins)), float(np.var(token_margins)),
            float(seq_length), float(perplexity), float(entropy_drift),
            consistency_score
        ]

        collected_data.append({
            "source": item["source"],
            "language": item["language"],
            "question": " ".join(item['question'].split()),
            "gold_standard": " ".join(item["gold_standard"].split()),
            "context": " ".join(item["context"].split()),
            "generated_response": " ".join(output.outputs[0].text.split()),
            "prompt_token_ids": output.prompt_token_ids,
            "gen_token_ids": list(output.outputs[0].token_ids),
            "logprob_features": logprob_features
        })

    # 1B. Batch Judge Hallucinations
    judge_prompts = []
    for record in collected_data:
        if record["source"] == "train":
            prompt = f"""Evaluate if the 'Generated Answer' is a hallucination based on the 'Reference Answer'.
    Question: {record['question']}
    Reference Answer: {record['gold_standard']}
    Generated Answer: {record['generated_response']}
    
    Is the Generated Answer factually inconsistent with the Reference Answer or the Question? 
    Respond with exactly one word: 'YES' (for hallucination) or 'NO' (for factual)."""

        else:
            # Special modified prompt for evaluation records handling Context inclusion
            prompt = f"""Evaluate if the 'Generated Answer' is a hallucination based on the 'Reference Answer' and the 'Context'.
    Question: {record['question']}
    Context: {record['context']}
    Reference Answer: {record['gold_standard']}
    Generated Answer: {record['generated_response']}

    Is the Generated Answer factually inconsistent with the Reference Answer, the Context, or the Question? 
    Respond with exactly one word: 'YES' (for hallucination) or 'NO' (for factual)."""

        messages = [{"role": "user", "content": prompt}]
        judge_prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    
    safe_judge_prompts = []

    # ZIP judge_prompts with collected_data so 'record' corresponds to the correct prompt
    for prompt, record in zip(judge_prompts, collected_data):
        token_count = len(tokenizer.encode(prompt))

        # Max model length is 2048 and max_tokens is 2, so the prompt can be at most 2046 tokens long
        if token_count <= 2046:
            safe_judge_prompts.append(prompt)
        else:
            short_prompt = f"""Evaluate if the 'Generated Answer' is a hallucination based on the 'Reference Answer' and the 'Context'.
        Question: {record['question']}
        Context: no context
        Reference Answer: {record['gold_standard']}
        Generated Answer: {record['generated_response']}

        Is the Generated Answer factually inconsistent with the Reference Answer, the Context, or the Question? 
        Respond with exactly one word: 'YES' (for hallucination) or 'NO' (for factual)."""

            messages = [{"role": "user", "content": short_prompt}]
            formatted_short = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            # 2nd safety check: Does it still exceed 2046 even without context?
            short_token_count = len(tokenizer.encode(formatted_short))
            if short_token_count > 2046:
                # Calculate how many tokens the chat template adds
                template_overhead = short_token_count - len(tokenizer.encode(short_prompt))
                # Leave a 5-token safety margin
                safe_content_length = 2046 - template_overhead - 10

                # Truncate the raw content tokens to the safe limit and decode
                raw_tokens = tokenizer.encode(short_prompt)
                truncated_content = tokenizer.decode(raw_tokens[:safe_content_length], skip_special_tokens=True)

                # Re-apply chat template onto the safely truncated content
                messages = [{"role": "user", "content": truncated_content}]
                formatted_short = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            safe_judge_prompts.append(formatted_short)


    judge_sampling_params = SamplingParams(max_tokens=2, temperature=0.0)
    print("Batch judging responses...")
    judge_outputs = llm.generate(safe_judge_prompts, judge_sampling_params)

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
        output_attentions=True
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
                output_hidden_states=True,
                output_attentions=True
            )

            # --- 1. HIDDEN STATES ---
            last_layer_hidden_states = outputs.hidden_states[-1][0]

            # Isolate Question vs Answer
            prompt_hidden_states = last_layer_hidden_states[:input_len]
            answer_hidden_states = last_layer_hidden_states[input_len:]

            if answer_hidden_states.shape[0] > 0:
                # Answer stats
                mean_state = answer_hidden_states.mean(dim=0).float().cpu().numpy()
                max_state = answer_hidden_states.max(dim=0).values.float().cpu().numpy()

                # --- NEW: SEMANTIC DRIFT (Cosine Similarity) ---
                # Compare the "concept" of the prompt to the "concept" of the answer
                mean_prompt_state = prompt_hidden_states.mean(dim=0).float()
                mean_answer_state_tensor = answer_hidden_states.mean(dim=0).float()

                # Cosine similarity (1 = identical meaning, -1 = opposite, 0 = unrelated)
                semantic_similarity = F.cosine_similarity(
                    mean_prompt_state.unsqueeze(0),
                    mean_answer_state_tensor.unsqueeze(0)
                ).item()
            else:
                # Fallbacks
                hidden_dim = last_layer_hidden_states.shape[-1]
                mean_state = np.zeros(hidden_dim, dtype=np.float32)
                max_state = np.zeros(hidden_dim, dtype=np.float32)
                semantic_similarity = 0.0

            last_layer_attn = outputs.attentions[-1][0]

            if answer_hidden_states.shape[0] > 0:
                gen_attn = last_layer_attn[:, input_len:, :]
                # Feature A: Prompt Grounding (How much does it look at the question?)
                # Sum attention weights falling on the prompt tokens (0 to input_len)
                prompt_focus = gen_attn[:, :, :input_len].sum(dim=-1)  # Shape: (heads, gen_tokens)
                mean_prompt_focus = prompt_focus.mean().item()

                # Feature B: Attention Entropy (How scattered is the attention?)
                # High entropy = confused/diffused attention
                attn_entropy = -torch.sum(gen_attn * torch.log(gen_attn + 1e-12), dim=-1)
                mean_attn_entropy = attn_entropy.mean().item()
                max_attn_entropy = attn_entropy.max().item()
            else:
                mean_prompt_focus, mean_attn_entropy, max_attn_entropy = 0.0, 0.0, 0.0

            attention_features = [mean_prompt_focus, mean_attn_entropy, max_attn_entropy]
            sequence_features = [semantic_similarity]

        # Combine Hidden States + Logprobs + Attention Features
        combined_features = np.concatenate([
            mean_state,
            max_state,
            record["logprob_features"],  # Now contains 11 features (8 + 3 seq)
            attention_features,  # Contains 3 features
            sequence_features  # Contains 1 feature
        ])
        record["features"] = combined_features.tolist()

        # Cleanup memory
        del record["prompt_token_ids"]
        del record["gen_token_ids"]
        del record["logprob_features"]
        del outputs
        del last_layer_attn
        if answer_hidden_states.shape[0] > 0: del gen_attn

    # Flush memory explicitly after the loop
    del model
    gc.collect()
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

    # Separate train and HF eval datasets using the stored label
    train_df = collected_data_csv[collected_data_csv["source"] == "train"]
    eval_df = collected_data_csv[collected_data_csv["source"] == "eval"]

    X_train_full = np.array(train_df["features"].tolist())
    y_train_full = np.array(train_df["label"])

    # Do normal Split/Train routines on original data ONLY
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    num_total_features = X_train.shape[1]
    num_scalar_features = 16  # We generated exactly 8 logprob statistical features
    num_hidden_features = num_total_features - num_scalar_features

    print(f"Training on {num_hidden_features} Hidden State features and {num_scalar_features} Logprob features.")

    preprocessor = ColumnTransformer(
        transformers=[
            # PCA on the heavy hidden states
            ('hidden_pca', PCA(n_components=100), slice(0, num_hidden_features)),
            # Pass all 16 scalar features through untouched
            ('scalar_pass', 'passthrough', slice(num_hidden_features, num_total_features))
        ]
    )

    pipeline = Pipeline([
        ('scaler_in', StandardScaler()),
        ('preprocessor', preprocessor),
        ('scaler_out', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, class_weight='balanced'))
    ])

    with threadpool_limits(limits=1, user_api='blas'):
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

    print(classification_report(y_test, preds))

    # --- NEW: Evaluate trained model PER LANGUAGE on the multi-lingual dataset ---
    if len(eval_df) > 0:
        print("\n=== 2. Multi-Lingual Evaluation (multi-wiki-qa-synthetic-hallucinations) ===")

        # Keep track of results for a clean summary
        lang_results = []
        languages = sorted(eval_df["language"].unique())

        for lang in languages:
            lang_df = eval_df[eval_df["language"] == lang]
            X_eval_lang = np.array(lang_df["features"].tolist())
            y_eval_lang = np.array(lang_df["label"])

            with threadpool_limits(limits=1, user_api='blas'):
                preds_eval_lang = pipeline.predict(X_eval_lang)

            # Using zero_division=0 in case small sample sizes result in only one class
            acc = accuracy_score(y_eval_lang, preds_eval_lang)
            macro_f1 = f1_score(y_eval_lang, preds_eval_lang, average="macro", zero_division=0)

            lang_results.append({
                "Language": lang,
                "Samples": len(lang_df),
                "Accuracy": acc,
                "Macro F1": macro_f1
            })

            # Optional: Uncomment to print the full classification report per language as it evaluates
            # print(f"\n--- Language: {lang} ---")
            # print(classification_report(y_eval_lang, preds_eval_lang, zero_division=0))

        # Convert results to a pandas dataframe and print a clean summary
        results_df = pd.DataFrame(lang_results)

        # Increase pandas display limits to print the whole table cleanly
        pd.set_option('display.max_rows', None)
        print("\n--- Summary of Performance Per Language ---")
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()


# Training on 7680 Hidden State features and 16 Logprob features.
#               precision    recall  f1-score   support
#
#            0       0.56      0.73      0.64       863
#            1       0.88      0.78      0.83      2262
#
#     accuracy                           0.77      3125
#    macro avg       0.72      0.76      0.73      3125
# weighted avg       0.80      0.77      0.78      3125
#
#
# === 2. Multi-Lingual Evaluation (multi-wiki-qa-synthetic-hallucinations) ===
#
# --- Summary of Performance Per Language ---
# Language  Samples  Accuracy  Macro F1
#       ab       58  0.775862  0.502310
#      ace       55  0.818182  0.634309
#      ady       44  0.750000  0.428571
#       af       45  0.622222  0.619214
#      als       39  0.692308  0.653846
#      alt       44  0.909091  0.642276
#       am       46  0.760870  0.507303
#      ami       47  0.765957  0.433735
#       an       49  0.693878  0.666667
#      ang       50  0.760000  0.650350
#      anp       47  0.553191  0.549932
#       ar       55  0.709091  0.659443
#      arc       48  0.916667  0.478261
#      ary       42  0.666667  0.654118
#      arz       47  0.680851  0.680272
#       as       52  0.865385  0.727341
#      ast       52  0.807692  0.630682
#      atj       55  0.800000  0.444444
#       av       52  0.846154  0.669841
#      avk       48  0.833333  0.553488
#      awa       55  0.672727  0.659091
#       ay       48  0.812500  0.599629
#       az       51  0.862745  0.773621
#      azb       51  0.921569  0.778261
#       ba       59  0.779661  0.503560
#      ban       50  0.760000  0.501661
#      bar       44  0.727273  0.636364
#      bcl       53  0.622642  0.598485
#       be       49  0.836735  0.455556
#       bg       51  0.882353  0.777616
#       bi       54  0.574074  0.513894
#      bjn       50  0.720000  0.702886
#      blk       49  0.795918  0.525194
#       bm       50  0.860000  0.462366
#       bn       60  0.833333  0.700000
#       bo       55  0.945455  0.485981
#      bpy       47  0.872340  0.590116
#       br       47  0.723404  0.601435
#       bs       50  0.680000  0.671593
#      bug       42  0.833333  0.564444
#      bxr       46  0.565217  0.543651
#       ca       48  0.854167  0.640642
#      cdo       47  0.787234  0.703283
#       ce       48  0.791667  0.441860
#      ceb       51  0.705882  0.514902
#       ch       49  0.693878  0.466231
#      chr       55  0.909091  0.476190
#      chy        8  0.750000  0.428571
#      ckb       47  0.893617  0.471910
#       co       59  0.576271  0.568335
#       cr       15  0.866667  0.464286
#      crh       50  0.800000  0.687500
#       cs       49  0.836735  0.666667
#      csb       56  0.839286  0.753786
#       cu       48  0.645833  0.481905
#       cv       51  0.784314  0.439560
#       cy       61  0.836066  0.455357
#       da       58  0.844828  0.782772
#      dag       45  0.800000  0.444444
#       de       44  0.863636  0.587500
#      din       48  0.687500  0.407407
#      diq       50  0.740000  0.539334
#      dsb       45  0.666667  0.561973
#      dty       51  0.803922  0.741903
#       dv       54  0.814815  0.531250
#       dz       50  0.980000  0.494949
#       ee       48  0.833333  0.454545
#       el       43  0.744186  0.653480
#       en       51  0.705882  0.514902
#       eo       51  0.686275  0.564103
#       es       48  0.750000  0.683516
#       et       42  0.785714  0.730577
#       eu       49  0.836735  0.554545
#      ext       54  0.814815  0.777961
#       fa       40  0.600000  0.583333
#      fat       53  0.867925  0.464646
#       ff       55  0.872727  0.466019
#       fi       46  0.804348  0.641558
#       fj       52  0.807692  0.446809
#       fo       56  0.750000  0.727778
#      fon       52  0.788462  0.440860
#       fr       60  0.700000  0.580745
#      frp       44  0.795455  0.671914
#      frr       53  0.811321  0.530142
#      fur       46  0.717391  0.482251
#       fy       59  0.830508  0.453704
#       ga       45  0.733333  0.492481
#      gag       58  0.827586  0.671202
#      gan       47  0.638298  0.613075
#      gcr       47  0.914894  0.477778
#       gd       54  0.814815  0.589666
#       gl       59  0.779661  0.726560
#      glk       61  0.754098  0.568600
#       gn       56  0.892857  0.471698
#      gom       54  0.870370  0.465347
#      gor       54  0.740741  0.625000
#      got       56  1.000000  1.000000
#      gpe       48  0.791667  0.757576
#       gu       47  0.659574  0.620968
#      guc       51  0.803922  0.527778
#      gur       61  0.770492  0.435185
#      guw       53  0.811321  0.447917
#       gv       48  0.833333  0.454545
#       ha       45  0.755556  0.560000
#      hak       50  0.720000  0.562500
#      haw       58  0.896552  0.472727
#       he       52  0.826923  0.710217
#       hi       50  0.800000  0.740125
#      hif       54  0.611111  0.587186
#       hr       59  0.745763  0.671858
#      hsb       44  0.795455  0.593846
#       ht       55  0.890909  0.471154
#       hu       50  0.840000  0.667774
#       hy       39  0.871795  0.465753
#      hyw       53  0.811321  0.530142
#       ia       55  0.636364  0.626359
#       id       52  0.769231  0.766117
#       ie       55  0.690909  0.662333
#       ig       48  0.916667  0.478261
#       ik       28  0.750000  0.428571
#      ilo       45  0.666667  0.584615
#      inh       48  0.791667  0.441860
#       io       57  0.666667  0.560649
#       is       49  0.673469  0.648746
#       it       55  0.836364  0.607454
#       iu       51  0.745098  0.426966
#       ja       54  0.833333  0.606478
#      jam       46  0.586957  0.537811
#      jbo       57  0.649123  0.393617
#       ka       49  0.897959  0.473118
#      kaa       51  0.745098  0.655584
#      kab       43  0.860465  0.462500
#      kbd       55  0.654545  0.440877
#      kbp       53  0.603774  0.416972
#      kcg       51  0.784314  0.439560
#       kg       50  0.800000  0.526515
#       ki       59  0.813559  0.448598
#       kk       48  0.833333  0.747368
#       kl       46  0.608696  0.378378
#       km       50  0.780000  0.642625
#       kn       48  0.812500  0.704716
#       ko       50  0.720000  0.666667
#      koi       42  0.809524  0.447368
#      krc       57  0.736842  0.526316
#       ks       50  0.700000  0.411765
#       ku       47  0.787234  0.522358
#       kv       54  0.814815  0.531250
#       kw       42  0.714286  0.416667
#       ky       50  0.780000  0.568627
#       la       50  0.860000  0.725490
#      lad       58  0.775862  0.767499
#       lb       52  0.730769  0.694118
#      lbe       52  0.673077  0.402299
#      lez       56  0.750000  0.573913
#      lfn       53  0.622642  0.611437
#       lg       55  0.909091  0.476190
#       li       52  0.788462  0.724868
#      lij       43  0.837209  0.743393
#      lld       42  0.809524  0.447368
#      lmo       46  0.543478  0.515789
#       ln       56  0.982143  0.495495
#       lo       47  0.765957  0.637193
#       lt       45  0.800000  0.533947
#      ltg       39  0.820513  0.560386
#       lv       49  0.693878  0.573913
#      mad       55  0.818182  0.532313
#      mai       47  0.638298  0.613075
#      mdf       50  0.820000  0.450549
#       mg       47  0.829787  0.552381
#      mhr       54  0.925926  0.480769
#       mi       44  0.977273  0.827451
#      min       55  0.709091  0.647436
#       mk       51  0.705882  0.622222
#       ml       53  0.830189  0.750131
#       mn       46  0.804348  0.739130
#      mni       56  1.000000  1.000000
#      mnw       45  0.888889  0.470588
#       mr       50  0.760000  0.501661
#      mrj       47  0.872340  0.465909
#       ms       45  0.666667  0.655788
#       mt       40  0.750000  0.725275
#      mwl       51  0.843137  0.803089
#       my       48  0.562500  0.458937
#      myv       59  0.915254  0.477876
#      mzn       46  0.760870  0.711681
#      nap       46  0.782609  0.701299
#      nds       45  0.888889  0.470588
#       ne       54  0.796296  0.674877
#      new       44  0.772727  0.517544
#      nia       59  0.796610  0.443396
#       nl       41  0.829268  0.740271
#       nn       51  0.843137  0.751220
#       no       51  0.627451  0.466703
#      nov       47  0.808511  0.724070
#      nqo       49  0.979592  0.494845
#      nso       57  0.964912  0.740909
#       nv       41  0.634146  0.546794
#       ny       42  0.666667  0.400000
#       oc       52  0.634615  0.433161
#      olo       48  0.833333  0.700000
#       om       50  0.720000  0.418605
#       or       50  0.920000  0.479167
#       os       54  0.796296  0.443299
#       pa       49  0.836735  0.666667
#      pag       48  0.729167  0.603810
#      pam       48  0.666667  0.528256
#      pap       45  0.844444  0.720000
#       pl       55  0.763636  0.663529
#       ro       42  0.714286  0.665782
#       sk       50  0.760000  0.714286
#       sl       47  0.702128  0.690789
#       sq       49  0.510204  0.484211
#       sr       43  0.697674  0.523444
#       sv       44  0.795455  0.593846
#       uk       58  0.810345  0.620915




