import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import json
import random
import numpy as np
from torch.optim import Adam

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"



def get_lora_parameters(model):
    """Return only LoRA parameters to keep gradient computation cheap."""
    return [p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]



def compute_gradient_vector(model, batch_encoding, lora_params):
    model.zero_grad()
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(**batch_encoding)
        loss = output.loss
    loss.backward()
    del output, loss
    grads = []
    for p in lora_params:
        if p.grad is not None:
            grads.append(p.grad.detach().float().view(-1).cpu())
        else:
            grads.append(torch.zeros(p.numel()))
    model.zero_grad()
    torch.cuda.empty_cache()
    return torch.cat(grads)


def retrieve_similar_val_examples(batch_texts, val_examples, val_embeddings, k, embedder):
    """
    For each text in the batch, retrieve k most similar examples from val set.
    Returns a flat deduplicated list of val examples for the whole batch (n*k pool).
    """
    batch_embeddings = embedder.encode(batch_texts, convert_to_tensor=True)  # (n, d)

    # Cosine similarity: (n, num_val)
    sims = F.cosine_similarity(
        batch_embeddings.unsqueeze(1),  # (n, 1, d)
        val_embeddings.unsqueeze(0),  # (1, num_val, d)
        dim=-1
    )

    # For each item in batch, get top-k val indices
    topk_indices = sims.topk(k, dim=-1).indices  # (n, k)

    # Deduplicate across the whole batch
    unique_indices = torch.unique(topk_indices)
    return [val_examples[i] for i in unique_indices.cpu().tolist()]


def to_ids_list(result):
    """Normalize apply_chat_template output to a plain Python list of ints."""
    if isinstance(result, list):
        # Could be list of ints, or list of lists
        if len(result) > 0 and isinstance(result[0], int):
            return result  # already flat list of ints
        return result[0]  # list of lists, take first
    if hasattr(result, 'input_ids'):
        ids = result.input_ids
        if isinstance(ids, list):
            return ids[0] if isinstance(ids[0], list) else ids
        return ids[0].tolist()
    if hasattr(result, 'tolist'):
        return result.tolist()
    return list(result)


def encode_examples_for_model(examples, tokenizer, device, max_length=2000):
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    all_labels = []

    for ex in examples:
        user_turn = ex['conversations'][0]['value']
        assistant_turn = ex['conversations'][1]['value']

        messages = [
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": assistant_turn},
        ]

        # Get prompt-only length for token_type_ids boundary
        prompt_messages = [{"role": "user", "content": user_turn}]
        prompt_ids = to_ids_list(tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,  # adds the <start_of_turn>model\n suffix
            tokenize=True,
            return_tensors=None,
        ))

        full_ids = to_ids_list(tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors=None,
            truncation=True,
            max_length=max_length,
        ))

        prompt_len = min(len(prompt_ids), len(full_ids))
        response_len = len(full_ids) - prompt_len

        token_type_ids = [0] * prompt_len + [1] * response_len
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        attention_mask = [1] * len(full_ids)

        all_input_ids.append(full_ids)
        all_attention_masks.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_labels.append(labels)

    # Right-padding
    max_len = max(len(x) for x in all_input_ids)
    pad_id = tokenizer.pad_token_id

    def pad_right(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))

    input_ids = torch.tensor([pad_right(x, pad_id) for x in all_input_ids])
    attention_mask = torch.tensor([pad_right(x, 0) for x in all_attention_masks])
    token_type_ids = torch.tensor([pad_right(x, 0) for x in all_token_type_ids])
    labels = torch.tensor([pad_right(x, -100) for x in all_labels])

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "token_type_ids": token_type_ids.to(device),
        "labels": labels.to(device),
    }

def compute_gradient_alignment_rewards(
        model,
        tokenizer,
        batch_texts,  # original raw texts
        responses,  # model's JSON responses for each text
        val_examples,  # full gams validation set
        val_embeddings,  # pre-computed embeddings of val_examples
        embedder,
        device,
        k=5
):
    """
    For each example in the batch:
      1. Parse model response to get fixed/unfixed text
      2. Retrieve k similar val examples
      3. Compute gradient alignment: dot(grad_train, grad_val)
         for both the original and fixed version
      4. Reward = alignment_fixed - alignment_original
         (positive = fix moved gradients in the right direction)
    """
    lora_params = get_lora_parameters(model)
    rewards = []

    similar_val = retrieve_similar_val_examples(
        batch_texts, val_examples, val_embeddings, k, embedder
    )
    val_encoding = encode_examples_for_model(similar_val, tokenizer, device)
    with torch.enable_grad():
        val_grad = compute_gradient_vector(model, val_encoding, lora_params)
    del val_encoding  # free immediately
    torch.cuda.empty_cache()

    for resp, orig_text in zip(responses, batch_texts):
        print("Response", resp)
        try:
            resp = resp.replace("```json", '').replace("```", "").strip
            data = json.loads(resp)
            status = data.get("status", "DIRTY").upper()
            fixed_text = data.get("fixed_text", orig_text)
            diag = data.get("diagnosis", "N/A")
            item_id = data.get("id")

            print("Status:", status)
            if status != "CLEAN":
                print("Fixed text", fixed_text)
            print('------------------------------------------')

            # --- Build tokenized inputs for original and fixed text ---
            # We wrap them as instruction-following examples for a fair loss comparison
            def make_example(text):
                return {
                    "conversations": [
                        {"from": "human", "value": "Refine this text if needed."},
                        {"from": "gpt", "value": text}
                    ]
                }

            orig_encoding = encode_examples_for_model([make_example(orig_text)], tokenizer, device)

            with torch.enable_grad():
                orig_grad = compute_gradient_vector(model, orig_encoding, lora_params)
            del orig_encoding
            torch.cuda.empty_cache()

            if status == "CLEAN":
                # Model left it alone — reward is just the raw alignment of original
                # Normalized dot product (cosine)
                alignment = F.cosine_similarity(
                    orig_grad.unsqueeze(0), val_grad.unsqueeze(0)
                ).item()
                reward = torch.tensor(float(np.clip(alignment, -1, 1)))
            else:
                # Model proposed a fix — compare alignment before and after
                fixed_encoding = encode_examples_for_model([make_example(fixed_text)], tokenizer, device)

                with torch.enable_grad():
                    fixed_grad = compute_gradient_vector(model, fixed_encoding, lora_params)
                del fixed_encoding
                torch.cuda.empty_cache()

                orig_alignment = F.cosine_similarity(
                    orig_grad.unsqueeze(0), val_grad.unsqueeze(0)
                ).item()
                fixed_alignment = F.cosine_similarity(
                    fixed_grad.unsqueeze(0), val_grad.unsqueeze(0)
                ).item()

                # Also gate on semantic faithfulness — fix shouldn't change meaning drastically
                with torch.no_grad():
                    orig_emb = embedder.encode(orig_text, convert_to_tensor=True).cpu()
                    fixed_emb = embedder.encode(fixed_text, convert_to_tensor=True).cpu()
                    semantic_sim = F.cosine_similarity(
                        orig_emb.unsqueeze(0), fixed_emb.unsqueeze(0)
                    ).item()

                faithfulness_gate = 1.0 if semantic_sim > 0.8 else 0.1

                # Core signal: did the fix improve gradient alignment?
                alignment_delta = fixed_alignment - orig_alignment

                reward = torch.tensor(
                    float(np.clip(alignment_delta * faithfulness_gate, -1, 2))
                )

                # Log if fix was accepted (positive signal)
                if alignment_delta > 0 and faithfulness_gate == 1.0:
                    log_refined_data(item_id, status, diag, fixed_text)

                del fixed_grad

            del orig_grad
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[reward error] {e}")
            reward = torch.tensor(-1.0)

        rewards.append(reward)

    del val_grad
    torch.cuda.empty_cache()
    return rewards


def train(dataset, val_examples, val_embeddings):
    old_log_probs = None

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i + batch_size]
        raw_texts = [item['conversations'][1]['value'] for item in batch]

        prompts = []
        for t in raw_texts:
            messages = [{"role": "user", "content":
                f"Determine if this text is 'CLEAN' or 'DIRTY'. "
                f"If 'DIRTY', fix it. Return ONLY JSON: "
                f'{{"id": "...", "status": "CLEAN"|"DIRTY", "fixed_text": "...", "diagnosis": "..."}}'
                f"\nText: {t}"
                         }]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # appends <start_of_turn>model\n
                tokenize=False,
            )
            prompts.append(prompt)

        query_tensors, response_tensors, responses = generate_responses(
            model, tokenizer, prompts, device
        )

        # Reward
        rewards = compute_gradient_alignment_rewards(
            model=model,
            tokenizer=tokenizer,
            batch_texts=raw_texts,
            responses=responses,
            val_examples=val_examples,
            val_embeddings=val_embeddings,
            embedder=embedder,
            device=device,
            k=5
        )

        # PPO update
        loss = ppo_step(
            model, query_tensors, response_tensors, rewards,
            optimizer, old_log_probs=old_log_probs
        )

        avg_reward = sum(r.item() for r in rewards) / len(rewards)
        print(f"Batch {i // batch_size} | Loss: {loss:.4f} | Avg Reward: {avg_reward:.4f}")


def ppo_step(model, query_tensors, response_tensors, rewards, optimizer,
             old_log_probs=None, clip_epsilon=0.2):
    model.train()

    rewards_tensor = torch.stack(rewards).to(device)
    rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for i, (query, response, reward) in enumerate(zip(query_tensors, response_tensors, rewards_tensor)):
        input_ids = torch.cat([query, response]).unsqueeze(0).to(device)
        labels = input_ids.clone()
        labels[0, :len(query)] = -100

        # Build token_type_ids: 0 for prompt, 1 for response
        token_type_ids = torch.cat([
            torch.zeros(len(query), dtype=torch.long),
            torch.ones(len(response), dtype=torch.long)
        ]).unsqueeze(0).to(device)

        attention_mask = torch.ones_like(input_ids)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
        log_prob = -output.loss

        if old_log_probs is not None:
            ratio = torch.exp(log_prob - old_log_probs[i].detach())
            clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            loss = -torch.min(ratio * reward, clipped * reward)
        else:
            loss = -log_prob * reward

        total_loss = total_loss + loss

    total_loss = total_loss / len(rewards)
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(get_lora_parameters(model), max_norm=1.0)
    optimizer.step()

    return total_loss.item()


def generate_responses(model, tokenizer, prompts, device, max_new_tokens=2000):
    model.eval()
    query_tensors = []
    response_tensors = []
    responses = []

    with torch.no_grad():
        for prompt in prompts:
            encoding = tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True
            ).to(device)

            output = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,  # pass it explicitly
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = output[0][encoding.input_ids.shape[1]:]
            query_tensors.append(encoding.input_ids[0])
            response_tensors.append(response)
            responses.append(tokenizer.decode(response, skip_special_tokens=True))

    return query_tensors, response_tensors, responses


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def log_refined_data(original_id, status, diagnosis, fixed_text):
    with open(SAVE_PATH, "a", encoding="utf-8") as f:
        entry = {
            "id": original_id,
            "status": status,
            "diagnosis": diagnosis,
            "conversations": [
                {"from": "human", "value": "Refine this text if needed."},
                {"from": "gpt", "value": fixed_text}
            ]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    model_id = "google/gemma-3-4b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    k_neighbors = 5
    SAVE_PATH = "data/refined_nemotron_dataset.jsonl"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    optimizer = Adam(get_lora_parameters(model), lr=1e-5)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    manual_data = load_jsonl("data/gams_ft_dataset.json")
    silver_data = load_jsonl("data/nemotron_sft_all_final_98k.json")

    val_anchor = manual_data[:500]
    train_manual = manual_data[500:]

    # Pre-compute val embeddings once — reused every batch
    print("Pre-computing validation embeddings...")
    val_texts = [ex['conversations'][1]['value'] for ex in val_anchor]
    val_embeddings = embedder.encode(val_texts, convert_to_tensor=True).to(device)
    print(f"Val embeddings shape: {val_embeddings.shape}")

    combined = silver_data + train_manual
    random.shuffle(combined)
    train(combined, val_anchor, val_embeddings)