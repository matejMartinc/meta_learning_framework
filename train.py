import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import json
import random
import numpy as np
from torch.optim import Adam
import gc

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# ── Gradient utilities ────────────────────────────────────────────────────────

def get_lora_parameters(model):
    """Return only LoRA parameters to keep gradient computation cheap."""
    return [p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]


def compute_gradient_vector(model, batch_encoding, lora_params):
    """
    Compute a flat gradient vector over LoRA params for the given batch.
    Uses no_grad context except for the single backward pass, then immediately
    frees the computation graph.

    Gradient checkpointing requires train() mode; we restore whatever mode
    the caller had afterwards.
    """
    was_training = model.training
    model.train()   # gradient checkpointing only works in train mode
    model.zero_grad()

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(**batch_encoding)
        loss = output.loss

    # Guard against NaN loss before backward (corrupted weights / empty labels)
    loss_val = loss.item()
    if not np.isfinite(loss_val):
        model.zero_grad()
        if not was_training:
            model.eval()
        torch.cuda.empty_cache()
        # Return a zero gradient vector so cosine similarity is 0, not NaN
        total_params = sum(p.numel() for p in lora_params)
        return torch.zeros(total_params), loss_val

    loss.backward()
    del output, loss

    grads = []
    for p in lora_params:
        g = p.grad.detach().float().view(-1).cpu() if p.grad is not None else torch.zeros(p.numel())
        grads.append(g)

    model.zero_grad()
    for p in lora_params:
        p.grad = None

    if not was_training:
        model.eval()

    torch.cuda.empty_cache()
    return torch.cat(grads), loss_val


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_similar_val_examples(batch_texts, val_examples, val_embeddings, k, embedder):
    """
    For each text in the batch, retrieve k most similar examples from val set.
    Returns a deduplicated list across the whole batch.
    """
    batch_embeddings = embedder.encode(batch_texts, convert_to_tensor=True)  # (n, d)

    sims = F.cosine_similarity(
        batch_embeddings.unsqueeze(1),  # (n, 1, d)
        val_embeddings.unsqueeze(0),    # (1, num_val, d)
        dim=-1
    )  # (n, num_val)

    topk_indices = sims.topk(k, dim=-1).indices   # (n, k)
    unique_indices = torch.unique(topk_indices)
    return [val_examples[i] for i in unique_indices.cpu().tolist()]


# ── Tokenisation helpers ──────────────────────────────────────────────────────

def to_ids_list(result):
    """Normalise apply_chat_template output to a plain Python list of ints."""
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], int):
            return result
        return result[0]
    if hasattr(result, 'input_ids'):
        ids = result.input_ids
        if isinstance(ids, list):
            return ids[0] if isinstance(ids[0], list) else ids
        return ids[0].tolist()
    if hasattr(result, 'tolist'):
        return result.tolist()
    return list(result)


def encode_examples_for_model(examples, tokenizer, device, max_length=2048):
    """
    Tokenise a list of conversation examples.
    Each example must have: example['conversations'][0]['value']  (user)
                            example['conversations'][1]['value']  (assistant)
    """
    all_input_ids, all_attention_masks, all_token_type_ids, all_labels = [], [], [], []

    for ex in examples:
        # Support both 'from'/'value' and 'role'/'content' key conventions
        turns = ex['conversations']
        user_turn = turns[0].get('value') or turns[0].get('content', '')
        assistant_turn = turns[1].get('value') or turns[1].get('content', '')

        prompt_messages = [{"role": "user", "content": user_turn}]
        full_messages = [
            {"role": "user",      "content": user_turn},
            {"role": "assistant", "content": assistant_turn},
        ]

        prompt_ids = to_ids_list(tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        ))

        full_ids = to_ids_list(tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors=None,
            truncation=True,
            max_length=max_length,
        ))

        prompt_len   = min(len(prompt_ids), len(full_ids))
        response_len = len(full_ids) - prompt_len

        all_input_ids.append(full_ids)
        all_attention_masks.append([1] * len(full_ids))
        all_token_type_ids.append([0] * prompt_len + [1] * response_len)
        all_labels.append([-100] * prompt_len + full_ids[prompt_len:])

    # Right-pad to longest sequence in the batch
    max_len = max(len(x) for x in all_input_ids)
    pad_id  = tokenizer.pad_token_id

    def pad_right(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))

    return {
        "input_ids":      torch.tensor([pad_right(x, pad_id) for x in all_input_ids]).to(device),
        "attention_mask": torch.tensor([pad_right(x, 0)      for x in all_attention_masks]).to(device),
        "token_type_ids": torch.tensor([pad_right(x, 0)      for x in all_token_type_ids]).to(device),
        "labels":         torch.tensor([pad_right(x, -100)   for x in all_labels]).to(device),
    }


def make_example(text):
    """Wrap a raw text string as a minimal conversation example."""
    return {
        "conversations": [
            {"value": "Refine this text if needed."},
            {"value": text},
        ]
    }


# ── Reward computation ────────────────────────────────────────────────────────

def compute_gradient_alignment_rewards(
    model,
    tokenizer,
    batch_texts,
    responses,
    val_examples,
    val_embeddings,
    embedder,
    device,
    k=5,
):
    """
    Gradient-alignment reward:
      - CLEAN  → cosine(grad_original, grad_val)
      - DIRTY  → cosine(grad_fixed, grad_val) − cosine(grad_original, grad_val),
                 gated by semantic faithfulness
    """
    lora_params = get_lora_parameters(model)

    # ── Compute val gradient once per batch ──────────────────────────────────
    similar_val  = retrieve_similar_val_examples(batch_texts, val_examples, val_embeddings, k, embedder)
    val_encoding = encode_examples_for_model(similar_val, tokenizer, device)

    with torch.enable_grad():
        val_grad, _ = compute_gradient_vector(model, val_encoding, lora_params)

    del val_encoding
    torch.cuda.empty_cache()

    rewards = []

    for resp, orig_text in zip(responses, batch_texts):

        try:
            # FIX: was `.strip` (property reference) — must be `.strip()`
            cleaned = resp.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)

            status = data.get("status", "DIRTY").upper()
            fixed_text = data.get("fixed_text", orig_text)
            diag = data.get("diagnosis", "N/A")
            item_id = data.get("id")

            print("Status:", status)
            if status != "CLEAN":
                print("Fixed text:", fixed_text)
                print('\n')
                print("Original text:", orig_text)

            print("------------------------------------------")

            # ── Original gradient ─────────────────────────────────────────
            orig_encoding = encode_examples_for_model([make_example(orig_text)], tokenizer, device)
            with torch.enable_grad():
                orig_grad, _ = compute_gradient_vector(model, orig_encoding, lora_params)
            del orig_encoding
            torch.cuda.empty_cache()

            orig_alignment = F.cosine_similarity(
                orig_grad.unsqueeze(0), val_grad.unsqueeze(0)
            ).item()

            if status == "CLEAN":
                reward = torch.tensor(float(np.clip(orig_alignment, -1, 1)))

            else:
                # ── Fixed gradient ────────────────────────────────────────
                fixed_encoding = encode_examples_for_model([make_example(fixed_text)], tokenizer, device)
                with torch.enable_grad():
                    fixed_grad, _ = compute_gradient_vector(model, fixed_encoding, lora_params)
                del fixed_encoding
                torch.cuda.empty_cache()

                fixed_alignment = F.cosine_similarity(
                    fixed_grad.unsqueeze(0), val_grad.unsqueeze(0)
                ).item()
                del fixed_grad

                # Semantic faithfulness gate
                with torch.no_grad():
                    orig_emb = embedder.encode(orig_text,  convert_to_tensor=True).cpu()
                    fixed_emb = embedder.encode(fixed_text, convert_to_tensor=True).cpu()
                    semantic_sim = F.cosine_similarity(
                        orig_emb.unsqueeze(0), fixed_emb.unsqueeze(0)
                    ).item()

                faithfulness_gate = 1.0 if semantic_sim > 0.8 else 0.1
                alignment_delta = fixed_alignment - orig_alignment
                print('Alignment delta', alignment_delta)
                print('Faithfulnes gate', faithfulness_gate)

                reward = torch.tensor(
                    float(np.clip(alignment_delta * faithfulness_gate, -1, 2))
                )

                if alignment_delta > 0 and faithfulness_gate == 1.0:
                    correction_worked = True
                else:
                    correction_worked = False

                log_refined_data(item_id, status, diag, fixed_text, correction_worked)

            del orig_grad
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[reward error] {e}")
            reward = torch.tensor(-1.0)

        rewards.append(reward)

    del val_grad
    torch.cuda.empty_cache()
    return rewards


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_step(model, query_tensors, response_tensors, rewards, optimizer,
             old_log_probs=None, clip_epsilon=0.2):
    model.train()

    rewards_tensor = torch.stack(rewards).to(device)

    # Safe normalisation: skip when batch has only one element (std is undefined)
    # Also guard against all-identical rewards (std == 0) which produces NaN
    if rewards_tensor.numel() > 1:
        std = rewards_tensor.std()
        if std.item() > 1e-6:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (std + 1e-8)
        else:
            rewards_tensor = rewards_tensor - rewards_tensor.mean()
    # else: single-element batch — use the raw (already clipped) reward as-is

    # FIX: accumulate into a plain Python float to avoid graph-reference leaks,
    #      then convert to tensor only for .backward()
    losses = []

    for i, (query, response, reward) in enumerate(zip(query_tensors, response_tensors, rewards_tensor)):
        input_ids = torch.cat([query, response]).unsqueeze(0).to(device)
        labels    = input_ids.clone()
        labels[0, :len(query)] = -100

        token_type_ids = torch.cat([
            torch.zeros(len(query),    dtype=torch.long),
            torch.ones(len(response),  dtype=torch.long),
        ]).unsqueeze(0).to(device)

        attention_mask = torch.ones_like(input_ids)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
        log_prob = -output.loss   # scalar tensor, graph attached

        # Guard: if loss is NaN (corrupted weights), skip this sample
        if not torch.isfinite(output.loss):
            losses.append(torch.tensor(0.0, device=device, requires_grad=False))
            continue

        if old_log_probs is not None:
            ratio   = torch.exp(log_prob - old_log_probs[i].detach())
            clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            loss_i  = -torch.min(ratio * reward, clipped * reward)
        else:
            loss_i = -log_prob * reward

        losses.append(loss_i)

    # FIX: sum then divide — all tensors stay in the same graph
    total_loss = torch.stack(losses).mean()

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(get_lora_parameters(model), max_norm=1.0)
    optimizer.step()

    return total_loss.item()


# ── Generation ────────────────────────────────────────────────────────────────

def generate_responses(model, tokenizer, prompts, device, max_new_tokens=2048):

    model.eval()
    query_tensors, response_tensors, responses = [], [], []

    with torch.no_grad():
        for prompt in prompts:
            encoding = tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)

            output = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = output[0][encoding.input_ids.shape[1]:]
            query_tensors.append(encoding.input_ids[0].cpu())
            response_tensors.append(response.cpu())
            responses.append(tokenizer.decode(response, skip_special_tokens=True))

            del output, encoding
            torch.cuda.empty_cache()

    # Restore train mode so gradient checkpointing works in reward computation
    model.train()
    return query_tensors, response_tensors, responses


# ── Training loop ─────────────────────────────────────────────────────────────

def train(dataset, val_examples, val_embeddings):
    old_log_probs = None

    for i in range(0, len(dataset), batch_size):
        batch     = dataset[i: i + batch_size]
        raw_texts = [item['conversations'][1]['value'] for item in batch]

        prompts = []
        for t in raw_texts:
            messages = [{
                "role": "user",
                "content": (
                    f"Determine if this text is 'CLEAN' or 'DIRTY'.\n"
                    f"- If CLEAN: return ONLY {{\"status\": \"CLEAN\"}}\n"
                    f"- If DIRTY: fix it and return ONLY {{\"status\": \"DIRTY\", \"fixed_text\": \"...\", \"diagnosis\": \"...\"}}\n"
                    f"Return raw JSON only, no markdown.\n"
                    f"Text: {t}"
                ),
            }]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(prompt)

        query_tensors, response_tensors, responses = generate_responses(
            model, tokenizer, prompts, device
        )

        rewards = compute_gradient_alignment_rewards(
            model=model,
            tokenizer=tokenizer,
            batch_texts=raw_texts,
            responses=responses,
            val_examples=val_examples,
            val_embeddings=val_embeddings,
            embedder=embedder,
            device=device,
            k=5,
        )

        loss = ppo_step(
            model, query_tensors, response_tensors, rewards,
            optimizer, old_log_probs=old_log_probs,
        )

        avg_reward = sum(r.item() for r in rewards) / len(rewards)

        # Detect NaN loss — if weights are corrupted, stop early rather than
        # cascading into a CUDA assertion crash on the next generate() call
        if not np.isfinite(loss):
            print(f"[WARNING] NaN/Inf loss detected at batch {i // batch_size} — skipping optimizer step and continuing")
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        print(f"Batch {i // batch_size} | Loss: {loss:.4f} | Avg Reward: {avg_reward:.4f}")

        # Explicit GC every batch to prevent fragmentation build-up
        gc.collect()
        torch.cuda.empty_cache()


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def log_refined_data(original_id, status, diagnosis, fixed_text, correction_worked):
    with open(SAVE_PATH, "a", encoding="utf-8") as f:
        entry = {
            "id": original_id,
            "status": status,
            "diagnosis": diagnosis,
            "fixed_text": fixed_text,
            "correction_worked": correction_worked
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model_id = "google/gemma-3-12b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    SAVE_PATH = "data/refined_nemotron_dataset.jsonl"

    # ── 4-bit quantised model (training target) ───────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        # Keeps quantisation statistics on CPU, saving ~200 MB VRAM
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",           # lets accelerate place layers optimally
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    optimizer = Adam(get_lora_parameters(model), lr=1e-5)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    manual_data = load_jsonl("data/gams_ft_dataset.json")
    silver_data = load_jsonl("data/nemotron_sft_all_final_98k.json")

    val_anchor = manual_data[:500]
    train_manual = manual_data[500:]

    print("Pre-computing validation embeddings...")
    val_texts = [ex['conversations'][1]['value'] for ex in val_anchor]
    # Keep val embeddings on CPU; move to GPU per-batch inside retrieval
    val_embeddings = embedder.encode(val_texts, convert_to_tensor=True)
    print(f"Val embeddings shape: {val_embeddings.shape}")

    combined = silver_data + train_manual
    random.shuffle(combined)
    train(combined, val_anchor, val_embeddings)