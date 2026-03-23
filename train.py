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
import copy

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Global stats for normalization
ppl_stats = {"mean": 0.0, "var": 1.0, "n": 0}
align_stats = {"mean": 0.0, "var": 1.0, "n": 0}


# ── Gradient utilities ────────────────────────────────────────────────────────

def update_and_normalize(value, stats, eps=1e-8):
    stats["n"] += 1
    delta = value - stats["mean"]
    stats["mean"] += delta / stats["n"]
    stats["var"] += (delta * (value - stats["mean"]) - stats["var"]) / stats["n"]
    std = max(stats["var"] ** 0.5, eps)
    return np.tanh(value / std)


def get_lora_parameters(model):
    return [p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]


def compute_gradient_vector(model, batch_encoding, lora_params):
    was_training = model.training
    model.train()
    model.zero_grad()

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(**batch_encoding)
        loss = output.loss

    loss_val = loss.item()
    if not np.isfinite(loss_val):
        model.zero_grad()
        if not was_training: model.eval()
        torch.cuda.empty_cache()
        total_params = sum(p.numel() for p in lora_params)
        return torch.zeros(total_params), loss_val

    loss.backward()
    del output, loss

    grads = []
    for p in lora_params:
        g = p.grad.detach().float().view(-1).cpu() if p.grad is not None else torch.zeros(p.numel())
        grads.append(g)

    model.zero_grad()
    for p in lora_params: p.grad = None
    if not was_training: model.eval()
    torch.cuda.empty_cache()
    return torch.cat(grads), loss_val


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_similar_val_examples(batch_texts, val_examples, val_embeddings, k, embedder):
    batch_embeddings = embedder.encode(batch_texts, convert_to_tensor=True)
    sims = F.cosine_similarity(batch_embeddings.unsqueeze(1), val_embeddings.unsqueeze(0), dim=-1)
    topk_indices = sims.topk(k, dim=-1).indices
    unique_indices = torch.unique(topk_indices)
    return [val_examples[i] for i in unique_indices.cpu().tolist()]


# ── Tokenisation helpers ──────────────────────────────────────────────────────

def to_ids_list(result):
    # Scalar int (single token)
    if isinstance(result, int):
        return [result]
    # Already a flat list of ints
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], int):
            return result
        # Nested list — unwrap one level
        if len(result) > 0 and isinstance(result[0], list):
            return result[0]
        return result
    # HuggingFace BatchEncoding or similar
    if hasattr(result, 'input_ids'):
        ids = result.input_ids
        if hasattr(ids, 'tolist'):
            return ids[0].tolist() if ids.dim() > 1 else ids.tolist()
        return list(ids[0]) if hasattr(ids[0], '__iter__') else list(ids)
    # Tensor
    if hasattr(result, 'tolist'):
        t = result.tolist()
        return t[0] if isinstance(t, list) and len(t) > 0 and isinstance(t[0], list) else t
    return list(result)


def encode_examples_for_model(examples, tokenizer, device, max_length=1024):
    all_input_ids, all_attention_masks, all_labels = [], [], []

    for ex in examples:
        turns = ex['conversations']
        user_turn = turns[0].get('value') or turns[0].get('content', '')
        assistant_turn = turns[1].get('value') or turns[1].get('content', '')

        prompt_messages = [{"role": "user", "content": user_turn}]
        full_messages = [{"role": "user", "content": user_turn}, {"role": "assistant", "content": assistant_turn}]

        prompt_ids = to_ids_list(tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True, truncation=True, max_length=max_length
        ))
        full_ids = to_ids_list(tokenizer.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=True, truncation=True, max_length=max_length
        ))

        prompt_len = min(len(prompt_ids), len(full_ids))

        all_input_ids.append(full_ids)
        all_attention_masks.append([1] * len(full_ids))
        all_labels.append([-100] * prompt_len + full_ids[prompt_len:])

    max_len = max(len(x) for x in all_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad_right(seq, pad_val): return seq + [pad_val] * (max_len - len(seq))

    input_ids = torch.tensor([pad_right(x, pad_id) for x in all_input_ids]).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": torch.tensor([pad_right(x, 0) for x in all_attention_masks]).to(device),
        "labels": torch.tensor([pad_right(x, -100) for x in all_labels]).to(device),
        "token_type_ids": torch.zeros_like(input_ids).to(device)
    }


def make_fixed_conversation(original_item, fixed_assistant_text):
    item = copy.deepcopy(original_item)
    turns = item["conversations"]
    for turn in reversed(turns):
        role = turn.get("from", turn.get("role", ""))
        if role in ("gpt", "assistant"):
            key = "value" if "value" in turn else "content"
            turn[key] = fixed_assistant_text
            break
    return item


# ── Reward computation ────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, item, device):
    encoding = encode_examples_for_model([item], tokenizer, device)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(**encoding)
    return output.loss.item()


def compute_quality_score(
        model, tokenizer, batch_items, orig_text, fixed_text,
        val_grad, lora_params, embedder, device
):
    """
    Compute a continuous quality score in (-1, 1) where:
        positive  → fix is BETTER than original
        negative  → fix is WORSE than original (or unfaithful)

    Components
    ----------
    alignment_delta : cosine-sim of fix gradient to val gradient minus that of original
    ppl_delta       : perplexity improvement (orig_ppl - fixed_ppl), normalised
    faithfulness    : semantic similarity between original and fixed text (gate)

    Returns
    -------
    score       : float in (-1, 1)
    components  : dict for logging
    """
    # --- Faithfulness gate (semantic similarity) ---
    with torch.no_grad():
        orig_emb = embedder.encode(orig_text, convert_to_tensor=True).cpu()
        fixed_emb = embedder.encode(fixed_text, convert_to_tensor=True).cpu()
        semantic_sim = F.cosine_similarity(orig_emb.unsqueeze(0), fixed_emb.unsqueeze(0)).item()

    # Hard gate: if the fix drifts too far from the original meaning, score is negative
    if semantic_sim < 0.80:
        return -1.0 * (1.0 - semantic_sim), {
            "alignment_delta": 0.0, "ppl_delta": 0.0,
            "semantic_sim": semantic_sim, "faithfulness_penalty": True
        }

    # --- Gradient alignment delta ---
    orig_item = batch_items
    fixed_item = make_fixed_conversation(batch_items, fixed_text)

    orig_encoding = encode_examples_for_model([orig_item], tokenizer, device)
    with torch.enable_grad():
        orig_grad, _ = compute_gradient_vector(model, orig_encoding, lora_params)
    del orig_encoding

    fixed_encoding = encode_examples_for_model([fixed_item], tokenizer, device)
    with torch.enable_grad():
        fixed_grad, _ = compute_gradient_vector(model, fixed_encoding, lora_params)
    del fixed_encoding

    orig_alignment = F.cosine_similarity(orig_grad.unsqueeze(0), val_grad.unsqueeze(0)).item()
    fixed_alignment = F.cosine_similarity(fixed_grad.unsqueeze(0), val_grad.unsqueeze(0)).item()
    del orig_grad, fixed_grad
    torch.cuda.empty_cache()

    # --- Perplexity delta ---
    orig_ppl = compute_perplexity(model, tokenizer, orig_item, device)
    fixed_ppl = compute_perplexity(model, tokenizer, fixed_item, device)

    # Normalise both deltas through running stats → tanh squash → (-1, 1)
    norm_align = update_and_normalize(fixed_alignment - orig_alignment, align_stats)
    norm_ppl = update_and_normalize(orig_ppl - fixed_ppl, ppl_stats)  # positive = fix lowers ppl

    # Combine with equal weight; result is in (-1, 1)
    score = 0.5 * norm_align + 0.5 * norm_ppl

    return score, {
        "alignment_delta": fixed_alignment - orig_alignment,
        "ppl_delta": orig_ppl - fixed_ppl,
        "norm_align": norm_align,
        "norm_ppl": norm_ppl,
        "semantic_sim": semantic_sim,
        "faithfulness_penalty": False,
        "score": score,
    }


def compute_rewards_and_decide_action(
        model, tokenizer, batch_texts, batch_items, responses,
        val_examples, val_embeddings, embedder, device, k=5,
):
    lora_params = get_lora_parameters(model)
    similar_val = retrieve_similar_val_examples(batch_texts, val_examples, val_embeddings, k, embedder)
    val_encoding = encode_examples_for_model(similar_val, tokenizer, device)

    with torch.enable_grad():
        val_grad, _ = compute_gradient_vector(model, val_encoding, lora_params)
    del val_encoding
    torch.cuda.empty_cache()

    actions = []

    for i, (resp, orig_text) in enumerate(zip(responses, batch_texts)):
        try:
            cleaned = resp.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)

            status = data.get("status", "DIRTY").upper()
            fixed_text = data.get("fixed_text", orig_text)
            diag = data.get("diagnosis", "N/A")
            item_id = batch_items[i]['id']
            old_conversation = batch_items[i]['conversations']

            print("Status:", status)

            if status == "CLEAN":
                # No fix attempted — straight SFT
                actions.append({
                    'action': 'SFT_CLEAN',
                    'item': batch_items[i],
                    'auditor_response': resp,
                    'quality_score': None,
                })
            else:
                print('***********************DIRTY*****************')
                print('Item id:', item_id)
                print("Fixed text:", fixed_text)
                print('\n\nOriginal text:', orig_text)

                # Compute continuous quality score
                score, components = compute_quality_score(
                    model, tokenizer, batch_items[i], orig_text, fixed_text,
                    val_grad, lora_params, embedder, device
                )

                print(
                    f"Quality score: {score:+.3f} | "
                    f"align_delta={components.get('alignment_delta', 0):.3f} | "
                    f"ppl_delta={components.get('ppl_delta', 0):.3f} | "
                    f"sem_sim={components['semantic_sim']:.2f}"
                )

                fixed_item = make_fixed_conversation(batch_items[i], fixed_text)
                fixed_conversation = fixed_item['conversations']

                # score > 0  → fix is better   → DPO: chosen=fixed, rejected=original
                # score <= 0 → fix is worse     → DPO: chosen=original, rejected=fixed  (flipped)
                fix_worked = bool(score > 0)

                if fix_worked:
                    print(f"Fix SUCCEEDED (score={score:+.3f}). DPO chosen=fixed.")
                else:
                    print(f"Fix FAILED (score={score:+.3f}). DPO chosen=original, penalising bad fix.")

                actions.append({
                    'action': 'DPO_FIX',
                    'item': batch_items[i],
                    'fixed_text': fixed_text,
                    'auditor_response': resp,
                    'quality_score': score,  # signed: positive=good, negative=bad
                    'fix_worked': fix_worked,
                })

                log_refined_data(item_id, status, diag, fixed_conversation, old_conversation, fix_worked)

        except Exception as e:
            print(f"JSON Error: {e}", resp)
            actions.append({'action': 'SKIP'})

    del val_grad
    torch.cuda.empty_cache()
    return actions

# ── Optimization Step (DPO + SFT) ─────────────────────────────────────────────

def get_batch_logps(logits, labels, label_pad_token_id=-100):
    """Compute log probabilities for DPO"""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll zero out loss later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)


def compute_dpo_loss(model, enc_chosen, enc_rejected, beta=0.1):
    """
    Standard DPO loss given pre-built encodings for chosen and rejected.
    Returns scalar loss tensor (with grad).
    """
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_chosen = model(**enc_chosen)
        out_rejected = model(**enc_rejected)

    logps_chosen = get_batch_logps(out_chosen.logits, enc_chosen['labels'])
    logps_rejected = get_batch_logps(out_rejected.logits, enc_rejected['labels'])

    with torch.no_grad():
        with model.disable_adapter():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                ref_out_chosen = model(**enc_chosen)
                ref_out_rejected = model(**enc_rejected)

    ref_logps_chosen = get_batch_logps(ref_out_chosen.logits, enc_chosen['labels'])
    ref_logps_rejected = get_batch_logps(ref_out_rejected.logits, enc_rejected['labels'])

    pi_logratios = logps_chosen - logps_rejected
    ref_logratios = ref_logps_chosen - ref_logps_rejected
    dpo_loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

    del out_chosen, out_rejected, ref_out_chosen, ref_out_rejected
    return dpo_loss


def hybrid_train_step(model, tokenizer, optimizer, actions, prompts, device, beta=0.1):
    """
    Handles:
    1. SFT on auditor output (keeps JSON format alive)
    2. SFT on clean conversation
    3. Quality-scaled DPO on dirty conversations

    Quality-scaled DPO logic
    ─────────────────────────
    Let q = quality_score ∈ (-1, 1)  (positive = fix is better)

    Case A  q > 0  (fix succeeded):
        chosen=fixed, rejected=original
        dpo_weight = 1.0 - q          # better fix → less correction needed → smaller weight
        total_dpo = dpo_weight * base_dpo_loss

    Case B  q ≤ 0  (fix failed):
        chosen=original, rejected=fixed  ← FLIPPED, penalises the bad fix
        dpo_weight = 1.0 + |q|         # worse fix → harder punishment → larger weight
        total_dpo = dpo_weight * base_dpo_loss
    """
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    valid_batch = False

    for action_data, prompt_text in zip(actions, prompts):
        action_type = action_data['action']

        if action_type == 'SKIP':
            continue

        valid_batch = True

        # ── 1. AUDITOR MAINTENANCE ──
        auditor_resp = action_data.get('auditor_response', '')
        if auditor_resp:
            auditor_text = prompt_text + auditor_resp
            auditor_enc = tokenizer(auditor_text, return_tensors='pt', truncation=True, max_length=1024).to(device)
            prompt_enc = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False)
            labels = auditor_enc.input_ids.clone()
            labels[:, :prompt_enc.input_ids.shape[1]] = -100
            if "token_type_ids" not in auditor_enc:
                auditor_enc["token_type_ids"] = torch.zeros_like(auditor_enc.input_ids)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                auditor_out = model(**auditor_enc, labels=labels)

            total_loss += 0.8 * auditor_out.loss
            del auditor_enc, auditor_out

        # ── 2. SFT on clean conversation ──
        if action_type == 'SFT_CLEAN':
            item = action_data['item']
            sft_encoding = encode_examples_for_model([item], tokenizer, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                sft_out = model(**sft_encoding)
            total_loss += sft_out.loss
            del sft_encoding, sft_out

        # ── 3. Quality-scaled DPO on dirty conversation ──
        elif action_type == 'DPO_FIX':
            item = action_data['item']
            fixed_text = action_data['fixed_text']
            q = action_data['quality_score']  # signed float in (-1, 1)
            fix_worked = action_data['fix_worked']  # bool

            item_fixed = make_fixed_conversation(item, fixed_text)

            enc_fixed = encode_examples_for_model([item_fixed], tokenizer, device)
            enc_original = encode_examples_for_model([item], tokenizer, device)

            if fix_worked:
                # Fix is better: reinforce fixed, suppress original
                # Weight < 1: the better the fix, the less extra pressure needed
                enc_chosen, enc_rejected = enc_fixed, enc_original
                dpo_weight = 1.0 - abs(q)  # q ∈ (0,1) → weight ∈ (0,1)
                print(f"  DPO (fix succeeded): weight={dpo_weight:.3f} (q={q:+.3f})")
            else:
                # Fix is worse: reinforce original, suppress bad fix  ← flipped polarity
                # Weight > 1: the worse the fix, the harder we punish
                enc_chosen, enc_rejected = enc_original, enc_fixed
                dpo_weight = 1.0 + abs(q)  # q ∈ (-1,0] → weight ∈ (1,2]
                print(f"  DPO (fix failed):    weight={dpo_weight:.3f} (q={q:+.3f})")

            base_dpo_loss = compute_dpo_loss(model, enc_chosen, enc_rejected, beta=beta)
            total_loss += dpo_weight * base_dpo_loss

            del enc_fixed, enc_original

    if valid_batch:
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_lora_parameters(model), max_norm=1.0)
        optimizer.step()
        return total_loss.item()
    else:
        return 0.0


# ── Generation ────────────────────────────────────────────────────────────────

def generate_responses(model, tokenizer, prompts, device, max_new_tokens=1024):
    model.eval()
    responses = []
    with torch.no_grad():
        for prompt in prompts:
            encoding = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, max_length=1024,
                                 add_special_tokens=False).to(device)
            output = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
            resp_ids = output[0][encoding.input_ids.shape[1]:]
            responses.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
            del output, encoding
    model.train()
    return responses


# ── Training loop ─────────────────────────────────────────────────────────────

def train(dataset, val_examples, val_embeddings):
    print("Starting Hybrid SFT/DPO Training...")

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i + batch_size]
        raw_texts = [item["conversations"][1].get("value", item["conversations"][1].get("content", "")) for item in
                     batch]

        # Auditor Prompt
        prompts = []
        for conv in raw_texts:
            messages = [{
                "role": "user",
                "content": (
                    f'''Your primary goal is to audit Slovenian text and identify and correct all morphosyntactic errors, specifically noun-adjective gender/number agreement and wrong case endings, which must be treated as critical errors.
                    Rules:
                    1. Reasoning: Briefly describe any errors found.
                    2. If grammatical errors exist, return {{"status": "DIRTY", "fixed_text": "...", "diagnosis": "..."}}
                    3. If perfect, return {{"status": "CLEAN"}}
                    4. Return ONLY valid JSON.
                    Text: \n "{conv}"'''
                ),
            }]
            prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

        # 1. Generate Auditor Feedback
        responses = generate_responses(model, tokenizer, prompts, device)

        # 2. Decide what to do (SFT Clean, DPO Fix, or Skip)
        actions = compute_rewards_and_decide_action(
            model=model, tokenizer=tokenizer,
            batch_texts=raw_texts, batch_items=batch, responses=responses,
            val_examples=val_examples, val_embeddings=val_embeddings, embedder=embedder,
            device=device, k=5
        )

        # 3. Update Model
        loss = hybrid_train_step(model, tokenizer, optimizer, actions, prompts, device)

        print(f"Batch {i // batch_size} | Loss: {loss:.4f}")
        gc.collect()
        torch.cuda.empty_cache()

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def is_short_enough(item, tokenizer, max_length=1024):
    turns = item["conversations"]
    total_tokens = sum(
        len(tokenizer.encode(t.get("value", t.get("content", ""))))
        for t in turns
    )
    return total_tokens <= max_length


def log_refined_data(original_id, status, diagnosis, fixed_text, old_text, correction_worked):
    with open(SAVE_PATH, "a", encoding="utf-8") as f:
        entry = {
            "id": original_id,
            "status": status,
            "diagnosis": diagnosis,
            "fixed_conversation": fixed_text,
            "old_conversation": old_text,
            "correction_worked": correction_worked
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ... (Keep your existing initialization code: Model loading, LoRA config, Dataset loading) ...
    # This part remains exactly the same as your original script
    model_id = "google/gemma-3-12b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    SAVE_PATH = "data/refined_nemotron_dataset.jsonl"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto", attn_implementation="flash_attention_2"
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    optimizer = Adam(get_lora_parameters(model), lr=1e-5)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Mock data loading for context (Replace with your actual loaders)
    manual_data = load_jsonl("data/gams_ft_dataset.json")
    silver_data = load_jsonl("data/nemotron_sft_all_final_98k.json")
    val_anchor = manual_data[:100]  # reduced for test
    train_manual = manual_data[100:]
    val_texts = [ex['conversations'][1]['value'] for ex in val_anchor]
    val_embeddings = embedder.encode(val_texts, convert_to_tensor=True)
    combined = silver_data + train_manual
    combined = [item for item in combined if is_short_enough(item, tokenizer)]
    random.shuffle(combined)

    train(combined, val_anchor, val_embeddings)