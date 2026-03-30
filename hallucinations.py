import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def detect_qa_hallucinations(model_id, system_prompt, user_question):
    # 1. Setup Model & Processor
    # Using bfloat16 for efficiency and accuracy on Gemma 3
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Prepare Input
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    messages.append({"role": "user", "content": [{"type": "text", "text": user_question}]})

    # Generate the initial prompt tokens
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[-1]

    # 3. Generate Answer (Inference Mode)
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False  # Greedy decoding for reproducibility
        )
        # Slicing: only the newly generated tokens
        answer_ids = generation[0][input_len:]

    answer_text = processor.decode(answer_ids, skip_special_tokens=True)
    print(f"\nModel Answer: {answer_text.strip()}")
    print("-" * 50)

    # 4. Probe for Contextual Grounding (Gradient Analysis)
    results = []

    for i in range(len(answer_ids)):
        token_id = answer_ids[i]

        # Build sequence: Prompt + Answer tokens up to the current one
        # current_sequence shape: [1, input_len + i]
        prefix_ids = generation[0][:input_len + i].unsqueeze(0).detach()

        # Enable gradients on embeddings for this specific pass
        inputs_embeds = model.get_input_embeddings()(prefix_ids).detach().requires_grad_(True)

        # Forward pass
        outputs = model(inputs_embeds=inputs_embeds)
        logits = outputs.logits[0, -1, :]  # Logits for the token at index 'i'

        # --- THE FIX: Use Log-Probability to avoid Softmax saturation ---
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_log_prob = log_probs[token_id]

        # Backward pass
        model.zero_grad()
        target_log_prob.backward()

        # Calculate grounding score (L2 Norm of gradients on the PROMPT tokens)
        # We slice from 0 to input_len to see how much the question influenced this token
        prompt_grads = inputs_embeds.grad[0, :input_len, :]
        grounding_score = prompt_grads.norm().item()

        token_str = processor.decode([token_id])
        results.append((token_str, grounding_score))

    return results


# --- Execution ---
# Note: Ensure you have access to the model on Hugging Face
MODEL_NAME = "google/gemma-3-12b-it"
sys_p = ""
question = "Napiši mi pesem od dekletu po imenu Maja, ki rada pleše."

scores = detect_qa_hallucinations(MODEL_NAME, sys_p, question)

print(f"{'Token':<12} | {'Grounding Score'}")
print("-" * 35)
for t, s in scores:
    # After switching to log_probs, true grounding usually yields HIGHER norms.
    # Hallucinations (relying on internal weights) yield LOWER norms.
    status = "✅ OK" if s > 0.01 else "⚠️ HALLUCINATION"
    print(f"{t:<12} | {s:.6f} {status}")