import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.linalg import eigvalsh
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. SETUP MODEL & DATA
model_id = "google/gemma-3-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_attentions=True  # Crucial for extracting maps
)

dataset = load_dataset("truthful_qa", "generation", split="validation[:50]")  # Small slice for example


# 2. FEATURE EXTRACTION (Spectral Features)
def get_laplacian_eigvals(attn_matrix, k=10):
    """Computes top-k eigenvalues of the Laplacian of an attention map."""
    # Symmetrize and remove batch dim
    adj = attn_matrix[0].mean(dim=0).detach().cpu().numpy()
    adj = (adj + adj.T) / 2

    # Degree matrix
    degree = np.diag(np.sum(adj, axis=-1))
    laplacian = degree - adj

    # Compute eigenvalues (only top-k for feature vector)
    try:
        evs = eigvalsh(laplacian)
        return evs[-k:]  # Take top-k largest
    except:
        return np.zeros(k)


all_features = []
all_responses = []

print("Generating responses and extracting features...")
for item in dataset:
    inputs = tokenizer(item['question'], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_attentions=True
        )

    # Grab attention from the last layer of the last generated token
    # shape: (num_layers, batch, num_heads, seq_len, seq_len)
    # 1. Capture all attention maps during generation
    # outputs.attentions is a tuple of tuples: (token_step, layer)
    all_step_features = []

    for step_attn in outputs.attentions:
        # step_attn is a tuple for each layer
        # We pick a middle-to-late layer (e.g., layer 28 for Gemma-12b)
        target_layer_attn = step_attn[28]

        # Compute eigenvalues for THIS token's view of the sequence
        step_evs = get_laplacian_eigvals(target_layer_attn)
        all_step_features.append(step_evs)

    # 2. Average the features across the whole response to get a single vector
    features = np.mean(all_step_features, axis=0)

    all_features.append(features)
    all_responses.append(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))


# 3. LLM-AS-A-JUDGE (Labeling)
# In a real scenario, use an API (GPT-4o) here. This is a mock judge logic.
def judge_hallucination(question, response, reference):
    """
    Mock LLM judge. In production, wrap an API call here.
    """
    # Placeholder: if response is too short or doesn't share keywords, mark as hallucination
    if len(response.split()) < 5: return 1
    return 0  # 0 = Factual, 1 = Hallucination


labels = [judge_hallucination(q['question'], r, q['best_answer'])
          for q, r in zip(dataset, all_responses)]

# 4. TRAIN & TEST CLASSIFIER
X = np.array(all_features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print("\nDetection Performance:")
print(classification_report(y_test, preds))