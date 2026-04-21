import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


# ---------------------------------------------------------------------------
# Configuration (Modified)
# ---------------------------------------------------------------------------

@dataclass
class FrameworkConfig:
    model_name: str = "google/gemma-3-12b-it"
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    data_path: str = ""
    max_seq_len: int = 2048
    learning_rate: float = 2e-5
    num_epochs: int = 1
    batch_size: int = 2
    grad_accumulation_steps: int = 4
    warmup_steps: int = 50
    output_dir: str = "./checkpoints_sft"


# ---------------------------------------------------------------------------
# Model loading (Same as provided)
# ---------------------------------------------------------------------------
def _build_bnb_config(cfg: FrameworkConfig) -> Optional[BitsAndBytesConfig]:
    if not cfg.load_in_4bit: return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _build_lora_config(cfg: FrameworkConfig) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )


def load_model_and_tokenizer(cfg: FrameworkConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=_build_bnb_config(cfg),
        device_map="auto",
        torch_dtype=torch.bfloat16 if not cfg.load_in_4bit else None,
        attn_implementation="flash_attention_2"
    )

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()

    lora_config = _build_lora_config(cfg)
    model = get_peft_model(model, lora_config, adapter_name="default")

    model.print_trainable_parameters()
    return model, tokenizer


class SFTDataset(Dataset):
    """
    Uses tokenizer.apply_chat_template to handle special tokens correctly.
    Expects data to have 'prompt' and 'response' strings.
    """

    def __init__(self, data, tokenizer, max_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        convs = item.get("conversations", [])

        # 1. Format as a conversation
        messages = [
            {"role": "user", "content": convs[0]["value"] if len(convs) >= 1 else ""},
            {"role": "assistant", "content": convs[1]["value"] if len(convs) >= 2 else ""}
        ]

        # 2. Tokenize the full conversation
        # We use tokenize() or apply_chat_template(tokenize=True)
        full_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.max_seq_len
        )

        # 3. Create Labels (Masking the User Prompt)
        # To mask the prompt, we tokenize JUST the user part to find the length
        user_messages = [{"role": "user", "content": convs[0]["value"] if len(convs) >= 1 else ""}]
        # add_generation_prompt=True includes the "<start_of_turn>model\n" prefix
        user_tokens = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True
        )

        prompt_len = len(user_tokens)
        labels = list(full_tokens)

        # Mask prompt tokens by setting them to -100
        for i in range(len(labels)):
            if i < prompt_len:
                labels[i] = -100

        return {
            "input_ids": torch.tensor(full_tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad to the right for training
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id)
    }


# ---------------------------------------------------------------------------
# Updated Training Function
# ---------------------------------------------------------------------------
def train_sft(model, tokenizer, train_data, cfg: FrameworkConfig):
    dataset = SFTDataset(train_data, tokenizer, cfg.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)

    # Calculate steps correctly for the scheduler
    total_steps = (len(dataloader) // cfg.grad_accumulation_steps) * cfg.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps
    )

    model.train()
    model.set_adapter("default")  # Ensure we are training the policy, not the ref

    print(f"Starting SFT... Steps: {total_steps}")

    for epoch in range(cfg.num_epochs):
        step_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(pbar):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accumulation_steps
            loss.backward()

            step_loss += loss.item()

            if (step + 1) % cfg.grad_accumulation_steps == 0:
                # Gradient clipping is often helpful for SFT
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                pbar.set_postfix({"loss": step_loss})
                step_loss = 0

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"SFT Complete. Weights saved to {cfg.output_dir}")

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg: FrameworkConfig):
    print("[Init] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    print(f"[Data] Reading {cfg.data_path}...")
    raw_data = load_jsonl(cfg.data_path)

    # Trigger SFT
    train_sft(model, tokenizer, raw_data, cfg)


if __name__ == "__main__":
    cfg = FrameworkConfig(
        model_name="google/gemma-3-12b-it",
        data_path="data/train_gams_nemotron.jsonl",
        num_epochs=1,
        load_in_4bit=False,
        batch_size=1,
        grad_accumulation_steps=2,
        output_dir="./checkpoints_sft"
    )
    main(cfg)