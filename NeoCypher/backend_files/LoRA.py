import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -----------------------------
# 1. Config
# -----------------------------
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_FILE = "converted.json"
OUTPUT_DIR = "./mistral-lora-output"

BATCH_SIZE = 2
EPOCHS = 3
LR = 2e-4
MAX_SEQ_LENGTH = 512


# -----------------------------
# 2. Load Dataset
# -----------------------------
def load_custom_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


dataset = load_custom_dataset(DATA_FILE)
dataset = dataset.train_test_split(test_size=0.2)

# -----------------------------
# 3. Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_fn(example):
    text = f"### Question:\n{example['prompt']}\n\n### Cypher:\n{example['completion']}"
    tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized = dataset.map(tokenize_fn, batched=False)

# -----------------------------
# 4. Model + LoRA
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


# -----------------------------
# 5. Live Loss Plot Callback
# -----------------------------
class LossPlotCallback(TrainerCallback):
    def __init__(self, output_path="loss_curve.png"):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.steps.append(state.global_step)
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])

        # Plot and save
        plt.figure(figsize=(8, 5))
        plt.plot(self.steps, self.train_losses, label="Training Loss")
        if len(self.eval_losses) > 0:
            plt.plot(
                np.linspace(0, self.steps[-1], len(self.eval_losses)),
                self.eval_losses,
                label="Validation Loss",
            )
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_path)
        plt.close()


# -----------------------------
# 6. Training Args
# -----------------------------
args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=100,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    callbacks=[LossPlotCallback(output_path="loss_curve.png")],
)

# -----------------------------
# 7. Train
# -----------------------------
trainer.train()

# Save LoRA weights
model.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA fine-tuned model saved to {OUTPUT_DIR}")
print("📊 Loss curve is being updated live at loss_curve.png")
