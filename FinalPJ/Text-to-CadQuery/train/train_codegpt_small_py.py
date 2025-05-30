from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from collections import Counter
import numpy as np

# Load raw dataset
train_dataset = load_dataset("json", data_files="data_train.jsonl", split="train")
val_dataset = load_dataset("json", data_files="data_val.jsonl", split="train")

# Load tokenizer and model
model_id = "microsoft/CodeGPT-small-py"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))

def count_length(example):
    prompt = f"{example['input']}\n{example['output']}"
    tokens = tokenizer(prompt, truncation=False)
    example["length"] = len(tokens["input_ids"])
    return example

train_with_length = train_dataset.map(count_length)
val_with_length = val_dataset.map(count_length)

train_filtered = train_with_length.filter(lambda x: x["length"] <= 1024)
val_filtered = val_with_length.filter(lambda x: x["length"] <= 1024)

print(f"filtered train samples: {len(train_filtered)}")
print(f"filtered val samples: {len(val_filtered)}")

def preprocess(example):
    prompt = f"{example['input']}\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)

train_tokenized = train_filtered.map(preprocess, remove_columns=train_filtered.column_names)
val_tokenized = val_filtered.map(preprocess, remove_columns=val_filtered.column_names)

training_args = TrainingArguments(
    output_dir="./checkpoints-codegpt",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    learning_rate=5e-5,
    num_train_epochs=2,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()