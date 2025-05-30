from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

model_path = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False,
    model_max_length=1024
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

train_raw = load_dataset("json", data_files="data_train.jsonl", split="train")
val_raw = load_dataset("json", data_files="data_val.jsonl", split="train")

def count_length(example):
    prompt = f"<start_of_turn>user\n{example['input']}\n<end_of_turn>\n<start_of_turn>model\n{example['output']}{tokenizer.eos_token}"
    tokens = tokenizer(prompt, truncation=False)
    example["length"] = len(tokens["input_ids"])
    return example

train_with_length = train_raw.map(count_length)
val_with_length = val_raw.map(count_length)

train_filtered = train_with_length.filter(lambda x: x["length"] <= 1024)
val_filtered = val_with_length.filter(lambda x: x["length"] <= 1024)

def preprocess(example):
    prompt = f"<start_of_turn>user\n{example['input']}\n<end_of_turn>\n<start_of_turn>model\n{example['output']}{tokenizer.eos_token}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)

train_dataset = train_filtered.map(preprocess, remove_columns=train_filtered.column_names)
val_dataset = val_filtered.map(preprocess, remove_columns=val_filtered.column_names)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

training_args = TrainingArguments(
    output_dir="./checkpoints_gemma1b",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    bf16=True,
    fp16=False,
    tf32=True,
    logging_steps=50,
    logging_dir="./logs_gemma1b",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()
