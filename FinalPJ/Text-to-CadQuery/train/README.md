# Training Scripts

This folder contains scripts to finetune six open-source language models for the task of generating CadQuery code from natural language descriptions.

## Models

- `train_codegpt_small_py.py` — Finetunes CodeGPT-small-py
- `train_gpt2_medium.py` — Finetunes GPT-2 Medium
- `train_gpt2_large.py` — Finetunes GPT-2 Large
- `train_gemma-1B.py` — Finetunes Gemma3 1B
- `train_qwen-3B.py` — Finetunes Qwen2.5 3B
- `train_mistral-7B_lora.py` — Finetunes Mistral-7B using LoRA

All models are trained using prompt-completion JSONL data in supervised fine-tuning (SFT) mode.

## Dataset

The dataset is split into three parts:

- `data_train.jsonl` — 90% for training  
- `data_val.jsonl` — 5% for validation  
- `data_test.jsonl` — 5% for held-out evaluation  

It is publicly available at:  
👉 [https://huggingface.co/ricemonster/NeurIPS11092/tree/main/data](https://huggingface.co/ricemonster/NeurIPS11092/tree/main/data)