from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
import json

test_dataset = '../test_filtered.jsonl'
#model_path = "../../train/checkpoints_qwen3b/checkpoint-23292"
#output_dir = '../generated_cadquery'
lora_adapter_path = "../../checkpoints_qwen_lora_4bit/checkpoints_qwen_lora_4bit/checkpoint-2911"
base_model_path = "../../Qwen/"
output_dir = '../generated_cadquery_lora'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    use_fast=False,
    model_max_length=1024
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_adapter_path)
model.eval()

batch_size = 50

with open(test_dataset, 'r') as f_in:
    lines = [json.loads(line) for line in f_in]

for i in range(0, len(lines), batch_size):
    batch = lines[i:i+batch_size]
    prompts = [
        f"### Instruction:\n{ex['input']}\n\n### Response:\n"
        for ex in batch
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    input_lengths = inputs["input_ids"].shape[1]
    max_new_tokens = max(1, 1024 - input_lengths)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for j, output in enumerate(decoded_outputs):
        if "### Response:" in output:
            response = output.split("### Response:", 1)[1].strip()
        else:
            response = output
        with open(os.path.join(output_dir, f"{i + j}.txt"), "w") as f_out:
            f_out.write(response.strip())
    