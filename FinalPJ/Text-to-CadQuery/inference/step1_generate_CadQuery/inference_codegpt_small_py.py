from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm

test_dataset = './test_filtered.jsonl'
model_path = "ricemonster/codegpt-small-sft"
output_dir = './txt'

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False,
    model_max_length=1024
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

batch_size = 64

with open(test_dataset, 'r') as f_in:
    lines = [json.loads(line) for line in f_in]

for i in tqdm(range(0, len(lines), batch_size)):
    batch = lines[i:i+batch_size]
    prompts = [ex["input"] for ex in batch]

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
        response = output.strip()
        with open(os.path.join(output_dir, f"{i + j}.txt"), "w") as f_out:
            f_out.write(response.strip())