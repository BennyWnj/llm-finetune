from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
import torch

# Load base + adapter
base_model = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_path = "./lora-adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model = PeftModel.from_pretrained(model, adapter_path)

# Set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Chat loop
print("🧠 Chat with your LoRA-tuned model. Type 'exit' to quit.")
while True:
    query = input("\nUser: ")
    if query.strip().lower() in {"exit", "quit"}:
        break

    prompt = f"### Instruction:\n{query}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n{response[len(prompt):].strip()}")
