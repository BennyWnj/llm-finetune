from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
peft_path = "./lora-adapter"
output_path = "./mistral-merged"

model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, peft_path)
model = model.merge_and_unload()
model.save_pretrained(output_path)