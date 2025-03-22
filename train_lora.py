import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files="train.jsonl")

    def preprocess(example):
        prompt = format_prompt(example)
        encoding = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    tokenized_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="./lora-out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

    model.save_pretrained("./lora-adapter")
    tokenizer.save_pretrained("./lora-adapter")

if __name__ == "__main__":
    main()
