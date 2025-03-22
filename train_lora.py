from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

def format(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

def main():
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = load_dataset("json", data_files="train.jsonl")
    dataset = dataset.map(format)

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
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

    model.save_pretrained("./lora-adapter")
    tokenizer.save_pretrained("./lora-adapter")

if __name__ == "__main__":
    main()
