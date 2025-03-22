
This repo helps you fine-tune an open-source LLM (like Mistral-7B) using LoRA and serve it using `vLLM`.

## ✅ Files
- `train_lora.py`: Fine-tunes with LoRA using `transformers + peft`
- `merge_lora.py`: Merges LoRA adapter into full model
- `serve.sh`: Launches OpenAI-compatible vLLM server
- `requirements.txt`: All dependencies

## 🧪 Dataset Format
Create `train.jsonl`:
```json
{"instruction": "What is your return policy?", "input": "", "output": "You can return within 30 days..."}
```

## 🚀 Quickstart
```bash
pip install -r requirements.txt
python train_lora.py
python merge_lora.py
bash serve.sh
```

## 🌐 Test the Model
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-merged", "messages": [{"role": "user", "content": "Tell me a joke."}]}'
```

## 🔧 Hardware Recommendation
A100 or 4090 with at least 40GB GPU memory for Mistral 7B + LoRA.

---

Let me know if you want this converted to a GitHub repo or Colab notebook!
