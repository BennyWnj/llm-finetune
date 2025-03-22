#!/bin/bash

# Activate your environment if needed
# source .venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model ./mistral-merged \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16
