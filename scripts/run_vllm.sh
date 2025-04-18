#!/bin/bash

# VLLM 설치
pip install vllm
pip install bitsandbytes

# 서버 직접 실행
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-lora \
  --qlora-adapter-name-or-path /qwen25-14b \
  --quantization bitsandbytes \
  --dtype half \
  --max-model-len 4096 \
  --trust-remote-code \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9