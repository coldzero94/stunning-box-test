from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

app = FastAPI()

# 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["ACCELERATE_USE_META_DEVICE"] = "0"
os.environ["ACCELERATE_OFFLOAD_WEIGHTS"] = "0"
os.environ["ACCELERATE_DISPATCH_MODEL"] = "0"  # 디스패치 기능 비활성화

# 모델과 토크나이저 초기화
BASE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "/qwen25-14b"

print("토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True
)

print("베이스 모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    offload_buffers=True
)

print("LoRA 어댑터 로드 중...")
model = PeftModel.from_pretrained(
    model, 
    ADAPTER_PATH,
    offload_buffers=True
)

class Query(BaseModel):
    text: str
    history: Optional[List[Dict[str, str]]] = []
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class Response(BaseModel):
    response: str

def format_chat_prompt(text: str, history: List[Dict[str, str]] = None) -> str:
    """채팅 프롬프트를 포맷팅합니다."""
    if history is None:
        history = []
    
    formatted_prompt = f"<|im_start|>system\n당신은 Qwen이라는 AI 어시스턴트입니다. 사용자의 질문에 친절하고 유용하게 답변해 주세요.<|im_end|>\n"
    
    # 대화 기록 추가 (최대 3개)
    recent_history = history[-3:] if history else []
    for msg in recent_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    # 현재 사용자 메시지 추가
    formatted_prompt += f"<|im_start|>user\n{text}<|im_end|>\n"
    formatted_prompt += "<|im_start|>assistant\n"
    
    return formatted_prompt

@app.post("/api/chat", response_model=Response)
async def chat(query: Query):
    try:
        # 입력 프롬프트 생성
        prompt = format_chat_prompt(query.text, query.history)
        
        # 입력 텍스트 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 입력 토큰 수 확인
        input_token_length = inputs.input_ids.shape[1]
        MAX_NEW_TOKENS = 3000
        
        if input_token_length > 3000:
            return Response(response="⚠️ 입력이 너무 깁니다. 더 짧은 텍스트로 시도해주세요.")
        
        # 생성 파라미터 설정
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": query.temperature,
            "top_p": query.top_p,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # 텍스트 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            response_text = response_text[len(prompt):].replace("<|im_end|>", "").strip()
        
        del inputs
        del outputs
        torch.cuda.empty_cache()
        
        # 출력이 최대 길이에 도달한 경우 알림
        if len(response_text.split()) >= MAX_NEW_TOKENS:
            response_text += "\n\n⚠️ 응답이 최대 길이에 도달했습니다."
        
        return Response(response=response_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"} 