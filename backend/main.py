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
ADAPTER_PATH = "qwen25-14b"

print("토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
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
    
    # 고정된 챗봇 설정 프롬프트
    chat_prompt = """당신은 Qwen이라는 AI 어시스턴트입니다. 사용자의 질문에 친절하고 유용하게 답변해 주세요. 
당신은 대화 모드로 작동하며, 번역 모드가 아닙니다. 번역을 요청받더라도 번역 형식(Human: ... Assistant: ...)으로 응답하지 말고, 
직접 번역 결과만 제공하세요. 번역 지시문을 포함하지 마세요."""
    
    formatted_prompt = f"<|im_start|>system\n{chat_prompt}<|im_end|>\n"
    
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

def clean_response(text: str) -> str:
    """응답 텍스트를 정리합니다."""
    # 번역 관련 패턴 제거
    patterns = [
        r"Translate the following .+?\.",
        r"Please translate .+?\.",
        r"다음 .+? 번역하세요\.",
        r"번역: ",
        r"^Human: .+?\n?Assistant: ",
        r"<\|im_end\|>.*"
    ]
    
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.DOTALL)
    
    # 응답의 시작과 끝의 공백 제거
    result = result.strip()
    
    return result

@app.post("/api/chat", response_model=Response)
async def chat(query: Query):
    try:
        # 입력 프롬프트 생성
        prompt = format_chat_prompt(query.text, query.history)
        
        # 입력 텍스트 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 생성 파라미터 설정
        gen_kwargs = {
            "max_new_tokens": min(query.max_length, 2048),  # 출력 토큰 수 제한
            "temperature": query.temperature,
            "top_p": query.top_p,
            "repetition_penalty": 1.2,  # 반복 방지
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # 텍스트 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # 생성된 텍스트 디코딩 및 입력 프롬프트 제거
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response_text = full_response[len(prompt):]
        
        # Qwen 모델의 특수 토큰 제거
        response_text = response_text.replace("<|im_end|>", "").strip()
        
        # 응답 정리
        clean_text = clean_response(response_text)
        
        # 번역 작업이 요청된 경우 직접 번역
        if "번역" in query.text or "translate" in query.text.lower():
            # 번역 패턴 검출
            translation_request = re.search(r"[\"'](.+?)[\"']", query.text)
            if translation_request:
                # 번역이 요청된 텍스트를 직접 응답으로 반환
                return Response(response=translation_request.group(1))
        
        return Response(response=clean_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"} 