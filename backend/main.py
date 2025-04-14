from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = FastAPI()

# 모델과 토크나이저 초기화
MODEL_PATH = "/qwen25-14b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True
).eval()

class Query(BaseModel):
    text: str
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class Response(BaseModel):
    response: str

@app.post("/api/chat", response_model=Response)
async def chat(query: Query):
    try:
        # 입력 텍스트 토큰화
        inputs = tokenizer(query.text, return_tensors="pt").to(model.device)
        
        # 생성 파라미터 설정
        gen_kwargs = {
            "max_length": query.max_length,
            "temperature": query.temperature,
            "top_p": query.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # 텍스트 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # 생성된 텍스트 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return Response(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"} 