import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

# 페이지 설정
st.set_page_config(
    page_title="Qwen Chat",
    page_icon="🤖",
    layout="wide"
)

def load_model():
    base_model_name = "Qwen/Qwen2.5-14B-Instruct"
    adapter_path = "/qwen25-14b"  # LoRA 어댑터 경로
    
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    print("베이스 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        offload_buffers=True
    )
    
    print("LoRA 어댑터 로드 중...")
    model = PeftModel.from_pretrained(
        model, 
        adapter_path,
        offload_buffers=True
    )
    
    return model, tokenizer

def format_chat_prompt(text: str, history=None):
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

def generate_response(prompt, history=None):
    if history is None:
        history = []
        
    if st.session_state.model is None or st.session_state.tokenizer is None:
        with st.spinner("Loading model..."):
            st.session_state.model, st.session_state.tokenizer = load_model()
    
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    # 입력 프롬프트 생성
    input_text = format_chat_prompt(prompt, history)
    
    # 토큰화 및 생성
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response_text = response_text[len(input_text):].replace("<|im_end|>", "").strip()
    
    # 메모리 정리
    del inputs
    del outputs
    torch.cuda.empty_cache()
    
    return response_text

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

# 앱 시작 시 모델 로드
if st.session_state.model is None or st.session_state.tokenizer is None:
    with st.spinner("모델을 로딩 중입니다..."):
        st.session_state.model, st.session_state.tokenizer = load_model()

# UI 구현
st.title("🤖 Qwen Chat")

# 사이드바에 모델 정보 표시
with st.sidebar:
    st.header("About")
    st.write("""
    This is a chat interface for the Qwen-25-14B model.
    The model has been fine-tuned for better performance.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 채팅 인터페이스
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("What would you like to know?"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, st.session_state.messages[:-1])
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response}) 