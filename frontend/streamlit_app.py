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
    adapter_path = "qwen25-14b"  # LoRA 어댑터 경로
    
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
            repetition_penalty=1.2,  # 반복 방지
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 생성된 텍스트 디코딩 및 입력 프롬프트 제거
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response_text = full_response[len(input_text):]
    
    # Qwen 모델의 특수 토큰 제거
    response_text = response_text.replace("<|im_end|>", "").strip()
    
    # 응답 정리
    clean_text = clean_response(response_text)
    
    # 번역 작업이 요청된 경우 직접 번역
    if "번역" in prompt or "translate" in prompt.lower():
        # 번역 패턴 검출
        translation_request = re.search(r"[\"'](.+?)[\"']", prompt)
        if translation_request:
            # 번역이 요청된 텍스트를 직접 응답으로 반환
            return translation_request.group(1)
    
    return clean_text

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