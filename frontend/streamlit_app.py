import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import asyncio

# 페이지 설정
st.set_page_config(
    page_title="Qwen Chat",
    page_icon="🤖",
    layout="wide"
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

def load_model():
    model_path = "/qwen25-14b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        state_dict_assign=True
    )
    return model, tokenizer

def generate_response(prompt, history):
    if st.session_state.model is None or st.session_state.tokenizer is None:
        with st.spinner("Loading model..."):
            st.session_state.model, st.session_state.tokenizer = load_model()
    
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    # 입력 텍스트 준비
    input_text = ""
    for h in history:
        input_text += f"Human: {h['user']}\nAssistant: {h['assistant']}\n"
    input_text += f"Human: {prompt}\nAssistant: "
    
    # 토큰화 및 생성
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 비동기 처리를 위한 설정
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

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