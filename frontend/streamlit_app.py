import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import asyncio

# 페이지 설정
st.set_page_config(
    page_title="Qwen Chat",
    page_icon="🤖",
    layout="wide"
)

def load_model():
    model_path = "/qwen25-14b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 모델 로딩 전에 환경 변수 설정
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    
    # 메타 디바이스 사용 비활성화
    os.environ["ACCELERATE_USE_META_DEVICE"] = "0"
    
    # 오프로딩 비활성화
    os.environ["ACCELERATE_OFFLOAD_WEIGHTS"] = "0"
    
    # 디스패치 기능 비활성화
    os.environ["ACCELERATE_DISPATCH_MODEL"] = "0"
    
    # 8비트 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True
    )
    
    # LoRA 파라미터 로딩 후 모델을 eval 모드로 설정
    model.eval()
    
    return model, tokenizer

def generate_response(prompt, history=None):
    if history is None:
        history = []
        
    if st.session_state.model is None or st.session_state.tokenizer is None:
        with st.spinner("Loading model..."):
            st.session_state.model, st.session_state.tokenizer = load_model()
    
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    # 간결한 대화 형식으로 변경
    input_text = ""
    for i, msg in enumerate(history):
        if i >= len(history) - 3:  # 최근 3개의 메시지만 포함
            try:
                role = msg.get('role', 'user' if i % 2 == 0 else 'assistant')
                content = msg.get('content', '')
                role_text = 'Human' if role == 'user' else 'Assistant'
                input_text += f"{role_text}: {content}\n"
            except Exception as e:
                st.error(f"메시지 처리 중 오류 발생: {str(e)}")
                
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