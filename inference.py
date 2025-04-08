import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def load_model_and_tokenizer(base_model_name="google/gemma-2b-it", adapter_path="/qlora_output"):
    """파인튜닝된 모델과 토크나이저를 로드합니다."""
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
        token=os.getenv('HUGGINGFACE_TOKEN')
    )
    
    print("베이스 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv('HUGGINGFACE_TOKEN')
    )
    
    print("LoRA 어댑터 로드 중...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_translation(text, model, tokenizer, max_length=512):
    """주어진 텍스트에 대한 번역을 생성합니다."""
    prompt = f"### 입력:\n{text}\n\n### 응답:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 프롬프트 제거하고 응답만 반환
    response = response.split("### 응답:\n")[-1].strip()
    return response

def main():
    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer()
    
    # 테스트할 예시 텍스트들
    test_texts = [
        "TABLE 9 ECTOPARASITICIDE DIPS MARKET, BY COUNTRY, 2018–2025 (USD MILLION)",
        "The cat sat on the mat.",
        "Hello, how are you today?"
    ]
    
    print("\n=== 번역 테스트 시작 ===")
    for text in test_texts:
        print("\n입력:", text)
        translation = generate_translation(text, model, tokenizer)
        print("번역:", translation)
        print("-" * 50)

if __name__ == "__main__":
    main() 