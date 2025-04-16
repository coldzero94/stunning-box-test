#!/usr/bin/env python3
import os
import argparse
import sys
from vllm.entrypoints.openai.api_server import main as vllm_api_server_main

# 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

def main():
    """
    VLLM의 내장 OpenAI 호환 서버를 실행합니다.
    파인튜닝된 Qwen 모델을 사용하여 OpenAI 호환 API를 제공합니다.
    """
    # 기본 모델 경로 설정
    base_model_name = "Qwen/Qwen2.5-14B-Instruct"
    adapter_path = "/qwen25-14b"  # 파인튜닝된 어댑터 경로
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="VLLM OpenAI 호환 API 서버")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--model", type=str, default=base_model_name, help="기본 모델 경로")
    parser.add_argument("--peft-model-path", type=str, default=adapter_path, help="PEFT/LoRA 어댑터 경로")
    args, unknown = parser.parse_known_args()
    
    print(f"🔧 기본 모델: {args.model}")
    print(f"🔄 파인튜닝 어댑터: {args.peft_model_path}")
    print(f"🚀 서버 시작 준비 중... http://{args.host}:{args.port}")
    
    # VLLM OpenAI 호환 서버 실행 인자 구성
    vllm_args = [
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--peft-model-path", args.peft_model_path,
        "--dtype", "half",  # float16로 변환
        "--max-model-len", "4096",
        "--trust-remote-code",  # 원격 코드 신뢰 (Qwen 모델에 필요)
        "--max-num-seqs", "256",  # 최대 동시 요청 수
        "--gpu-memory-utilization", "0.9"
    ]
    
    # VLLM 내장 OpenAI API 서버 실행
    print("✨ VLLM OpenAI 호환 서버를 시작합니다...")
    print(f"💡 API 호출 예시: curl http://{args.host}:{args.port}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\n  \"model\": \"{args.model}\",\n  \"messages\": [{{\n    \"role\": \"user\",\n    \"content\": \"안녕하세요\"\n  }}],\n  \"stream\": true\n}}'")
    
    # VLLM API 서버 메인 함수 호출
    sys.argv = ["vllm_server.py"] + vllm_args
    vllm_api_server_main()

if __name__ == "__main__":
    main() 