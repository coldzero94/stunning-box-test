#!/bin/bash

# 스크립트 디렉토리를 기준으로 프로젝트 루트 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 디버깅 정보 출력
echo "스크립트 디렉토리: $SCRIPT_DIR"
echo "프로젝트 루트: $PROJECT_ROOT"

# 필요한 패키지 설치
pip install requests sseclient-py gradio

# VLLM 서버 설정
VLLM_API_URL="http://localhost:8000"
GRADIO_PORT=7860

# VLLM 서버가 실행 중인지 확인
echo "VLLM 서버 연결 테스트 중..."
if curl -s "$VLLM_API_URL/v1/models" > /dev/null; then
    echo "VLLM 서버가 실행 중입니다: $VLLM_API_URL"
else
    echo "경고: VLLM 서버에 연결할 수 없습니다!"
    echo "VLLM 서버가 실행 중인지 확인하세요."
    echo "VLLM 서버를 시작하려면 다음 명령어를 실행하세요:"
    echo "  ./scripts/run_vllm.sh"
    
    # 계속 진행할지 확인
    read -p "VLLM 서버 없이 계속 진행할까요? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "종료 중..."
        exit 1
    fi
fi

# frontend 디렉토리로 이동
cd "${PROJECT_ROOT}/frontend"
echo "현재 디렉토리: $(pwd)"

# Gradio 실행
echo "Gradio 시작 중..."
python app.py \
    --api-base-url "$VLLM_API_URL" \
    --port "$GRADIO_PORT"
