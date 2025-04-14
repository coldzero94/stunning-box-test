#!/bin/bash

# 스크립트 디렉토리를 기준으로 프로젝트 루트 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 디버깅 정보 출력
echo "스크립트 디렉토리: $SCRIPT_DIR"
echo "프로젝트 루트: $PROJECT_ROOT"

# 가상환경 활성화
if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "가상환경 활성화됨"
else
    echo "가상환경을 찾을 수 없습니다. setup.sh를 먼저 실행해주세요."
    exit 1
fi

# 필요한 패키지 설치 확인 및 설치
echo "필요한 패키지 설치 확인 중..."
pip install -q gradio vllm transformers

# frontend 디렉토리로 이동
cd "${PROJECT_ROOT}/frontend"
echo "현재 디렉토리: $(pwd)"

# Gradio 실행
echo "Gradio 시작 중..."
python app.py --model-id Qwen/Qwen2.5-14B-Instruct --port 8000
