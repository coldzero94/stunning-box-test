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

# 모델 설정
MODEL_PATH="/qwen25-14b"  # 로컬 모델 경로
PORT=8000
MAX_NUM_SEQS=16
MAX_MODEL_LEN=32767
DTYPE="bfloat16"  # 또는 "float16", "auto"

# 모델 디렉토리 확인
if [ ! -d "$MODEL_PATH" ]; then
    echo "모델 디렉토리를 찾을 수 없습니다: $MODEL_PATH"
    exit 1
fi

echo "모델 경로: $MODEL_PATH"

# frontend 디렉토리로 이동
cd "${PROJECT_ROOT}/frontend"
echo "현재 디렉토리: $(pwd)"

# Gradio 실행
echo "Gradio 시작 중..."
python app.py \
    --model-id "$MODEL_PATH" \
    --port "$PORT" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE"
