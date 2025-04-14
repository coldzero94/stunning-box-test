#!/bin/bash

# 에러가 발생하면 스크립트 실행을 중단
set -e

# CUDA 환경 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 작업 디렉토리 설정 (VESSL 환경)
WORK_DIR="/root"
cd "$WORK_DIR"

# 현재 디렉토리를 프로젝트 루트로 설정
PROJECT_ROOT="$(pwd)"

echo "=== FastAPI 서버 시작 ==="
echo "작업 디렉토리: $(pwd)"
echo "프로젝트 루트: $PROJECT_ROOT"

# 가상환경 활성화
echo "가상환경 활성화 중..."
source venv/bin/activate

# FastAPI 실행
echo "FastAPI 서버 시작 중..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300 --workers 1 