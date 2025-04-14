#!/bin/bash

# 에러가 발생하면 스크립트 실행을 중단
set -e

# CUDA 환경 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 작업 디렉토리 설정 (VESSL 환경)
WORK_DIR="/workspace"
cd "$WORK_DIR"

# 현재 디렉토리를 프로젝트 루트로 설정
PROJECT_ROOT="$(pwd)"

echo "=== 환경 설정 시작 ==="
echo "작업 디렉토리: $(pwd)"
echo "프로젝트 루트: $PROJECT_ROOT"

# 사용 가능한 Python 버전 확인
echo "사용 가능한 Python 버전 확인 중..."
if command -v python3 &> /dev/null; then
    echo "Python3 사용 가능:"
    python3 --version
elif command -v python &> /dev/null; then
    echo "Python 사용 가능:"
    python --version
else
    echo "Python이 설치되어 있지 않습니다."
    exit 1
fi

# pip 업그레이드
echo "pip 업그레이드 중..."
if command -v python3 &> /dev/null; then
    python3 -m pip install --upgrade pip
else
    python -m pip install --upgrade pip
fi

# 가상환경 생성 및 활성화
echo "가상환경 생성 중..."
if command -v python3 &> /dev/null; then
    python3 -m venv venv
else
    python -m venv venv
fi
echo "가상환경 활성화 중..."
source venv/bin/activate

# 필요한 패키지 설치
echo "필요한 패키지 설치 중..."
if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    pip install -r "${PROJECT_ROOT}/requirements.txt"
else
    echo "requirements.txt 파일을 찾을 수 없습니다. 기본 패키지만 설치합니다."
    pip install streamlit torch transformers accelerate sentencepiece protobuf safetensors
fi

# CUDA 사용 가능 여부 확인
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA 사용 가능:"
    nvidia-smi
else
    echo "경고: CUDA가 감지되지 않았습니다. GPU 가속이 없을 수 있습니다."
fi

echo "=== 환경 설정 완료 ==="

