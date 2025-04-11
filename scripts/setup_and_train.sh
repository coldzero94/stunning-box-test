#!/bin/bash

# 에러가 발생하면 스크립트 실행을 중단
set -e

# CUDA 환경 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


echo "=== 환경 설정 시작 ==="

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
source venv/bin/activate

# 필요한 패키지 설치
echo "필요한 패키지 설치 중..."
pip install -r requirements.txt

# CUDA 사용 가능 여부 확인
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA 사용 가능:"
    nvidia-smi
else
    echo "경고: CUDA가 감지되지 않았습니다. GPU 가속이 없을 수 있습니다."
fi

echo "=== 데이터셋 준비 시작 ==="

# training_data 디렉토리 생성
mkdir -p training_data

echo "=== 파인튜닝 시작 ==="

# 파인튜닝 실행
python train_qlora.py

echo "=== 파인튜닝 완료 ==="

echo "모든 작업이 완료되었습니다."