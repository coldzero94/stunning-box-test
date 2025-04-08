#!/bin/bash

# 에러가 발생하면 스크립트 실행을 중단
set -e

# CUDA 환경 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "=== 환경 설정 시작 ==="

# Python 3.11 설치 확인
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 설치 중..."
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# pip 업그레이드
python3.11 -m pip install --upgrade pip

# 가상환경 생성 및 활성화
python3.11 -m venv venv
source venv/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt

# CUDA 사용 가능 여부 확인
python3.11 -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available())"
if [ $? -ne 0 ]; then
    echo "경고: CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다."
fi

echo "=== 데이터셋 준비 시작 ==="

# training_data 디렉토리 생성
mkdir -p training_data

echo "=== 파인튜닝 시작 ==="

# 파인튜닝 실행
python3.11 train_qlora.py

echo "=== 파인튜닝 완료 ==="

echo "모든 작업이 완료되었습니다."