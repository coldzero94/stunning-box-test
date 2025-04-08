#!/bin/bash

# 에러가 발생하면 스크립트 실행을 중단
set -e

echo "=== 환경 설정 시작 ==="

# Python 3.11 설치 (필요한 경우)
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 설치 중..."
    # Ubuntu/Debian 기준
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# pip 업그레이드
python3.11 -m pip install --upgrade pip

# 가상환경 생성 및 활성화
echo "가상환경 생성 중..."
python3.11 -m venv venv
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

# 기존 training_data 파일 복사 (필요한 경우)
# scp -r user@source_server:/path/to/training_data/* training_data/

echo "=== 파인튜닝 시작 ==="

# 파인튜닝 실행
python train_qlora.py

echo "=== 파인튜닝 완료 ==="

# 결과 모델 저장 (필요한 경우)
# tar -czf qlora_output.tar.gz qlora_output/

echo "모든 작업이 완료되었습니다." 