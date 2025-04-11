#!/bin/bash

# 에러가 발생하면 스크립트 실행을 중단
set -e

# CUDA 환경 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 기본값 설정
START_PAGE=30
END_PAGE=40
USE_GPT=true  # 기본값을 true로 변경
NO_INTERACTIVE=true  # 기본값을 true로 변경
BASE_MODEL_NAME="google/gemma-2b-it"  # 기본 모델
ADAPTER_PATH="gemma-2b"  # 기본 어댑터 경로
KOREAN_PDF="korean.pdf"  # 한글 PDF 파일 경로
ENGLISH_PDF="english.pdf"  # 영어 PDF 파일 경로
OUTPUT_DIR="/evaluation_results"  # 결과 저장 디렉토리 (상대 경로로 수정)

# 명령줄 인자 파싱
while [[ $# -gt 0 ]]; do
  case $1 in
    --start-page)
      START_PAGE="$2"
      shift 2
      ;;
    --end-page)
      END_PAGE="$2"
      shift 2
      ;;
    --use-gpt)
      USE_GPT=true
      shift
      ;;
    --no-use-gpt)  # GPT 사용 안 함 옵션 추가
      USE_GPT=false
      shift
      ;;
    --interactive)  # 대화형 모드 활성화 옵션 추가
      NO_INTERACTIVE=false
      shift
      ;;
    --base-model)  # 베이스 모델 이름 옵션 추가
      BASE_MODEL_NAME="$2"
      shift 2
      ;;
    --adapter-path)  # 어댑터 경로 옵션 추가
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --korean-pdf)  # 한글 PDF 파일 경로 옵션 추가
      KOREAN_PDF="$2"
      shift 2
      ;;
    --english-pdf)  # 영어 PDF 파일 경로 옵션 추가
      ENGLISH_PDF="$2"
      shift 2
      ;;
    --output-dir)  # 출력 디렉토리 옵션 추가
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "알 수 없는 옵션: $1"
      exit 1
      ;;
  esac
done

echo "=== 환경 설정 시작 ==="
echo "시작 페이지: $START_PAGE"
echo "끝 페이지: $END_PAGE"
echo "베이스 모델: $BASE_MODEL_NAME"
echo "어댑터 경로: $ADAPTER_PATH"
echo "한글 PDF: $KOREAN_PDF"
echo "영어 PDF: $ENGLISH_PDF"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "GPT 사용: $([ "$USE_GPT" = true ] && echo "활성화" || echo "비활성화")"
echo "대화형 모드: $([ "$NO_INTERACTIVE" = true ] && echo "비활성화" || echo "활성화")"

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

# NLTK 데이터 다운로드
echo "NLTK 데이터 다운로드 중..."
python -c "import nltk; nltk.download('punkt')"

# CUDA 사용 가능 여부 확인
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA 사용 가능:"
    nvidia-smi
else
    echo "경고: CUDA가 감지되지 않았습니다. GPU 가속이 없을 수 있습니다."
fi

echo "=== 환경 설정 완료 ==="

echo "=== 번역 평가 시작 ==="

# OpenAI API 키 확인
echo "OpenAI API 키 확인 중..."
python check_openai_key.py

# 번역 평가 실행 명령어 구성
EVAL_CMD="python evaluate_translation.py --start-page $START_PAGE --end-page $END_PAGE --base-model-name $BASE_MODEL_NAME --adapter-path $ADAPTER_PATH --korean-pdf \"$KOREAN_PDF\" --english-pdf \"$ENGLISH_PDF\" --output-dir $OUTPUT_DIR --use-gpt"

# 추가 옵션 설정
if [ "$USE_GPT" = true ]; then
    EVAL_CMD="$EVAL_CMD --use-gpt"
fi

if [ "$NO_INTERACTIVE" = true ]; then
    EVAL_CMD="$EVAL_CMD --no-interactive"
fi

# 번역 평가 실행
echo "번역 평가 실행 중... (시작 페이지: $START_PAGE, 끝 페이지: $END_PAGE)"
echo "실행 명령어: $EVAL_CMD"
eval $EVAL_CMD

echo "=== 번역 평가 완료 ==="

echo "모든 작업이 완료되었습니다."