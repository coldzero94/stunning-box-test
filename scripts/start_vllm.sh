#!/bin/bash

# VLLM OpenAI 호환 서버 실행 스크립트
# 파인튜닝된 Qwen 모델을 사용한 LLM 서버를 시작합니다.

# 스크립트 디렉토리를 기준으로 프로젝트 루트 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 기본 설정값
HOST="0.0.0.0"
PORT=8000
BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH="$PROJECT_ROOT/qwen25-14b"  # 프로젝트 루트 기준 상대 경로
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/vllm_server.log"
DTYPE="half"  # 또는 "bfloat16", "float16"
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=256
GPU_MEM_UTIL=0.9

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 디버깅 정보 출력
echo "================ 환경 정보 ================"
echo "스크립트 디렉토리: $SCRIPT_DIR"
echo "프로젝트 루트: $PROJECT_ROOT"
echo "모델 파인튜닝 경로: $ADAPTER_PATH"
echo "==========================================="

# 가상환경 활성화
if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "✅ 가상환경 활성화됨"
elif [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
    echo "✅ 가상환경(.venv) 활성화됨"
else
    echo "⚠️ 가상환경을 찾을 수 없습니다. 시스템 Python을 사용합니다."
fi

# 모델 디렉토리 확인 (파인튜닝된 모델인 경우)
if [[ "$ADAPTER_PATH" != *"/"* ]]; then
    echo "⚠️ 파인튜닝 어댑터 경로가 지정되지 않았습니다. 기본 모델만 사용합니다."
    USE_PEFT=false
elif [ ! -d "$ADAPTER_PATH" ]; then
    echo "⚠️ 파인튜닝 어댑터 디렉토리를 찾을 수 없습니다: $ADAPTER_PATH"
    echo "⚠️ 기본 모델만 사용합니다."
    USE_PEFT=false
else
    echo "✅ 파인튜닝 어댑터 경로 확인됨: $ADAPTER_PATH"
    USE_PEFT=true
fi

# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# 도움말 출력
function show_help {
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  -h, --help               도움말 표시"
    echo "  -m, --model MODEL        기본 모델 경로 (기본값: $BASE_MODEL)"
    echo "  -a, --adapter PATH       파인튜닝 어댑터 경로 (기본값: $ADAPTER_PATH)"
    echo "  --host HOST              서버 호스트 (기본값: $HOST)"
    echo "  -p, --port PORT          서버 포트 (기본값: $PORT)"
    echo "  --dtype TYPE             데이터 타입 [half, float16, bfloat16] (기본값: $DTYPE)"
    echo "  --max-model-len LEN      최대 모델 길이 (기본값: $MAX_MODEL_LEN)"
    echo "  --max-num-seqs NUM       최대 동시 요청 수 (기본값: $MAX_NUM_SEQS)"
    echo "  --gpu-mem-util FLOAT     GPU 메모리 사용률 0.0-1.0 (기본값: $GPU_MEM_UTIL)"
    echo ""
    exit 0
}

# 명령줄 인자 파싱
while (( "$#" )); do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -m|--model)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                BASE_MODEL=$2
                shift 2
            else
                echo "오류: --model 인자 다음에 값이 필요합니다."
                exit 1
            fi
            ;;
        -a|--adapter)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                ADAPTER_PATH=$2
                # 모델 디렉토리 다시 확인
                if [ ! -d "$ADAPTER_PATH" ]; then
                    echo "⚠️ 파인튜닝 어댑터 디렉토리를 찾을 수 없습니다: $ADAPTER_PATH"
                    USE_PEFT=false
                else
                    USE_PEFT=true
                fi
                shift 2
            else
                echo "오류: --adapter 인자 다음에 값이 필요합니다."
                exit 1
            fi
            ;;
        --host)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                HOST=$2
                shift 2
            else
                echo "오류: --host 인자 다음에 값이 필요합니다."
                exit 1
            fi
            ;;
        -p|--port)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                PORT=$2
                shift 2
            else
                echo "오류: --port 인자 다음에 값이 필요합니다."
                exit 1
            fi
            ;;
        --dtype)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                DTYPE=$2
                shift 2
            else
                echo "오류: --dtype 인자 다음에 값이 필요합니다."
                exit 1
            fi
            ;;
        --max-model-len)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                MAX_MODEL_LEN=$2
                shift 2
            else
                echo "오류: --max-model-len 인자 다음에 값이 필요합니다."
                exit 1
            fi
            ;;
        --max-num-seqs)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                MAX_NUM_SEQS=$2
                shift 2
            else
                echo "오류: --max-num-seqs 인자 다음에 값이 필요합니다."
                exit 1
            fi
            ;;
        --gpu-mem-util)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                GPU_MEM_UTIL=$2
                shift 2
            else
                echo "오류: --gpu-mem-util 인자 다음에 값이 필요합니다."
                exit 1
            fi
            ;;
        *)
            shift
            ;;
    esac
done

echo "======================= VLLM 서버 시작 ======================="
echo "📂 기본 모델: $BASE_MODEL"
if [ "$USE_PEFT" = true ]; then
    echo "📂 파인튜닝 어댑터: $ADAPTER_PATH"
else
    echo "⚠️ 파인튜닝 어댑터 없음 - 기본 모델만 사용합니다"
fi
echo "🌐 서버 주소: http://$HOST:$PORT"
echo "📊 데이터 타입: $DTYPE"
echo "📝 로그 파일: $LOG_FILE"
echo "=============================================================="

echo "🚀 서버를 시작합니다..."

# 현재 디렉토리를 프로젝트 루트로 변경
cd "$PROJECT_ROOT"
echo "현재 디렉토리: $(pwd)"

# VLLM OpenAI 호환 서버 직접 실행
if [ "$USE_PEFT" = true ]; then
    # 파인튜닝 모델 사용
    python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --peft-model-path "$ADAPTER_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        2>&1 | tee -a "$LOG_FILE"
else
    # 기본 모델만 사용
    python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        2>&1 | tee -a "$LOG_FILE"
fi

# 서버 종료시 메시지
echo "✅ VLLM 서버가 종료되었습니다."
