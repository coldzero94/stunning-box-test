#!/bin/bash

# 스크립트 디렉토리를 기준으로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "====== 스터닝 박스 시작 ======"
echo "작업 디렉토리: $(pwd)"

# VLLM 설치
pip install vllm
pip install bitsandbytes
pip install requests sseclient-py gradio

# 스크립트 실행 권한 부여
chmod -R +x "$SCRIPT_DIR"

# 로그 파일 설정
LOG_FILE="/tmp/vllm_server.log"
FIFO="/tmp/vllm_fifo"

# FIFO 생성
rm -f "$FIFO"
mkfifo "$FIFO"

# 첫 번째 터미널: VLLM 서버 시작
echo "1. VLLM 서버를 시작합니다..."
echo "모델 로딩에는 약 7분 정도 소요될 수 있습니다."

# 백그라운드에서 서버 실행하고 로그 모니터링
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-lora \
  --qlora-adapter-name-or-path /qwen25-14b \
  --quantization bitsandbytes \
  --dtype half \
  --max-model-len 4096 \
  --trust-remote-code \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.6 > "$LOG_FILE" 2>&1 &

VLLM_PID=$!
echo "VLLM 서버가 백그라운드로 시작되었습니다. (PID: $VLLM_PID)"

# 로그를 모니터링하면서 서버 준비 확인
echo "모델 로딩 상태를 확인합니다..."
(
    tail -f "$LOG_FILE" | while read -r line; do
        echo "$line"
        
        # 첫 번째 조건: API 서버 시작됨
        if [[ "$line" == *"Application startup complete"* ]]; then
            echo "API 서버가 시작되었습니다. 이제 모델 로딩을 기다립니다..."
        fi
        
        # 두 번째 조건: 모델 로딩 완료됨
        if [[ "$line" == *"Engine 000: Avg prompt throughput"* ]]; then
            echo "모델 로딩이 완료되었습니다!"
            echo "GO" > "$FIFO" &
            break
        fi
    done
) &

LOGGER_PID=$!

# FIFO에서 신호 기다리기 또는 일정 시간 후 자동 진행
echo "서버가 준비될 때까지 기다리는 중..."

# 최대 15분 기다림 (900초)
TIMEOUT=900
count=0

(
    while [ $count -lt $TIMEOUT ]; do
        # 5초마다 준비 상태 확인
        sleep 5
        count=$((count + 5))
        
        # API 요청 테스트
        if curl -s -X POST "http://localhost:8000/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"Qwen/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":1}' \
            -o /dev/null -w "%{http_code}" | grep -q "200"; then
            
            echo "API 요청 테스트 성공! 모델 로딩 완료됨."
            echo "GO" > "$FIFO" &
            break
        fi
        
        # 진행상황 표시
        if [ $((count % 60)) -eq 0 ]; then
            echo "아직 기다리는 중... ($count 초 경과)"
        fi
    done
    
    if [ $count -ge $TIMEOUT ]; then
        echo "타임아웃: $TIMEOUT 초 동안 서버가 준비되지 않았습니다."
        echo "그래도 계속 진행합니다..."
        echo "GO" > "$FIFO" &
    fi
) &

TIMEOUT_PID=$!

# FIFO에서 신호 읽기
read -r line < "$FIFO"

# 모니터링 프로세스 종료
kill $LOGGER_PID 2>/dev/null
kill $TIMEOUT_PID 2>/dev/null

echo "-------------------------------------------------------------"
echo "VLLM 서버가 실행 중입니다. Gradio 웹 인터페이스를 시작합니다."

# Gradio 앱 시작
echo "2. Gradio 웹 인터페이스 시작 중..."
echo "====== Gradio 시작 ======"

# 현재 디렉토리를 frontend로 변경
cd "${PROJECT_ROOT}/frontend" || exit 1
echo "현재 디렉토리: $(pwd)"

# Gradio 실행
python app.py \
    --api-base-url "http://localhost:8000" \
    --port 7860

echo "====== 종료 ======"

# 정리
echo "애플리케이션을 종료합니다..."
if ps -p "$VLLM_PID" > /dev/null; then
    echo "VLLM 서버 종료 중 (PID: $VLLM_PID)..."
    kill "$VLLM_PID"
    echo "VLLM 서버가 종료되었습니다."
fi

# 임시 파일 정리
rm -f "$FIFO" 