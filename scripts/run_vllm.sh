#!/bin/bash

# VLLM 설치
pip install vllm
pip install bitsandbytes

# PID 파일 설정
PID_FILE="/tmp/vllm_server.pid"

# 이미 실행 중인 서버가 있는지 확인
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null; then
        echo "VLLM 서버가 이미 실행 중입니다. (PID: $OLD_PID)"
        echo "기존 서버를 사용하려면 계속 진행하세요."
        echo "다시 시작하려면 먼저 다음 명령어로 종료하세요:"
        echo "  kill $OLD_PID && rm $PID_FILE"
        exit 0
    else
        echo "이전 PID 파일 발견, 하지만 서버가 실행 중이지 않습니다. 파일을 제거합니다."
        rm "$PID_FILE"
    fi
fi

# 서버 백그라운드로 실행
echo "VLLM 서버를 백그라운드로 시작합니다..."
nohup python -m vllm.entrypoints.openai.api_server \
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
  --gpu-memory-utilization 0.6 > /tmp/vllm_server.log 2>&1 &

# 프로세스 ID 저장
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"
echo "VLLM 서버가 백그라운드로 시작되었습니다. (PID: $SERVER_PID)"

# 서버가 시작될 때까지 잠시 대기
echo "서버 시작을 확인하는 중..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:8000/v1/models" > /dev/null; then
        echo "VLLM 서버가 성공적으로 시작되었습니다!"
        echo "서버 로그: /tmp/vllm_server.log"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "서버 시작 확인 중... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "VLLM 서버 시작 시간이 초과되었습니다. 로그를 확인하세요: /tmp/vllm_server.log"
    echo "백그라운드에서 계속 시작을 시도합니다."
fi