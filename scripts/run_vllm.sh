#!/bin/bash

# VLLM 설치
pip install vllm
pip install bitsandbytes

# PID 파일 설정
PID_FILE="/tmp/vllm_server.pid"
LOG_FILE="/tmp/vllm_server.log"
READY_FILE="/tmp/vllm_server_ready"

# 기존 로그 및 준비 파일 제거
rm -f "$LOG_FILE" "$READY_FILE"

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
  --gpu-memory-utilization 0.6 > "$LOG_FILE" 2>&1 &

# 프로세스 ID 저장
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"
echo "VLLM 서버가 백그라운드로 시작되었습니다. (PID: $SERVER_PID)"

# 서버가 초기 시작되었는지 확인
echo "서버 시작을 확인하는 중..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:8000/ping" > /dev/null; then
        echo "VLLM 서버 초기 시작 확인됨 - API 서버는 활성화되었습니다."
        echo "이제 모델이 로드될 때까지 기다립니다. (약 7분 소요)"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "서버 시작 확인 중... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "VLLM 서버 초기 시작에 실패했습니다. 로그를 확인하세요: $LOG_FILE"
    echo "백그라운드에서 계속 시작을 시도합니다."
else
    # 로그 모니터링 함수를 백그라운드로 실행하여 모델 로딩 완료 감지
    (
        # 로딩 완료 메시지 패턴들
        PATTERNS=(
            "Engine 000: Avg prompt throughput"
            "Application startup complete"
        )
        
        echo "모델 로딩 진행 상황을 모니터링합니다..."
        
        while true; do
            # 15초마다 로그 확인
            sleep 15
            
            # 모든 패턴을 확인
            for PATTERN in "${PATTERNS[@]}"; do
                if grep -q "$PATTERN" "$LOG_FILE"; then
                    echo "$(date) - VLLM 서버 준비 완료! 모델이 성공적으로 로드되었습니다."
                    # 준비 완료 파일 생성
                    echo "$(date)" > "$READY_FILE"
                    
                    # 로컬에서 실행 중이라면 데스크톱 알림 표시
                    if command -v notify-send &> /dev/null; then
                        notify-send "VLLM 서버 준비 완료" "모델이 성공적으로 로드되었습니다!"
                    fi
                    
                    exit 0
                fi
            done
            
            # 실제 요청을 보내서 확인 (가벼운 요청)
            if curl -s -X POST "http://localhost:8000/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{"model":"Qwen/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":1}' \
                -o /dev/null -w "%{http_code}" | grep -q "200"; then
                
                echo "$(date) - VLLM 서버 준비 완료! API 요청 테스트 성공."
                # 준비 완료 파일 생성
                echo "$(date)" > "$READY_FILE"
                
                # 로컬에서 실행 중이라면 데스크톱 알림 표시
                if command -v notify-send &> /dev/null; then
                    notify-send "VLLM 서버 준비 완료" "모델이 성공적으로 로드되었습니다!"
                fi
                
                exit 0
            fi
            
            echo "모델 로딩 중... 기다려 주세요."
        done
    ) &
    
    # 모니터링 프로세스 ID 저장
    MONITOR_PID=$!
    echo "모델 로딩 모니터링이 백그라운드에서 실행 중입니다. (PID: $MONITOR_PID)"
    echo "서버가 준비되면 알림이 표시됩니다."
    echo "준비 상태 확인: cat $READY_FILE"
fi

echo "서버 로그: $LOG_FILE"