#!/bin/bash

# 스크립트 디렉토리를 기준으로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# 백그라운드에서 VLLM 서버 실행
echo "1. VLLM 서버를 백그라운드에서 시작합니다..."
echo "모델 로딩에는 약 7분 정도 소요될 수 있습니다."

# 명령어 직접 실행 (백그라운드 실행 대신)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 7000 \
  --enable-lora \
  --qlora-adapter-name-or-path /qwen25-14b \
  --quantization bitsandbytes \
  --dtype half \
  --max-model-len 4096 \
  --trust-remote-code \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.6 > "$LOG_FILE" 2>&1 &

# VLLM 서버 PID 저장
VLLM_PID=$!
echo $VLLM_PID > "/tmp/vllm_server.pid"

echo "VLLM 서버가 백그라운드로 시작되었습니다. (PID: $VLLM_PID)"
echo "서버가 준비될 때까지 기다립니다..."

# 서버가 준비될 때까지 대기
echo "모델 로딩 확인 중 (약 7분 소요)..."

# 첫번째 - API 서버가 응답하는지 확인
MAX_RETRIES=100
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:7000/ping" > /dev/null; then
        echo "API 서버 응답 확인됨 - 모델 로딩 중..."
        SERVER_RESPONDING=true
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "API 서버 응답 대기 중... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 5
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "경고: API 서버가 응답하지 않습니다. Gradio를 계속 실행합니다."
    echo "나중에 서버가 준비되면 자동으로 연결됩니다."
    SERVER_RESPONDING=false
fi

# 서버가 응답하면 모델 로딩 완료 확인
if [ "$SERVER_RESPONDING" = true ]; then
    # 두번째 - 모델이 완전히 로드되었는지 확인
    echo "모델 로딩이 완료될 때까지 기다립니다..."
    
    # 최대 5분간 모델 로딩 확인 (300초)
    MAX_WAIT=300
    WAIT_COUNT=0
    MODEL_LOADED=false
    
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        # 서버 로그 확인
        if grep -q "Engine 000: Avg prompt throughput" "$LOG_FILE"; then
            echo "모델 로딩 완료됨!"
            MODEL_LOADED=true
            break
        fi
        
        # 실제 요청으로 확인
        if curl -s -X POST "http://localhost:7000/v1/chat/completions" \
           -H "Content-Type: application/json" \
           -d '{"model":"Qwen/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":1}' \
           -o /dev/null -w "%{http_code}" | grep -q "200"; then
            
            echo "모델 로딩 완료! API 요청 테스트 성공."
            MODEL_LOADED=true
            break
        fi
        
        WAIT_COUNT=$((WAIT_COUNT + 10))
        echo "모델 로딩 중... 기다려 주세요. ($WAIT_COUNT/$MAX_WAIT 초)"
        sleep 10
    done
    
    if [ "$MODEL_LOADED" = false ]; then
        echo "모델 로딩 시간이 초과되었습니다. Gradio를 계속 실행합니다."
    else
        echo "VLLM 서버가 완전히 준비되었습니다!"
    fi
fi

# Gradio 앱 시작
echo "2. Gradio 웹 인터페이스 시작 중..."
echo "====== Gradio 시작 ======"

# 현재 디렉토리를 frontend로 변경
cd "/root/frontend" || exit 1
echo "현재 디렉토리: $(pwd)"

# Gradio 실행
python app.py \
    --api-base-url "http://localhost:7000" \
    --port 8000
echo "Gradio 웹 인터페이스가 시작되었습니다."

# 종료 시 VLLM 서버도 함께 종료
echo "애플리케이션을 종료합니다..."
if ps -p "$VLLM_PID" > /dev/null; then
    echo "VLLM 서버 종료 중 (PID: $VLLM_PID)..."
    kill "$VLLM_PID"
    rm "/tmp/vllm_server.pid"
    echo "VLLM 서버가 종료되었습니다."
fi

echo "====== 종료 ======" 