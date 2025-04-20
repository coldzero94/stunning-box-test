#!/bin/bash

# 스크립트 디렉토리를 기준으로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "====== 스터닝 박스 시작 ======"
echo "작업 디렉토리: $(pwd)"

# 스크립트 실행 권한 부여
chmod -R +x "$SCRIPT_DIR"

# VLLM 서버 시작
echo "1. VLLM 서버 시작 중..."
"$SCRIPT_DIR/run_vllm.sh"

# 서버가 시작되었는지 확인
echo "2. VLLM 서버 상태 확인 중..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:8000/v1/models" > /dev/null; then
        echo "VLLM 서버가 준비되었습니다!"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "서버 준비 확인 중... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "경고: VLLM 서버가 준비되지 않았습니다. 로그를 확인하세요: /tmp/vllm_server.log"
    echo "Gradio 인터페이스를 계속 시작합니다..."
fi

# Gradio 앱 시작
echo "3. Gradio 웹 인터페이스 시작 중..."
echo "====== Gradio 시작 ======"
"$SCRIPT_DIR/start_app.sh"

echo "====== 종료 ======" 