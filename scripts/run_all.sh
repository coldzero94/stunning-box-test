#!/bin/bash

# 스크립트 디렉토리를 기준으로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "====== 스터닝 박스 시작 ======"
echo "작업 디렉토리: $(pwd)"

# 스크립트 실행 권한 부여
chmod -R +x "$SCRIPT_DIR"

# 준비 파일 설정
READY_FILE="/tmp/vllm_server_ready"
# 기존 준비 파일 제거
rm -f "$READY_FILE"

# VLLM 서버 시작
echo "1. VLLM 서버 시작 중..."
"$SCRIPT_DIR/run_vllm.sh"

# 사용자에게 상태 알림
echo ""
echo "======================================"
echo "VLLM 서버가 백그라운드에서 시작 중입니다."
echo "모델 로딩에는 약 7분 정도 소요될 수 있습니다."
echo ""
echo "모델 로딩이 완료되면 다음 파일에 기록됩니다:"
echo "  $READY_FILE"
echo ""
echo "로딩 완료 확인 방법: cat $READY_FILE"
echo "로그 확인 방법: tail -f /tmp/vllm_server.log"
echo "======================================"
echo ""

# Gradio 앱 시작
echo "2. Gradio 웹 인터페이스 시작 중..."
echo "====== Gradio 시작 ======"
"$SCRIPT_DIR/start_app.sh"

echo "====== 종료 ======"

# 서버 준비 상태 확인 코드
# (백그라운드에서 이미 모니터링 중이므로 여기서는 추가 작업 불필요) 