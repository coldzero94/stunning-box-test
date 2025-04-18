#!/bin/bash

# VLLM 서버 실행 스크립트
# 사용법: ./run_vllm.sh [start|stop|status|help]

# 스크립트 위치 디렉토리로 이동
cd "$(dirname "$0")"

# 함수 정의
start_server() {
  echo "🚀 VLLM 서버를 시작합니다..."
  docker compose up -d
  echo "✅ 서버가 백그라운드에서 실행 중입니다."
  echo "🔗 API 엔드포인트: http://localhost:8000"
  echo ""
  echo "📝 로그 확인: docker compose logs -f"
}

stop_server() {
  echo "🛑 VLLM 서버를 중지합니다..."
  docker compose down
  echo "✅ 서버가 중지되었습니다."
}

check_status() {
  if docker compose ps | grep -q "vllm-server"; then
    echo "✅ VLLM 서버가 실행 중입니다."
    echo "📊 서버 상태:"
    docker compose ps
  else
    echo "❌ VLLM 서버가 실행 중이 아닙니다."
  fi
}

show_help() {
  echo "📚 VLLM 서버 실행 스크립트 사용법"
  echo ""
  echo "사용법: ./run_vllm.sh [명령어]"
  echo ""
  echo "명령어:"
  echo "  start   - 서버 시작 (백그라운드 모드)"
  echo "  stop    - 서버 중지"
  echo "  status  - 서버 상태 확인"
  echo "  help    - 이 도움말 표시"
  echo ""
  echo "예시:"
  echo "  ./run_vllm.sh start"
  echo ""
  echo "API 호출 예시:"
  echo "  curl http://localhost:8000/v1/chat/completions \\"
  echo "    -H 'Content-Type: application/json' \\"
  echo "    -d '{ \"model\": \"Qwen/Qwen2.5-14B-Instruct\", \"messages\": [{ \"role\": \"user\", \"content\": \"안녕하세요\" }], \"stream\": true }'"
}

# 메인 로직
case "$1" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  status)
    check_status
    ;;
  help|"")
    show_help
    ;;
  *)
    echo "❌ 알 수 없는 명령어: $1"
    echo ""
    show_help
    exit 1
    ;;
esac

exit 0 