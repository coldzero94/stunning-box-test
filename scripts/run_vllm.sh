#!/bin/bash

# VLLM 서버 실행 스크립트
# 사용법: ./run_vllm.sh [start|stop|status|help]

# 환경 설정
# Docker 환경 변수 로드 (필요한 경우 주석 해제)
# source /etc/profile
# source ~/.bashrc

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 스크립트 위치 디렉토리로 이동
cd "$(dirname "$0")"

# Docker compose 파일 경로 설정
DOCKER_COMPOSE_PATH="../backend/docker-compose.yml"

# OS 감지 함수
detect_os() {
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
  elif type lsb_release >/dev/null 2>&1; then
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
  elif [ -f /etc/lsb-release ]; then
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
  else
    OS=$(uname -s)
    VER=$(uname -r)
  fi
  echo "$OS"
}

# Docker 설치 함수
install_docker() {
  echo -e "${YELLOW}🔄 Docker를 설치합니다...${NC}"
  
  OS=$(detect_os)
  echo -e "${BLUE}🖥️ 감지된 OS: $OS${NC}"
  
  if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    echo -e "${BLUE}📦 Ubuntu/Debian 계열 시스템에 Docker 설치 중...${NC}"
    sudo apt update
    sudo apt install -y docker.io docker-compose-plugin
    sudo systemctl enable --now docker
  elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
    echo -e "${BLUE}📦 CentOS/RHEL 계열 시스템에 Docker 설치 중...${NC}"
    sudo yum install -y yum-utils
    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo systemctl enable --now docker
  elif [[ "$OS" == *"Darwin"* ]] || [[ "$OS" == *"macOS"* ]]; then
    echo -e "${YELLOW}⚠️ macOS에는 Docker Desktop이 필요합니다.${NC}"
    echo -e "${BLUE}📥 다음 URL에서 Docker Desktop을 설치해주세요: https://www.docker.com/products/docker-desktop${NC}"
    exit 1
  else
    echo -e "${RED}❌ 지원되지 않는 OS: $OS${NC}"
    echo -e "${YELLOW}📄 Docker 수동 설치 가이드: https://docs.docker.com/engine/install/${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}✅ Docker 설치가 완료되었습니다.${NC}"
  sudo docker --version
  echo ""
}

# Docker Compose 설치 함수 (필요한 경우)
install_docker_compose() {
  echo -e "${YELLOW}🔄 Docker Compose를 설치합니다...${NC}"
  
  # 최신 Docker 버전에는 docker compose 플러그인이 포함되어 있지만
  # 혹시 필요한 경우를 대비해 별도 설치 함수 제공
  OS=$(detect_os)
  
  if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    sudo apt update
    sudo apt install -y docker-compose-plugin
  elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
    sudo yum install -y docker-compose-plugin
  fi
  
  echo -e "${GREEN}✅ Docker Compose 설치가 완료되었습니다.${NC}"
}

# Docker 확인 및 설치 함수
check_and_install_docker() {
  # Docker 명령어 경로 설정
  if [ -x "/usr/bin/docker" ]; then
    DOCKER_CMD="/usr/bin/docker"
  elif [ -x "/usr/local/bin/docker" ]; then
    DOCKER_CMD="/usr/local/bin/docker"
  else
    DOCKER_CMD="docker"
  fi
  
  echo -e "${BLUE}🔍 Docker 설치 여부 확인 중...${NC}"
  
  if command -v $DOCKER_CMD &> /dev/null; then
    echo -e "${GREEN}✅ Docker가 설치되어 있습니다.${NC}"
    $DOCKER_CMD --version
    echo ""
    
    if $DOCKER_CMD compose version &> /dev/null; then
      echo -e "${GREEN}✅ Docker Compose가 설치되어 있습니다.${NC}"
      $DOCKER_CMD compose version
    else
      echo -e "${YELLOW}⚠️ Docker Compose가 설치되어 있지 않습니다. 자동으로 설치합니다...${NC}"
      install_docker_compose
    fi
  else
    echo -e "${YELLOW}⚠️ Docker가 설치되어 있지 않습니다. 자동으로 설치합니다...${NC}"
    install_docker
    # Docker 설치 후 경로 재설정
    if [ -x "/usr/bin/docker" ]; then
      DOCKER_CMD="/usr/bin/docker"
    elif [ -x "/usr/local/bin/docker" ]; then
      DOCKER_CMD="/usr/local/bin/docker"
    else
      DOCKER_CMD="docker"
    fi
    
    # Docker Compose 확인
    if ! $DOCKER_CMD compose version &> /dev/null; then
      install_docker_compose
    fi
  fi
  
  # Docker 그룹 권한 안내
  if ! groups | grep -q docker; then
    echo -e "${YELLOW}⚠️ 현재 사용자가 docker 그룹에 없습니다.${NC}"
    echo -e "${YELLOW}💡 sudo 없이 Docker를 사용하려면 다음 명령어를 실행하세요:${NC}"
    echo -e "${BLUE}   sudo usermod -aG docker $USER${NC}"
    echo -e "${YELLOW}   (명령어 실행 후 재로그인 필요)${NC}"
    echo ""
  fi
  
  return 0
}

# 서버 시작 함수
start_server() {
  echo -e "${BLUE}🚀 VLLM 서버를 시작합니다...${NC}"
  
  # Docker 확인 및 설치
  check_and_install_docker
  
  echo -e "${BLUE}📂 Docker compose 설정 확인 중...${NC}"
  if [ ! -f "$DOCKER_COMPOSE_PATH" ]; then
    echo -e "${RED}❌ Docker compose 파일을 찾을 수 없습니다: $DOCKER_COMPOSE_PATH${NC}"
    exit 1
  fi
  
  echo -e "${BLUE}🔄 서버 시작 중...${NC}"
  $DOCKER_CMD compose -f $DOCKER_COMPOSE_PATH up -d
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker 명령 실행 실패.${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}✅ 서버가 백그라운드에서 실행 중입니다.${NC}"
  echo -e "${GREEN}🔗 API 엔드포인트: http://localhost:8000${NC}"
  echo ""
  echo -e "${BLUE}📝 로그 확인: $DOCKER_CMD compose -f $DOCKER_COMPOSE_PATH logs -f${NC}"
}

# 서버 중지 함수
stop_server() {
  echo -e "${BLUE}🛑 VLLM 서버를 중지합니다...${NC}"
  
  # Docker 확인
  check_and_install_docker
  
  $DOCKER_CMD compose -f $DOCKER_COMPOSE_PATH down
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 서버가 중지되었습니다.${NC}"
  else
    echo -e "${RED}❌ 서버 중지 실패.${NC}"
    exit 1
  fi
}

# 서버 상태 확인 함수
check_status() {
  # Docker 확인
  check_and_install_docker
  
  if $DOCKER_CMD compose -f $DOCKER_COMPOSE_PATH ps | grep -q "vllm-server"; then
    echo -e "${GREEN}✅ VLLM 서버가 실행 중입니다.${NC}"
    echo -e "${BLUE}📊 서버 상태:${NC}"
    $DOCKER_CMD compose -f $DOCKER_COMPOSE_PATH ps
  else
    echo -e "${YELLOW}❌ VLLM 서버가 실행 중이 아닙니다.${NC}"
  fi
}

# 도움말 표시 함수
show_help() {
  echo -e "${BLUE}📚 VLLM 서버 실행 스크립트 사용법${NC}"
  echo ""
  echo -e "사용법: ${GREEN}./run_vllm.sh [명령어]${NC}"
  echo ""
  echo "명령어:"
  echo -e "  ${GREEN}start${NC}   - 서버 시작 (백그라운드 모드)"
  echo -e "  ${GREEN}stop${NC}    - 서버 중지"
  echo -e "  ${GREEN}status${NC}  - 서버 상태 확인"
  echo -e "  ${GREEN}help${NC}    - 이 도움말 표시"
  echo ""
  echo -e "예시: ${GREEN}./run_vllm.sh start${NC}"
  echo ""
  echo "API 호출 예시:"
  echo -e "${BLUE}  curl http://localhost:8000/v1/chat/completions \\${NC}"
  echo -e "${BLUE}    -H 'Content-Type: application/json' \\${NC}"
  echo -e "${BLUE}    -d '{ \"model\": \"Qwen/Qwen2.5-14B-Instruct\", \"messages\": [{ \"role\": \"user\", \"content\": \"안녕하세요\" }], \"stream\": true }'${NC}"
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
    echo -e "${RED}❌ 알 수 없는 명령어: $1${NC}"
    echo ""
    show_help
    exit 1
    ;;
esac

exit 0 