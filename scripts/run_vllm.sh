#!/bin/bash

# VLLM 서버 실행 스크립트
# 사용법: ./run_vllm.sh [start|stop|status|help]

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

# sudo 명령어 설정 (루트인 경우 빈 문자열)
SUDO=""
if [ "$(id -u)" != "0" ]; then
  if command -v sudo &> /dev/null; then
    SUDO="sudo"
  else
    echo -e "${RED}❌ 루트 권한이 필요하지만 sudo 명령어를 찾을 수 없습니다.${NC}"
    echo -e "${YELLOW}루트 사용자로 다시 실행하거나 sudo를 설치하세요.${NC}"
    exit 1
  fi
fi

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
    apt update
    apt install -y docker.io docker-compose-plugin
    systemctl enable --now docker
  elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
    echo -e "${BLUE}📦 CentOS/RHEL 계열 시스템에 Docker 설치 중...${NC}"
    yum install -y yum-utils
    yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker
  elif [[ "$OS" == *"Darwin"* ]] || [[ "$OS" == *"macOS"* ]]; then
    echo -e "${YELLOW}⚠️ macOS에는 Docker Desktop이 필요합니다.${NC}"
    echo -e "${BLUE}📥 다음 URL에서 Docker Desktop을 설치해주세요: https://www.docker.com/products/docker-desktop${NC}"
    exit 1
  else
    echo -e "${RED}❌ 지원되지 않는 OS: $OS${NC}"
    echo -e "${YELLOW}📄 Docker 수동 설치 가이드: https://docs.docker.com/engine/install/${NC}"
    exit 1
  fi
  
  # PATH에 Docker 디렉토리 추가
  export PATH=$PATH:/usr/bin:/usr/local/bin
  
  # Docker 서비스 시작 확인
  if command -v systemctl &> /dev/null; then
    systemctl start docker || true
  fi
  
  echo -e "${GREEN}✅ Docker 설치가 완료되었습니다.${NC}"
  if command -v docker &> /dev/null; then
    docker --version
  else
    echo -e "${YELLOW}⚠️ Docker 명령어를 찾을 수 없습니다. PATH를 확인하세요.${NC}"
  fi
  echo ""
}

# Docker Compose 설치 함수
install_docker_compose() {
  echo -e "${YELLOW}🔄 Docker Compose를 설치합니다...${NC}"
  
  # 최신 Docker 버전에는 docker compose 플러그인이 포함되어 있지만
  # 혹시 필요한 경우를 대비해 별도 설치 함수 제공
  OS=$(detect_os)
  
  if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    apt update
    apt install -y docker-compose-plugin
  elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
    yum install -y docker-compose-plugin
  fi
  
  # 바이너리 직접 다운로드 (대안)
  if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${YELLOW}📥 Docker Compose 바이너리 직접 다운로드 시도...${NC}"
    COMPOSE_VERSION="v2.18.1"
    mkdir -p /usr/local/bin
    curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
  fi
  
  echo -e "${GREEN}✅ Docker Compose 설치가 완료되었습니다.${NC}"
}

# Docker 확인 및 설치 함수
check_and_install_docker() {
  # Docker 명령어 경로 설정
  export PATH=$PATH:/usr/bin:/usr/local/bin
  
  echo -e "${BLUE}🔍 Docker 설치 여부 확인 중...${NC}"
  
  if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker가 설치되어 있습니다.${NC}"
    docker --version
    echo ""
    
    if docker compose version &> /dev/null || command -v docker-compose &> /dev/null; then
      echo -e "${GREEN}✅ Docker Compose가 설치되어 있습니다.${NC}"
      docker compose version || docker-compose --version
    else
      echo -e "${YELLOW}⚠️ Docker Compose가 설치되어 있지 않습니다. 자동으로 설치합니다...${NC}"
      install_docker_compose
    fi
  else
    echo -e "${YELLOW}⚠️ Docker가 설치되어 있지 않습니다. 자동으로 설치합니다...${NC}"
    install_docker
    
    # PATH 업데이트 및 Docker 실행 확인
    export PATH=$PATH:/usr/bin:/usr/local/bin
    
    # 설치 후 Docker 검증
    if ! command -v docker &> /dev/null; then
      echo -e "${RED}❌ Docker 설치 실패 또는 PATH에 없습니다.${NC}"
      echo -e "${YELLOW}스크립트 실행 후 다음 명령어를 시도해보세요:${NC}"
      echo -e "${BLUE}  source /etc/profile && source ~/.bashrc${NC}"
      echo -e "${BLUE}  export PATH=\$PATH:/usr/bin:/usr/local/bin${NC}"
      exit 1
    fi
    
    # Docker Compose 확인
    if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
      install_docker_compose
    fi
  fi
  
  # Docker가 실행 중인지 확인
  if command -v systemctl &> /dev/null; then
    if ! systemctl is-active --quiet docker; then
      echo -e "${YELLOW}⚠️ Docker 서비스가 실행 중이 아닙니다. 시작합니다...${NC}"
      systemctl start docker || true
    fi
  fi
  
  # Docker 실행 확인
  if ! docker info &> /dev/null; then
    echo -e "${RED}⚠️ Docker 데몬이 실행 중이 아닙니다.${NC}"
    echo -e "${YELLOW}다음 명령어로 시작해보세요: ${BLUE}systemctl start docker${NC}"
    return 1
  fi
  
  return 0
}

# 서버 시작 함수
start_server() {
  echo -e "${BLUE}🚀 VLLM 서버를 시작합니다...${NC}"
  
  # Docker 확인 및 설치
  check_and_install_docker || exit 1
  
  echo -e "${BLUE}📂 Docker compose 설정 확인 중...${NC}"
  if [ ! -f "$DOCKER_COMPOSE_PATH" ]; then
    echo -e "${RED}❌ Docker compose 파일을 찾을 수 없습니다: $DOCKER_COMPOSE_PATH${NC}"
    exit 1
  fi
  
  echo -e "${BLUE}🔄 서버 시작 중...${NC}"
  docker compose -f $DOCKER_COMPOSE_PATH up -d
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker 명령 실행 실패.${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}✅ 서버가 백그라운드에서 실행 중입니다.${NC}"
  echo -e "${GREEN}🔗 API 엔드포인트: http://localhost:8000${NC}"
  echo ""
  echo -e "${BLUE}📝 로그 확인: docker compose -f $DOCKER_COMPOSE_PATH logs -f${NC}"
}

# 서버 중지 함수
stop_server() {
  echo -e "${BLUE}🛑 VLLM 서버를 중지합니다...${NC}"
  
  # Docker 확인
  if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker가 설치되어 있지 않습니다.${NC}"
    exit 1
  fi
  
  docker compose -f $DOCKER_COMPOSE_PATH down
  
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
  if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker가 설치되어 있지 않습니다.${NC}"
    exit 1
  fi
  
  if docker compose -f $DOCKER_COMPOSE_PATH ps | grep -q "vllm-server"; then
    echo -e "${GREEN}✅ VLLM 서버가 실행 중입니다.${NC}"
    echo -e "${BLUE}📊 서버 상태:${NC}"
    docker compose -f $DOCKER_COMPOSE_PATH ps
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