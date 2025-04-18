#!/bin/bash

# Docker 내부에서 vllm-openai 이미지를 실행하는 스크립트
# 사용법: ./run_docker_vllm.sh [start|stop|status|help]

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker 컨테이너 이름 설정
CONTAINER_NAME="vllm-server"
IMAGE_NAME="vllm/vllm-openai:latest"

# 설정 변수
HOST="0.0.0.0"
PORT="8000"
MODEL="Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH="/qwen25-14b"

# Docker 소켓 경로 확인
DOCKER_SOCKET="/var/run/docker.sock"

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")"

# Docker 소켓 및 CLI 확인
check_docker() {
    echo -e "${BLUE}🔍 Docker 소켓 및 CLI 확인 중...${NC}"
    
    # Docker 소켓 확인
    if [ ! -e "$DOCKER_SOCKET" ]; then
        echo -e "${RED}❌ Docker 소켓을 찾을 수 없습니다: $DOCKER_SOCKET${NC}"
        echo -e "${YELLOW}💡 컨테이너 시작 시 아래 볼륨 마운트가 필요합니다:${NC}"
        echo -e "${BLUE}   -v /var/run/docker.sock:/var/run/docker.sock${NC}"
        return 1
    fi
    
    # Docker CLI 확인
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}⚠️ Docker CLI가 설치되어 있지 않습니다. 설치를 시도합니다...${NC}"
        
        # apt 또는 apk 사용
        if command -v apt-get &> /dev/null; then
            apt-get update && apt-get install -y docker.io
        elif command -v apk &> /dev/null; then
            apk add --no-cache docker-cli
        else
            echo -e "${RED}❌ 패키지 관리자를 찾을 수 없습니다. Docker CLI를 수동으로 설치하세요.${NC}"
            return 1
        fi
        
        # 설치 확인
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}❌ Docker CLI 설치 실패${NC}"
            return 1
        fi
    fi
    
    echo -e "${GREEN}✅ Docker 소켓과 CLI 확인 완료${NC}"
    return 0
}

# 서버 시작 함수
start_server() {
    echo -e "${BLUE}🚀 VLLM 서버를 시작합니다...${NC}"
    
    # Docker 확인
    check_docker || return 1
    
    # 이미 실행 중인지 확인
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}⚠️ VLLM 서버가 이미 실행 중입니다.${NC}"
        docker ps --filter name=${CONTAINER_NAME}
        return 0
    fi
    
    # 중지된 컨테이너 제거
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}⚠️ 중지된 VLLM 서버 컨테이너를 제거합니다.${NC}"
        docker rm ${CONTAINER_NAME}
    fi
    
    echo -e "${BLUE}📂 어댑터 경로 확인 중...${NC}"
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo -e "${YELLOW}⚠️ 경고: 어댑터 경로($ADAPTER_PATH)를 찾을 수 없습니다.${NC}"
        echo -e "${YELLOW}⚠️ 경로가 맞는지 확인하세요.${NC}"
    fi
    
    echo -e "${BLUE}🔄 Docker 이미지 확인/다운로드 중...${NC}"
    docker pull ${IMAGE_NAME}
    
    echo -e "${BLUE}🚀 VLLM 서버 컨테이너 시작 중...${NC}"
    docker run -d \
        --name ${CONTAINER_NAME} \
        --gpus all \
        -p ${PORT}:80 \
        -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True \
        -v ${ADAPTER_PATH}:${ADAPTER_PATH} \
        ${IMAGE_NAME} \
        --model ${MODEL} \
        --enable-lora \
        --qlora-adapter-name-or-path ${ADAPTER_PATH} \
        --quantization bitsandbytes \
        --dtype half \
        --max-model-len 4096 \
        --trust-remote-code \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.9
    
    # 시작 확인
    sleep 3
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}✅ VLLM 서버가 성공적으로 시작되었습니다.${NC}"
        echo -e "${GREEN}🔗 API 엔드포인트: http://${HOST}:${PORT}${NC}"
        echo ""
        echo -e "${BLUE}📝 로그 확인: docker logs -f ${CONTAINER_NAME}${NC}"
    else
        echo -e "${RED}❌ VLLM 서버 시작 실패. 로그를 확인하세요:${NC}"
        docker logs ${CONTAINER_NAME}
        return 1
    fi
}

# 서버 중지 함수
stop_server() {
    echo -e "${BLUE}🛑 VLLM 서버를 중지합니다...${NC}"
    
    # Docker 확인
    check_docker || return 1
    
    # 실행 중인지 확인
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}⚠️ VLLM 서버가 실행 중이 아닙니다.${NC}"
        return 0
    fi
    
    # 중지 및 제거
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
    
    echo -e "${GREEN}✅ VLLM 서버가 중지되었습니다.${NC}"
}

# 서버 상태 확인 함수
check_status() {
    echo -e "${BLUE}📊 VLLM 서버 상태 확인 중...${NC}"
    
    # Docker 확인
    check_docker || return 1
    
    # 실행 중인지 확인
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}✅ VLLM 서버가 실행 중입니다.${NC}"
        docker ps --filter name=${CONTAINER_NAME}
        echo ""
        echo -e "${GREEN}🔗 API 엔드포인트: http://${HOST}:${PORT}${NC}"
    else
        echo -e "${YELLOW}❌ VLLM 서버가 실행 중이 아닙니다.${NC}"
        
        # 중지된 컨테이너 확인
        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo -e "${YELLOW}⚠️ 중지된 VLLM 서버 컨테이너가 있습니다:${NC}"
            docker ps -a --filter name=${CONTAINER_NAME}
        fi
    fi
}

# 도움말 표시 함수
show_help() {
    echo -e "${BLUE}📚 Docker로 VLLM 서버 실행 스크립트 사용법${NC}"
    echo ""
    echo -e "사용법: ${GREEN}./run_docker_vllm.sh [명령어]${NC}"
    echo ""
    echo "명령어:"
    echo -e "  ${GREEN}start${NC}   - 서버 시작"
    echo -e "  ${GREEN}stop${NC}    - 서버 중지"
    echo -e "  ${GREEN}status${NC}  - 서버 상태 확인"
    echo -e "  ${GREEN}help${NC}    - 이 도움말 표시"
    echo ""
    echo -e "예시: ${GREEN}./run_docker_vllm.sh start${NC}"
    echo ""
    echo "API 호출 예시:"
    echo -e "${BLUE}  curl http://${HOST}:${PORT}/v1/chat/completions \\${NC}"
    echo -e "${BLUE}    -H 'Content-Type: application/json' \\${NC}"
    echo -e "${BLUE}    -d '{ \"model\": \"${MODEL}\", \"messages\": [{ \"role\": \"user\", \"content\": \"안녕하세요\" }], \"stream\": true }'${NC}"
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