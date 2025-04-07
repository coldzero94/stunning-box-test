#!/bin/bash

# Docker Hub 사용자명 설정
DOCKER_USERNAME="coldzero94"
IMAGE_NAME="pdf-translation-tool"
VERSION="1.0.0"

# Docker 이미지 빌드
docker build -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} .

# Docker Hub에 로그인
docker login

# 이미지 푸시
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}

echo "이미지가 성공적으로 빌드되고 푸시되었습니다." 