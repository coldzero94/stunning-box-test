# VLLM 서버 실행 가이드

## 도커 사용 방법 (권장)

최적화된 VLLM 서버를 실행하기 위한 도커 설정이 제공됩니다. 이 방법은 Flash Attention과 Page Attention이 자동으로 적용되어 성능이 10-15배 향상됩니다.

### 사전 요구 사항
- Docker와 Docker Compose 설치
- NVIDIA Docker 지원 (GPU 사용을 위해)

### 실행 방법

```bash
# backend 디렉토리로 이동
cd backend

# 도커 컴포즈로 서버 실행
docker compose up
```

서버가 실행되면 다음 주소로 접근할 수 있습니다: http://localhost:8000

### API 호출 예시

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "Qwen/Qwen2.5-14B-Instruct",
  "messages": [{
    "role": "user",
    "content": "안녕하세요"
  }],
  "stream": true
}'
```

## Python 스크립트 사용 방법 (최적화 없음)

기존 Python 스크립트로도 서버를 실행할 수 있습니다:

```bash
python vllm_server.py
```

하지만 이 방법은 Flash Attention과 Page Attention 최적화가 적용되지 않아 성능이 저하될 수 있습니다. 