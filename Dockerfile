FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 파일 복사
COPY . .

# 환경 변수 설정
ENV PYTHONPATH=/app

# 기본 명령어 설정
CMD ["python", "evaluate_translation.py"] 