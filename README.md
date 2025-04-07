# PDF 번역 데이터셋 생성 및 모델 파인튜닝

이 프로젝트는 PDF 문서에서 영어-한국어 번역 데이터셋을 생성하고, 이를 이용하여 LLM 모델을 파인튜닝하는 도구를 제공합니다.

## 주요 기능

1. PDF 텍스트 추출 및 매칭
   - 영어와 한국어 PDF에서 텍스트 추출
   - OpenAI API를 활용한 텍스트 매칭
   - 신뢰도 점수 기반 데이터 필터링

2. 번역 평가
   - 일반 LLM의 번역 성능 평가
   - 문장 임베딩 기반 유사도 계산
   - 상세한 평가 결과 리포트

3. QLoRA 파인튜닝
   - Gemma-3/Qwen-2.5-12b 모델 지원
   - 4비트 양자화 학습
   - TensorBoard 모니터링

## 설치 방법

1. Poetry 설치
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 의존성 설치
```bash
poetry install
```

## 사용 방법

1. PDF 텍스트 추출
```bash
poetry run python pdf_extractor_openai.py
```

2. 번역 평가
```bash
poetry run python evaluate_translation.py
```

3. 모델 파인튜닝
```bash
poetry run python train_qlora.py
```

## 환경 요구사항

- Python 3.10 이상
- CUDA 지원 GPU
- OpenAI API 키

## 라이선스

MIT License
