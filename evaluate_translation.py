import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import os
from openai import OpenAI
from tqdm import tqdm
import time
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def load_dataset(csv_path: str):
    """데이터셋을 로드합니다."""
    df = pd.read_csv(csv_path)
    return df[['source', 'target', 'confidence']].copy()

def calculate_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """두 텍스트 간의 코사인 유사도를 계산합니다."""
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def evaluate_with_openai(df: pd.DataFrame, api_key: str, output_path: str):
    """OpenAI API를 사용하여 번역을 생성하고 평가합니다."""
    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)
    
    # 문장 임베딩 모델 로드
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 프롬프트 생성
        prompt = f"""Translate the following English text to Korean:
{row['source']}

Translation:"""
        
        try:
            # OpenAI API 호출
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional English to Korean translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # 번역 추출
            generated_translation = response.choices[0].message.content.strip()
            
            # 유사도 계산
            similarity = calculate_similarity(row['target'], generated_translation, embedder)
            
            results.append({
                'source': row['source'],
                'target': row['target'],
                'generated': generated_translation,
                'similarity': similarity,
                'confidence': row['confidence']
            })
            
            # API 호출 간 딜레이 추가 (속도 제한 방지)
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    # 평가 통계 출력
    print("\n=== 평가 결과 ===")
    print(f"평균 유사도: {results_df['similarity'].mean():.3f}")
    print(f"유사도 분포:\n{results_df['similarity'].describe()}")
    
    return results_df

def main():
    # 설정
    dataset_path = "parallel_corpus_openai.csv"
    output_path = "translation_evaluation_openai.csv"
    
    # OpenAI API 키 설정
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("1. .env 파일을 생성하고 OPENAI_API_KEY=your_api_key_here 형식으로 API 키를 추가하세요.")
        print("2. 또는 환경 변수를 직접 설정하세요: export OPENAI_API_KEY=your_api_key_here")
        return
    
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    df = load_dataset(dataset_path)
    
    # 평가 실행
    print("번역 평가 중...")
    results = evaluate_with_openai(df, api_key, output_path)
    
    print(f"\n평가 결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main() 