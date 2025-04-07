import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

def load_dataset(csv_path: str):
    """데이터셋을 로드합니다."""
    df = pd.read_csv(csv_path)
    return df[['source', 'target', 'confidence']].copy()

def calculate_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """두 텍스트 간의 코사인 유사도를 계산합니다."""
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def evaluate_with_llm(df: pd.DataFrame, model_name: str, output_path: str):
    """LLM을 사용하여 번역을 생성하고 평가합니다."""
    # 문장 임베딩 모델 로드
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # LLM 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 프롬프트 생성
        prompt = f"""Translate the following English text to Korean:
{row['source']}

Translation:"""
        
        # 번역 생성
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        generated_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 유사도 계산
        similarity = calculate_similarity(row['target'], generated_translation, embedder)
        
        results.append({
            'source': row['source'],
            'target': row['target'],
            'generated': generated_translation,
            'similarity': similarity,
            'confidence': row['confidence']
        })
    
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
    output_path = "translation_evaluation.csv"
    model_name = "mistralai/Mistral-7B-v0.1"  # 오픈 모델로 변경
    
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    df = load_dataset(dataset_path)
    
    # 평가 실행
    print("번역 평가 중...")
    results = evaluate_with_llm(df, model_name, output_path)
    
    print(f"\n평가 결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main() 