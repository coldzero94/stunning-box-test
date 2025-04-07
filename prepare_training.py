import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_dataset(csv_path: str, output_dir: str, test_size: float = 0.1, min_confidence: float = 0.85):
    """
    QLoRA 파인튜닝을 위한 데이터셋을 준비합니다.
    
    Args:
        csv_path: 원본 CSV 파일 경로
        output_dir: 출력 디렉토리
        test_size: 테스트 세트 비율
        min_confidence: 최소 신뢰도 점수
    """
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV 파일 로드
    print(f"데이터 로드 중: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 신뢰도 필터링
    df = df[df['confidence'] >= min_confidence].copy()
    print(f"신뢰도 {min_confidence} 이상 데이터: {len(df)}개")
    
    # 학습/테스트 분할
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    print(f"학습 데이터: {len(train_df)}개")
    print(f"테스트 데이터: {len(test_df)}개")
    
    # JSONL 형식으로 변환
    def create_jsonl(df, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                item = {
                    "instruction": "다음 영어 텍스트를 한국어로 번역하세요:",
                    "input": row['source'],
                    "output": row['target']
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 데이터 저장
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    
    create_jsonl(train_df, train_path)
    create_jsonl(test_df, test_path)
    
    print(f"\n데이터셋이 {output_dir}에 저장되었습니다.")
    print(f"학습 데이터: {train_path}")
    print(f"테스트 데이터: {test_path}")
    
    # 데이터셋 통계
    print("\n=== 데이터셋 통계 ===")
    print(f"평균 신뢰도: {df['confidence'].mean():.3f}")
    print(f"신뢰도 분포:\n{df['confidence'].describe()}")

if __name__ == "__main__":
    prepare_dataset(
        csv_path="parallel_corpus_openai.csv",
        output_dir="training_data",
        test_size=0.1,
        min_confidence=0.85
    ) 