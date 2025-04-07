import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

def prepare_dataset(
    csv_path: str = "parallel_corpus_openai.csv",
    output_dir: str = "training_data",
    test_size: float = 0.1,
    min_confidence: float = 0.85
):
    """데이터셋을 준비하고 train/test로 분할합니다."""
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # CSV 파일 로드
    df = pd.read_csv(csv_path)
    
    # confidence score로 필터링
    df = df[df['confidence'] >= min_confidence]
    
    # train/test 분할
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # 데이터 포맷 변환
    def convert_to_jsonl(df, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                data = {
                    "instruction": "Translate the following English text to Korean.",
                    "input": row['source'],
                    "output": row['target']
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # JSONL 파일로 저장
    convert_to_jsonl(train_df, output_path / 'train.jsonl')
    convert_to_jsonl(test_df, output_path / 'test.jsonl')
    
    # 데이터셋 통계 출력
    print(f"데이터셋 준비 완료:")
    print(f"- 전체 데이터: {len(df)}개")
    print(f"- 학습 데이터: {len(train_df)}개")
    print(f"- 테스트 데이터: {len(test_df)}개")
    print(f"- 평균 confidence score: {df['confidence'].mean():.3f}")
    print(f"\n데이터가 {output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    prepare_dataset() 