import fitz
import pandas as pd
from pathlib import Path

def extract_parallel_text(en_pdf_path: str, ko_pdf_path: str, output_path: str = None):
    """
    영문과 한글 PDF 파일에서 텍스트를 추출하고 병렬 코퍼스 CSV 형식으로 저장합니다.
    
    Args:
        en_pdf_path (str): 영문 PDF 파일 경로
        ko_pdf_path (str): 한글 PDF 파일 경로
        output_path (str, optional): 결과를 저장할 CSV 파일 경로
    
    Returns:
        pandas.DataFrame: 추출된 텍스트 내용
    """
    try:
        # PDF 파일 열기
        en_doc = fitz.open(en_pdf_path)
        ko_doc = fitz.open(ko_pdf_path)
        
        # 텍스트 추출을 위한 리스트
        text_pairs = []
        
        # 페이지 수 확인
        if len(en_doc) != len(ko_doc):
            print(f"경고: 영문({len(en_doc)}페이지)과 한글({len(ko_doc)}페이지)의 페이지 수가 다릅니다.")
        
        # 각 페이지별로 처리
        for en_page, ko_page in zip(en_doc, ko_doc):
            # 텍스트 추출 (블록 단위로)
            en_blocks = en_page.get_text("blocks")
            ko_blocks = ko_page.get_text("blocks")
            
            # 블록을 텍스트로 변환하고 정리
            en_texts = [block[4].strip() for block in en_blocks if block[4].strip()]
            ko_texts = [block[4].strip() for block in ko_blocks if block[4].strip()]
            
            # 영어와 한글 텍스트 매칭
            for en_text, ko_text in zip(en_texts, ko_texts):
                # 최소 길이 체크 (빈 문장 제외)
                if len(en_text) > 10 and len(ko_text) > 10:
                    text_pairs.append({
                        'source': en_text,
                        'target': ko_text
                    })
        
        # PDF 파일 닫기
        en_doc.close()
        ko_doc.close()
        
        # DataFrame 생성
        df = pd.DataFrame(text_pairs)
        
        # CSV 파일로 저장 (지정된 경우)
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"결과가 {output_path}에 저장되었습니다.")
            print(f"총 {len(df)} 개의 문장 쌍이 추출되었습니다.")
        
        return df
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return None

def main():
    # PDF 파일 경로
    en_pdf = "animal parasiticides market - global forecast to 2025.pdf"
    ko_pdf = "animal parasiticides market - global forecast to 2025 영문.pdf"
    output_path = "parallel_corpus.csv"
    
    df = extract_parallel_text(en_pdf, ko_pdf, output_path)
    if df is not None and not df.empty:
        print("\n데이터 미리보기:")
        print(df.head())
        
        # 기본 통계
        print("\n기본 통계:")
        print(f"평균 source 길이: {df['source'].str.len().mean():.1f} 문자")
        print(f"평균 target 길이: {df['target'].str.len().mean():.1f} 문자")
        
        # 문장 길이 분포
        print("\n문장 길이 분포:")
        print("영문:")
        print(df['source'].str.len().describe())
        print("\n한글:")
        print(df['target'].str.len().describe())

if __name__ == "__main__":
    main() 