import fitz
import pandas as pd
from pathlib import Path
from openai import OpenAI
from typing import List, Dict
import json
import time

def setup_openai(api_key: str) -> OpenAI:
    """
    OpenAI 클라이언트 설정
    """
    return OpenAI(api_key=api_key)

def should_merge_blocks(block1: tuple, block2: tuple) -> bool:
    """
    두 블록을 병합해야 하는지 판단합니다.
    
    Args:
        block1: 첫 번째 블록 (x0, y0, x1, y1, text)
        block2: 두 번째 블록 (x0, y0, x1, y1, text)
    """
    # 블록의 y 좌표가 비슷한 범위에 있는지 확인
    y_threshold = 3  # y 좌표 차이 허용 범위 (포인트)
    y_diff = abs(block1[3] - block2[1])  # block1의 하단과 block2의 상단 비교
    
    # 첫 번째 블록이 문장 중간에서 끝나는지 확인
    ends_with_sentence = block1[4].strip().endswith(('.', '?', '!', '"', ''', '"', ''', '」', '.', '?', '!', '"', ''', '"', ''', '」'))
    
    # 두 번째 블록이 소문자로 시작하는지 확인 (영어의 경우)
    starts_with_lowercase = block2[4].strip() and block2[4].strip()[0].islower()
    
    return (y_diff <= y_threshold and not ends_with_sentence) or starts_with_lowercase

def merge_text_blocks(blocks: List[tuple]) -> List[str]:
    """
    인접한 텍스트 블록들을 병합합니다.
    """
    if not blocks:
        return []
    
    # y 좌표를 기준으로 블록 정렬
    sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    
    merged_texts = []
    current_text = sorted_blocks[0][4].strip()
    current_block = sorted_blocks[0]
    
    for block in sorted_blocks[1:]:
        if should_merge_blocks(current_block, block):
            # 블록 사이에 공백 추가
            if not current_text.endswith(' ') and not block[4].strip().startswith(' '):
                current_text += ' '
            current_text += block[4].strip()
            current_block = block
        else:
            if current_text:
                merged_texts.append(current_text)
            current_text = block[4].strip()
            current_block = block
    
    if current_text:
        merged_texts.append(current_text)
    
    # 최소 길이 필터링 및 후처리
    min_length = 10  # 최소 문자 길이
    return [text for text in merged_texts if len(text) >= min_length]

def extract_texts_from_page(pdf_page) -> List[str]:
    """
    PDF 페이지에서 텍스트 블록을 추출하고 자연스럽게 병합합니다.
    """
    blocks = pdf_page.get_text("blocks")
    # 빈 블록 제거 및 병합
    non_empty_blocks = [block for block in blocks if block[4].strip()]
    return merge_text_blocks(non_empty_blocks)

def split_texts_into_chunks(en_texts: List[str], ko_texts: List[str], max_tokens_per_chunk: int = 4000) -> List[tuple]:
    """
    텍스트를 토큰 제한에 맞게 청크로 나눕니다.
    
    Args:
        en_texts: 영어 텍스트 리스트
        ko_texts: 한글 텍스트 리스트
        max_tokens_per_chunk: 청크당 최대 토큰 수 (기본값: 4000)
    
    Returns:
        List[tuple]: (en_chunk, ko_chunk) 형태의 리스트
    """
    chunks = []
    current_en = []
    current_ko = []
    current_tokens = 0
    
    # 대략적인 토큰 수 추정 (영어: 단어당 1.3 토큰, 한글: 문자당 1.5 토큰)
    def estimate_tokens(en_text: str, ko_text: str) -> int:
        en_tokens = len(en_text.split()) * 1.3
        ko_tokens = len(ko_text) * 1.5
        return int(en_tokens + ko_tokens)
    
    for en_text, ko_text in zip(en_texts, ko_texts):
        estimated_tokens = estimate_tokens(en_text, ko_text)
        
        # 현재 청크가 토큰 제한을 초과하면 새로운 청크 시작
        if current_tokens + estimated_tokens > max_tokens_per_chunk and current_en:
            chunks.append((current_en.copy(), current_ko.copy()))
            current_en = []
            current_ko = []
            current_tokens = 0
        
        current_en.append(en_text)
        current_ko.append(ko_text)
        current_tokens += estimated_tokens
    
    # 마지막 청크 추가
    if current_en:
        chunks.append((current_en, current_ko))
    
    return chunks

def match_texts_with_llm(client: OpenAI, en_texts: List[str], ko_texts: List[str], page_num: int) -> tuple:
    """
    OpenAI를 사용하여 영문과 한글 텍스트를 매칭합니다.
    """
    # 텍스트를 청크로 나누기
    chunks = split_texts_into_chunks(en_texts, ko_texts)
    
    all_matches = []
    all_unmatched = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    for chunk_idx, (en_chunk, ko_chunk) in enumerate(chunks):
        print(f"  청크 {chunk_idx + 1}/{len(chunks)} 처리 중...")
        
        prompt = f"""You are a professional translator validator. Your task is to match English texts with their corresponding Korean translations.
I will provide you with two lists: English texts and Korean texts from the same PDF page (page {page_num}, chunk {chunk_idx + 1}).
Please analyze and match them based on their meaning and context.

Important Guidelines:
1. Only match texts that are direct translations of each other
2. Consider the context and meaning, not just the position
3. Ensure the matched texts convey the same information
4. Assign confidence scores based on the following criteria:
   - 0.95-1.00: Perfect or near-perfect translation match
   - 0.85-0.94: Good match with minor differences
   - 0.75-0.84: Acceptable match but with some discrepancies
   - Below 0.75: Do not include in matches

English texts:
{json.dumps(en_chunk, indent=2, ensure_ascii=False)}

Korean texts:
{json.dumps(ko_chunk, indent=2, ensure_ascii=False)}

Please provide the matches in the following JSON format:
{{
    "matches": [
        {{
            "source": "English text",
            "target": "Korean text",
            "confidence": 0.95,
            "notes": "Optional notes about the match quality"
        }},
        ...
    ],
    "unmatched": [
        {{
            "text": "Unmatched text",
            "language": "en/ko",
            "reason": "Why this text couldn't be matched"
        }}
    ]
}}

Only provide the JSON output, no other text. Only include high-confidence matches (0.75 or higher)."""

        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional translator validator specialized in English-Korean translation. Always respond in JSON format and focus on accuracy over quantity."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 페이지 번호와 청크 번호 추가
            for match in result.get('matches', []):
                match['page'] = page_num
                match['chunk'] = chunk_idx + 1
            
            all_matches.extend(result.get('matches', []))
            all_unmatched.extend(result.get('unmatched', []))
            
            # 토큰 사용량 누적
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            
            # API 레이트 리밋 방지
            time.sleep(1)
            
        except Exception as e:
            print(f"  청크 {chunk_idx + 1} 처리 중 오류 발생: {str(e)}")
    
    return all_matches, all_unmatched, total_prompt_tokens, total_completion_tokens

def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    """
    API 사용 비용을 계산합니다.
    
    Args:
        prompt_tokens (int): 입력 토큰 수
        completion_tokens (int): 출력 토큰 수
        model (str): 모델 이름 (기본값: "gpt-4o-mini")
    
    Returns:
        float: 예상 비용 (USD)
    """
    if model == "gpt-4o-mini":
        # GPT-4o mini 가격:
        # $0.60 / 1M input tokens
        # $0.30 / 1M cached input tokens (캐시된 경우는 현재 고려하지 않음)
        # $2.40 / 1M output tokens
        input_cost = (prompt_tokens * 0.60) / 1_000_000  # $0.60 per 1M tokens
        output_cost = (completion_tokens * 2.40) / 1_000_000  # $2.40 per 1M tokens
        return input_cost + output_cost
    else:  # gpt-3.5-turbo
        return (prompt_tokens * 0.0005 + completion_tokens * 0.0015) / 1000

def extract_parallel_text_llm(en_pdf_path: str, ko_pdf_path: str, api_key: str, 
                            output_path: str = None, page_numbers: List[int] = None,
                            start_page: int = None, end_page: int = None,
                            resume_from: str = None):
    """
    OpenAI를 사용하여 PDF에서 병렬 코퍼스를 추출합니다.
    
    Args:
        en_pdf_path (str): 영문 PDF 경로
        ko_pdf_path (str): 한글 PDF 경로
        api_key (str): OpenAI API 키
        output_path (str, optional): 결과 저장 경로
        page_numbers (List[int], optional): 처리할 특정 페이지 번호 리스트 (1-based)
        start_page (int, optional): 처리 시작 페이지 (1-based)
        end_page (int, optional): 처리 종료 페이지 (1-based)
        resume_from (str, optional): 이어서 처리할 중간 결과 파일 경로
    """
    try:
        # 기존 결과 로드 (있는 경우)
        all_matches = []
        if resume_from and Path(resume_from).exists():
            existing_df = pd.read_csv(resume_from)
            all_matches = existing_df.to_dict('records')
            print(f"\n기존 결과를 불러왔습니다: {len(all_matches)}개의 매칭")
            
            # 마지막으로 처리된 페이지 확인
            if len(all_matches) > 0:
                last_page = max(match['page'] for match in all_matches)
                if start_page is None and page_numbers is None:
                    start_page = last_page + 1
                    print(f"마지막 처리 페이지: {last_page}, 다음 페이지부터 시작합니다.")

        # OpenAI 클라이언트 설정
        client = setup_openai(api_key)
        
        # PDF 파일 열기
        en_doc = fitz.open(en_pdf_path)
        ko_doc = fitz.open(ko_pdf_path)
        
        # 전체 페이지 수 확인
        max_pages = min(len(en_doc), len(ko_doc))
        print(f"\n전체 페이지 수: {max_pages}페이지")
        
        # 처리할 페이지 범위 결정
        if page_numbers:
            # 특정 페이지 리스트가 제공된 경우
            page_indices = [p - 1 for p in page_numbers if 0 < p <= max_pages]
            print(f"지정된 페이지 처리: {page_numbers}")
        else:
            # 시작/종료 페이지로 범위 지정
            if start_page is None:
                start_page = 1
            if end_page is None:
                end_page = max_pages
                
            # 페이지 범위 유효성 검사 및 조정
            start_page = max(1, min(start_page, max_pages))
            end_page = max(start_page, min(end_page, max_pages))
            
            page_indices = list(range(start_page - 1, end_page))
            print(f"처리 페이지 범위: {start_page}~{end_page}페이지")
        
        if not page_indices:
            raise ValueError(f"처리할 유효한 페이지가 없습니다. PDF는 {max_pages}페이지까지 있습니다.")
        
        # 결과 저장용 리스트
        all_unmatched = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # 진행 상황 표시
        total_pages = len(page_indices)
        
        for idx, page_idx in enumerate(page_indices, 1):
            print(f"\n=== 페이지 {page_idx + 1} 처리 중... ({idx}/{total_pages}) ===")
            
            # 이미 처리된 페이지 건너뛰기
            if resume_from and any(match['page'] == page_idx + 1 for match in all_matches):
                print(f"페이지 {page_idx + 1}는 이미 처리되었습니다. 건너뜁니다.")
                continue
            
            # 텍스트 추출
            en_texts = extract_texts_from_page(en_doc[page_idx])
            ko_texts = extract_texts_from_page(ko_doc[page_idx])
            
            print(f"추출된 텍스트 블록: 영어 {len(en_texts)}개, 한글 {len(ko_texts)}개")
            
            # LLM으로 매칭
            result, unmatched, prompt_tokens, completion_tokens = match_texts_with_llm(
                client, en_texts, ko_texts, page_idx + 1
            )
            
            # 토큰 사용량 누적
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            
            # 현재까지의 비용 계산
            current_cost = calculate_cost(total_prompt_tokens, total_completion_tokens)
            
            # 결과 저장
            all_matches.extend(result)
            all_unmatched.extend(unmatched)
            
            # 진행 상황 출력
            print(f"매칭된 텍스트 쌍: {len(result)}개")
            print(f"매칭되지 않은 텍스트: {len(unmatched)}개")
            print(f"현재까지 사용된 토큰: {total_prompt_tokens + total_completion_tokens:,}개")
            print(f"현재까지 예상 비용: ${current_cost:.4f}")
            
            # 중간 결과 저장 (매 10페이지마다 또는 마지막 페이지)
            if output_path and (idx % 10 == 0 or idx == total_pages):
                interim_df = pd.DataFrame(all_matches)
                interim_path = output_path.replace('.csv', f'_interim_{idx}.csv')
                interim_df.to_csv(interim_path, index=False)
                print(f"중간 결과가 {interim_path}에 저장되었습니다.")
            
            # API 레이트 리밋 방지
            time.sleep(1)
        
        # PDF 파일 닫기
        en_doc.close()
        ko_doc.close()
        
        # 최종 비용 계산 및 출력
        final_cost = calculate_cost(total_prompt_tokens, total_completion_tokens)
        print(f"\n=== 처리 완료 ===")
        print(f"처리된 총 페이지: {total_pages}페이지")
        print(f"총 매칭 쌍: {len(all_matches)}개")
        print(f"총 미매칭: {len(all_unmatched)}개")
        print(f"\n토큰 사용량:")
        print(f"Prompt 토큰: {total_prompt_tokens:,}개")
        print(f"Completion 토큰: {total_completion_tokens:,}개")
        print(f"총 예상 비용: ${final_cost:.4f}")
        
        # DataFrame 생성 및 최종 저장
        if all_matches:
            df = pd.DataFrame(all_matches)
            if output_path:
                df.to_csv(output_path, index=False)
                print(f"\n최종 결과가 {output_path}에 저장되었습니다.")
                
                # 매칭되지 않은 텍스트 저장
                if all_unmatched:
                    unmatched_output = output_path.replace('.csv', '_unmatched.csv')
                    pd.DataFrame(all_unmatched).to_csv(unmatched_output, index=False)
                    print(f"매칭되지 않은 텍스트가 {unmatched_output}에 저장되었습니다.")
            
            return df
        
        return None
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return None

def main():
    # 설정
    en_pdf = "animal parasiticides market - global forecast to 2025.pdf"
    ko_pdf = "animal parasiticides market - global forecast to 2025 영문.pdf"
    output_path = "parallel_corpus_openai.csv"
    api_key = "sk-proj-lcj80N4im628R7dNIRl7T3BlbkFJm68gdJADgfCoS7DzUyc7"
    
    # 처리 방식 선택
    print("\n처리 방식을 선택하세요:")
    print("1. 전체 페이지 처리")
    print("2. 페이지 범위 지정")
    print("3. 특정 페이지 처리")
    print("4. 이전 작업 이어서 처리")
    
    choice = input("선택 (1-4): ").strip()
    
    if choice == "1":
        # 전체 페이지 처리
        print("\nOpenAI를 사용한 PDF 텍스트 매칭 시작...")
        df = extract_parallel_text_llm(en_pdf, ko_pdf, api_key, output_path)
    
    elif choice == "2":
        # 페이지 범위 지정
        start = int(input("시작 페이지 (1부터): ").strip())
        end_input = input("종료 페이지 (빈 값 입력시 PDF 끝까지): ").strip()
        end = int(end_input) if end_input else None
        print(f"\n{start}~{end if end else '마지막'} 페이지 처리 시작...")
        df = extract_parallel_text_llm(en_pdf, ko_pdf, api_key, output_path, 
                                     start_page=start, end_page=end)
    
    elif choice == "3":
        # 특정 페이지 처리
        pages = input("처리할 페이지 번호들 (쉼표로 구분, 예: 1,3,5): ").strip()
        page_numbers = [int(p.strip()) for p in pages.split(",")]
        print(f"\n페이지 {page_numbers} 처리 시작...")
        df = extract_parallel_text_llm(en_pdf, ko_pdf, api_key, output_path, 
                                     page_numbers=page_numbers)
    
    elif choice == "4":
        # 이전 작업 이어서 처리
        resume_file = input("이어서 처리할 중간 결과 파일 (예: parallel_corpus_openai_interim_200.csv): ").strip()
        end_input = input("종료 페이지 (빈 값 입력시 PDF 끝까지): ").strip()
        end = int(end_input) if end_input else None
        print(f"\n{resume_file}에서 이어서 처리 시작...")
        if end:
            print(f"종료 페이지: {end}")
        else:
            print("PDF 마지막 페이지까지 처리합니다.")
        df = extract_parallel_text_llm(en_pdf, ko_pdf, api_key, output_path,
                                     end_page=end, resume_from=resume_file)
    
    else:
        print("잘못된 선택입니다.")
        return
    
    # 결과 통계 출력
    if df is not None and not df.empty:
        print("\n=== 결과 통계 ===")
        print(f"총 매칭 쌍: {len(df)}개")
        print(f"평균 신뢰도: {df['confidence'].mean():.2f}")
        
        print("\n신뢰도 분포:")
        print(df['confidence'].describe())
        
        print("\n페이지별 매칭 수:")
        print(df['page'].value_counts().sort_index())

if __name__ == "__main__":
    main() 