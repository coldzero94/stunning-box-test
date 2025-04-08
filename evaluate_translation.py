import os
import pandas as pd
import fitz  # pymupdf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
import time
import sys

# 환경 변수 로드
load_dotenv()

def get_user_input():
    """사용자로부터 시작/끝 페이지를 입력받습니다."""
    print("=" * 50)
    print("PDF 번역 평가 프로그램")
    print("=" * 50)
    
    # 미리 지정된 PDF 파일 경로
    korean_pdf = "animal parasiticides market - global forecast to 2025.pdf"  # 한글 PDF 파일 경로
    english_pdf = "animal parasiticides market - global forecast to 2025 영문.pdf"  # 영어 PDF 파일 경로
    
    # PDF가 존재하는지 확인
    if not os.path.exists(korean_pdf):
        print(f"오류: {korean_pdf} 파일을 찾을 수 없습니다.")
        sys.exit(1)
    if not os.path.exists(english_pdf):
        print(f"오류: {english_pdf} 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    # PDF 총 페이지 수 확인
    doc_korean = fitz.open(korean_pdf)
    total_pages_korean = doc_korean.page_count
    doc_korean.close()
    
    doc_english = fitz.open(english_pdf)
    total_pages_english = doc_english.page_count
    doc_english.close()
    
    print(f"한글 PDF 총 페이지 수: {total_pages_korean}")
    print(f"영어 PDF 총 페이지 수: {total_pages_english}")
    
    # 시작 페이지와 끝 페이지 입력 받기
    while True:
        try:
            start_page = input("시작 페이지를 입력하세요 (기본값: 1): ").strip()
            start_page = int(start_page) if start_page else 1
            
            end_page_input = input(f"끝 페이지를 입력하세요 (기본값: {min(total_pages_korean, total_pages_english)}): ").strip()
            end_page = int(end_page_input) if end_page_input else min(total_pages_korean, total_pages_english)
            
            if start_page < 1 or start_page > min(total_pages_korean, total_pages_english):
                print(f"시작 페이지는 1과 {min(total_pages_korean, total_pages_english)} 사이여야 합니다.")
                continue
                
            if end_page < start_page or end_page > min(total_pages_korean, total_pages_english):
                print(f"끝 페이지는 시작 페이지({start_page})와 {min(total_pages_korean, total_pages_english)} 사이여야 합니다.")
                continue
                
            break
            
        except ValueError:
            print("유효한 숫자를 입력하세요.")
    
    output_dir = "evaluation_results"  # 결과 저장 디렉토리 미리 지정
    
    return {
        "korean_pdf": korean_pdf,
        "english_pdf": english_pdf,
        "start_page": start_page,
        "end_page": end_page,
        "output_dir": output_dir
    }

def get_args():
    """커맨드 라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='PDF 번역 평가 스크립트')
    
    # 미리 지정된 PDF 파일 경로
    korean_pdf = "animal parasiticides market - global forecast to 2025.pdf"  # 한글 PDF 파일 경로
    english_pdf = "animal parasiticides market - global forecast to 2025 영문.pdf"  # 영어 PDF 파일 경로
    output_dir = "evaluation_results"  # 결과 저장 디렉토리
    
    parser.add_argument('--start-page', type=int, help='시작 페이지 (1부터 시작)')
    parser.add_argument('--end-page', type=int, help='끝 페이지 (미지정시 마지막 페이지까지)')
    
    args = parser.parse_args()
    
    # 사용자 입력 받기 (커맨드 라인에 시작/끝 페이지가 지정되지 않은 경우)
    if args.start_page is None or args.end_page is None:
        user_input = get_user_input()
        args.start_page = user_input["start_page"]
        args.end_page = user_input["end_page"]
    
    # 미리 지정된 파일 경로 설정
    args.korean_pdf = korean_pdf
    args.english_pdf = english_pdf
    args.output_dir = output_dir
    
    return args

def extract_text_from_pdf(pdf_path, start_page=None, end_page=None):
    """PDF에서 텍스트를 추출합니다."""
    texts = []
    try:
        # PDF 파일 열기
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        # 페이지 범위 조정
        start = (start_page - 1) if start_page else 0
        end = min(end_page if end_page else total_pages, total_pages)
        
        if start < 0 or start >= total_pages:
            raise ValueError(f"시작 페이지가 유효하지 않습니다. (1-{total_pages})")
        if end <= start:
            raise ValueError("끝 페이지는 시작 페이지보다 커야 합니다.")
        
        print(f"PDF 추출 범위: {start + 1}페이지 부터 {end}페이지 까지")
        
        # 각 페이지에서 텍스트 추출
        for page_num in range(start, end):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:  # 빈 페이지 제외
                texts.append({
                    'page': page_num + 1,
                    'text': text
                })
        
        # PDF 파일 닫기
        doc.close()
        
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    except Exception as e:
        raise Exception(f"PDF 처리 중 오류 발생: {str(e)}")
    
    return texts

def save_page_results(results, output_dir, page_num):
    """페이지별 결과를 CSV 파일로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'page_{page_num:03d}_results.csv')
    
    df = pd.DataFrame([results])
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    return output_file

def save_summary_results(all_results, output_dir):
    """전체 평가 결과 요약을 저장합니다."""
    summary_file = os.path.join(output_dir, 'summary_results.csv')
    df = pd.DataFrame(all_results)
    df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    return summary_file

def load_finetuned_model(base_model_name="google/gemma-2b-it", adapter_path="qlora_output"):
    """파인튜닝된 모델을 로드합니다."""
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
        token=os.getenv('HUGGINGFACE_TOKEN')
    )
    
    print("베이스 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv('HUGGINGFACE_TOKEN')
    )
    
    print("LoRA 어댑터 로드 중...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def translate_with_finetuned(text, model, tokenizer, max_length=1024):
    """파인튜닝된 모델로 번역을 수행합니다."""
    prompt = f"### 입력:\n{text}\n\n### 응답:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 토큰 길이가 너무 길 경우 처리
    input_length = inputs.input_ids.shape[1]
    if input_length > max_length - 100:  # 응답을 위한 여유 공간 확보
        print(f"경고: 입력 텍스트가 너무 깁니다 ({input_length} 토큰). 텍스트를 잘라서 처리합니다.")
        # 앞부분 일부와 뒷부분 일부만 사용
        prefix_len = (max_length - 100) // 2
        suffix_len = (max_length - 100) - prefix_len
        
        # 새 입력 생성 (앞부분 + 뒷부분)
        inputs_truncated = {
            "input_ids": torch.cat([
                inputs.input_ids[0, :prefix_len], 
                inputs.input_ids[0, -suffix_len:]
            ]).unsqueeze(0),
            "attention_mask": torch.cat([
                inputs.attention_mask[0, :prefix_len], 
                inputs.attention_mask[0, -suffix_len:]
            ]).unsqueeze(0)
        }
        inputs = inputs_truncated
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # max_length 대신 max_new_tokens 사용
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### 응답:\n")[-1].strip()

def translate_with_gpt(text):
    """GPT를 사용하여 번역을 수행합니다."""
    try:
        # OpenAI API v1.0 이상 호환 방식으로 수정
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the given Korean text to English accurately and naturally."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT 번역 오류: {e}")
        return ""

def calculate_metrics(reference, hypothesis):
    """번역 품질 평가 지표를 계산합니다."""
    # BLEU 점수 계산
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
    
    # ROUGE 점수 계산
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    
    return {
        'bleu': bleu_score,
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure
    }

def display_progress_bar(current, total, bar_length=50):
    """진행 상황을 보여주는 프로그레스 바를 표시합니다."""
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write('\r진행: [%s%s] %d%%' % (arrow, spaces, percent))
    sys.stdout.flush()

def summarize_and_print_results(df):
    """결과를 요약하고 출력합니다."""
    print("\n\n" + "=" * 80)
    print("번역 평가 결과 요약")
    print("=" * 80)
    
    # 페이지별 점수 출력
    print("\n[페이지별 BLEU 점수]")
    for i, row in df.iterrows():
        page = row['page']
        gpt_bleu = row['gpt_bleu']
        ft_bleu = row['finetuned_bleu']
        winner = "GPT" if gpt_bleu > ft_bleu else "파인튜닝모델" if ft_bleu > gpt_bleu else "동일"
        print(f"페이지 {page:3d}: GPT={gpt_bleu:.4f}, 파인튜닝={ft_bleu:.4f} (승자: {winner})")
    
    # 종합 평균 점수
    print("\n[종합 평균 점수]")
    print("GPT 번역:")
    print(f"BLEU: {df['gpt_bleu'].mean():.4f}")
    print(f"ROUGE-1: {df['gpt_rouge1'].mean():.4f}")
    print(f"ROUGE-2: {df['gpt_rouge2'].mean():.4f}")
    print(f"ROUGE-L: {df['gpt_rougeL'].mean():.4f}")
    
    print("\n파인튜닝된 모델 번역:")
    print(f"BLEU: {df['finetuned_bleu'].mean():.4f}")
    print(f"ROUGE-1: {df['finetuned_rouge1'].mean():.4f}")
    print(f"ROUGE-2: {df['finetuned_rouge2'].mean():.4f}")
    print(f"ROUGE-L: {df['finetuned_rougeL'].mean():.4f}")
    
    # 승자 판정
    gpt_avg_bleu = df['gpt_bleu'].mean()
    ft_avg_bleu = df['finetuned_bleu'].mean()
    
    print("\n[최종 결과]")
    if gpt_avg_bleu > ft_avg_bleu:
        winner = "GPT"
        diff = ((gpt_avg_bleu - ft_avg_bleu) / ft_avg_bleu) * 100 if ft_avg_bleu > 0 else 0
        print(f"GPT가 파인튜닝 모델보다 {diff:.2f}% 더 나은 결과를 보였습니다.")
    elif ft_avg_bleu > gpt_avg_bleu:
        winner = "파인튜닝 모델"
        diff = ((ft_avg_bleu - gpt_avg_bleu) / gpt_avg_bleu) * 100 if gpt_avg_bleu > 0 else 0
        print(f"파인튜닝 모델이 GPT보다 {diff:.2f}% 더 나은 결과를 보였습니다.")
    else:
        winner = "동일"
        print("두 모델의 성능이 동일합니다.")
    
    print(f"\n최종 승자: {winner}")
    print("=" * 80)

def main():
    args = get_args()
    
    # 결과를 저장할 리스트 초기화
    all_results = []
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 시작 시간 기록
    start_time = time.time()
    
    # PDF에서 텍스트 추출
    print("\nPDF에서 텍스트 추출 중...")
    korean_texts = extract_text_from_pdf(args.korean_pdf, args.start_page, args.end_page)
    english_texts = extract_text_from_pdf(args.english_pdf, args.start_page, args.end_page)
    
    if len(korean_texts) != len(english_texts):
        print(f"경고: 한글 PDF ({len(korean_texts)}페이지)와 영어 PDF ({len(english_texts)}페이지)의 페이지 수가 다릅니다.")
        if len(korean_texts) == 0 or len(english_texts) == 0:
            print("오류: 처리할 페이지가 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)
    
    total_pages = min(len(korean_texts), len(english_texts))
    print(f"총 {total_pages}개 페이지를 처리합니다.")
    
    # 실행 여부 확인
    confirm = input("\n평가를 시작하시겠습니까? (y/n): ").lower()
    if confirm != 'y' and confirm != 'yes':
        print("프로그램을 종료합니다.")
        sys.exit(0)
    
    # OpenAI API 키 설정 및 확인
    openai_api_key = os.getenv('OPENAI_API_KEY')
    use_gpt = False
    
    if not openai_api_key:
        print("\n경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("GPT 번역은 건너뛰고 파인튜닝된 모델만 사용합니다.")
    else:
        use_gpt_input = input("GPT 번역을 포함하시겠습니까? (y/n, 기본값: n): ").lower()
        if use_gpt_input == 'y' or use_gpt_input == 'yes':
            use_gpt = True
            print("GPT 번역을 포함합니다.")
        else:
            print("GPT 번역은 건너뛰고 파인튜닝된 모델만 사용합니다.")
    
    try:
        # 파인튜닝된 모델 로드
        print("\n파인튜닝된 모델 로드 중...")
        model, tokenizer = load_finetuned_model()
        
        print("\n번역 및 평가 시작...")
        for i, (k_text, e_text) in enumerate(zip(korean_texts, english_texts), 1):
            page_num = k_text['page']
            korean = k_text['text']
            english_reference = e_text['text']
            
            print(f"\n=== {page_num}페이지 처리 중 ({i}/{total_pages}) ===")
            
            # 중간 진행 상황 표시
            display_progress_bar(i, total_pages)
            
            # GPT 번역
            gpt_translation = ""
            if use_gpt:
                print("\nGPT로 번역 중...")
                gpt_translation = translate_with_gpt(korean)
            
            # 파인튜닝된 모델 번역
            print("파인튜닝된 모델로 번역 중...")
            finetuned_translation = translate_with_finetuned(korean, model, tokenizer)
            
            # 평가 지표 계산
            print("평가 지표 계산 중...")
            gpt_metrics = calculate_metrics(english_reference, gpt_translation) if gpt_translation else {
                'bleu': 0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0
            }
            finetuned_metrics = calculate_metrics(english_reference, finetuned_translation)
            
            # 결과 저장
            result = {
                'page': page_num,
                'korean_text': korean,
                'english_reference': english_reference,
                'gpt_translation': gpt_translation,
                'finetuned_translation': finetuned_translation,
                'gpt_bleu': gpt_metrics['bleu'],
                'gpt_rouge1': gpt_metrics['rouge1'],
                'gpt_rouge2': gpt_metrics['rouge2'],
                'gpt_rougeL': gpt_metrics['rougeL'],
                'finetuned_bleu': finetuned_metrics['bleu'],
                'finetuned_rouge1': finetuned_metrics['rouge1'],
                'finetuned_rouge2': finetuned_metrics['rouge2'],
                'finetuned_rougeL': finetuned_metrics['rougeL']
            }
            
            # 페이지별 결과 저장
            output_file = save_page_results(result, args.output_dir, page_num)
            print(f"\n{page_num}페이지 결과가 저장되었습니다: {output_file}")
            
            # 중간 결과 미리보기
            print(f"\n[{page_num}페이지 번역 결과 미리보기]")
            print(f"원본 한글 (일부): {korean[:100]}..." if len(korean) > 100 else korean)
            print(f"영어 참조 (일부): {english_reference[:100]}..." if len(english_reference) > 100 else english_reference)
            if gpt_translation:
                print(f"GPT 번역 (일부): {gpt_translation[:100]}..." if len(gpt_translation) > 100 else gpt_translation)
            print(f"파인튜닝 번역 (일부): {finetuned_translation[:100]}..." if len(finetuned_translation) > 100 else finetuned_translation)
            
            print(f"\n[{page_num}페이지 평가 결과]")
            print(f"GPT BLEU: {gpt_metrics['bleu']:.4f}, 파인튜닝 BLEU: {finetuned_metrics['bleu']:.4f}")
            
            all_results.append(result)
            
            # 중간 진행 상황을 요약 파일로 저장
            save_summary_results(all_results, args.output_dir)
    
    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("프로그램을 종료합니다.")
        sys.exit(1)
    
    # 전체 결과 요약 저장
    summary_file = save_summary_results(all_results, args.output_dir)
    print(f"\n전체 평가 결과가 저장되었습니다: {summary_file}")
    
    # 경과 시간 계산
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n총 소요 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")
    
    # 평균 점수 출력
    df = pd.DataFrame(all_results)
    summarize_and_print_results(df)

if __name__ == "__main__":
    main() 