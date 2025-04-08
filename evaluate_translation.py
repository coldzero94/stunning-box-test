import os
import pandas as pd
import fitz  # pymupdf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import openai
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

# 환경 변수 로드
load_dotenv()

def get_args():
    """커맨드 라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='PDF 번역 평가 스크립트')
    parser.add_argument('--korean-pdf', type=str, required=True, help='한글 PDF 파일 경로')
    parser.add_argument('--english-pdf', type=str, required=True, help='영어 PDF 파일 경로')
    parser.add_argument('--start-page', type=int, default=1, help='시작 페이지 (1부터 시작)')
    parser.add_argument('--end-page', type=int, help='끝 페이지 (미지정시 마지막 페이지까지)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='결과 저장 디렉토리')
    return parser.parse_args()

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

def translate_with_finetuned(text, model, tokenizer, max_length=512):
    """파인튜닝된 모델로 번역을 수행합니다."""
    prompt = f"### 입력:\n{text}\n\n### 응답:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
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
        response = openai.ChatCompletion.create(
            model="gpt-4",
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

def main():
    args = get_args()
    
    # 결과를 저장할 리스트 초기화
    all_results = []
    
    # PDF에서 텍스트 추출
    print("PDF에서 텍스트 추출 중...")
    korean_texts = extract_text_from_pdf(args.korean_pdf, args.start_page, args.end_page)
    english_texts = extract_text_from_pdf(args.english_pdf, args.start_page, args.end_page)
    
    if len(korean_texts) != len(english_texts):
        print(f"경고: 한글 PDF ({len(korean_texts)}페이지)와 영어 PDF ({len(english_texts)}페이지)의 페이지 수가 다릅니다.")
    
    # 파인튜닝된 모델 로드
    print("파인튜닝된 모델 로드 중...")
    model, tokenizer = load_finetuned_model()
    
    # OpenAI API 키 설정
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    print("번역 및 평가 시작...")
    for k_text, e_text in tqdm(zip(korean_texts, english_texts), total=len(korean_texts)):
        page_num = k_text['page']
        korean = k_text['text']
        english_reference = e_text['text']
        
        print(f"\n=== {page_num}페이지 처리 중 ===")
        
        # GPT 번역
        print("GPT로 번역 중...")
        gpt_translation = translate_with_gpt(korean)
        
        # 파인튜닝된 모델 번역
        print("파인튜닝된 모델로 번역 중...")
        finetuned_translation = translate_with_finetuned(korean, model, tokenizer)
        
        # 평가 지표 계산
        print("평가 지표 계산 중...")
        gpt_metrics = calculate_metrics(english_reference, gpt_translation)
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
        print(f"{page_num}페이지 결과가 저장되었습니다: {output_file}")
        
        all_results.append(result)
    
    # 전체 결과 요약 저장
    summary_file = save_summary_results(all_results, args.output_dir)
    print(f"\n전체 평가 결과가 저장되었습니다: {summary_file}")
    
    # 평균 점수 출력
    df = pd.DataFrame(all_results)
    print("\n=== 평균 평가 점수 ===")
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

if __name__ == "__main__":
    main() 