import os
from pathlib import Path
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
import torch
from dotenv import load_dotenv
import json

# 환경 변수 로드
load_dotenv()

def load_and_process_data(data_dir: str):
    """데이터셋을 로드하고 전처리합니다."""
    # JSONL 파일에서 데이터셋 로드
    dataset = load_dataset('json', 
                         data_files={
                             'train': str(Path(data_dir) / 'train.jsonl'),
                             'test': str(Path(data_dir) / 'test.jsonl')
                         })
    
    # 데이터셋 형식 확인
    print("데이터셋 컬럼:", dataset["train"].column_names)
    print("데이터셋 샘플:", dataset["train"][0])
    
    return dataset

def prepare_model_and_tokenizer(model_name: str):
    """모델과 토크나이저를 준비합니다."""
    # 모델 로드 (양자화 없이)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv('HUGGINGFACE_TOKEN')
    )
    model.config.use_cache = False
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.getenv('HUGGINGFACE_TOKEN')
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def create_peft_config():
    """LoRA 설정을 생성합니다."""
    # Gemma 모델에 대한 target_modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return LoraConfig(
        r=8,  # 어텐션 헤드의 차원
        lora_alpha=32,  # 스케일링 파라미터
        target_modules=target_modules,  # 타겟 모듈
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def prepare_model_for_training(model, peft_config):
    """모델을 학습을 위해 준비합니다."""
    print("Applying PEFT LoRA configuration...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # 학습 가능한 파라미터 수 출력
    return model

def preprocess_function(examples, tokenizer, max_length=256):
    """데이터를 전처리합니다."""
    # 데이터셋 형식에 따라 프롬프트 생성
    prompts = []
    
    # 데이터셋의 컬럼 이름에 따라 처리
    if 'instruction' in examples and 'input' in examples and 'output' in examples:
        # instruction, input, output 형식
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i] if examples['instruction'][i] is not None else ""
            input_text = examples['input'][i] if examples['input'][i] is not None else ""
            output_text = examples['output'][i] if examples['output'][i] is not None else ""
            
            prompt = (
                f"### 지시문: {instruction}\n\n"
                f"### 입력:\n{input_text}\n\n"
                f"### 응답:\n{output_text}{tokenizer.eos_token}"
            )
            prompts.append(prompt)
    elif 'source' in examples and 'target' in examples:
        # source, target 형식 (번역 데이터셋)
        for i in range(len(examples['source'])):
            source = examples['source'][i] if examples['source'][i] is not None else ""
            target = examples['target'][i] if examples['target'][i] is not None else ""
            
            prompt = (
                f"### 입력:\n{source}\n\n"
                f"### 응답:\n{target}{tokenizer.eos_token}"
            )
            prompts.append(prompt)
    else:
        # 기타 형식 (데이터셋의 첫 번째 샘플을 출력하여 디버깅)
        print("지원되지 않는 데이터셋 형식입니다. 데이터셋 샘플:", examples)
        raise ValueError("지원되지 않는 데이터셋 형식입니다.")
    
    # 토크나이징
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 레이블 설정
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def main():
    # 설정
    model_name = "google/gemma-2b-it"  # 또는 "Qwen/Qwen-2.5-12B"
    data_dir = "training_data"
    output_dir = "qlora_output"
    
    print("데이터 로드 중...")
    dataset = load_and_process_data(data_dir)
    
    print("모델 및 토크나이저 준비 중...")
    model, tokenizer = prepare_model_and_tokenizer(model_name)
    
    print("LoRA 설정 준비 중...")
    peft_config = create_peft_config()
    model = prepare_model_for_training(model, peft_config)
    
    print("데이터 전처리 중...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 학습 설정 (매개변수 업데이트)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer 초기화 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )
    
    print("학습 시작...")
    trainer.train()
    
    print("모델 저장 중...")
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"모델을 {output_dir}에 저장합니다...")
    
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"학습이 완료되었습니다. 모델이 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main() 