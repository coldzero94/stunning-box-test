import os
from pathlib import Path
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    return dataset

def prepare_model_and_tokenizer(model_name: str, load_in_4bit: bool = True):
    """모델과 토크나이저를 준비합니다."""
    # BitsAndBytes 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def create_peft_config():
    """LoRA 설정을 생성합니다."""
    return LoraConfig(
        r=8,  # 어텐션 헤드의 차원
        lora_alpha=32,  # 스케일링 파라미터
        target_modules=["query_key_value"],  # 타겟 모듈
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def prepare_model_for_training(model, peft_config):
    """모델을 학습을 위해 준비합니다."""
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model

def preprocess_function(examples, tokenizer, max_length=512):
    """데이터를 전처리합니다."""
    # 프롬프트 형식 지정
    prompts = [
        f"### 지시문: {example['instruction']}\n\n"
        f"### 입력:\n{example['input']}\n\n"
        f"### 응답:\n{example['output']}"
        for example in examples
    ]
    
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

def compute_metrics(eval_preds):
    """평가 메트릭을 계산합니다."""
    preds, labels = eval_preds
    # 패딩 토큰 마스킹
    mask = labels != -100
    
    # perplexity 계산
    loss = torch.nn.CrossEntropyLoss(reduction='none')(
        torch.tensor(preds[mask]).view(-1, preds.shape[-1]),
        torch.tensor(labels[mask]).view(-1)
    )
    perplexity = torch.exp(torch.mean(loss)).item()
    
    return {"perplexity": perplexity}

def main():
    # 설정
    model_name = "google/gemma-2b"  # 또는 "Qwen/Qwen-2.5-12B"
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
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
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
        report_to="tensorboard"
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
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("학습 시작...")
    trainer.train()
    
    print("모델 저장 중...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"학습이 완료되었습니다. 모델이 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main() 