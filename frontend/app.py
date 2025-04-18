import argparse
import os
import torch
from typing import Optional, List, Dict, Any
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from threading import Lock, Thread
import time
import asyncio

# CUDA 메모리 설정 (최소화)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class LLMChatHandler():
    def __init__(self, model_id: str, max_num_seqs: int, max_model_len: int, dtype: str):
        # GPU 메모리 초기화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 로컬 모델 경로인지 확인
        is_local_path = os.path.exists(model_id) and os.path.isdir(model_id)
        
        # 토크나이저 및 모델 로드 (evaluate_translation.py와 동일하게)
        if is_local_path:
            print(f"로컬 모델 경로 사용: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            has_adapter = os.path.exists(os.path.join(model_id, "adapter_model.safetensors"))
            if has_adapter:
                print("어댑터 모델 감지됨. 기본 모델과 결합합니다.")
                base_model = "Qwen/Qwen2.5-14B-Instruct"
                print(f"기본 모델 로드 중: {base_model}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    offload_buffers=True
                )
                print(f"어댑터 로드 중: {model_id}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    model_id,
                    offload_buffers=True
                )
            else:
                print("어댑터 모델이 감지되지 않았습니다. 기본 모델을 사용합니다.")
                self.model = model_id
        else:
            print(f"Hugging Face 모델 ID 사용: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = model_id

        self.lock = Lock()
        self.max_new_tokens = 1024
        self.temperature = 0.3
        self.top_p = 0.95
        self.top_k = 50
        self.repetition_penalty = 1.1

    def format_chat_history(self, history):
        """채팅 기록을 모델 입력에 맞게 포맷팅합니다."""
        formatted = []
        for message in history:
            if isinstance(message, dict) and "role" in message and "content" in message:
                formatted.append(message)
        return formatted

    def generate_response(self, message, history):
        """동기 함수로 응답을 생성합니다."""
        # 스트리밍 시작 시간 기록
        start_time = time.time()
        
        # 대화 기록 형식화
        history_format = self.format_chat_history(history)
        
        # 현재 메시지 추가
        history_format.append({"role": "user", "content": message})
        
        # 채팅 템플릿 적용
        prompt = self.tokenizer.apply_chat_template(history_format, tokenize=False, add_generation_prompt=True)
        
        with self.lock:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # 입력 토큰 수 확인
                input_token_length = inputs.input_ids.shape[1]
                if input_token_length > 3000:
                    yield "⚠️ 입력이 너무 깁니다. 더 짧은 텍스트로 시도해주세요."
                    return
                
                # 스트리머 설정 (토큰 단위 스트리밍)
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                # 생성 파라미터 설정
                gen_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "repetition_penalty": self.repetition_penalty,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "streamer": streamer
                }
                
                # 별도 스레드에서 생성 실행
                thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
                thread.start()
                
                # 응답 누적
                generated_text = ""
                is_first_token = True
                token_count = 0
                
                for new_text in streamer:
                    # 첫 토큰 시간 기록 (지연 시간 계산용)
                    if is_first_token:
                        first_token_time = time.time()
                        is_first_token = False
                    
                    generated_text += new_text
                    token_count += 1
                    
                    # 진행 중임을 나타내는 표시
                    yield generated_text + " ⌛"
                
                # 스레드 종료 대기
                thread.join()
                
                # 생성 완료 시간 계산
                end_time = time.time()
                generation_time = end_time - start_time
                first_token_latency = first_token_time - start_time if not is_first_token else 0
                
                # 생성 속도 계산 (초당 토큰)
                if generation_time > 0:
                    tokens_per_second = token_count / generation_time
                else:
                    tokens_per_second = 0
                
                # 생성 통계 추가
                stats = f"\n\n---\n✅ 생성 완료 (토큰: {token_count}개, 시간: {generation_time:.2f}초, 속도: {tokens_per_second:.1f}토큰/초, 첫 토큰: {first_token_latency:.2f}초)"
                
                # 최종 텍스트 반환 (진행 중 표시 제거하고 통계 추가)
                yield generated_text + stats
                
                # 메모리 정리
                del inputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                yield f"Error: {str(e)}"

def main(args):
    print(f"Loading the model {args.model_id}...")
    hdlr = LLMChatHandler(model_id=args.model_id, max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len, dtype=args.dtype)

    with gr.Blocks(title=f"🤗 스터닝 박스 챗봇", fill_height=True) as demo:
        gr.Markdown(
            f"<h2>번역을 위한 에이전트</h2>"
        )
        
        # 상태 표시 추가
        status = gr.Markdown("✨ 준비 완료", elem_id="status")
        
        # 메시지 형식 사용으로 경고 제거
        chatbot = gr.Chatbot(type='messages', scale=20, render_markdown=True)
        
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="메시지를 입력하세요...",
                container=False,
                scale=8
            )
            submit = gr.Button("전송", scale=1)
        
        clear = gr.Button("대화 지우기")
        
        def user_message(message, history, status_text=None):
            """사용자 메시지를 추가하는 함수"""
            if message.strip() == "":
                return "", history, "✨ 준비 완료"
            
            # 이미 history가 있으면 그대로 사용
            if history is None:
                history = []
            
            # 메시지 형식으로 추가
            history.append({"role": "user", "content": message})
            return "", history, "⌛ 응답 생성 중..."
        
        def bot_response(history, status_text=None):
            """봇 응답을 생성하는 함수"""
            if not history:
                yield history, "✨ 준비 완료"
                return
            
            # 마지막 사용자 메시지 가져오기
            last_user_message = history[-1]["content"]
            history_so_far = history[:-1]
            
            # 응답 생성
            for new_text in hdlr.generate_response(last_user_message, history_so_far):
                new_history = history.copy()
                
                # 스트리밍 중인지 완료된건지 확인
                if new_text.endswith("⌛"):
                    status_update = "⌛ 응답 생성 중..."
                    new_text = new_text[:-1]  # 진행 중 표시 제거
                elif new_text.endswith("초)"):
                    status_update = "✅ 응답 생성 완료"
                else:
                    status_update = "✨ 준비 완료"
                
                # 봇 응답 추가 (메시지 형식)
                if len(new_history) > 0 and new_history[-1]["role"] == "assistant":
                    new_history[-1]["content"] = new_text
                else:
                    new_history.append({"role": "assistant", "content": new_text})
                
                yield new_history, status_update
        
        # 이벤트 연결
        msg.submit(
            user_message, 
            [msg, chatbot, status], 
            [msg, chatbot, status],
            queue=False
        ).then(
            bot_response,
            [chatbot, status],
            [chatbot, status]
        )
        
        submit.click(
            user_message, 
            [msg, chatbot, status], 
            [msg, chatbot, status],
            queue=False
        ).then(
            bot_response,
            [chatbot, status],
            [chatbot, status]
        )
        
        clear.click(lambda: ([], "✨ 준비 완료"), None, [chatbot, status], queue=False)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OSS Chatbot",
        description="Run open source LLMs from HuggingFace with a simple chat interface")

    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct", help="HuggingFace model name for LLM.")
    parser.add_argument("--port", default=7860, type=int, help="Port number for the Gradio app.")
    parser.add_argument("--dtype", default="auto", type=str, help="Data type for model weights and activations.")
    parser.add_argument("--max-num-seqs", default=16, type=int, help="")
    parser.add_argument("--max-model-len", default=32767, type=int, help="")
    args = parser.parse_args()

    main(args) 