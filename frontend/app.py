import argparse
import os
import torch
from typing import Optional, List, Dict, Any
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from threading import Lock, Thread
import time

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

    async def chat_function(self, message, history):
        history_format = []
        for h in history:
            history_format.append({"role": "user", "content": h[0]})
            if h[1] is not None:
                history_format.append({"role": "assistant", "content": h[1]})
        
        history_format.append({"role": "user", "content": message})
        prompt = self.tokenizer.apply_chat_template(history_format, tokenize=False, add_generation_prompt=True)
        
        with self.lock:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # 입력 토큰 수 확인
                input_token_length = inputs.input_ids.shape[1]
                if input_token_length > 3000:
                    yield "⚠️ 입력이 너무 깁니다. 더 짧은 텍스트로 시도해주세요."
                    return
                
                # 답변 생성 중 메시지 실시간 표시
                yield "⏳ 답변을 생성 중입니다..."
                
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
                
                # 첫 번째 토큰 대기
                first_token = True
                generated_text = ""
                
                # 토큰 스트리밍
                for new_text in streamer:
                    if first_token:
                        # 첫 번째 토큰이면 "생성 중" 메시지를 대체
                        generated_text = new_text
                        first_token = False
                    else:
                        # 이후 토큰은 누적
                        generated_text += new_text
                    
                    # 각 토큰 출력
                    yield generated_text
                
                # 스레드 종료 대기
                thread.join()
                
                # 메모리 정리
                del inputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                yield f"Error: {str(e)}"

def main(args):
    print(f"Loading the model {args.model_id}...")
    hdlr = LLMChatHandler(model_id=args.model_id, max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len, dtype=args.dtype)

    with gr.Blocks(title=f"🤗 Chatbot with {args.model_id}", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(
                f"<h2>Chatbot with 🤗 {args.model_id} 🤗</h2>"
                "<h3>Interact with LLM using chat interface!<br></h3>"
                f"<h3>Original model: <a href='https://huggingface.co/{args.model_id}' target='_blank'>{args.model_id}</a></h3>")
        
        chatbot = gr.Chatbot(scale=20, render_markdown=True)
        
        with gr.Row():
            with gr.Column(scale=8):
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="메시지를 입력하세요...",
                    container=False
                )
            with gr.Column(scale=1):
                submit = gr.Button("전송")
        
        clear = gr.Button("대화 지우기")
        
        def user(user_message, history):
            return "", history + [[user_message, None]]
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            hdlr.chat_function, [chatbot[-1][0], chatbot], chatbot[-1][1]
        )
        
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            hdlr.chat_function, [chatbot[-1][0], chatbot], chatbot[-1][1]
        )
        
        clear.click(lambda: None, None, chatbot, queue=False)

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