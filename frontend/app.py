import argparse
import os
import torch
from typing import Optional
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from threading import Lock

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
        history.append({"role": "user", "content": message})
        prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        
        with self.lock:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # 입력 토큰 수 확인
                input_token_length = inputs.input_ids.shape[1]
                if input_token_length > 3000:
                    yield "⚠️ 입력이 너무 깁니다. 더 짧은 텍스트로 시도해주세요."
                    return
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # 메모리 정리
                del inputs
                del outputs
                torch.cuda.empty_cache()
                
                yield response
                
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
        cb = gr.Chatbot(type="messages", scale=20, render_markdown=False)
        gr.ChatInterface(hdlr.chat_function, chatbot=cb)

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