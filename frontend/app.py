import argparse
import os
import torch
from typing import Optional
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

class LLMChatHandler():
    def __init__(self, model_id: str, max_num_seqs: int, max_model_len: int, dtype: str):
        # 로컬 모델 경로인지 확인
        is_local_path = os.path.exists(model_id) and os.path.isdir(model_id)
        
        # 토크나이저 로드
        if is_local_path:
            print(f"로컬 모델 경로 사용: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # 어댑터 모델인지 확인
            has_adapter = os.path.exists(os.path.join(model_id, "adapter_model.safetensors"))
            if has_adapter:
                print("어댑터 모델 감지됨. 기본 모델과 결합합니다.")
                base_model = "Qwen/Qwen2.5-14B-Instruct"
                print(f"기본 모델 로드 중: {base_model}")
                
                # CUDA를 사용하고 GPU 메모리를 중간 이상으로 사용
                print("CUDA를 사용하고 GPU 메모리를 중간 이상으로 사용합니다.")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",  # CUDA 사용
                    offload_folder=None,  # 오프로드 폴더 사용 안 함
                    offload_state_dict=False  # 오프로드 상태 딕셔너리 사용 안 함
                )
                
                # 어댑터 로드 및 적용
                print(f"어댑터 로드 중: {model_id}")
                model.load_adapter(model_id)
                
                # 모델 병합 후 사용
                print("모델 병합 중...")
                # 병합된 모델을 저장할 디렉토리 생성
                merged_model_dir = model_id + "_merged"
                os.makedirs(merged_model_dir, exist_ok=True)
                
                # 모델 병합 및 저장
                print("모델 병합 및 저장 중...")
                # 병합된 모델을 저장하기 전에 모든 파라미터를 CPU로 로드
                for name, param in model.named_parameters():
                    if param.device.type == "meta":
                        print(f"메타 텐서 감지: {name}")
                        # 메타 텐서를 CPU로 로드
                        param.to("cpu")
                
                # 모델 저장
                print("병합된 모델 저장 중...")
                model.save_pretrained(merged_model_dir)
                
                # 모델 ID를 병합된 모델 디렉토리로 설정
                model_id = merged_model_dir
                print(f"병합된 모델 저장됨: {model_id}")
            else:
                print("어댑터 모델이 감지되지 않았습니다. 기본 모델을 사용합니다.")
        else:
            print(f"Hugging Face 모델 ID 사용: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        def _guess_quantization(model_id) -> Optional[str]:
            if "awq" in model_id or "AWQ" in model_id:
                return "awq"
            if "bnb" in model_id:
                return "bitsandbytes"
            return "awq"  # 기본적으로 AWQ 양자화 사용

        def _guess_load_format(model_id) -> Optional[str]:
            if "bnb" in model_id:
                return "bitsandbytes"
            return "auto"

        # GPU 개수 확인
        num_gpus = torch.cuda.device_count()
        print(f"사용 가능한 GPU 개수: {num_gpus}")
        
        # GPU 메모리 확인
        if num_gpus > 0:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB 단위
            print(f"GPU 메모리: {gpu_memory:.2f} GB")
            
            # GPU 메모리가 충분하면 병렬화를 사용하지 않음
            tensor_parallel_size = 1
            # GPU 메모리 활용도 높임
            gpu_memory_utilization = 0.95
            # 배치 처리 최적화
            max_num_batched_tokens = 16384
        else:
            tensor_parallel_size = 1
            gpu_memory_utilization = 0.9
            max_num_batched_tokens = 4096
        
        engine_args = AsyncEngineArgs(
            model=model_id,
            task="generate",
            tokenizer=None,
            tokenizer_mode="auto",
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            load_format=_guess_load_format(model_id=model_id),
            quantization=_guess_quantization(model_id=model_id),
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,  # GPU 메모리 활용도 높임
            tensor_parallel_size=tensor_parallel_size,  # 단일 GPU 사용
            max_num_batched_tokens=max_num_batched_tokens,  # 배치 처리 최적화
        )
        self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def chat_function(self, message, history):
        history.append({"role": "user", "content": message})
        prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        sampling_params = SamplingParams(
            stop_token_ids=self.terminators,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1)
        results_generator = self.vllm_engine.generate(prompt, sampling_params, random_uuid())
        async for request_output in results_generator:
            response_txt = ""
            for output in request_output.outputs:
                if output.text not in self.terminators:
                    response_txt += output.text
            yield response_txt

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