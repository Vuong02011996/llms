from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

# Set the environment variable to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# model_id = "akjindal53244/Llama-3.1-Storm-8B"  # FP8 model: "akjindal53244/Llama-3.1-Storm-8B-FP8-Dynamic"
model_id = "akjindal53244/Llama-3.1-Storm-8B-FP8-Dynamic" 
num_gpus = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = LLM(model=model_id, tensor_parallel_size=num_gpus, max_num_batched_tokens = 4944,max_num_seqs=4944)
# llm = LLM(model=model_id, tensor_parallel_size=num_gpus, kv_cache_size=131072)
sampling_params = SamplingParams(max_tokens=128, temperature=0.01, top_k=100, top_p=0.95)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize = False)
print(llm.generate([prompt], sampling_params)[0].outputs[0].text.strip())

# Error: OutofMemoryError: CUDA out of memory. 