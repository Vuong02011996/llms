import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B"
access_token = "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"
custom_cache_dir = "/home/labelling/llms/llama/models"
# custom_cache_dir = "/home/labelling/llms/llama/models" # ~/.cache/huggingface/transformers

pipeline = transformers.pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    token=access_token, 
    device_map="auto",
    # cache_dir=custom_cache_dir
)

# # Load the model configuration
# model = pipeline.model
# print("Default token length:", model.config.max_position_embeddings)

output = pipeline("Hey how are you doing today?", max_length=50)
print(output[0]["generated_text"])
print("Done!")

# Save the tokenizer and model
# Save the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# tokenizer.save_pretrained(custom_cache_dir)
# model.save_pretrained(custom_cache_dir)

# print("Model and tokenizer saved to", custom_cache_dir)