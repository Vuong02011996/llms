import transformers
import torch

# Define the custom cache directory
custom_cache_dir = "/home/labelling/llms/llama/models"

# Load the tokenizer and model from the custom directory
tokenizer = transformers.AutoTokenizer.from_pretrained(custom_cache_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(custom_cache_dir)

# Initialize the pipeline with the loaded model and tokenizer
pipeline = transformers.pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    model_kwargs={"torch_dtype": torch.float16}, 
    device=1,
    device_map="auto",  # Enable multi-GPU support
)

# Generate text
output = pipeline("Hey how are you doing today?", max_length=50)
print(output[0]["generated_text"])
print("Done!")