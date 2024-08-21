import transformers
import torch
from accelerate import init_empty_weights, infer_auto_device_map


# Define the custom cache directory
custom_cache_dir = "/home/labelling/llms/llama/models"


def test_multi_gpu():
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


def test_multi_gpu1():
    prompt = "How are you doing today?"
    print("Prompt:", prompt)
    torch.cuda.empty_cache() 
    
    # Load the tokenizer and model from the custom directory
    tokenizer = transformers.AutoTokenizer.from_pretrained(custom_cache_dir, local_files_only=True)
    # model = transformers.AutoModelForCausalLM.from_pretrained(custom_cache_dir, local_files_only=True)
    # Initialize the model with empty weights
    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_pretrained(custom_cache_dir, local_files_only=True)

    # Infer the device map for multi-GPU setup
    device_map = infer_auto_device_map(model)

    # Load the model with the inferred device map and set torch_dtype to float16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        custom_cache_dir,
        local_files_only=True,
        device_map=device_map,
        torch_dtype=torch.float16
    )

    # # Initialize the pipeline with the loaded model and tokenizer
    # pipeline = transformers.pipeline(
    #     "text-generation", 
    #     model=model, 
    #     tokenizer=tokenizer,
    #     device_map=device_map
    # )

    # # Generate text
    # output = pipeline("Hey how are you doing today?", max_length=50)
    # print(output[0]["generated_text"])
    # print("Done!")

    # # Load the model with the inferred device map
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     custom_cache_dir,
    #     local_files_only=True,
    #     device_map=device_map,
    #     torch_dtype=torch.bfloat16
    # )

    # # Initialize the pipeline with the loaded model and tokenizer
    # # pipeline = transformers.pipeline(
    # #     "text-generation", 
    # #     model=model, 
    # #     tokenizer=tokenizer,
    # #     # model_kwargs={"torch_dtype": torch.bfloat16}, 
    # #     # device_map="auto",
    # #     device_map=device_map
    # #     # device=2
    # # )

    # # Generate text
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda:1") 
    # inputs = {key: value.to(model.device) for key, value in inputs.items()}
    # # output = pipeline(prompt)
    # # print(output[0]["generated_text"])
    # # print("Done!")

    #     # Generate text using the model
    # output = model.generate(**inputs)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # print("Generated Text:", generated_text)

    prompt = "Hey how are you doing today?"

    # Tokenize the input and move to the appropriate device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}

    # Generate text using the model
    output = model.generate(**inputs)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Generated Text:", generated_text)


if __name__ == "__main__":
    test_multi_gpu1()

