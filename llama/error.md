# Access to model meta-llama/Meta-Llama-3.1-8B is restricted. You must be authenticated to access it.
+ Create account hugging face and accepted.
+ Go to https://huggingface.co/settings/tokens (account -> setting -> access token)create token and pass to code:
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, token=access_token, device_map="auto"
)

# OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like meta-llama/Meta-Llama-3.1-8B is not the path to a directory containing a file named config.json
+ Create new Access Token with permissions is WRITE not 

# ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`
+ pip install accelerate

# Error when combine to source ai-service

## huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/labelling/llms/llama/models'. Use `repo_type` argument if needed
no answers
+ Error model path need to save in container, load model from host get this error: example `model_path = '/app/core/model/gemma-embedding' `
+   custom_cache_dir = "/home/dxtech/Project/llm_local/tanika-ai-service/app/core/model/llama" => custom_cache_dir = "/app/core/model/model_llama"
    

## ValueError: `rope_scaling` must be a dictionary with with two fields, `name` and `factor`, got {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
no answers
+ https://github.com/meta-llama/llama3/issues/299
+ pip install transformers==4.43.1

# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:2 and cuda:1!`
