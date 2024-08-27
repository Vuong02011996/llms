# ValueError: The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (4944)
+  The error indicates that the model's maximum sequence length (131072) exceeds the maximum number of tokens that can be stored in the Key-Value (KV) cache (4944). To resolve this, you can either reduce the model's maximum sequence length or increase the KV cache size if possible.
+ max-num-seqs
    Maximum number of sequences per iteration.
    Default: 256
    
+ `llm = LLM(model=model_id, tensor_parallel_size=num_gpus, max_num_seqs=4944)`
# ValueError: max_num_batched_tokens (512) must be greater than or equal to max_num_seqs (4944).
