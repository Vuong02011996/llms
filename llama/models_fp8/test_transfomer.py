import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def save_model(model_id, custom_cache_dir):
    # Save the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer.save_pretrained(custom_cache_dir)
    model.save_pretrained(custom_cache_dir)
    print("Model and tokenizer saved to", custom_cache_dir)



def test_model():
    # Check if the custom cache directory exists, and create it if it does not
    custom_cache_dir = "/home/labelling/llms/llama/models_fp8/models_storm"
    if not os.path.exists(custom_cache_dir):
        os.makedirs(custom_cache_dir)
        model_id = "akjindal53244/Llama-3.1-Storm-8B"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        save_model(model_id, custom_cache_dir)

    else:
        # Load the tokenizer and model from the custom directory
        print("Loading model from: ", custom_cache_dir)
        tokenizer = transformers.AutoTokenizer.from_pretrained(custom_cache_dir)
        model = transformers.AutoModelForCausalLM.from_pretrained(custom_cache_dir, device_map = 'auto')
        # Initialize the pipeline with the loaded model and tokenizer
        pipeline = transformers.pipeline(
            "question-answering", 
            model=model, 
            tokenizer=tokenizer, 
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            device_map="auto",
        )

    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "What is 2+2?"}
    # ]
    messages = "How are you doing today?"
    outputs = pipeline(messages, max_new_tokens=128, do_sample=True, temperature=0.01, top_k=100, top_p=0.95)
    print(outputs[0]["generated_text"])
    # print(outputs[0]["generated_text"][-1])  # Expected Output: {'role': 'assistant', 'content': '2 + 2 = 4'}

    # Run oke on 2 GPU 1, 2


def question_answering_llama3():
    custom_cache_dir = "/home/labelling/llms/llama/models_fp8/models_storm"
    task = "question-answering"
    if not os.path.exists(custom_cache_dir):
        os.makedirs(custom_cache_dir)
        model_id = "akjindal53244/Llama-3.1-Storm-8B"
        pipeline = transformers.pipeline(
            task,
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        save_model(model_id, custom_cache_dir)

    else:
        # Load the tokenizer and model from the custom directory
        print("Loading model from: ", custom_cache_dir)
        tokenizer = transformers.AutoTokenizer.from_pretrained(custom_cache_dir)
        model = transformers.AutoModelForCausalLM.from_pretrained(custom_cache_dir, device_map = 'auto')
        # Initialize the pipeline with the loaded model and tokenizer
        pipeline = transformers.pipeline(
            task, 
            model=model, 
            tokenizer=tokenizer, 
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            device_map="auto",
        )

    paragraph = """
                    Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. 
                    The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
                """
    # if isinstance(paragraph, str):
    #     paragraph_dict = {"context": paragraph}
    #     pipeline(paragraph_dict)
    # else:
    #     pipeline(paragraph)

    question = "Where do I live?"
    context = "My name is Merve and I live in İstanbul."
    outputs = pipeline(question = question, context = context)
            
    # outputs = pipeline(paragraph)
    print("outputs: ", outputs)

if __name__ == "__main__":
    question_answering_llama3()