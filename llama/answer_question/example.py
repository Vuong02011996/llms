from transformers import pipeline
import transformers
import torch


def question_answering_llama3_1():
    task = "question-answering"
    custom_cache_dir = "/home/labelling/llms/llama/models_fp8/models_storm"
    tokenizer = transformers.AutoTokenizer.from_pretrained(custom_cache_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(custom_cache_dir, device_map = 'auto')
    # Initialize the pipeline with the loaded model and tokenizer
    model_id = "akjindal53244/Llama-3.1-Storm-8B"
    qa_pipeline = pipeline(
        task, 
        model=model_id, 
        # tokenizer=tokenizer, 
        model_kwargs={"torch_dtype": torch.bfloat16}, 
        device_map="auto",
    )

    # Initialize the question-answering pipeline
    # qa_pipeline = pipeline("question-answering")

    # Define the context and the question
    context = """
    Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.
    The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
    """
    question = "What is AI?"

    # Get the answer from the pipeline
    result = qa_pipeline(question=question, context=context)

    # Print the result
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']}")
    print(f"Start: {result['start']}")
    print(f"End: {result['end']}")

def question_answering_example():
    """
    No model was supplied, defaulted to distilbert/distilbert-base-cased-distilled-squad and revision 626af31 (https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad).
    """
    # Initialize the question-answering pipeline
    qa_pipeline = pipeline("question-answering")

    # Define the context and the question
    context = """
    Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.
    The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
    """
    question = "What is AI?"

    # Get the answer from the pipeline
    result = qa_pipeline(question=question, context=context)

    # Print the result
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']}")
    print(f"Start: {result['start']}")
    print(f"End: {result['end']}")

if __name__ == "__main__":
    question_answering_llama3_1()