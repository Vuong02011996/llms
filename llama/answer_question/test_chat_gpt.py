from transformers import pipeline

# Load a pre-trained question generation model (e.g., GPT-based)
generator = pipeline("question-generation")

# Input paragraph
paragraph = """
    Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. 
    The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
"""

# Generate QA pairs
qa_pairs = generator(paragraph)

# Output QA pairs
for qa in qa_pairs:
    print(f"Question: {qa['question']}")
    print(f"Answer: {qa['answer']}")
