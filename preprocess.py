from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
ds = load_dataset("Amod/mental_health_counseling_conversations")

# Load tokenizer for Falcon 7B
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['input'], truncation=True, padding="max_length")

tokenized_ds = ds.map(tokenize_function, batched=True)
