from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
ds = load_dataset("Amod/mental_health_counseling_conversations")

# Load tokenizer for Falcon 7B
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

# Manually set pad_token to eos_token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define preprocessing function
def preprocess_function(examples):
    # Format each conversation correctly
    conversation = [f"User: {context} Counselor: {response}" for context, response in zip(examples['Context'], examples['Response'])]
    
    # Tokenize the conversation
    return tokenizer(conversation, padding=True, truncation=True, max_length=512)

# Apply preprocessing to the 'train' split
tokenized_datasets = ds.map(preprocess_function, batched=True, remove_columns=["Context", "Response"])

# Check the columns of the tokenized dataset
print(tokenized_datasets['train'].column_names)
