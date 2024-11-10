from datasets import load_dataset

# Replace the name if it's different or check if there are any versioning issues
ds = load_dataset("Amod/mental_health_counseling_conversations")

# Check the structure of the dataset
print(ds)

# Inspect the first entry in the 'train' split
print(ds['train'][0])
