from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def fine_tune_model():
    # Load the tokenized dataset from disk
    tokenized_datasets = load_dataset("tokenized_mental_health_conversations")  # Correct path to your tokenized data

    # Load the model and tokenizer
    model_name = "tiiuae/falcon-7b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./model",  # Save the model here after training
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        learning_rate=5e-5,
        per_device_train_batch_size=1,  # Use a small batch size due to the model size
        per_device_eval_batch_size=1,
        num_train_epochs=3,  # Fine-tune for a few epochs
        weight_decay=0.01,
        logging_dir="./logs",  # Directory to save logs
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,  # Use mixed-precision for faster training
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,  # The model to train
        args=training_args,  # The training arguments
        train_dataset=tokenized_datasets['train'],  # Training dataset (using 'train' split)
        eval_dataset=tokenized_datasets['train'],  # Validation dataset (optional, can be separated)
        tokenizer=tokenizer,  # Tokenizer for text preprocessing
    )

    # Start training
    trainer.train()

    # Save the trained model
    model.save_pretrained("./model/falcon-7b-mental-health")

if __name__ == "__main__":
    fine_tune_model()
