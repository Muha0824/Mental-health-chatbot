from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

def fine_tune_model():
    # Load the tokenized dataset from disk
    tokenized_datasets = load_dataset("tokenized_mental_health_conversations")  # Correct path to your tokenized data

    # Load the model and tokenizer
    model_name = "tiiuae/falcon-7b"  # You can also try smaller models like "falcon-3b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up training arguments for slower but manageable training
    training_args = TrainingArguments(
        output_dir="./model",  # Save the model here after training
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        learning_rate=5e-5,
        per_device_train_batch_size=1,  # Reduce batch size
        per_device_eval_batch_size=1,   # Reduce batch size for evaluation as well
        num_train_epochs=2,  # Reduce number of epochs to speed up training
        weight_decay=0.01,
        logging_dir="./logs",  # Directory to save logs
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch size
        fp16=False,  # Optionally, disable mixed precision to avoid potential issues
        dataloader_num_workers=2,  # Limit the number of workers to avoid slowdowns
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,  # The model to train
        args=training_args,  # The training arguments
        train_dataset=tokenized_datasets['train'],  # Training dataset
        eval_dataset=tokenized_datasets['train'],  # Validation dataset (optional)
        tokenizer=tokenizer,  # Tokenizer for text preprocessing
    )

    # Start training
    trainer.train()

    # Save the trained model
    model.save_pretrained("./model/falcon-7b-mental-health")

if __name__ == "__main__":
    fine_tune_model()
