import pandas as pd
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Step 1: Prepare Initial Data
def prepare_initial_data():
    # Simulate small initial labeled dataset
    data = {
        "text": ["Document about AI", "File related to finance", "Tech trends of 2023", "Market analysis report"],
        "label": [0, 1, 0, 1]  # Example: 0 = AI/Tech, 1 = Finance
    }
    return pd.DataFrame(data)

# Step 2: Train Model with Optimizations
def train_model(train_data, model_name="distilbert-base-uncased", num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Tokenize Data with Shortened Sequence Length
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)  # Shortened length
    
    dataset = Dataset.from_pandas(train_data)
    tokenized_data = dataset.map(tokenize_function, batched=True)
    train_data, val_data = tokenized_data.train_test_split(test_size=0.2, seed=42).values()

    # Training Arguments with Optimizations
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,  # Fewer epochs
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=True,  # Enable mixed precision
        gradient_accumulation_steps=2,  # Simulate larger batch size
        load_best_model_at_end=True,  # Resume best checkpoint
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )
    trainer.train()
    return trainer, tokenizer, model

# Step 3: Classify Unlabeled Documents
def classify_documents(unlabeled_data, trainer, tokenizer):
    tokenized_data = tokenizer(unlabeled_data["text"].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
    outputs = trainer.model(**tokenized_data)
    predictions = torch.argmax(outputs.logits, dim=1)
    
    # Map predictions to category names
    label_mapping = {0: "AI", 1: "Finance"}
    unlabeled_data["predicted_label"] = predictions.tolist()
    unlabeled_data["predicted_category"] = unlabeled_data["predicted_label"].map(label_mapping)
    return unlabeled_data

# Step 4: Human Refinement (Simulated)
def refine_labels(predictions):
    # Simulate manual refinement
    refined_data = predictions.copy()
    refined_data.loc[0, "predicted_label"] = 0  # Example refinement
    refined_data["label"] = refined_data["predicted_label"]  # Treat as labeled for next iteration
    return refined_data

# Step 5: Iterative Retraining with Sampling
def iterative_training():
    # Check for initial labeled dataset
    labeled_data_path = "randomized_initial_labeled_data.csv"
    if not os.path.exists(labeled_data_path):
        print("Initial labeled data not found. Generating a small initial dataset...")
        labeled_data = prepare_initial_data()
        labeled_data.to_csv(labeled_data_path, index=False)
    else:
        labeled_data = pd.read_csv(labeled_data_path)
    
    # Load unlabeled data
    unlabeled_data_path = "new_unlabled_data.csv"
    if not os.path.exists(unlabeled_data_path):
        raise FileNotFoundError(f"Unlabeled data file '{unlabeled_data_path}' not found.")

    unlabeled_data = pd.read_csv(unlabeled_data_path)

    iteration = 0
    while iteration < 3:  # Example: Perform 3 iterations
        print(f"=== Iteration {iteration + 1} ===")
        
        # Use a subset of the labeled data for faster training
        if len(labeled_data) > 1000:
            labeled_data = labeled_data.sample(n=1000, random_state=42)  # Sample 1,000 examples

        # Train model
        print("Training model...")
        trainer, tokenizer, model = train_model(labeled_data)
        
        # Classify unlabeled data
        print("Classifying unlabeled documents...")
        classified_data = classify_documents(unlabeled_data, trainer, tokenizer)
        
        # Simulate human refinement
        print("Refining labels...")
        refined_data = refine_labels(classified_data)
        
        # Add refined data to training set
        labeled_data = pd.concat([labeled_data, refined_data])
        
        iteration += 1

    # Save the trained model and tokenizer after the final iteration
    output_dir = "./final_model"
    print(f"Saving the final model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model

# Run the iterative training process
if __name__ == "__main__":
    final_model = iterative_training()
