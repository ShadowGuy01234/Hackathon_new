import pandas as pd
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

# Step 2: Train Initial Model
def train_model(train_data, model_name="distilbert-base-uncased", num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Tokenize Data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    dataset = Dataset.from_pandas(train_data)
    tokenized_data = dataset.map(tokenize_function, batched=True)
    train_data, val_data = tokenized_data.train_test_split(test_size=0.2).values()
    
    # Training Arguments
    training_args = TrainingArguments(
    output_dir="/home/shadow-guy/Hackathon_New/results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,  # Try slightly higher or lower
    per_device_train_batch_size=4,  # Adjust for small dataset
    num_train_epochs=5,  # Increase epochs
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,  # Load best model automatically
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
    tokenized_data = tokenizer(unlabeled_data["text"].tolist(), padding=True, truncation=True, return_tensors="pt")
    outputs = trainer.model(**tokenized_data)
    predictions = torch.argmax(outputs.logits, dim=1)
    
    # Map predictions to category names
    label_mapping = {0: "AI", 1: "Finance"}
    unlabeled_data["predicted_label"] = predictions.tolist()
    unlabeled_data["predicted_category"] = unlabeled_data["predicted_label"].map(label_mapping)
    return unlabeled_data


# Step 4: Human Refinement (Simulated for this code)
def refine_labels(predictions):
    # Simulate manual refinement (e.g., update some labels based on human input)
    # This can be replaced with a user interface for actual human input.
    refined_data = predictions.copy()
    refined_data.loc[0, "predicted_label"] = 0  # Example refinement
    refined_data["label"] = refined_data["predicted_label"]  # Treat as labeled for next iteration
    return refined_data

# Step 5: Iterative Retraining
def iterative_training():
    # Load datasets from files
    labeled_data = pd.read_csv("initial_labeled_data.csv")
    unlabeled_data = pd.read_csv("unlabeled_data.csv")

    iteration = 0
    while iteration < 3:  # Example: Perform 3 iterations
        print(f"=== Iteration {iteration + 1} ===")
        
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
    output_dir = "/home/shadow-guy/Hackathon_New/final_model"  # Specify the directory to save the model
    print(f"Saving the final model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)  # Save the model, configuration, and weights
    tokenizer.save_pretrained(output_dir)  # Save the tokenizer files

    return model


# Run the iterative training process
if __name__ == "__main__":
    final_model = iterative_training()

    
