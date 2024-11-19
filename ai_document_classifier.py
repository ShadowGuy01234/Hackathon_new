import os
from data_loader import load_data, load_multiple_files
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


def classify_documents(documents, model_dir="final_model"):
    """
    Classify a list of documents.

    Args:
        documents (list): A list of strings to classify.
        model_dir (str): Path to the saved model directory.

    Returns:
        pd.DataFrame: DataFrame with the original text and predictions.
    """
    # Load the saved model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Tokenize the input documents
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt", max_length=128)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).tolist()

    # Map predictions to categories
    label_mapping = {0: "AI", 1: "Finance"}
    categories = [label_mapping[pred] for pred in predictions]

    # Prepare the results as a DataFrame
    results_df = pd.DataFrame({"text": documents, "predicted_label": predictions, "predicted_category": categories})
    return results_df


def classify_files(directory, model_dir="final_model"):
    """
    Classify all supported files in a directory.

    Args:
        directory (str): Path to the directory containing files.
        model_dir (str): Path to the saved model directory.

    Returns:
        dict: Dictionary of classified DataFrames keyed by file name.
    """
    print(f"Loading files from directory: {directory}")
    all_data = load_multiple_files(directory)
    results = {}

    for file_name, data in all_data.items():
        print(f"\nClassifying data from {file_name}...")
        try:
            documents = data["text"].tolist()  # Extract text column
            results[file_name] = classify_documents(documents, model_dir)
        except Exception as e:
            print(f"Error classifying {file_name}: {e}")

    return results


def plot_classification_results(results):
    """
    Plot a bar graph of classification results.

    Args:
        results (pd.DataFrame): Combined DataFrame with 'predicted_category' column.
    """
    # Count categories
    category_counts = results["predicted_category"].value_counts()

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
    plt.xlabel("Categories", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Classification Results", fontsize=15)
    plt.xticks(fontsize=10)
    plt.show()


if __name__ == "__main__":
    # Path to the directory containing test files
    test_dir = "/home/shadow-guy/Hackathon_New/Hackathon_new/test"  # Replace with your directory path
    model_dir = "/home/shadow-guy/Hackathon_New/Hackathon_new/final_model"  # Replace with your model directory path

    print("Classifying files from the test directory...")
    results = classify_files(test_dir, model_dir)

    # Combine all results for visualization
    combined_results = pd.concat(results.values(), ignore_index=True)

    # Display combined results
    print("\nCombined classification results:")
    print(combined_results)

    # Plot classification results
    plot_classification_results(combined_results)
