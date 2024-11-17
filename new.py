import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def classify_documents_from_file(file_path, model_dir="final_model"):
    """
    Classify documents from a single file.

    Args:
        file_path (str): Full path to the document file. Supported formats: .txt, .csv, .json.
        model_dir (str): Path to the directory containing the saved model and tokenizer.

    Returns:
        pd.DataFrame: DataFrame containing the text and predicted labels.
    """
    # Load the saved model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Read the file based on its format
    if file_path.endswith(".txt"):
        with open(file_path, "r") as file:
            documents = file.readlines()
        documents = [doc.strip() for doc in documents if doc.strip()]  # Remove empty lines
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        if "text" not in df.columns:
            raise ValueError("CSV file must contain a 'text' column.")
        documents = df["text"].tolist()
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
        if "text" not in df.columns:
            raise ValueError("JSON file must contain a 'text' column.")
        documents = df["text"].tolist()
    else:
        raise ValueError("Unsupported file format. Use .txt, .csv, or .json.")

    # Tokenize the input documents
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).tolist()

    # Prepare the results as a DataFrame
    results_df = pd.DataFrame({"text": documents, "predicted_label": predictions})
    return results_df

def classify_multiple_files(directory="Hackathon_new/test", model_dir="final_model"):
    """
    Classify documents from multiple files in a directory.

    Args:
        directory (str): Directory containing the document files.
        model_dir (str): Path to the directory containing the saved model and tokenizer.

    Returns:
        dict: A dictionary with file names as keys and classified DataFrames as values.
    """
    results = {}
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith((".txt", ".csv", ".json")):
            print(f"Processing file: {file_name}")
            try:
                results[file_name] = classify_documents_from_file(file_path, model_dir)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    return results

if __name__ == "__main__":
    # Base directory where the test files are located
    base_dir = "./test"

    # Path to the directory where the trained model is stored
    model_dir = "final_model"

    # Classify documents from multiple files and print the results
    print("Classifying documents from multiple files...")
    try:
        results = classify_multiple_files(base_dir, model_dir)
        for file_name, result_df in results.items():
            print(f"\nResults for {file_name}:\n")
            print(result_df)
    except Exception as e:
        print(f"Error: {e}")
