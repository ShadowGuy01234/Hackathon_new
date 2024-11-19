import os
import pandas as pd

def load_data(file_path):
    """
    Load data from a single file (.csv, .json, or .txt).

    Args:
        file_path (str): Path to the data file.

    Returns:
        pd.DataFrame: A DataFrame with a 'text' column.
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        if "text" not in df.columns:
            raise ValueError("CSV file must contain a 'text' column.")
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
        if "text" not in df.columns:
            raise ValueError("JSON file must contain a 'text' column.")
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            lines = f.readlines()
        df = pd.DataFrame({"text": [line.strip() for line in lines if line.strip()]})
    else:
        raise ValueError("Unsupported file format. Please use .csv, .json, or .txt files.")
    
    return df


def load_multiple_files(directory):
    """
    Load data from multiple files in a directory.

    Args:
        directory (str): Path to the directory containing the files.

    Returns:
        dict: A dictionary where keys are file names and values are DataFrames.
    """
    all_data = {}
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith((".csv", ".json", ".txt")):
            print(f"Loading data from {file_name}...")
            try:
                all_data[file_name] = load_data(file_path)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    return all_data
