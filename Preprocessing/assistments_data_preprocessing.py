import os
import pandas as pd
import numpy as np

def preprocess_assistments_data(base_filename, input_data_dir):
    """
    Load and encode data from the original dataset.
    """
    print("Loading data...")

    # Paths
    file_path = os.path.join(input_data_dir, base_filename + ".txt")
    print(f"Input file path: {file_path}")

    # Load data
    try:
        data_source = pd.read_csv(file_path, sep='\t')
        print(f"Loaded data with columns: {data_source.columns}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Columns to use
    columns_list = ["Anon Student Id", "Problem Name", "Step Name", "Outcome", "Time"]
    
    try:
        filter_data = data_source[columns_list].dropna(subset=["Problem Name", "Step Name", "Outcome"])
    except KeyError as e:
        print(f"Missing required columns: {e}")
        return None
    
    # Convert Time to a numeric type (if not already)
    filter_data['Time'] = pd.to_numeric(filter_data['Time'], errors='coerce')
    
    # Sort by Student, KC, and Time
    filter_data = filter_data.sort_values(by=["Anon Student Id", "Step Name", "Time"])
    
    # Assign attempt numbers starting from 0 for each Student-KC pair
    filter_data['Attempts'] = filter_data.groupby(["Anon Student Id", "Step Name"]).cumcount()
    
    filter_data['Answer_Score'] = filter_data["Outcome"].map(lambda x: 1 if x == 'CORRECT' else 0)

    # Encode Student and Question
    filter_data['Student_Id'] = filter_data['Anon Student Id'].astype('category').cat.codes
    filter_data['Question_Id'] = filter_data['Step Name'].astype('category').cat.codes # Step Name, KC (Unique-step)

    return filter_data


def build_tensor_from_data(data, valid_attempts):
    """
    Build a tensor from the original dataset, considering only valid attempts.
    """
    # Filter data for valid attempts
    data = data[data['Attempts'].isin(valid_attempts)]

    # Determine dimensions
    max_student = data['Student_Id'].max() + 1
    max_question = data['Question_Id'].max() + 1
    max_attempt = max(valid_attempts)

    # Initialize dense tensor
    tensor = np.full((max_student, max_question, max_attempt), np.nan)

    # Populate the tensor
    for _, row in data.iterrows():
        s = int(row['Student_Id'])
        q = int(row['Question_Id'])
        a = int(row['Attempts']) - 1  # Adjust to 0-based indexing
        tensor[s, q, a] = row['Answer_Score']

    return tensor


def calculate_sparsity(tensor):
    """
    Calculate the sparsity of the tensor.
    """
    total_elements = tensor.size
    non_nan_elements = np.count_nonzero(~np.isnan(tensor))
    sparsity = 1 - (non_nan_elements / total_elements)
    return sparsity


if __name__ == '__main__':
    base_filename = "Assistments_Math_2007-2008_Symb-DFA_(69_Students)"
    input_data_dir = os.path.join(os.getcwd(), "dataset", "source_dataset", "ASSISTMENTS")

    # Define valid attempts to include
    valid_attempts = {1,2}

    # Step 1: Load original data
    data = preprocess_assistments_data(base_filename, input_data_dir)
    if data is None:
        exit("Data loading failed.")

    # Step 2: Build tensor considering valid attempts
    dense_tensor = build_tensor_from_data(data, valid_attempts)

    # Step 3: Calculate sparsity of the tensor
    sparsity = calculate_sparsity(dense_tensor)
    print(f"Sparsity of the dense tensor with attempts {valid_attempts}: {sparsity:.2%}")
    
    