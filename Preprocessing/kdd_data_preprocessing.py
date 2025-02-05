import os
import pandas as pd
import numpy as np
import json


def preprocess_kdd_data(base_filename, input_data_dir):
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
        return None, None, None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None

    # Columns to use
    columns_list = [
        "Anon Student Id", 
        "Problem Name", 
        "Step Name",  
        "Step Duration (sec)",
        "Correct Step Duration (sec)", 
        "Step Start Time"
    ]
    
    # Filter out rows with missing key columns
    try:
        filter_data = data_source[columns_list].dropna(subset=[
            'Step Duration (sec)', 
            'Anon Student Id', 
            'Problem Name'
        ])
    except KeyError as e:
        print(f"Missing required columns: {e}")
        return None, None, None
    
    # Sort by "Anon Student Id", "Step Name", and "Time"
    filter_data = filter_data.sort_values(by=["Anon Student Id", "Step Name", "Step Start Time"])
    
    print("filter_data Correct Step Duration (sec) is ", filter_data["Correct Step Duration (sec)"]) 
    
    # Assign attempt numbers
    filter_data['Attempts'] = filter_data.groupby(["Anon Student Id", "Step Name"]).cumcount()
    filter_data['Answer_Score'] = filter_data["Correct Step Duration (sec)"].map(lambda x: 1 if pd.notnull(x) else 0)

    print("filter_data is ", filter_data)
        # Build the mappings
    student_cat = filter_data['Anon Student Id'].astype('category')
    student_categories = student_cat.cat.categories
    student_mapping = {student_id: code for code, student_id in enumerate(student_categories)}
    
    step_cat = filter_data['Step Name'].astype('category')
    step_categories = step_cat.cat.categories
    step_mapping = {step_name: code for code, step_name in enumerate(step_categories)}
    
    # Apply numeric codes
    filter_data['Student_Id'] = filter_data['Anon Student Id'].map(student_mapping)
    filter_data['Question_Id'] = filter_data['Step Name'].map(step_mapping)
    
    print("filter_data['Student_Id'] is ", filter_data['Student_Id'].nunique())
    print("filter_data['Question_Id'] is ", filter_data['Question_Id'].nunique())

    return filter_data, student_mapping, step_mapping

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

    # Initialize dense tensor with NaNs
    tensor = np.full((max_student, max_question, max_attempt), np.nan)

    # Populate the tensor
    for _, row in data.iterrows():
        s = int(row['Student_Id'])
        q = int(row['Question_Id'])
        a = int(row['Attempts']) - 1  # 0-based index
        tensor[s, q, a] = row['Answer_Score']

    print(f"Tensor shape: {tensor.shape}")
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
    ''' -------------------------------
    # Step 0: Setup
    # All base files: 
    # algebra_2005_2006_train
    # -------------------------------
    '''
    
    # Lessons: 1) Using_Scale_Factor_2019-2020, 2) Modeling_Two-Step_Expressions_2019-2020_v2, 3) Analyzing_Models_of_Two-Step_Linear_Relationships_2019-2020
    base_filename = "bridge_to_algebra_2008_2009_train"
    input_data_dir = os.path.join(os.getcwd(), "dataset", "source_dataset", "KDD")
    output_data_dir = os.path.join(os.getcwd(), "dataset", "dense_dataset", "KDD")
    os.makedirs(output_data_dir, exist_ok=True)  # Create the output dir if not exists

    # Define valid attempts to include
    valid_attempts = {1, 2, 3}

    # -------------------------------
    # Step 1: Load original data
    # -------------------------------
    data, student_mapping, step_mapping = preprocess_kdd_data(base_filename, input_data_dir)
    if data is None:
        exit("Data loading failed.")
    
        # -------------------------------
    # Step 2: Build sparse tensor
    # -------------------------------
    sparse_tensor = build_tensor_from_data(data, valid_attempts)
    print(sparse_tensor[1, :, :])

    # -------------------------------
    # Step 3: Calculate sparsity
    # -------------------------------
    sparsity = calculate_sparsity(sparse_tensor)
    print(f"Sparsity of the sparse tensor with attempts {valid_attempts}: {sparsity:.2%}")
    
    # -------------------------------
    # Step 4: Check null value dims
    # -------------------------------
    null_count = np.isnan(sparse_tensor).sum()
    print(f"Total null values in the tensor: {null_count}")
    
    null_positions = np.argwhere(np.isnan(sparse_tensor))
    print(f"Total null positions: {null_positions}")
    
    students_with_null = np.unique(null_positions[:, 0])
    print("Total students with null value:", len(students_with_null))
    print("Unique student indices (numeric) with null values:", students_with_null)
    
    questions_with_null = np.unique(null_positions[:, 1])
    print("Total questions with null value:", len(questions_with_null))
    print("Unique question indices (numeric) with null values:", questions_with_null)
    
    # -------------------------------
    # Step 5: Remove null students -> Dense tensor
    # -------------------------------
    all_students = np.arange(sparse_tensor.shape[0])
    students_without_null = np.setdiff1d(all_students, students_with_null)

    dense_tensor = sparse_tensor[students_without_null, :, :]
    print(f"Original sparse_tensor shape: {sparse_tensor.shape}")
    print(f"Dense tensor shape (after removing null-students): {dense_tensor.shape}")

    dense_null_count = np.isnan(dense_tensor).sum()
    print(f"Total null values in dense_tensor: {dense_null_count}")

    # -------------------------------
    # Step 6: Save the dense tensor
    # -------------------------------
    dense_filename = base_filename + "_dense.npy"
    dense_file_path = os.path.join(output_data_dir, dense_filename)
    np.save(dense_file_path, dense_tensor)
    print(f"Dense tensor saved to {dense_file_path}")

    # -------------------------------
    # Step 7: Build a "new" student mapping (for the dense tensor)
    # -------------------------------
    # 7.1) Reverse the original mapping (number -> original Student ID)
    reverse_student_mapping = {v: k for k, v in student_mapping.items()}

    # 7.2) Create a new mapping from the old Student IDs that survived
    #      to a new 0..N-1 range
    new_student_mapping = {}
    for new_code, old_code in enumerate(students_without_null):
        old_student_id_str = reverse_student_mapping[old_code]
        new_student_mapping[old_student_id_str] = new_code

    # We didn't remove any questions, so we keep the same step_mapping as is.
    # If you also removed certain questions for a fully-dense question dimension,
    # you'd need to create a new_step_mapping as well, similar to the above.

    # -------------------------------
    # Step 8: Save the new mappings
    # -------------------------------
    # 8.1) Save the new student mapping
    new_student_mapping_path = os.path.join(
        output_data_dir, base_filename + "_dense_student_mapping.json"
    )
    with open(new_student_mapping_path, "w") as f:
        json.dump(new_student_mapping, f, indent=2)

    print(f"New student mapping saved to {new_student_mapping_path}")

    # 8.2) Save the original step mapping (unchanged)
    new_step_mapping_path = os.path.join(
        output_data_dir, base_filename + "_dense_step_mapping.json"
    )
    with open(new_step_mapping_path, "w") as f:
        json.dump(step_mapping, f, indent=2)

    print(f"Step mapping saved to {new_step_mapping_path}")