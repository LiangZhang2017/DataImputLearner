import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    print("Preprocess")
    
    # Adjust file_path as necessary
    base_filename = "scale_drawings_3"
    data_dir = os.path.join(os.getcwd(), "dataset", "source_dataset", "MATHia")
    file_path = os.path.join(data_dir, base_filename + ".txt")

    print("File path is:", file_path)

    # Load data
    data_source = pd.read_csv(file_path, sep='\t')
    print("Data columns are:", data_source.columns)

    # Select specific columns
    columns_list = ["Anon Student Id", "Step Name", "Attempt At Step", "Outcome"]
    filter_data = data_source[columns_list].dropna(subset=['Attempt At Step', 'Anon Student Id', 'Step Name'])

    # Ensure 'Attempt At Step' is integer
    filter_data['Attempt At Step'] = filter_data['Attempt At Step'].astype(int)

    # Map Outcome to binary 'Answer_Score'
    filter_data['Answer_Score'] = filter_data["Outcome"].map(lambda x: 1 if x == 'OK' else 0)

    # Encode 'Anon Student Id' and 'Step Name' to numeric IDs (initially)
    filter_data['Student_Id'] = filter_data['Anon Student Id'].astype('category').cat.codes
    filter_data['Question_Id'] = filter_data['Step Name'].astype('category').cat.codes

    # Rename 'Attempt At Step' to 'Question_Attempt'
    filter_data['Question_Attempt'] = filter_data['Attempt At Step']

    # Required attempts
    required_attempts = {1, 2, 3, 4}

    # Filter to only the required attempts
    filter_data = filter_data[filter_data['Question_Attempt'].isin(required_attempts)]

    # Identify valid (Student_Id, Question_Id) pairs that have exactly these four attempts
    pairs = (filter_data
             .groupby(['Student_Id', 'Question_Id'])['Question_Attempt']
             .apply(lambda attempts: set(attempts)))
    valid_pairs = pairs[pairs == required_attempts].index  # Only pairs with {1,2,3,4}

    # Keep only valid pairs
    complete_data = filter_data.set_index(['Student_Id','Question_Id']).loc[valid_pairs].reset_index()

    # Recode Student_Id
    complete_data['Student_Id'] = complete_data['Student_Id'].astype('category').cat.codes

    # Re-derive Question_Id based on Step Name categories after filtering
    complete_data['Step Name'] = complete_data['Step Name'].astype('category')
    complete_data['Question_Id'] = complete_data['Step Name'].cat.codes

    # Extract the category mapping from codes to original step names
    question_categories = complete_data['Step Name'].cat.categories
    question_id_to_name = dict(enumerate(question_categories))

    # Build tensor
    max_student = complete_data['Student_Id'].max() + 1
    max_question = complete_data['Question_Id'].max() + 1
    max_attempt = len(required_attempts)  # 4

    tensor = np.full((max_student, max_question, max_attempt), np.nan)

    # Fill the tensor
    for _, row in complete_data.iterrows():
        student_id = int(row['Student_Id'])
        question_id = int(row['Question_Id'])
        attempt = int(row['Question_Attempt']) - 1  # zero-based index
        tensor[student_id, question_id, attempt] = row['Answer_Score']

    # Check if there's any NaN
    nan_elements = np.isnan(tensor).sum()
    total_elements = tensor.size
    sparsity_level = nan_elements / total_elements
    print(f"Tensor Shape: ({max_student}, {max_question}, {max_attempt})")
    print(f"Sparsity Level: {sparsity_level:.4f}")

    # Post-filter if needed (remove incomplete pairs)
    if np.isnan(tensor).any():
        complete_mask = ~np.isnan(tensor).any(axis=2)
        valid_students, valid_questions = np.where(complete_mask)

        unique_students = np.unique(valid_students)
        unique_questions = np.unique(valid_questions)
        student_map = {s: i for i, s in enumerate(unique_students)}
        question_map = {q: j for j, q in enumerate(unique_questions)}

        new_tensor = np.empty((len(unique_students), len(unique_questions), max_attempt))
        for s, q in zip(valid_students, valid_questions):
            new_s = student_map[s]
            new_q = question_map[q]
            new_tensor[new_s, new_q, :] = tensor[s, q, :]

        tensor = new_tensor
        nan_elements = np.isnan(tensor).sum()
        total_elements = tensor.size
        sparsity_level = nan_elements / total_elements
        print("After post-filtering:")
        print(f"New Tensor Shape: {tensor.shape}")
        print(f"Sparsity Level: {sparsity_level:.4f}")

        # Update question_id_to_name mapping for filtered questions
        # We must map only the questions that survived filtering
        final_question_id_to_name = {}
        for old_id, new_id in question_map.items():
            # old_id corresponds to the original coded question index before filtering
            if old_id in question_id_to_name:
                final_question_id_to_name[new_id] = question_id_to_name[old_id]
        question_id_to_name = final_question_id_to_name

    # Print a slice for inspection
    if tensor.shape[0] > 0:
        print("Tensor slice for Student_Id 0:")
        print(tensor[0, :, :])

    output_data_dir = os.path.join(os.getcwd(), "dataset", "dense_dataset", "MATHia")
    
    # Save the tensor as a .npy file
    tensor_filename = os.path.join(output_data_dir, base_filename + "_tensor.npy")
    np.save(tensor_filename, tensor)
    print(f"Tensor saved to {tensor_filename}")

    # Save the question_id_name mapping to a CSV
    mapping_filename = os.path.join(output_data_dir, base_filename + "_question_id_mapping.csv")
    mapping_df = pd.DataFrame(list(question_id_to_name.items()), columns=['Question_Id', 'Question_Name'])
    mapping_df.to_csv(mapping_filename, index=False)
    print(f"Question ID mapping saved to {mapping_filename}")