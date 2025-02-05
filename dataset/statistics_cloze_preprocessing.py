
import os
import pandas as pd

def preprocess_statistic_data(base_filename, input_data_dir):
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
        "Attempt At Step", 
        "Outcome", 
        "Time"
    ]
    
    # Filter out rows with missing key columns
    try:
        filter_data = data_source[columns_list].dropna(subset=[
            'Anon Student Id', 
            'Step Name'
        ])
    except KeyError as e:
        print(f"Missing required columns: {e}")
        return None, None, None

    # Sort by "Anon Student Id", "Step Name", and "Time"
    filter_data = filter_data.sort_values(by=["Anon Student Id", "Step Name", "Time"])
    # Assign attempt numbers
    filter_data['Attempts'] = filter_data.groupby(["Anon Student Id", "Step Name"]).cumcount()
    
    print("filter_data is ", filter_data)
    
    outcome_map = {
        "CORRECT": 1,
        "INCORRECT": 0
        # You can add more outcome types here if needed
    }
    
    filter_data['Answer_Score'] = filter_data['Outcome'].map(lambda x: outcome_map.get(x.upper(), 0))
    
    print("filter_data['Attempts']", filter_data['Attempts'].unique())
    
    

if __name__ == '__main__':
    # -------------------------------
    # Step 0: Setup
    # All base files: ds5513_tx_All_Data_7784_2023_0414_032745 
    # -------------------------------
    base_filename = "ds5513_tx_All_Data_7784_2023_0414_032745"
    input_data_dir = os.path.join(os.getcwd(), "dataset", "source_dataset", "Statistics_Cloze")
    output_data_dir = os.path.join(os.getcwd(), "dataset", "dense_dataset", "Statistics_Cloze")
    os.makedirs(output_data_dir, exist_ok=True)  # Create the output dir if not exists
    
    # Define valid attempts to include
    valid_attempts = {1, 2, 3, 4}
    
     # -------------------------------
    # Step 1: Load original data
    # -------------------------------
    # data, student_mapping, step_mapping = 
    preprocess_statistic_data(base_filename, input_data_dir)
    # if data is None:
    #     exit("Data loading failed.")