import os
import numpy as np

def apply_exact_sparsity(tensor, sparsity=0.2, random_state=None):
    """
    Replaces exactly 'sparsity' fraction of elements in 'tensor' with NaN.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Flatten for easier indexing
    flat_tensor = tensor.ravel()
    N = flat_tensor.size

    # Number of elements to replace with NaN
    num_to_nan = int(round(sparsity * N))

    # Randomly choose 'num_to_nan' unique positions
    all_indices = np.arange(N)
    chosen_indices = np.random.choice(all_indices, size=num_to_nan, replace=False)
    
    # Copy so you don't modify the original
    flat_copy = flat_tensor.copy()
    flat_copy[chosen_indices] = np.nan

    # Reshape back to original shape
    return flat_copy.reshape(tensor.shape)


if __name__ == '__main__':
    ''' -------------------------------
    1) Using_Scale_Factor_2019-2020_dense
    2) Modeling_Two-Step_Expressions_2019-2020_v2_dense 
    3) Analyzing_Models_of_Two-Step_Linear_Relationships_2019-2020_dense
    ------------------------------'''
    
    base_filename = "Analyzing_Models_of_Two-Step_Linear_Relationships_2019-2020_dense"
    input_data_dir = os.path.join(os.getcwd(), "dataset", "dense_dataset", "MATHia")
    output_data_dir = os.path.join(os.getcwd(), "dataset", "sparse_dataset_sim", "MATHia", )
    os.makedirs(output_data_dir, exist_ok=True)  # Create the output dir if not exists 
    
    # Paths
    file_path = os.path.join(input_data_dir, base_filename + ".npy")
    print(f"Input file path: {file_path}")
    
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    print("Data loaded. Shape:", data.shape)
    
    # List of sparsity levels to apply
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Simulate these different sparsities
    for sp in sparsity_levels:
        sparse_data = apply_exact_sparsity(data, sparsity=sp, random_state=42)
        
        # ---- NEW: Verify the actual sparsity ----
        total_count = sparse_data.size
        nan_count   = np.isnan(sparse_data).sum()
        actual_sparsity = nan_count / total_count

        print(f"\nRequested Sparsity={sp:.2f}")
        print(f"NaN count={nan_count}, total elements={total_count}")
        print(f"Actual Sparsity={actual_sparsity:.4f}")
        
        # Construct output file name
        out_file_name = f"{base_filename}_sparsity_{int(sp*100)}.npy"
        out_file_path = os.path.join(output_data_dir, out_file_name)

        # Save the sparse data
        np.save(out_file_path, sparse_data)
        print(f"Saved sparse data to: {out_file_path}")