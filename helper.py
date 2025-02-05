import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def Partition(tensor,slice_size,mode,filter):
    """
    Partition a tensor into sub-tensors along the first dimension.

    :param tensor: The input tensor to be partitioned.
    :param slice_size: The size of the first dimension of each sub-tensor.
    :param mode: If "Average", partitions the tensor into sub-tensors of equal size.
    :param filter: If "normal", the rest dimension after partition will be filtered.
    :return: A list of sub-tensors.
    """

    sub_tensors = []
    sub_tensor = []
    sub_tensor_missing_indices = []

    if mode == "Average" and filter == "normal":
        total_elements = tensor.shape[0]
        print("tensor.shape[0] is {}".format(tensor.shape[0]))
        num_slices = total_elements // slice_size
        for i in range(num_slices):
            start_idx = i * slice_size
            sub_tensor = tensor[start_idx:(start_idx + slice_size)]
            # sub_tensor, tensor_missing_indices = svd_impute(sub_tensor)
            sub_tensors.append(sub_tensor)
            # sub_tensor_missing_indices.append(tensor_missing_indices)

    return sub_tensors, tf.size(sub_tensor)


def k_fold_split(sub_tensors, n_splits=5):
    """
    Split sub-tensors into training and testing sets using k-fold cross-validation
    and also yield the indices of the train and test sets.

    :param sub_tensors: List of sub-tensors to be split.
    :param n_splits: Number of folds (default is 5).
    :return: A generator that yields tuples containing training-test pairs of sub-tensor lists
             and their corresponding indices for each fold.
    """

    kf = KFold(n_splits=n_splits)
    for train_indices, test_indices in kf.split(sub_tensors):
        train_set = [sub_tensors[i] for i in train_indices]
        test_set = [sub_tensors[i] for i in test_indices]
        yield (train_set, test_set), (train_indices, test_indices)


def generate_hints(mask, hint_rate):
    hint_mask = np.random.binomial(1, hint_rate, size=mask.shape)
    return mask * hint_mask


def reshape_for_pyBKT(dense_tensor):
    """
    Reshape a 3D NumPy array of shape (num_students, num_questions, num_attempts)
    into a Pandas DataFrame suitable for pyBKT.

    dense_tensor: np.ndarray
        - A NumPy array of shape (S, Q, A)
          S = number of students
          Q = number of questions (skills)
          A = number of attempts per question
        - Each entry is typically 0 or 1 indicating correctness,
          but can also be other numeric values.

    Returns: pd.DataFrame
        - Columns: 'user_id', 'skill_name', 'attempt_ix', 'correct'
    """
    if len(dense_tensor.shape) != 3:
        raise ValueError(
            f"dense_tensor must have shape (students, questions, attempts). "
            f"Got shape={dense_tensor.shape}."
        )

    S, Q, A = dense_tensor.shape  # unpack dimensions
    records = []

    # Loop over students, questions, attempts
    for student_id in range(S):
        for question_id in range(Q):
            for attempt_id in range(A):
                correctness = dense_tensor[student_id, question_id, attempt_id]

                # Skip NaN values
                if np.isnan(correctness):
                    continue  

                # Ensure 0/1 integer if needed
                correctness = int(correctness)

                records.append({
                    "user_id": "S"+str(student_id),
                    "skill_name": "Q_"+str(question_id)+"_S_"+str(student_id),    # treat question as skill
                    "attempt_ix": attempt_id,
                    "correct": correctness,
                    "template_id": "S"+str(student_id)
                })

    # Create a DataFrame
    df = pd.DataFrame(records)
    
    return df