import numpy as np
import tensorly as tl
import os
import csv
import time
import tensorflow as tf
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from scipy.special import expit
from tensorly.cp_tensor import cp_to_tensor

from helper import save_summary
from TC.TC_helper import tensor_to_numpy, perform_k_fold_cross_validation_tf, save_indices, save_metrics, save_tensor, get_missing_indices

# Helper function to save hyperparameters and elapsed training time for each fold.
def save_model_parameters(params, elapsed_time, fold_save_path):
    """
    Save the model parameters and elapsed training time to a CSV file in the given fold directory.
    The elapsed_time is in seconds.
    """
    csv_path = os.path.join(fold_save_path, "model_parameters.csv")
    params['elapsed_time'] = elapsed_time
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = list(params.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(params)
    print(f"Saved model parameters and elapsed time to {csv_path}")

class CPDecomposition:
    def __init__(self, sparse_tensor, dense_tensor, parameters, rank_similarity_tensors, args, sparsity_level):
        print("CP Decomposition")
        np.random.seed(22)
        
        # Basic model info (from args and parameters)
        self.Lesson_Id = args.Lesson_Id[0]
        self.Imputation_model = args.Imputation_model[0]
        self.sparsity_level = sparsity_level
        self.rank_similarity_tensors = rank_similarity_tensors
        
        self.sparse_tensor = sparse_tensor
        self.dense_tensor = dense_tensor
        self.parameters = parameters
        self.num_features = 6
        
        # Using constant hyperparameters as provided.
        self.lambda_t = 0.000001
        self.lambda_q = 0.001
        self.lr = 0.000008 # 0.000001
        self.lambda_w = 0.1
        
        # Factor matrices, biases, and tensor reconstruction
        self.U = None
        self.V = None
        self.W = None
        self.T = None
        
        self.use_bias = True
        self.binarized_question = True
        self.is_rank = True
        
        self.bias_s = None
        self.bias_t = None
        self.bias_q = None
        self.global_bias = None
        
        self.global_max_iter = 200
        self.Version = 0
        
        self.train_tensor = None
        self.test_tensor = None
        self.train_tensor_np = None
        self.test_tensor_np = None
        self.missing_indices = None
        
        # Setup directory to save models and metrics (similar to Standard_TC)
        self.save_path_global_training = os.path.join(os.getcwd(), 'results', 'global_training_models', 
                                                      str(self.Imputation_model), str(self.Lesson_Id), str(self.sparsity_level))
        os.makedirs(self.save_path_global_training, exist_ok=True)

    # -------------------------------
    # CP Decomposition Prediction and Gradients
    # -------------------------------
    def get_question_prediction(self, learner, attempt, question):
        # Prediction: element-wise product of factor vectors plus biases.
        pred = np.sum(self.U[learner, :] * self.V[attempt, :] * self.W[question, :])
        if self.use_bias:
            pred += self.bias_s[learner] + self.bias_q[question] + self.bias_t[attempt] + self.global_bias
        if self.binarized_question:
            pred = expit(pred)
        return pred

    def grad_U(self, learner, attempt, question, obs=None):
        grad = np.zeros_like(self.U[learner, :])
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad = -2.0 * (obs - pred) * self.V[attempt, :] * self.W[question, :] + 2.0 * self.lambda_t * self.U[learner, :]
        return grad

    def grad_V(self, learner, attempt, question, obs=None):
        grad = np.zeros_like(self.V[attempt, :])
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad = -2.0 * (obs - pred) * self.U[learner, :] * self.W[question, :] + 2.0 * self.lambda_t * self.V[attempt, :]
        return grad

    def grad_W(self, learner, attempt, question, obs=None):
        grad = np.zeros_like(self.W[question, :])
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad = -2.0 * (obs - pred) * self.U[learner, :] * self.V[attempt, :] + 2.0 * self.lambda_q * self.W[question, :]
        return grad

    def grad_bias_q(self, learner, attempt, question, obs=None):
        grad = 0.0
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad -= 2.0 * (obs - pred)
            if self.binarized_question:
                grad *= pred * (1.0 - pred)
        return grad

    def grad_bias_s(self, learner, attempt, question, obs=None):
        grad = 0.0
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad -= 2.0 * (obs - pred)
            if self.binarized_question:
                grad *= pred * (1.0 - pred)
        return grad

    def grad_bias_t(self, learner, attempt, question, obs=None):
        grad = 0.0
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            grad -= 2.0 * (obs - pred)
            if self.binarized_question:
                grad *= pred * (1.0 - pred)
        return grad

    def grad_global_bias(self, learner, attempt, question, obs=None):
        grad = 0.0
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad -= 2.0 * (obs - pred) * pred * (1.0 - pred)
            else:
                grad -= 2.0 * (obs - pred)
        return grad

    def optimize_sgd(self, learner, attempt, question, obs=None):
        self.U[learner, :] -= self.lr * self.grad_U(learner, attempt, question, obs)
        self.V[attempt, :] -= self.lr * self.grad_V(learner, attempt, question, obs)
        self.W[question, :] -= self.lr * self.grad_W(learner, attempt, question, obs)
        self.bias_q[question] -= self.lr * self.grad_bias_q(learner, attempt, question, obs)
        self.bias_s[learner] -= self.lr * self.grad_bias_s(learner, attempt, question, obs)
        self.bias_t[attempt] -= self.lr * self.grad_bias_t(learner, attempt, question, obs)
        self.global_bias -= self.lr * self.grad_global_bias(learner, attempt, question, obs)

    # -------------------------------
    # Loss Computation
    # -------------------------------
    def get_loss(self):
        loss = 0.0
        train_obs = []
        train_pred = []
        for (learner, question, attempt, obs) in self.train_tensor_np:
            learner = int(learner)
            question = int(question)
            attempt = int(attempt)
            pred = self.get_question_prediction(learner, attempt, question)
            if pred is not None:
                train_obs.append(obs)
                train_pred.append(pred)
        # Filter out NaN observations.
        train_obs, train_pred = zip(*[(obs, pred) for obs, pred in zip(train_obs, train_pred) if not np.isnan(obs)])
        q_mae = mean_absolute_error(train_obs, train_pred)
        q_auc = roc_auc_score(train_obs, train_pred)
        q_rmse = sqrt(mean_squared_error(train_obs, train_pred))
        mse = mean_squared_error(train_obs, train_pred)
        q_rse = mse / np.mean(np.square(train_obs))
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        cross_entropy = bce_loss(train_obs, train_pred)
        square_loss_q = q_rmse

        # Reconstruct tensor from CP factors.
        factors = [self.U, self.V, self.W]
        weights = np.ones(self.U.shape[1])
        cp_tensor = (weights, factors)
        pred_tensor = cp_to_tensor(cp_tensor)
        sum_n = 0.0
        if self.is_rank:
            for attempt in range(self.num_attempt):
                if attempt > 0:
                    for n in range(attempt - 1, attempt):
                        slice_n = np.subtract(pred_tensor[:, attempt, :], pred_tensor[:, n, :])
                        slice_sig = np.log(expit(slice_n))
                        sum_n += np.sum(slice_sig)
                    ranking_gain = self.lambda_w * sum_n
                else:
                    ranking_gain = 0.0
        loss = square_loss_q + self.lambda_t * np.linalg.norm(self.U)**2 + self.lambda_q * np.linalg.norm(self.W)**2 - ranking_gain
        print("Overall Loss: {}".format(loss))
        metrics_all = [q_mae, q_rmse, q_rse, q_auc, cross_entropy.numpy()]
        return loss, metrics_all

    # -------------------------------
    # Training Procedure for a Fold
    # -------------------------------
    def train(self, fold, fold_save_path, early_stopping_patience=5):
        print("Training CP Decomposition tensor factorization")
        csv_path = os.path.join(fold_save_path, f'epoch_metrics_fold_{fold+1}.csv')
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ['epoch', 'MAE', 'RMSE', 'RSE', 'AUC', 'Cross_Entropy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        train_perf = []
        no_improve_epochs = 0
        best_metric = float('inf')
        self.best_epoch = 0

        loss, metrics_all = self.get_loss()
        loss_list = [loss]

        best_U, best_V, best_W = None, None, None
        best_bias_s, best_bias_t, best_bias_q = None, None, None

        for epoch in range(1, self.global_max_iter + 1):
            # Update each training sample using SGD.
            for (learner, question, attempt, obs) in self.train_tensor_np:
                learner = int(learner)
                question = int(question)
                attempt = int(attempt)
                self.optimize_sgd(learner, attempt, question, obs)
            
            loss, metrics_all = self.get_loss()
            loss_list.append(loss)
            train_perf.append([metrics_all[0], metrics_all[1], metrics_all[2], metrics_all[3], metrics_all[4]])
            print(epoch, "MAE:", metrics_all[0], "RMSE:", metrics_all[1],
                  "RSE:", metrics_all[2], "AUC:", metrics_all[3], "Cross Entropy:", metrics_all[4])
            
            with open(csv_path, "a", newline="") as csvfile:
                fieldnames = ['epoch', 'MAE', 'RMSE', 'RSE', 'AUC', 'Cross_Entropy']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                row = {
                    'epoch': epoch,
                    'MAE': metrics_all[0],
                    'RMSE': metrics_all[1],
                    'RSE': metrics_all[2],
                    'AUC': metrics_all[3],
                    'Cross_Entropy': metrics_all[4]
                }
                writer.writerow(row)
            
            # Early stopping based on RMSE improvement.
            if metrics_all[1] < best_metric:
                best_metric = metrics_all[1]
                no_improve_epochs = 0
                self.best_epoch = epoch
                best_U = np.copy(self.U)
                best_V = np.copy(self.V)
                best_W = np.copy(self.W)
                best_bias_s = np.copy(self.bias_s)
                best_bias_t = np.copy(self.bias_t)
                best_bias_q = np.copy(self.bias_q)
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} for fold {fold+1}")
                break

        # Set factor matrices and biases to the best found.
        self.U = best_U
        self.V = best_V
        self.W = best_W
        self.bias_s = best_bias_s
        self.bias_t = best_bias_t
        self.bias_q = best_bias_q

        print("Factor matrices shapes:", self.U.shape, self.V.shape, self.W.shape)
        dim1 = self.U.shape[0]
        dim2 = self.V.shape[0]
        dim3 = self.W.shape[0]
        T = np.zeros((dim1, dim3, dim2))
        for r in range(self.U.shape[1]):
            T += np.outer(np.outer(self.U[:, r], self.W[:, r]), self.V[:, r]).reshape(dim1, dim3, dim2)
        print("Reconstructed tensor shape: {}".format(T.shape))
        print("Bias shapes:", self.bias_s.shape, self.bias_t.shape, self.bias_q.shape)
        bias_s_expanded = self.bias_s[:, np.newaxis, np.newaxis]
        bias_t_expanded = self.bias_t[np.newaxis, np.newaxis, :]
        bias_q_expanded = self.bias_q[np.newaxis, :, np.newaxis]
        T = T + bias_s_expanded + bias_t_expanded + bias_q_expanded
        T = np.where(T > 100, 1, T)
        T = np.where(T < -100, 0, T)
        T = expit(T)
        self.T = T
        mode = "trained"
        
        save_metrics(train_perf, self.global_max_iter, mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)

    # -------------------------------
    # Testing Procedure
    # -------------------------------
    def test(self, fold, fold_save_path, test_indices):
        perf_dict = []
        print("Testing CP Decomposition tensor factorization")
        test_indices_tf = tf.constant(test_indices)
        test_real_values = tf.gather_nd(self.test_tensor, test_indices_tf)
        test_real_values_np = test_real_values.numpy()
        pred_test_values = tf.gather_nd(self.T, test_indices_tf)
        pred_test_values_np = pred_test_values.numpy()
        curr_obs_list, curr_pred_list = zip(*[(obs, pred) for obs, pred in zip(test_real_values_np, pred_test_values_np) if not np.isnan(obs)])
        test_mae = mean_absolute_error(curr_obs_list, curr_pred_list)
        test_rmse = sqrt(mean_squared_error(curr_obs_list, curr_pred_list))
        mse = mean_squared_error(curr_obs_list, curr_pred_list)
        test_rse = mse / np.mean(np.square(curr_obs_list))
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        cross_entropy = bce_loss(curr_obs_list, curr_pred_list)
        try:
            test_auc_score = roc_auc_score(curr_obs_list, curr_pred_list)
            test_auc_score = tf.reduce_mean(test_auc_score)
        except ValueError as e:
            if 'Only one class present' in str(e):
                test_auc_score = np.nan
            else:
                raise
        # Append as a list (instead of a tuple) to avoid concatenation errors.
        perf_dict.append([test_mae, test_rmse, test_rse, float(test_auc_score), float(cross_entropy)])
        mode = "test"
        save_metrics(perf_dict, self.global_max_iter, mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
        return perf_dict

    # -------------------------------
    # Imputation Evaluation
    # -------------------------------
    def evaluate_imputation(self):
        """
        Evaluate imputation performance on missing entries using ground truth from dense_tensor.
        """
        if self.missing_indices is None:
            self.missing_indices = get_missing_indices(self.sparse_tensor)
        missing_indices_tf = tf.constant(self.missing_indices)
        predicted_values = tf.gather_nd(self.T, missing_indices_tf).numpy()
        ground_truth_values = tf.gather_nd(self.dense_tensor, missing_indices_tf).numpy()
        mae = mean_absolute_error(ground_truth_values, predicted_values)
        rmse = sqrt(mean_squared_error(ground_truth_values, predicted_values))
        mse = mean_squared_error(ground_truth_values, predicted_values)
        rse = mse / np.mean(np.square(ground_truth_values))
        try:
            auc = roc_auc_score(ground_truth_values, predicted_values)
        except ValueError:
            auc = np.nan
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        ce = float(bce_loss(ground_truth_values, predicted_values).numpy())
        return mae, rmse, rse, auc, ce

    # -------------------------------
    # Save Best Fold Metrics
    # -------------------------------
    def save_fold_best_metrics(self, fold_best_metrics_list):
        csv_path = os.path.join(self.save_path_global_training, 'global_training_best_metrics_all_folds.csv')
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ['fold', 'best_epoch', 'best_rmse', 'best_mae', 'best_rse', 'best_auc', 'best_cross_entropy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in fold_best_metrics_list:
                row = {
                    'fold': metrics['fold'],
                    'best_epoch': metrics['best_epoch'],
                    'best_rmse': float(metrics['best_rmse']),
                    'best_mae': float(metrics['best_mae']),
                    'best_rse': float(metrics['best_rse']),
                    'best_auc': float(metrics['best_auc']),
                    'best_cross_entropy': float(metrics['best_cross_entropy'])
                }
                writer.writerow(row)
        print(f"Saved best metrics for all folds to: {csv_path}")

    # -------------------------------
    # Run Model with K-Fold Cross Validation
    # -------------------------------
    def RunModel(self, early_stopping_patience=5):
        print("Start of CP Decomposition Model")
        self.sparse_tensor = tf.convert_to_tensor(self.sparse_tensor)
        self.dense_tensor = tf.convert_to_tensor(self.dense_tensor)
        start_time = time.time()
        cv_results = []
        fold_summary = {}
        fold_best_metrics_list = []

        for fold, (train_tensor, test_tensor, train_indices, test_indices) in enumerate(perform_k_fold_cross_validation_tf(self.sparse_tensor, k=5)):
            print(f"Training on fold {fold + 1}")
            print("train_tensor.shape:", train_tensor.shape)
            print("test_tensor.shape:", test_tensor.shape)
            print("self.sparse_tensor shape:", self.sparse_tensor.shape)
            
            self.missing_indices = get_missing_indices(self.sparse_tensor)
            fold_save_path = os.path.join(self.save_path_global_training, f"fold_{fold+1}")
            os.makedirs(fold_save_path, exist_ok=True)
            
            tl.set_backend('numpy')
            self.num_learner = self.sparse_tensor.shape[0]
            self.num_question = self.sparse_tensor.shape[1]
            self.num_attempt = self.sparse_tensor.shape[2]
            self.U = np.random.random_sample((self.num_learner, self.num_features))
            self.V = np.random.random_sample((self.num_attempt, self.num_features))
            self.W = np.random.random_sample((self.num_question, self.num_features))
            self.bias_s = np.zeros(self.num_learner)
            self.bias_t = np.zeros(self.num_attempt)
            self.bias_q = np.zeros(self.num_question)
            self.global_bias = np.nanmean(train_tensor)
            self.train_tensor_np = tensor_to_numpy(train_tensor)
            self.test_tensor_np = tensor_to_numpy(test_tensor)
            self.test_tensor = test_tensor
            self.train_tensor = train_tensor

            # Save original indices and tensor data.
            train_mode = "origin_train"
            save_indices(train_indices, self.global_max_iter, train_mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
            save_tensor(self.train_tensor_np, self.global_max_iter, train_mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)

            test_mode = "origin_test"
            save_indices(test_indices, self.global_max_iter, test_mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
            save_tensor(self.test_tensor_np, self.global_max_iter, test_mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
            
            # Record the start time for this fold.
            fold_start_time = time.time()
            
            self.train(fold, fold_save_path, early_stopping_patience)
            current_best_T = self.T
            print("current_best_T is ", current_best_T.shape)
            
            mae, rmse, rse, auc, ce = self.evaluate_imputation()
            print("Imputation evaluation on missing entries:")
            print("Fold:", fold+1, "MAE:", mae, "RMSE:", rmse, "RSE:", rse, "AUC:", auc, "Cross Entropy:", ce)
            fold_best = {
                'fold': fold+1,
                'best_epoch': self.best_epoch,
                'best_rmse': rmse,
                'best_mae': mae,
                'best_rse': rse,
                'best_auc': auc,
                'best_cross_entropy': ce
            }
            fold_best_metrics_list.append(fold_best)
            
            test_results = self.test(fold, fold_save_path, test_indices)
            cv_results.append((fold, test_results[0][0], test_results[0][1],
                               test_results[0][2], test_results[0][3],
                               test_results[0][4], self.global_max_iter, self.Version))
            
            # Compute elapsed time for this fold (in seconds)
            fold_elapsed_time = time.time() - fold_start_time
            fold_summary[fold] = fold_elapsed_time

            # Prepare hyperparameters and training parameters to save.
            params = {
                'lambda_t': self.lambda_t,
                'lambda_q': self.lambda_q,
                'lr': self.lr,
                'lambda_w': self.lambda_w,
                'global_max_iter': self.global_max_iter,
                'num_features': self.num_features,
                'use_bias': self.use_bias,
                'binarized_question': self.binarized_question,
                'is_rank': self.is_rank
            }
            save_model_parameters(params, fold_elapsed_time, fold_save_path)
            
            # Optionally, update summary using save_summary function.
            summary = save_summary(cv_results, fold_elapsed_time)
            
        self.save_fold_best_metrics(fold_best_metrics_list)
        total_elapsed_time = time.time() - start_time
        print("Total elapsed time for RunModel: {:.2f} seconds".format(total_elapsed_time))
        return cv_results, fold_summary