import time
import os
import csv
import numpy as np
import tensorflow as tf
import tensorly as tl
from math import sqrt
from scipy.special import expit
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from scipy.stats import wishart
from numpy.linalg import inv, solve
from scipy.linalg import khatri_rao as kr_prod, cholesky as cholesky_upper, solve_triangular as solve_ut
from helper import save_summary
from TC.TC_helper import tensor_to_numpy, perform_k_fold_cross_validation_tf, save_indices, save_tensor, get_missing_indices
import random
random.seed(22)
tf.random.set_seed(22)

#####################################
# Helper Save Functions (CSV based)
#####################################
def save_metrics_csv(perf_list, max_iter, mode, model_name, lesson_id, sparsity_level, fold, fold_save_path):
    """
    Save a list of performance rows into a CSV file.
    Each row is expected to be a list: [MAE, RMSE, RSE, AUC, Cross_Entropy].
    """
    csv_path = os.path.join(fold_save_path, f"{model_name}_{lesson_id}_{sparsity_level}_{mode}_MaxIter{max_iter}_fold{fold}_metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ['epoch', 'MAE', 'RMSE', 'RSE', 'AUC', 'Cross_Entropy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for i, row in enumerate(perf_list, start=1):
            writer.writerow({
                'epoch': i,
                'MAE': row[0],
                'RMSE': row[1],
                'RSE': row[2],
                'AUC': row[3],
                'Cross_Entropy': row[4]
            })
    print(f"Saved metrics to CSV file: {csv_path}")

def save_fold_best_metrics_csv(fold_best_metrics_list, save_path):
    """
    Save a summary CSV file with best metrics for all folds.
    """
    csv_path = os.path.join(save_path, 'global_training_best_metrics_all_folds.csv')
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ['fold', 'best_epoch', 'best_rmse', 'best_mae', 'best_rse', 'best_auc', 'best_cross_entropy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in fold_best_metrics_list:
            writer.writerow({
                'fold': metrics['fold'],
                'best_epoch': metrics['best_epoch'],
                'best_rmse': float(metrics['best_rmse']),
                'best_mae': float(metrics['best_mae']),
                'best_rse': float(metrics['best_rse']),
                'best_auc': float(metrics['best_auc']),
                'best_cross_entropy': float(metrics['best_cross_entropy'])
            })
    print(f"Saved best metrics for all folds to: {csv_path}")

# New helper function to save model parameters and elapsed training time
def save_model_parameters(params, elapsed_time, fold_save_path):
    """
    Save the model parameters and elapsed training time to a CSV file in the given fold directory.
    """
    csv_path = os.path.join(fold_save_path, "model_parameters.csv")
    params['elapsed_time'] = elapsed_time
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = list(params.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(params)
    print(f"Saved model parameters and elapsed time to {csv_path}")

#####################################
# BPTF Class Definition
#####################################
class BPTF:
    def __init__(self, sparse_tensor, dense_tensor, parameters, rank_similarity_tensors, args, sparsity_level):
        print("Standard Tensor Factorization (CP Decomposition) - BPTF Version")
        np.random.seed(22)
        
        # Basic model info
        self.Lesson_Id = args.Lesson_Id[0]
        self.Imputation_model = args.Imputation_model[0]
        self.sparsity_level = sparsity_level
        
        # Data: sparse tensor (with missing values) and dense tensor (ground truth)
        self.sparse_tensor = sparse_tensor
        self.dense_tensor = dense_tensor
        self.parameters = parameters
        self.rank_similarity_tensors = rank_similarity_tensors
        self.args = args
        
        # Number of features provided in parameters.
        self.num_features = 6
        
        # Hyperparameters specific to BPTF.
        self.lambda_u = 0.001    # Regularization for U (learners)
        self.lambda_v = 0.001    # Regularization for V (questions)
        self.lambda_x = 0.001    # Regularization for X (attempts)
        self.lambda_bias = 0.001  # Base weight for biases
        
        # Introduce weight factors for biases (as in the old version)
        self.weight_u_bias = self.lambda_bias
        self.weight_v_bias = self.lambda_bias
        self.weight_x_bias = self.lambda_bias
        self.weight_global_bias = self.lambda_bias
        
        self.lambda_w = 0.01
        self.lr = 0.000008 # 0.000001
        self.global_max_iter = 200
        self.gibbs_iteration = 500
        self.burn_iter = 500
        
        # Bias usage flags.
        self.use_bias_t = True
        self.use_global_bias = True
        self.binarized_question = True
        
        self.is_rank = True
        
        # Factor matrices and biases.
        # U: (num_learners, num_features)
        # V: (num_questions, num_features)
        # X: (num_attempts, num_features)
        self.U = None
        self.V = None
        self.X = None
        
        self.U_bias = None
        self.V_bias = None
        self.X_bias = None
        self.global_bias = 0.0
        
        self.T = None  # Reconstructed tensor.
        self.Version = 0
        
        self.missing_indices = None
        
        # Placeholders for train/test tensors in the current fold.
        self.train_tensor = None   # Full tensor (e.g., shape (num_learners, num_questions, num_attempts))
        self.test_tensor = None
        self.train_tensor_np = None  # Flattened version for get_loss.
        self.test_tensor_np = None
        
        # For storing best epoch in each fold.
        self.best_epoch = 0
        
        # Setup directory to save models and metrics.
        self.save_path_global_training = os.path.join(
            os.getcwd(), 'results', 'global_training_models',
            str(self.Imputation_model), str(self.Lesson_Id), str(self.sparsity_level)
        )
        os.makedirs(self.save_path_global_training, exist_ok=True)
    
    # -------------------------------
    # Get Prediction
    # -------------------------------
    def get_question_prediction(self, learner, question, attempt):
        # Compute prediction as dot product between U[learner, :] and (V[question, :] * X[attempt, :])
        pred = np.dot(self.U[learner, :], self.V[question, :] * self.X[attempt, :])
        if self.use_bias_t:
            if self.use_global_bias:
                pred += self.U_bias[learner] + self.V_bias[question] + self.X_bias[attempt] + self.global_bias
            else:
                pred += self.U_bias[learner] + self.V_bias[question] + self.X_bias[attempt]
        else:
            if self.use_global_bias:
                pred += self.U_bias[learner] + self.V_bias[question] + self.global_bias
            else:
                pred += self.U_bias[learner] + self.V_bias[question]
        if self.binarized_question:
            pred = expit(pred)
        return pred
    
    # -------------------------------
    # Sampling and Helper Methods
    # -------------------------------
    def mvnrnd_pre(self, mu, Lambda):
        src = np.random.normal(size=(mu.shape[0],))
        L_upper = cholesky_upper(Lambda, overwrite_a=True, check_finite=False)
        return solve_ut(L_upper, src, lower=False, check_finite=False, overwrite_b=True) + mu

    def cov_mat(self, mat, mat_bar):
        mat = mat - mat_bar
        return mat.T @ mat

    def ten2mat(self, tensor, mode):
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')
    
    def sample_factor_u(self, tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
        dim1, rank = U.shape  # dim1 = num_learners
        U_bar = np.mean(U, axis=0)
        temp = dim1 / (dim1 + beta0)
        var_mu_hyper = temp * U_bar
        var_U_hyper = inv(np.eye(rank) + self.cov_mat(U, U_bar) + temp * beta0 * np.outer(U_bar, U_bar))
        reg_term = self.lambda_u * np.eye(rank)
        var_U_hyper = inv(inv(var_U_hyper) + reg_term)
        var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_U_hyper)
        var_mu_hyper = self.mvnrnd_pre(var_mu_hyper, (dim1 + beta0) * var_Lambda_hyper)
        tau_ind_array = np.full(tau_sparse_tensor.shape, tau_ind)
        var1 = kr_prod(X, V).T  # Expected shape: (rank, num_attempts*num_questions)
        var2 = kr_prod(var1, var1)  # Expected shape: (rank*rank, num_attempts*num_questions)
        var3 = (var2 @ self.ten2mat(tau_ind_array, 0).T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
        var4 = var1 @ self.ten2mat(tau_sparse_tensor, 0).T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
        for i in range(dim1):
            U[i, :] = self.mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
        return U

    def sample_factor_v(self, tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
        dim2, rank = V.shape  # dim2 = num_questions
        V_bar = np.mean(V, axis=0)
        temp = dim2 / (dim2 + beta0)
        var_mu_hyper = temp * V_bar
        var_V_hyper = inv(np.eye(rank) + self.cov_mat(V, V_bar) + temp * beta0 * np.outer(V_bar, V_bar))
        reg_term = self.lambda_v * np.eye(rank)
        var_V_hyper = inv(inv(var_V_hyper) + reg_term)
        var_Lambda_hyper = wishart.rvs(df=dim2 + rank, scale=var_V_hyper)
        var_mu_hyper = self.mvnrnd_pre(var_mu_hyper, (dim2 + beta0) * var_Lambda_hyper)
        tau_ind_array = np.full(tau_sparse_tensor.shape, tau_ind)
        var1 = kr_prod(X, U).T  # Expected shape: (rank, num_attempts*num_learners)
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ self.ten2mat(tau_ind_array, 1).T).reshape([rank, rank, dim2]) + var_Lambda_hyper[:, :, None]
        var4 = var1 @ self.ten2mat(tau_sparse_tensor, 1).T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
        for j in range(dim2):
            V[j, :] = self.mvnrnd_pre(solve(var3[:, :, j], var4[:, j]), var3[:, :, j])
        return V

    # def sample_factor_x(self, tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
    #     dim3, rank = X.shape  # dim3 = num_attempts
    #     var_mu_hyper = X[0, :] / (beta0 + 1)
    #     if dim3 > 1:
    #         dx = X[1:, :] - X[:-1, :]
    #     else:
    #         dx = np.zeros((dim3, rank))
    #     var_V_hyper = inv(np.eye(rank) + dx.T @ dx + beta0 * np.outer(X[0, :], X[0, :]) / (beta0 + 1))
    #     reg_term = self.lambda_x * np.eye(rank)
    #     var_V_hyper = inv(inv(var_V_hyper) + reg_term)
    #     var_Lambda_hyper = wishart.rvs(df=dim3 + rank, scale=var_V_hyper)
    #     var_mu_hyper = self.mvnrnd_pre(var_mu_hyper, (beta0 + 1) * var_Lambda_hyper)
    #     tau_ind_array = np.full(tau_sparse_tensor.shape, tau_ind)
    #     var1 = kr_prod(V, U).T
    #     var2 = kr_prod(var1, var1)
    #     var3 = (var2 @ self.ten2mat(tau_sparse_tensor, 2).T).reshape([rank, rank, dim3])
    #     var4 = var1 @ self.ten2mat(tau_sparse_tensor, 2).T
    #     for t in range(dim3):
    #         if dim3 == 1:
    #             temp1 = var4[:, t] + var_Lambda_hyper @ var_mu_hyper
    #             temp2 = var3[:, :, t] + var_Lambda_hyper
    #             X[t, :] = self.mvnrnd_pre(solve(temp2, temp1), temp2)
    #         elif t == 0:
    #             X[t, :] = self.mvnrnd_pre((X[t + 1, :] + var_mu_hyper) / 2, var3[:, :, t] + 2 * var_Lambda_hyper)
    #         elif t == dim3 - 1:
    #             temp1 = var4[:, t] + var_Lambda_hyper @ X[t - 1, :]
    #             temp2 = var3[:, :, t] + var_Lambda_hyper
    #             X[t, :] = self.mvnrnd_pre(solve(temp2, temp1), temp2)
    #         else:
    #             temp1 = var4[:, t] + var_Lambda_hyper @ (X[t - 1, :] + X[t + 1, :])
    #             temp2 = var3[:, :, t] + 2 * var_Lambda_hyper
    #             X[t, :] = self.mvnrnd_pre(solve(temp2, temp1), temp2)
    #     return X
    
    def sample_factor_x(self, tau_sparse_tensor, tau_ind, U, V, X, beta0=1):
        dim3, rank = X.shape  # dim3 = num_attempts
        var_mu_hyper = X[0, :] / (beta0 + 1)
        if dim3 > 1:
            dx = X[1:, :] - X[:-1, :]
        else:
            dx = np.zeros((dim3, rank))
            
        # Compute an initial version of var_V_hyper based on the temporal differences and initial state
        var_V_hyper = inv(np.eye(rank) + dx.T @ dx + beta0 * np.outer(X[0, :], X[0, :]) / (beta0 + 1))
        reg_term = self.lambda_x * np.eye(rank)
        var_V_hyper = inv(inv(var_V_hyper) + reg_term)
        
        # Increase regularization and enforce symmetry to help ensure positive definiteness
        epsilon = 1e-4  # Increased epsilon for more stability than 1e-6
        var_V_hyper += np.eye(rank) * epsilon
        var_V_hyper = (var_V_hyper + var_V_hyper.T) / 2  # enforce symmetry

        # Sample the precision matrix from a Wishart distribution
        var_Lambda_hyper = wishart.rvs(df=dim3 + rank, scale=var_V_hyper)
        
        # Sample the hypermean using our custom multivariate normal function
        var_mu_hyper = self.mvnrnd_pre(var_mu_hyper, (beta0 + 1) * var_Lambda_hyper)
        
        tau_ind_array = np.full(tau_sparse_tensor.shape, tau_ind)
        var1 = kr_prod(V, U).T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ self.ten2mat(tau_sparse_tensor, 2).T).reshape([rank, rank, dim3])
        var4 = var1 @ self.ten2mat(tau_sparse_tensor, 2).T
        
        # Update X for each attempt (time slice)
        for t in range(dim3):
            if dim3 == 1:
                temp1 = var4[:, t] + var_Lambda_hyper @ var_mu_hyper
                temp2 = var3[:, :, t] + var_Lambda_hyper
                X[t, :] = self.mvnrnd_pre(solve(temp2, temp1), temp2)
            elif t == 0:
                X[t, :] = self.mvnrnd_pre((X[t + 1, :] + var_mu_hyper) / 2, var3[:, :, t] + 2 * var_Lambda_hyper)
            elif t == dim3 - 1:
                temp1 = var4[:, t] + var_Lambda_hyper @ X[t - 1, :]
                temp2 = var3[:, :, t] + var_Lambda_hyper
                X[t, :] = self.mvnrnd_pre(solve(temp2, temp1), temp2)
            else:
                temp1 = var4[:, t] + var_Lambda_hyper @ (X[t - 1, :] + X[t + 1, :])
                temp2 = var3[:, :, t] + 2 * var_Lambda_hyper
                X[t, :] = self.mvnrnd_pre(solve(temp2, temp1), temp2)
        
        return X

    def sample_precision_tau(self, sparse_tensor, tensor_hat, ind):
        var_alpha = 1e-6 + 0.5 * np.sum(ind)
        var_beta = 1e-6 + 0.5 * np.sum(((sparse_tensor - tensor_hat) ** 2) * ind)
        return np.random.gamma(var_alpha, 1 / var_beta)
    
    # -------------------------------
    # Evaluation Metrics
    # -------------------------------
    def compute_rmse(self, var, var_hat):
        return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])
    
    def compute_binary_cross_entropy(self, var, var_hat):
        epsilon = 1e-15
        var_hat = np.clip(var_hat, epsilon, 1 - epsilon)
        return -np.mean(var * np.log(var_hat) + (1 - var) * np.log(1 - var_hat))
    
    def compute_mae(self, var, var_hat):
        return np.mean(np.abs(var - var_hat))
    
    def compute_auc(self, var, var_hat):
        return roc_auc_score(var, var_hat)
    
    def compute_rse(self, var, var_hat):
        numerator = np.sum((var - var_hat) ** 2)
        mean_var = np.mean(var)
        denominator = np.sum((var - mean_var) ** 2)
        return numerator / denominator

    # -------------------------------
    # Bias Updates and Reconstruction
    # -------------------------------
    def update_biases(self, sparse_tensor, tensor_hat, ind):
        residuals = (sparse_tensor - tensor_hat) * ind
        self.U_bias = np.mean(residuals, axis=(1, 2))
        self.V_bias = np.mean(residuals, axis=(0, 2))
        self.X_bias = np.mean(residuals, axis=(0, 1))
        self.global_bias = np.mean(residuals)
    
    def reconstruct(self, U, V, X):
        """
        Reconstruct the tensor with biases included, applying weight factors to each bias.
        This implementation follows the old version's approach.
        """
        bias_matrix = (self.U_bias[:, np.newaxis, np.newaxis] * self.weight_u_bias +
                       self.V_bias[np.newaxis, :, np.newaxis] * self.weight_v_bias +
                       self.X_bias[np.newaxis, np.newaxis, :] * self.weight_x_bias)
        tensor_hat = np.einsum('is, js, ts -> ijt', U, V, X) + bias_matrix + self.global_bias * self.weight_global_bias
        tensor_hat = expit(tensor_hat)
        return tensor_hat
    
    # -------------------------------
    # get_loss Method
    # -------------------------------
    def get_loss(self):
        loss = 0.0
        train_obs = []
        train_pred = []
        for row in self.train_tensor_np:
            if len(row) < 4:
                continue
            learner, question, attempt, obs = row
            if not np.isnan(obs):
                pred = self.get_question_prediction(int(learner), int(question), int(attempt))
                if pred is not None:
                    train_obs.append(obs)
                    train_pred.append(pred)
        train_obs, train_pred = zip(*[(obs, pred) for obs, pred in zip(train_obs, train_pred) if not np.isnan(obs)])
        q_mae = mean_absolute_error(train_obs, train_pred)
        q_auc = roc_auc_score(train_obs, train_pred)
        q_rmse = sqrt(mean_squared_error(train_obs, train_pred))
        mse_val = mean_squared_error(train_obs, train_pred)
        q_rse = mse_val / np.mean(np.square(train_obs))
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        cross_entropy = bce_loss(train_obs, train_pred)
        square_loss_q = q_rmse
        reg_U = np.linalg.norm(self.U)**2
        reg_V = np.linalg.norm(self.V)**2
        reg_features = self.lambda_u * reg_U + self.lambda_v * reg_V
        if self.lambda_bias:
            if self.use_bias_t:
                reg_bias = self.lambda_bias * (np.linalg.norm(self.U_bias)**2 + np.linalg.norm(self.X_bias)**2 + np.linalg.norm(self.V_bias)**2)
            else:
                reg_bias = self.lambda_bias * (np.linalg.norm(self.U_bias)**2 + np.linalg.norm(self.V_bias)**2)
        else:
            reg_bias = 0
        trans_V = np.transpose(self.V, (1, 0))
        pred_tensor = np.dot(self.U, trans_V)
        sum_n = 0.0
        if self.is_rank:
            for attempt in range(self.sparse_tensor.shape[2]):
                if attempt > 0:
                    for n in range(attempt - 1, attempt):
                        slice_n = np.subtract(pred_tensor[:, attempt], pred_tensor[:, n])
                        slice_sig = np.log(expit(slice_n))
                        sum_n += np.sum(slice_sig)
                    ranking_gain = self.lambda_w * sum_n
                else:
                    ranking_gain = 0.0
            loss = square_loss_q + reg_features + reg_bias - ranking_gain
        else:
            loss = square_loss_q + reg_features + reg_bias
        print("Overall Loss: {}".format(loss))
        metrics_all = [q_mae, q_rmse, q_rse, q_auc, cross_entropy.numpy()]
        return loss, metrics_all
    
    # -------------------------------
    # Training Procedure for a Fold
    # -------------------------------
    def train(self, train_tensor, burn_iter, gibbs_iter, fold, fold_save_path, early_stopping_patience):
        print("Training BPTF")
        csv_path = os.path.join(self.save_path_global_training, f"fold_{fold+1}", f'epoch_metrics_fold_{fold+1}.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Open CSV file in write mode once and write header
        csvfile = open(csv_path, "w", newline="")
        fieldnames = ['epoch', 'MAE', 'RMSE', 'RSE', 'AUC', 'Cross_Entropy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()  # flush after writing header

        best_metric = float('inf')
        no_improve_epochs = 0

        # Create a copy of the training tensor and get its flattened version.
        full_train_tensor = train_tensor.copy()  # shape: (num_learner, num_question, num_attempt)
        self.train_tensor_np = tensor_to_numpy(train_tensor)
        
        # Replace NaNs with zeros for training
        full_train_tensor[np.isnan(full_train_tensor)] = 0

        # 'pos_test' holds the indices of non-NaN entries.
        pos_test = np.where(~np.isnan(train_tensor))
        
        best_epoch = 0
        best_U, best_V = None, None
        best_U_bias, best_V_bias, best_X_bias = None, None, None

        for it in range(burn_iter + gibbs_iter):
            tau_ind = self.sample_precision_tau(full_train_tensor, full_train_tensor, (~np.isnan(full_train_tensor)))
            tau_sparse_tensor = tau_ind * full_train_tensor

            self.U = self.sample_factor_u(tau_sparse_tensor, tau_ind, self.U, self.V, self.X)
            self.V = self.sample_factor_v(tau_sparse_tensor, tau_ind, self.U, self.V, self.X)
            self.X = self.sample_factor_x(tau_sparse_tensor, tau_ind, self.U, self.V, self.X)

            tensor_hat = self.reconstruct(self.U, self.V, self.X)

            # Update biases only after burn-in period.
            if it + 1 > burn_iter:
                self.update_biases(full_train_tensor, tensor_hat, (~np.isnan(full_train_tensor)))

            # Compute metrics on the current reconstruction
            MAE = self.compute_mae(full_train_tensor[pos_test], tensor_hat[pos_test])
            RMSE = self.compute_rmse(full_train_tensor[pos_test], tensor_hat[pos_test])
            RSE = self.compute_rse(full_train_tensor[pos_test], tensor_hat[pos_test])
            AUC = self.compute_auc(full_train_tensor[pos_test], tensor_hat[pos_test])
            cross_entropy = self.compute_binary_cross_entropy(full_train_tensor[pos_test], tensor_hat[pos_test])

            print(f"Iter: {it+1}  MAE: {MAE}  RMSE: {RMSE}  RSE: {RSE}  AUC: {AUC}")

            # Write the current iteration's metrics to the CSV immediately.
            writer.writerow({
                'epoch': it+1,
                'MAE': MAE,
                'RMSE': RMSE,
                'RSE': RSE,
                'AUC': AUC,
                'Cross_Entropy': cross_entropy
            })
            csvfile.flush()

            # Update best model if needed.
            if RMSE < best_metric:
                best_metric = RMSE
                best_epoch = it + 1
                best_U = np.copy(self.U)
                best_V = np.copy(self.V)
                best_U_bias = np.copy(self.U_bias)
                best_V_bias = np.copy(self.V_bias)
                best_X_bias = np.copy(self.X_bias)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if (it + 1) >= (burn_iter + gibbs_iter) or no_improve_epochs >= early_stopping_patience:
                if no_improve_epochs >= early_stopping_patience:
                    print(f"Early stopping triggered at iteration {it+1} for fold {fold+1}")
                break

        # Close the CSV file after training is complete.
        csvfile.close()

        # Restore best parameters.
        self.U = best_U
        self.V = best_V
        self.U_bias = best_U_bias
        self.V_bias = best_V_bias
        self.X_bias = best_X_bias
        self.best_epoch = best_epoch

        self.T = tensor_hat  # Final reconstructed tensor from the last iteration.

        # Save final tensor and metrics summary.
        mode = "train"
        save_tensor(self.T, self.global_max_iter, mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)

    # -------------------------------
    # Testing Procedure
    # -------------------------------
    def test(self, test_tensor, test_indices, fold, fold_save_path):
        perf_dict = []
        print("Testing tensor factorization")
        test_indices_tf = tf.constant(test_indices)
        test_real_values = tf.gather_nd(self.test_tensor, test_indices_tf)
        test_real_values_np = test_real_values.numpy()
        pred_test_values = tf.gather_nd(self.T, test_indices_tf)
        pred_test_values_np = pred_test_values.numpy()
        curr_obs_list, curr_pred_list = zip(*[(obs, pred) for obs, pred in zip(test_real_values_np, pred_test_values_np) if not np.isnan(obs)])
        
        test_mae = mean_absolute_error(curr_obs_list, curr_pred_list)
        test_rmse = sqrt(mean_squared_error(curr_obs_list, curr_pred_list))
        mse_val = mean_squared_error(curr_obs_list, curr_pred_list)
        test_rse = mse_val / np.mean(np.square(curr_obs_list))
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
        perf_dict.append((test_mae, test_rmse, test_rse, float(test_auc_score), float(cross_entropy)))
        mode = "test"
        save_metrics_csv(perf_dict, self.global_max_iter, mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
        return perf_dict
    
    # -------------------------------
    # Evaluation Function
    # -------------------------------
    def evaluate_imputation(self):
        if self.missing_indices is None:
            self.missing_indices = get_missing_indices(self.sparse_tensor)
        missing_indices_tf = tf.constant(self.missing_indices)
        predicted_values = tf.gather_nd(self.T, missing_indices_tf).numpy()
        ground_truth_values = tf.gather_nd(self.dense_tensor, missing_indices_tf).numpy()
        mae = mean_absolute_error(ground_truth_values, predicted_values)
        rmse = sqrt(mean_squared_error(ground_truth_values, predicted_values))
        mse_val = mean_squared_error(ground_truth_values, predicted_values)
        rse = mse_val / np.mean(np.square(ground_truth_values))
        try:
            auc = roc_auc_score(ground_truth_values, predicted_values)
        except ValueError:
            auc = np.nan
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        ce = float(bce_loss(ground_truth_values, predicted_values).numpy())
        return mae, rmse, rse, auc, ce
    
    # -------------------------------
    # Save Best Fold Metrics Globally
    # -------------------------------
    def save_fold_best_metrics(self, fold_best_metrics_list):
        csv_path = os.path.join(self.save_path_global_training, 'global_training_best_metrics_all_folds.csv')
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ['fold', 'best_epoch', 'best_rmse', 'best_mae', 'best_rse', 'best_auc', 'best_cross_entropy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in fold_best_metrics_list:
                writer.writerow({
                    'fold': metrics['fold'],
                    'best_epoch': metrics['best_epoch'],
                    'best_rmse': float(metrics['best_rmse']),
                    'best_mae': float(metrics['best_mae']),
                    'best_rse': float(metrics['best_rse']),
                    'best_auc': float(metrics['best_auc']),
                    'best_cross_entropy': float(metrics['best_cross_entropy'])
                })
        print(f"Saved best metrics for all folds to: {csv_path}")
    
    # -------------------------------
    # Run Model with K-Fold Cross Validation
    # -------------------------------
    def RunModel(self):
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
            print("Missing indices in original tensor:", self.missing_indices)
            
            fold_save_path = os.path.join(self.save_path_global_training, f"fold_{fold+1}")
            os.makedirs(fold_save_path, exist_ok=True)
            
            tl.set_backend('numpy')
            self.num_learner = self.sparse_tensor.shape[0]
            self.num_question = self.sparse_tensor.shape[1]
            self.num_attempt = self.sparse_tensor.shape[2]
            
            self.U_bias = np.zeros(self.num_learner)
            self.V_bias = np.zeros(self.num_question)
            self.X_bias = np.zeros(self.num_attempt)
            
            self.U = 0.1 * np.random.randn(self.num_learner, self.num_features)
            self.V = 0.1 * np.random.randn(self.num_question, self.num_features)
            self.X = 0.1 * np.random.randn(self.num_attempt, self.num_features)
            
            self.train_tensor = train_tensor.numpy()
            self.test_tensor = test_tensor.numpy()
            self.train_tensor_np = tensor_to_numpy(train_tensor)
            self.test_tensor_np = tensor_to_numpy(test_tensor)
            
            burn_iter = self.burn_iter
            gibbs_iter = self.gibbs_iteration
            
            # Record start time for this fold
            fold_start_time = time.time()
            
            self.train(self.train_tensor, burn_iter, gibbs_iter, fold, fold_save_path, early_stopping_patience=5)
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
            
            test_results = self.test(self.test_tensor, test_indices, fold, fold_save_path)
            cv_results.append((fold, test_results[0][0], test_results[0][1], test_results[0][2], test_results[0][3],
                               test_results[0][4], self.global_max_iter, self.Version))
            
            # Record elapsed time for this fold
            fold_elapsed_time = time.time() - fold_start_time
            fold_summary[fold] = fold_elapsed_time
            
            # Save model parameters and elapsed time to CSV file for this fold
            params = {
                'lambda_u': self.lambda_u,
                'lambda_v': self.lambda_v,
                'lambda_x': self.lambda_x,
                'lambda_bias': self.lambda_bias,
                'weight_u_bias': self.weight_u_bias,
                'weight_v_bias': self.weight_v_bias,
                'weight_x_bias': self.weight_x_bias,
                'weight_global_bias': self.weight_global_bias,
                'lambda_w': self.lambda_w,
                'lr': self.lr,
                'global_max_iter': self.global_max_iter,
                'gibbs_iteration': self.gibbs_iteration,
                'burn_iter': self.burn_iter,
                'use_bias_t': self.use_bias_t,
                'use_global_bias': self.use_global_bias,
                'binarized_question': self.binarized_question,
                'is_rank': self.is_rank,
                'num_features': self.num_features
            }
            save_model_parameters(params, fold_elapsed_time, fold_save_path)
            
            # Optionally, update summary using save_summary function
            summary = save_summary(cv_results, fold_elapsed_time)
            
        self.save_fold_best_metrics(fold_best_metrics_list)
        return cv_results, fold_summary