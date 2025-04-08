import numpy as np
import tensorly as tl
from TC.TC_helper import tensor_to_numpy, perform_k_fold_cross_validation_tf, save_indices, save_metrics, save_tensor, get_missing_indices
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from math import sqrt
from scipy.special import expit
import tensorflow as tf
import os
import csv
import time
from helper import save_summary

# Helper function to save hyperparameters and training parameters for each fold.
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

class Standard_TC:
    def __init__(self, sparse_tensor, dense_tensor, parameters, rank_similarity_tensors, args, sparsity_level):
        print("Standard Tensor Factorization (CP Decomposition) - Modified Version")
        np.random.seed(22)
        
        # Basic model info
        self.Lesson_Id = args.Lesson_Id[0]
        self.Imputation_model = args.Imputation_model[0]
        self.sparsity_level = sparsity_level
        
        self.sparse_tensor = sparse_tensor
        self.dense_tensor = dense_tensor
        self.parameters = parameters
        self.rank_similarity_tensors = rank_similarity_tensors
        self.args = args
        
        # In this algorithm, the number of features is set in parameters.
        self.num_features = 6
        
        # Hyperparameters (make sure 'lambda_bias' is provided in parameters)
        self.lambda_t = 0.000001
        self.lambda_q = 0.001
        self.lambda_bias = 0.001
        self.lambda_w = 0.1
        self.lr = 0.000008 # 0.000001
        
        # Bias usage flags
        self.use_bias_t = True
        self.use_global_bias = True
        self.binarized_question = True
        
        self.is_rank = True
        
        # Factor matrices:
        # U: (num_learner x num_features)
        # V: (num_features x num_attempt x num_question)
        self.U = None
        self.V = None
        self.T = None  # Reconstructed tensor
        
        self.bias_s = None  # learner bias, shape: (num_learner,)
        self.bias_t = None  # attempt bias, shape: (num_attempt,)
        self.bias_q = None  # question bias, shape: (num_question,)
        self.global_bias = None
        
        self.global_max_iter = 200
        self.Version = 0
        
        self.train_tensor = None
        self.test_tensor = None
        self.train_tensor_np = None
        self.test_tensor_np = None
        self.missing_indices = None
        
        # Setup directory to save models and metrics
        self.save_path_global_training = os.path.join(os.getcwd(), 'results', 'global_training_models', 
                                                      str(self.Imputation_model), str(self.Lesson_Id), str(self.sparsity_level))
        os.makedirs(self.save_path_global_training, exist_ok=True)
        
    # -------------------------------
    # Prediction and Gradients
    # -------------------------------
    def get_question_prediction(self, learner, attempt, question):
        # Compute prediction as dot product between U[learner, :] and V[:, attempt, question]
        pred = np.dot(self.U[learner, :], self.V[:, attempt, question])
        if self.use_bias_t:
            if self.use_global_bias:
                pred += self.bias_s[learner] + self.bias_t[attempt] + self.bias_q[question] + self.global_bias
            else:
                pred += self.bias_s[learner] + self.bias_t[attempt] + self.bias_q[question]
        else:
            if self.use_global_bias:
                pred += self.bias_s[learner] + self.bias_q[question] + self.global_bias
            else:
                pred += self.bias_s[learner] + self.bias_q[question]
        if self.binarized_question:
            pred = expit(pred)
        return pred

    def grad_Q_K(self, learner, attempt, question, obs=None):
        # Gradient for V[:, attempt, question] (question–related factor)
        grad = np.zeros_like(self.V[:, attempt, question])
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad = -2.0 * (obs - pred) * pred * (1.0 - pred) * self.U[learner, :] + 2.0 * self.lambda_q * self.V[:, attempt, question]
            else:
                grad = -2.0 * (obs - pred) * self.U[learner, :] + 2.0 * self.lambda_q * self.V[:, attempt, question]
        return grad

    def grad_T_ij(self, learner, attempt, question, obs=None):
        # Gradient for U[learner, :] (learner–related factor)
        grad = np.zeros_like(self.U[learner, :])
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad = -2.0 * (obs - pred) * pred * (1.0 - pred) * self.V[:, attempt, question] + 2.0 * self.lambda_t * self.U[learner, :]
            else:
                grad = -2.0 * (obs - pred) * self.V[:, attempt, question] + 2.0 * self.lambda_t * self.U[learner, :]
        return grad

    def grad_bias_q(self, learner, attempt, question, obs=None):
        grad = 0.0
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad = -2.0 * (obs - pred) * pred * (1.0 - pred) + 2.0 * self.lambda_bias * self.bias_q[question]
            else:
                grad = -2.0 * (obs - pred) + 2.0 * self.lambda_bias * self.bias_q[question]
        return grad

    def grad_bias_s(self, learner, attempt, question, obs=None):
        grad = 0.0
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad = -2.0 * (obs - pred) * pred * (1.0 - pred) + 2.0 * self.lambda_bias * self.bias_s[learner]
            else:
                grad = -2.0 * (obs - pred) + 2.0 * self.lambda_bias * self.bias_s[learner]
        return grad

    def grad_bias_t(self, learner, attempt, question, obs=None):
        grad = 0.0
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad = -2.0 * (obs - pred) * pred * (1.0 - pred) + 2.0 * self.lambda_bias * self.bias_t[attempt]
            else:
                grad = -2.0 * (obs - pred) + 2.0 * self.lambda_bias * self.bias_t[attempt]
        return grad

    def grad_global_bias(self, learner, attempt, question, obs=None):
        grad = 0.0
        if obs is not None and not np.isnan(obs):
            pred = self.get_question_prediction(learner, attempt, question)
            if self.binarized_question:
                grad = -2.0 * (obs - pred) * pred * (1.0 - pred) + 2.0 * self.lambda_bias * self.global_bias
            else:
                grad = -2.0 * (obs - pred) + 2.0 * self.lambda_bias * self.global_bias
        return grad

    def optimize_sgd(self, learner, attempt, question, obs=None):
        # Update V (question-related factor)
        grad_v = self.grad_Q_K(learner, attempt, question, obs)
        self.V[:, attempt, question] -= self.lr * grad_v
        # Enforce non-negativity and normalize V if lambda_q is zero
        self.V[:, attempt, question][self.V[:, attempt, question] < 0] = 0.0
        if self.lambda_q == 0:
            sum_val = np.sum(self.V[:, attempt, question])
            if sum_val != 0:
                self.V[:, attempt, question] /= sum_val
        # Update U (learner-related factor)
        grad_u = self.grad_T_ij(learner, attempt, question, obs)
        self.U[learner, :] -= self.lr * grad_u
        # Update biases
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
        
        # Regularization terms
        reg_U = np.linalg.norm(self.U)**2
        reg_V = np.linalg.norm(self.V)**2
        reg_features = self.lambda_t * reg_U + self.lambda_q * reg_V
        if self.lambda_bias:
            if self.use_bias_t:
                reg_bias = self.lambda_bias * (np.linalg.norm(self.bias_s)**2 + np.linalg.norm(self.bias_t)**2 + np.linalg.norm(self.bias_q)**2)
            else:
                reg_bias = self.lambda_bias * (np.linalg.norm(self.bias_s)**2 + np.linalg.norm(self.bias_q)**2)
        else:
            reg_bias = 0
        
        # Reconstruct tensor: 
        # trans_V: shape (num_attempt, num_features, num_question)
        trans_V = np.transpose(self.V, (1, 0, 2))
        pred_tensor = np.dot(self.U, trans_V)  # shape: (num_learner, num_attempt, num_question)
        
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
            loss = square_loss_q + reg_features + reg_bias - ranking_gain
        else:
            loss = square_loss_q + reg_features + reg_bias
        
        print("Overall Loss: {}".format(loss))
        metrics_all = [q_mae, q_rmse, q_rse, q_auc, cross_entropy.numpy()]
        return loss, metrics_all
    
    # -------------------------------
    # Training Procedure for a Fold
    # -------------------------------
    def train(self, fold, fold_save_path, early_stopping_patience):
        print("Training tensor factorization")
        csv_path = os.path.join(fold_save_path, f'epoch_metrics_fold_{fold+1}.csv')
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ['epoch', 'MAE', 'RMSE', 'RSE', 'AUC', 'Cross_Entropy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        train_perf = []
        no_improve_epochs = 0
        best_metric = float('inf')
        
        loss, metrics_all = self.get_loss()
        loss_list = [loss]
        
        best_U, best_V = None, None
        best_bias_s, best_bias_t, best_bias_q = None, None, None
        
        for epoch in range(1, self.global_max_iter + 1):
            for (learner, question, attempt, obs) in self.train_tensor_np:
                learner = int(learner)
                question = int(question)
                attempt = int(attempt)
                self.optimize_sgd(learner, attempt, question, obs)
            
            loss, metrics_all = self.get_loss()
            loss_list.append(loss)
            train_perf.append([metrics_all[0], metrics_all[1], metrics_all[2], metrics_all[3], metrics_all[4]])
            print(epoch, "MAE:", metrics_all[0], "RMSE:", metrics_all[1], "RSE:", metrics_all[2],
                  "AUC:", metrics_all[3], "Cross Entropy:", metrics_all[4])
            
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'MAE', 'RMSE', 'RSE', 'AUC', 'Cross_Entropy'])
                row = {
                    'epoch': epoch,
                    'MAE': metrics_all[0],
                    'RMSE': metrics_all[1],
                    'RSE': metrics_all[2],
                    'AUC': metrics_all[3],
                    'Cross_Entropy': metrics_all[4]
                }
                writer.writerow(row)
            
            if metrics_all[1] < best_metric:
                best_metric = metrics_all[1]
                no_improve_epochs = 0
                self.best_epoch = epoch
                best_U = np.copy(self.U)
                best_V = np.copy(self.V)
                best_bias_s = np.copy(self.bias_s)
                best_bias_t = np.copy(self.bias_t)
                best_bias_q = np.copy(self.bias_q)
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} for fold {fold+1}")
                break
        
        self.U = best_U
        self.V = best_V
        self.bias_s = best_bias_s
        self.bias_t = best_bias_t
        self.bias_q = best_bias_q
        
        print("Factor matrices shapes:", self.U.shape, self.V.shape)
        # Reconstruct the tensor T:
        # First, transpose V to shape (num_attempt, num_features, num_question)
        trans_V = np.transpose(self.V, (1, 0, 2))
        T = np.dot(self.U, trans_V) + self.global_bias
        for i in range(self.U.shape[0]):
            T[i, :, :] += self.bias_s[i]
        for j in range(self.num_attempt):
            T[:, j, :] += self.bias_t[j]
        for k in range(self.num_question):
            T[:, :, k] += self.bias_q[k]
        T = np.where(T > 100, 1, T)
        T = np.where(T < -100, 0, T)
        T = expit(T)
        self.T = T
        
        mode = "trained"
        save_metrics(train_perf, self.global_max_iter, mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
    
    # -------------------------------
    # Testing Procedure
    # -------------------------------
    def test(self, fold, test_indices, fold_save_path):
        perf_dict = []
        print("Testing tensor factorization")
        test_indices_tf = tf.constant(test_indices)
        test_real_values = tf.gather_nd(self.test_tensor, test_indices_tf)
        test_real_values_np = test_real_values.numpy()
        pred_test_values = tf.gather_nd(self.T, test_indices_tf)
        pred_test_values_np = pred_test_values.numpy()
        # Append as a list (not a tuple) to avoid concatenation errors in save_metrics
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
        perf_dict.append([test_mae, test_rmse, test_rse, float(test_auc_score), float(cross_entropy)])
        mode = "test"
        save_metrics(perf_dict, self.global_max_iter, mode, self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
        return perf_dict
    
    def evaluate_imputation(self):
        # Evaluate imputation performance on missing entries
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
            
            fold_save_path = os.path.join(self.save_path_global_training, f"fold_{fold+1}")
            os.makedirs(fold_save_path, exist_ok=True)
            
            tl.set_backend('numpy')
            self.num_learner = self.sparse_tensor.shape[0]
            self.num_question = self.sparse_tensor.shape[1]
            self.num_attempt = self.sparse_tensor.shape[2]
            # Initialize U and V according to the algorithm:
            self.U = np.random.random_sample((self.num_learner, self.num_features))
            self.V = np.random.random_sample((self.num_features, self.num_attempt, self.num_question))
            self.bias_s = np.zeros(self.num_learner)
            self.bias_t = np.zeros(self.num_attempt)
            self.bias_q = np.zeros(self.num_question)
            self.global_bias = np.nanmean(train_tensor)
            self.train_tensor_np = tensor_to_numpy(train_tensor)
            self.test_tensor_np = tensor_to_numpy(test_tensor)
            self.test_tensor = test_tensor
            self.train_tensor = train_tensor
            
            # Uncomment these lines if you want to save original indices and tensor data:
            # save_indices(train_indices, self.global_max_iter, "origin_train", self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
            # save_tensor(self.train_tensor_np, self.global_max_iter, "origin_train", self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
            # save_indices(test_indices, self.global_max_iter, "origin_test", self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
            # save_tensor(self.test_tensor_np, self.global_max_iter, "origin_test", self.Imputation_model, self.Lesson_Id, self.sparsity_level, fold, fold_save_path)
            
            self.train(fold, fold_save_path, early_stopping_patience=5)
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
            
            test_results = self.test(fold, test_indices, fold_save_path)
            # Build cv_results as a list (not a tuple) to avoid concatenation issues in save_metrics
            cv_results.append([fold, 
                               test_results[0][0], 
                               test_results[0][1],
                               test_results[0][2],
                               test_results[0][3],
                               test_results[0][4],
                               self.global_max_iter, self.Version])
            # Compute elapsed time for this fold (in seconds)
            fold_elapsed_time = time.time() - start_time
            summary = save_summary(cv_results, fold_elapsed_time)
            fold_summary[fold] = fold_elapsed_time
            
            # Prepare hyperparameters and training parameters to save.
            params = {
                'lambda_t': self.lambda_t,
                'lambda_q': self.lambda_q,
                'lambda_bias': self.lambda_bias,
                'lambda_w': self.lambda_w,
                'lr': self.lr,
                'global_max_iter': self.global_max_iter,
                'num_features': self.num_features,
                'use_bias_t': self.use_bias_t,
                'use_global_bias': self.use_global_bias,
                'binarized_question': self.binarized_question,
                'is_rank': self.is_rank
            }
            save_model_parameters(params, fold_elapsed_time, fold_save_path)
            
        self.save_fold_best_metrics(fold_best_metrics_list)
        return cv_results, fold_summary