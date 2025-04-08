
import warnings
warnings.filterwarnings('ignore')

import time

import tensorflow as tf
import numpy as np
import os

from tensorflow import keras
from sklearn.metrics import roc_auc_score

from helper import Partition, k_fold_split,save_summary
from InfoGAN.InfoGAN_Algorithms import InfoGANInputPreparation, InfoGAN_Standard_Generator, \
    InfoGAN_Standard_Discriminator, InfoGANInputPreparationWithTensor, InfoGAN_Standard_Generator_WithTensor

from TC.TC_helper import save_tensor
from InfoGAN.infoGAN_helper import save_metrics, save_indices, save_ori_tensor
import shutil
from sklearn.model_selection import KFold
from tqdm import tqdm
import csv

tf.random.set_seed(22)
np.random.seed(22)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # if you have multiple GPUs

# Helper function to save hyperparameters and training parameters for each fold.
def save_model_parameters(params, elapsed_time, fold_save_path):
    """
    Save the model hyperparameters and training parameters along with the elapsed training time (in seconds)
    to a CSV file in the given fold directory.
    """
    csv_path = os.path.join(fold_save_path, "model_parameters.csv")
    params['elapsed_time'] = elapsed_time
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = list(params.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(params)
    print(f"Saved model parameters and elapsed time to {csv_path}")

# Helper function to save the parameter counts of a model (with a final row for the total count).
def save_model_parameter_counts(model, csv_path):
    param_list = []
    total_count = 0
    for var in model.trainable_variables:
        count = np.prod(var.shape)
        total_count += count
        param_list.append({'name': var.name, 'shape': var.shape.as_list(), 'count': count})
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['name', 'shape', 'count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in param_list:
            writer.writerow(p)
        writer.writerow({'name': 'TOTAL', 'shape': '', 'count': total_count})
    print(f"Total parameters: {total_count}. Saved parameter details to {csv_path}")

class InfoGAN:
    def __init__(self, sparse_tensor, dense_tensor, parameters, rank_similarity_tensors, args, sparsity_level):
        self.sparse_tensor = sparse_tensor
        self.dense_tensor = dense_tensor
        self.parameters = parameters
        self.slice_size = 1
        self.batch_size = 10  # 10
        self.learning_rate = 0.000008  # =0.000001
        self.global_max_iter = 200

        self.hint_rate = 0.8  # 0.9
        self.alpha = 0.5 # 1
        
        self.noise_dim = 100
        self.cont_code1_dim = 10
        self.cont_code2_dim = 20

        self.NaTransfer = 'zeros'  # ones or zeros

        self.GAIN_neworks = "CNN"
        self.loss_function_type = "divergence_loss"  # 'error_loss', 'divergence_loss'
        self.rank_similarity_tensors = rank_similarity_tensors

        self.Lesson_Id = args.Lesson_Id[0]
        self.Imputation_model = args.Imputation_model[0]
        self.sparsity_level = sparsity_level

        # Save the args for later use (needed by RunModel, etc.)
        self.args = args
        
        self.save_path_global_training = os.path.join(os.getcwd(), 'results', 'global_training_models', 
                                                      str(self.Imputation_model), str(self.Lesson_Id), str(self.sparsity_level))
        os.makedirs(self.save_path_global_training, exist_ok=True)

    # -------------------------------
    # Loss Functions
    # -------------------------------
    def compute_generator_error_loss(self, fake_out, fake_cont_out1, fake_cont_out2, 
                                       cont_code1, cont_code2, train_set_tensor, fake_data):
        g_loss_adv = -tf.reduce_mean(tf.math.log(fake_out + 1e-8))
        cont1_loss = tf.reduce_mean(tf.square(fake_cont_out1 - cont_code1))
        cont2_loss = tf.reduce_mean(tf.square(fake_cont_out2 - cont_code2))
        mse_loss = tf.reduce_mean(tf.square(train_set_tensor - fake_data))
        rmse_loss = tf.sqrt(mse_loss)
        g_loss = g_loss_adv + self.alpha * rmse_loss + cont1_loss + cont2_loss
        return g_loss

    def compute_discriminator_error_loss(self, real_out, real_cont_out1, real_cont_out2,
                                           fake_out, cont_code1, cont_code2):
        d_loss_real = -tf.reduce_mean(tf.math.log(real_out + 1e-8))
        d_loss_fake = -tf.reduce_mean(tf.math.log(1. - fake_out + 1e-8))
        cont1_loss = tf.reduce_mean(tf.square(real_cont_out1 - cont_code1))
        cont2_loss = tf.reduce_mean(tf.square(real_cont_out2 - cont_code2))
        d_loss = d_loss_real + d_loss_fake + cont1_loss + cont2_loss
        return d_loss

    def compute_generator_divergence_loss(self, fake_out, fake_cont_out1, fake_cont_out2, 
                                          cont_code1, cont_code2, train_set_tensor, fake_data):
        wasserstein_loss = -tf.reduce_mean(fake_out)
        cont1_loss = tf.reduce_mean(tf.square(fake_cont_out1 - cont_code1))
        cont2_loss = tf.reduce_mean(tf.square(fake_cont_out2 - cont_code2))
        mse_loss = tf.reduce_mean(tf.square(train_set_tensor - fake_data))
        rmse_loss = tf.sqrt(mse_loss)
        g_loss = wasserstein_loss + self.alpha * rmse_loss + cont1_loss + cont2_loss
        return g_loss

    def compute_discriminator_divergence_loss(self, real_out, real_cont_out1, real_cont_out2,
                                              fake_out, cont_code1, cont_code2):
        d_loss_adv = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
        cont1_loss = tf.reduce_mean(tf.square(real_cont_out1 - cont_code1))
        cont2_loss = tf.reduce_mean(tf.square(real_cont_out2 - cont_code2))
        d_loss = d_loss_adv + cont1_loss + cont2_loss
        return d_loss

    # -------------------------------
    # Evaluation Metrics
    # -------------------------------
    def compute_rmse(self, true, pred, mask):
        epsilon = 1e-8
        squared_error = tf.square(true - pred) * mask
        mse = tf.reduce_sum(squared_error) / (tf.reduce_sum(mask) + epsilon)
        rmse = tf.sqrt(mse)
        return rmse.numpy()
    
    def compute_mae(self, true, pred, mask):
        epsilon = 1e-8
        absolute_error = tf.abs(true - pred) * mask
        mae = tf.reduce_sum(absolute_error) / (tf.reduce_sum(mask) + epsilon)
        return mae.numpy()

    def compute_rse(self, true, pred, mask):
        epsilon = 1e-8
        masked_true = true * mask
        true_mean = tf.reduce_sum(masked_true) / (tf.reduce_sum(mask) + epsilon)
        total_variance = tf.reduce_sum(tf.square(masked_true - true_mean)) / (tf.reduce_sum(mask) + epsilon)
        mse = tf.reduce_sum(tf.square((true - pred) * mask)) / (tf.reduce_sum(mask) + epsilon)
        rse = mse / (total_variance + epsilon)
        return rse.numpy()

    def compute_auc(self, true, pred, mask):
        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(true * mask, pred * mask)
        return auc_metric.result().numpy()
    
    def compute_cross_entropy(self, true, pred, mask):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true * mask, logits=pred * mask)
        return tf.reduce_mean(loss).numpy()

    # -------------------------------
    # Save Metrics Functions
    # -------------------------------
    def save_epoch_metrics(self, epoch_metrics, fold, fold_save_path):
        csv_path = os.path.join(fold_save_path, f'epoch_metrics_fold_{fold+1}.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'val_rmse', 'val_mae', 'val_rse', 'val_auc', 'val_cross_entropy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in epoch_metrics:
                writer.writerow(metrics)
        print(f"Saved epoch metrics for fold {fold+1} to: {csv_path}")

    def save_fold_best_metrics(self, fold_best_metrics_list):
        csv_path = os.path.join(self.save_path_global_training, 'global_training_best_metrics_all_folds.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['fold', 'best_epoch', 'best_rmse', 'best_mae', 'best_rse', 'best_auc', 'best_cross_entropy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in fold_best_metrics_list:
                writer.writerow(metrics)
        print(f"Saved best metrics for all folds to: {csv_path}")

    # -------------------------------
    # Training Step
    # -------------------------------
    def train_step(self, train_batch, generator, discriminator, g_optimizer, d_optimizer):
        train_set_tensor = tf.cast(train_batch, tf.float32)
        mask_train_x = 1 - tf.cast(tf.math.is_nan(train_set_tensor), tf.float32)
        
        if self.NaTransfer == 'zeros':
            train_set_tensor = tf.where(tf.math.is_nan(train_set_tensor), tf.zeros_like(train_set_tensor), train_set_tensor)
        else:
            train_set_tensor = tf.where(tf.math.is_nan(train_set_tensor), -tf.ones_like(train_set_tensor), train_set_tensor)
        
        preparation = InfoGANInputPreparation(self.noise_dim, self.cont_code1_dim, self.cont_code2_dim)
        combined_input = preparation([train_set_tensor, tf.shape(train_set_tensor)[0]])
        noise, cont_code1, cont_code2 = combined_input
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_data = generator(combined_input, training=True)
            d_fake, fake_cont_out1, fake_cont_out2 = discriminator(fake_data, training=True)
            d_real, real_cont_out1, real_cont_out2 = discriminator(train_set_tensor, training=True)
            
            if self.loss_function_type == "error_loss": 
                g_loss = self.compute_generator_error_loss(fake_out=d_fake, fake_cont_out1=fake_cont_out1, fake_cont_out2=fake_cont_out2,
                                                          cont_code1=cont_code1, cont_code2=cont_code2,
                                                          train_set_tensor=train_set_tensor, fake_data=fake_data)
                d_loss = self.compute_discriminator_error_loss(real_out=d_real, real_cont_out1=real_cont_out1, real_cont_out2=real_cont_out2,
                                                              fake_out=d_fake, cont_code1=cont_code1, cont_code2=cont_code2)
            elif self.loss_function_type == "divergence_loss": 
                g_loss = self.compute_generator_divergence_loss(fake_out=d_fake, fake_cont_out1=fake_cont_out1, fake_cont_out2=fake_cont_out2,
                                                               cont_code1=cont_code1, cont_code2=cont_code2,
                                                               train_set_tensor=train_set_tensor, fake_data=fake_data)
                d_loss = self.compute_discriminator_divergence_loss(real_out=d_real, real_cont_out1=real_cont_out1, real_cont_out2=real_cont_out2,
                                                                   fake_out=d_fake, cont_code1=cont_code1, cont_code2=cont_code2)
        
        gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
        disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        g_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        
        return {'g_loss': g_loss, 'd_loss': d_loss}, {'fake_data': fake_data}

    # -------------------------------
    # Training and Evaluation per Fold
    # -------------------------------
    def train_and_evaluate_fold(self, fold, train_index, val_index, early_stopping_patience):
        print(f"Parallel Processing Fold {fold+1}/5")
        train_sparse, val_sparse = self.sparse_tensor[train_index], self.sparse_tensor[val_index]
        train_dense, val_dense = self.dense_tensor[train_index], self.dense_tensor[val_index]
        
        train_sparse = tf.expand_dims(tf.convert_to_tensor(train_sparse, dtype=tf.float32), axis=1)
        val_sparse = tf.expand_dims(tf.convert_to_tensor(val_sparse, dtype=tf.float32), axis=1)
        train_dense = tf.convert_to_tensor(train_dense, dtype=tf.float32)
        val_dense = tf.convert_to_tensor(val_dense, dtype=tf.float32)
        
        train_dataset = tf.data.Dataset.from_tensor_slices(train_sparse)\
                                        .batch(self.batch_size)\
                                        .cache()\
                                        .prefetch(tf.data.AUTOTUNE)
        
        generator_global = InfoGAN_Standard_Generator(
            channel=train_sparse.shape[-1],
            output_size=np.prod(train_sparse.shape[1:]),
            input_shape=train_sparse.shape[1:]
        )
        discriminator_global = InfoGAN_Standard_Discriminator(self.cont_code1_dim, self.cont_code2_dim)
        
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        fold_save_path = os.path.join(self.save_path_global_training, f"fold_{fold+1}")
        os.makedirs(fold_save_path, exist_ok=True)
        
        best_rmse = float('inf')
        no_improve_epochs = 0
        epoch_metrics = []
        best_metrics_record = None
        
        fold_start_time = time.time()
        
        for epoch in tqdm(range(self.global_max_iter), desc=f"Fold {fold+1} Training Progress"):
            for batch in train_dataset:
                self.train_step(batch, generator_global, discriminator_global, g_optimizer, d_optimizer)
            
            val_dataset = tf.data.Dataset.from_tensor_slices(val_sparse).batch(self.batch_size)
            val_generated_data = []
            for val_batch in val_dataset:
                batch_size_val = tf.shape(val_batch)[0]
                dummy_original = tf.zeros_like(val_batch)
                preparation_val = InfoGANInputPreparation(self.noise_dim, self.cont_code1_dim, self.cont_code2_dim)
                combined_input_val = preparation_val([dummy_original, batch_size_val])
                pred_val = generator_global(combined_input_val, training=False)
                val_generated_data.append(pred_val)
            
            full_val_pred = tf.concat(val_generated_data, axis=0)
            val_pred = tf.cast(tf.squeeze(full_val_pred, axis=1), tf.float32)
            mask_val_dense = tf.squeeze(tf.cast(tf.math.is_nan(val_sparse), tf.float32), axis=1)
            
            val_rmse = self.compute_rmse(val_dense, val_pred, mask_val_dense)
            val_mae = self.compute_mae(val_dense, val_pred, mask_val_dense)
            val_rse = self.compute_rse(val_dense, val_pred, mask_val_dense)
            val_auc = self.compute_auc(val_dense, val_pred, mask_val_dense)
            val_entropy = self.compute_cross_entropy(val_dense, val_pred, mask_val_dense)
            
            current_epoch_metrics = {
                'epoch': epoch + 1,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_rse': val_rse,
                'val_auc': val_auc,
                'val_cross_entropy': val_entropy
            }
            print("Epoch metrics:", current_epoch_metrics)
            epoch_metrics.append(current_epoch_metrics)
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_metrics_record = {
                    'fold': fold + 1,
                    'best_epoch': epoch + 1,
                    'best_rmse': val_rmse,
                    'best_mae': val_mae,
                    'best_rse': val_rse,
                    'best_auc': val_auc,
                    'best_cross_entropy': val_entropy
                }
                generator_global.save_weights(os.path.join(fold_save_path, 'generator_global_weights.h5'))
                discriminator_global.save_weights(os.path.join(fold_save_path, 'discriminator_global_weights.h5'))
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= early_stopping_patience:
                tqdm.write(f"Early stopping at epoch {epoch+1} for fold {fold+1}")
                break
        
        print("Best RMSE for fold", fold+1, "is", best_rmse)
        self.save_epoch_metrics(epoch_metrics, fold, fold_save_path)
        fold_elapsed_time = time.time() - fold_start_time
        
        # Save hyperparameters and training parameters.
        params = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'global_max_iter': self.global_max_iter,
            'hint_rate': self.hint_rate,
            'alpha': self.alpha,
            'noise_dim': self.noise_dim,
            'cont_code1_dim': self.cont_code1_dim,
            'cont_code2_dim': self.cont_code2_dim,
            'NaTransfer': self.NaTransfer,
            'GAIN_neworks': self.GAIN_neworks,
            'loss_function_type': self.loss_function_type
        }
        save_model_parameters(params, fold_elapsed_time, fold_save_path)
        
        # Save parameter counts (with total count) for generator and discriminator.
        save_model_parameter_counts(generator_global, os.path.join(fold_save_path, 'generator_parameter_counts.csv'))
        save_model_parameter_counts(discriminator_global, os.path.join(fold_save_path, 'discriminator_parameter_counts.csv'))
        
        return best_metrics_record

    def global_training_cv(self, early_stopping_patience=5):
        print("Starting Parallel Global Training with 5-fold CV")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_best_metrics_list = []
        for fold, (train_index, val_index) in enumerate(kf.split(self.sparse_tensor)):
            best_metrics = self.train_and_evaluate_fold(fold, train_index, val_index, early_stopping_patience)
            fold_best_metrics_list.append(best_metrics)

        best_fold_idx = np.argmin([metrics['best_rmse'] for metrics in fold_best_metrics_list])
        print(f"Best fold overall: {best_fold_idx+1} with RMSE: {fold_best_metrics_list[best_fold_idx]['best_rmse']}")

        best_fold_save_path = os.path.join(self.save_path_global_training, f"fold_{best_fold_idx+1}")

        shutil.copy(
            os.path.join(best_fold_save_path, 'generator_global_weights.h5'),
            os.path.join(self.save_path_global_training, 'best_generator_global_weights.h5')
        )
        shutil.copy(
            os.path.join(best_fold_save_path, 'discriminator_global_weights.h5'),
            os.path.join(self.save_path_global_training, 'best_discriminator_global_weights.h5')
        )
        print(f"Best weights saved from fold {best_fold_idx+1} to global directory.")

        self.save_fold_best_metrics(fold_best_metrics_list)

        return fold_best_metrics_list[best_fold_idx]['best_rmse']
    
    def RunModel(self):
        print("AmbientGAN Model Start")
        print("Current path is", os.getcwd())
        print("Sparse tensor shape:", self.sparse_tensor.shape)
        print("Dense tensor shape:", self.dense_tensor.shape)
        
        final_val_rmse = self.global_training_cv(early_stopping_patience=5)
        print("Final global training RMSE with default hyperparameters:", final_val_rmse)