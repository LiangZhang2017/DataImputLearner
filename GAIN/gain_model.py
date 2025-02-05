
from helper import Partition, k_fold_split, generate_hints
from GAIN.GAIN_algorithm import GAIN_Gennerator_CNN, GAIN_Discriminator_CNN
import tensorflow as tf

class GAIN:
    def __init__(self, sparse_tensor, dense_tensor, parameters):

        # self.embedding_value = None
        self.output_size = None
        self.train_tensor = None
        self.sparse_tensor = sparse_tensor
        self.dense_tensor = dense_tensor
        self.parameters = parameters
        self.slice_size=1
        self.batch_size=10
        self.learning_rate=0.00001
        self.max_iter=10
        
        self.hint_rate=0.9
        self.alpha=1
        self.h_dim = 6
        
        self.GAIN_neworks="CNN"
    
    def compute_generator_loss(self, d_fake, mask_train_x, G_sample, train_set_tensor):
        # Generator tries to fool discriminator so we want the discriminator to output 1 for fake data
        g_loss_temp = -tf.reduce_mean((1 - mask_train_x) * tf.math.log(d_fake + 1e-8))

        # Mean Squared Error loss for the generator
        mse_loss = tf.reduce_mean((mask_train_x * train_set_tensor - mask_train_x*G_sample) ** 2) / tf.reduce_mean(mask_train_x)
        rmse_loss = tf.sqrt(mse_loss)

        # Total generator loss: adversarial loss + alpha * reconstruction loss
        g_loss = g_loss_temp + self.alpha * rmse_loss

        return g_loss

    def compute_discriminator_loss(self, d_real, d_fake):
        # Log loss for real data, discriminator should output 1 for real data
        real_loss = -tf.reduce_mean(tf.math.log(d_real + 1e-8))

        # Log loss for fake data, discriminator should output 0 for fake data
        fake_loss = -tf.reduce_mean(tf.math.log(1. - d_fake + 1e-8))

        # Total discriminator loss
        d_loss = real_loss + fake_loss
        return d_loss

    def compute_mae(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        mae = tf.reduce_mean(tf.abs(true_data[mask] - predicted_data[mask]))
        return mae.numpy()

    def compute_rmse(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        mse = tf.reduce_mean(tf.square(true_data[mask] - predicted_data[mask]))
        rmse = tf.sqrt(mse)
        return rmse.numpy()

    def compute_rse(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        true_mean = tf.reduce_mean(true_data[mask])
        total_variance = tf.reduce_mean(tf.square(true_data[mask] - true_mean))
        mse = tf.reduce_mean(tf.square(true_data[mask] - predicted_data[mask]))
        rse = mse / total_variance
        return rse.numpy()

    def compute_auc(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(true_data[mask], predicted_data[mask])
        return auc_metric.result().numpy()

    def compute_cross_entropy(self, true_data, predicted_data):
        mask = ~tf.math.is_nan(true_data)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_data[mask], logits=predicted_data[mask]))
        return cross_entropy.numpy()
    
    def train_step(self, train_batch, generator, discriminator, g_optimizer, d_optimizer):
        # Convert types as necessary
        train_set_tensor = tf.cast(train_batch, tf.float32)

        # Compute the mask based on NaN values in the tensor
        mask_train_x = 1 - tf.cast(tf.math.is_nan(train_set_tensor), tf.float32)

        # Replace NaNs with zeros in the input tensor for processing
        train_set_tensor = tf.where(tf.math.is_nan(train_set_tensor), tf.zeros_like(train_set_tensor), train_set_tensor)

        # Generate noise
        noise_z = tf.random.uniform(shape=train_set_tensor.shape, minval=0, maxval=0.01)
        noise_X = mask_train_x * train_set_tensor + (1 - mask_train_x) * noise_z  # Noise Matrix

        # Generate hints
        hints_train = generate_hints(mask_train_x, hint_rate=self.hint_rate)
        hints_train_tensor = tf.convert_to_tensor(hints_train, dtype=tf.float32)

        # Prepare inputs for the generator
        generator_inputs = [noise_X, mask_train_x]
        
        # print("generator_inputs shape is ", generator_inputs.shape)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate synthetic data with generator
            G_sample = generator(generator_inputs, training=True)

            fake_data = mask_train_x * train_set_tensor + (1 - mask_train_x) * G_sample

            # Predict with discriminator
            d_fake = discriminator(fake_data, hints_train_tensor, training=True)
            d_real = discriminator(train_set_tensor, hints_train_tensor, training=True)

            # Calculate losses
            g_loss = self.compute_generator_loss(d_fake, mask_train_x, G_sample, train_set_tensor)
            d_loss = self.compute_discriminator_loss(d_real, d_fake)

        # Compute gradients
        gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
        disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)

        # Apply gradients
        g_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        return {'g_loss': g_loss.numpy(), 'd_loss': d_loss.numpy()}, {'G_sample': G_sample.numpy(), 'fake_data': fake_data.numpy()}

        
    def RunModel(self):
        
        print("Model Start")
        
        print(self.sparse_tensor.shape)
        print(self.dense_tensor.shape)
        
        all_slice_tensors, output_size_tf = Partition(self.sparse_tensor, slice_size=self.slice_size, mode="Average", filter='normal')
        
        print("slice_tensors generated")
        print(all_slice_tensors[1].shape)
        print(len(all_slice_tensors)) 
        self.output_size = int(output_size_tf.numpy())
        
        print("self.output_size is ", self.output_size)
        
        # Select first 50 slices for training (indices 0–49)
        train_sub_tensors = all_slice_tensors[:50]
        test_sub_tensor = all_slice_tensors[50]
        
        train_set_tensor_sparse = tf.convert_to_tensor(train_sub_tensors)
        test_set_tensor_sparse = tf.convert_to_tensor(test_sub_tensor)
        
        print("train_set_tensor_sparse shape is ", train_set_tensor_sparse.shape)
        print("test_set_tensor_sparse shape is ", test_set_tensor_sparse.shape)
        
        input_shape = [train_set_tensor_sparse.shape[1], train_set_tensor_sparse.shape[2], train_set_tensor_sparse.shape[3]]
        
        print("input_shape is ", input_shape)
        print("train_set_tensor_sparse.shape[-1] is ", train_set_tensor_sparse.shape[-1])
        
        train_set_tensor = tf.where(tf.math.is_nan(train_set_tensor_sparse), tf.zeros_like(train_set_tensor_sparse), train_set_tensor_sparse)
        train_set_tensor = tf.cast(train_set_tensor, tf.float32) 

        test_set_tensor = tf.where(tf.math.is_nan(test_set_tensor_sparse), tf.zeros_like(test_set_tensor_sparse), test_set_tensor_sparse)
        test_set_tensor = tf.cast(test_set_tensor, tf.float32)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_set_tensor).batch(self.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_set_tensor).batch(self.batch_size)
        
        print("self.output_size is ", self.output_size)
        
        if self.GAIN_neworks=='CNN':
            
            print("CNN type")
            generator = GAIN_Gennerator_CNN(channel=train_set_tensor_sparse.shape[-1], output_size=self.output_size, input_shape=input_shape)
            discriminator = GAIN_Discriminator_CNN()
            
            g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)
            d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_2=0.5)

            # Initialize performance storage
            train_perf = []
            test_perf = []

            # Early stopping parameters
            patience = 5
            epochs_since_last_improvement = 0
            best_rmse = 0.1
            best_weights = None
            improvement_flag = False  # Flag to indicate if an improvement was mad

            for epoch in range(self.max_iter):
                print("epoch is {}".format(epoch))

                train_generated_data=[]

                ori_train_tensor=tf.cast(tf.squeeze(train_set_tensor_sparse,axis=1), tf.float32)
                ori_test_tensor=tf.cast(tf.squeeze(train_set_tensor_sparse,axis=1), tf.float32)

                for train_batch in train_dataset:
                    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                        train_loss, train_generate = self.train_step(train_batch, generator, discriminator, g_optimizer, d_optimizer)
                        train_generated_data.append(train_generate['G_sample'])

                full_generated_train_data = tf.concat(train_generated_data, axis=0)
                full_generated_train_data = tf.cast(tf.squeeze(full_generated_train_data, axis=1), tf.float32)

                train_mae = self.compute_mae(ori_train_tensor, full_generated_train_data)
                train_rmse = self.compute_rmse(ori_train_tensor, full_generated_train_data)
                train_rse = self.compute_rse(ori_train_tensor, full_generated_train_data)
                train_auc = self.compute_auc(ori_train_tensor, full_generated_train_data)
                train_cross_entropy = self.compute_cross_entropy(ori_train_tensor, full_generated_train_data)

                print("mae: ", train_mae, "rmse: ", train_rmse, "rse: ", train_rse, "auc: ", train_auc, "cross_entropy: ", train_cross_entropy)
                
                