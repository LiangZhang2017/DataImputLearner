import tensorflow as tf
from tensorflow import keras
from keras import layers

class GAIN_Gennerator_CNN(keras.Model):
    def __init__(self, channel, output_size, input_shape):
        super(GAIN_Gennerator_CNN, self).__init__()
        
        # We'll output eventually to shape = input_shape (e.g. [1, 28, 4])
        self.final_output_shape = input_shape  
        self.channel = channel
        self.output_size = output_size
        
        # --- Layers ---
        # Flatten + Dense
        self.fc = layers.Dense(3 * 3 * 512)
        
        # Conv2DTranspose layers
        self.conv1 = layers.Conv2DTranspose(256, 3, strides=2, padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.dp1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv2DTranspose(128, 5, strides=2, padding='valid')
        self.bn2 = layers.BatchNormalization()
        self.dp2 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2DTranspose(64, 4, strides=2, padding='valid')
        self.bn3 = layers.BatchNormalization()
        self.dp3 = layers.Dropout(0.5)

        self.conv4 = layers.Conv2DTranspose(32, 4, strides=2, padding='valid')
        self.bn4 = layers.BatchNormalization()
        self.dp4 = layers.Dropout(0.5)

        # Final transpose conv
        self.conv5 = layers.Conv2DTranspose(channel, 4, strides=2, padding='valid')
        self.bn5 = layers.BatchNormalization()
        self.dp5 = layers.Dropout(0.5)

        # Flatten and final Dense to match output_size if needed
        self.flatten = layers.Flatten()
        self.fc_output = layers.Dense(int(self.output_size))

        # Reshape to the final output shape, e.g., [1, 28, 4]
        self.output_layer = layers.Reshape(target_shape=self.final_output_shape)

    def call(self, inputs, training=None):
        """
        inputs = [train_set_tensor, mask], each of shape (batch, 1, 28, channels)
        """
        train_set_tensor, mask = inputs  # shapes = (b,1,28,4) each
        batch_size = tf.shape(train_set_tensor)[0]

        # Step 1: Concatenate along the last channel axis => (b,1,28,8)
        x = tf.concat([train_set_tensor, mask], axis=-1)

        # Step 2: Flatten to 2D => (b, 1*28*8)
        x = tf.reshape(x, [batch_size, -1])  # => (b, 224) if 1*28*8 = 224

        # Step 3: Pass through Dense => (b, 3*3*512) = (b, 4608)
        x = self.fc(x) 
        x = tf.nn.leaky_relu(x)

        # Step 4: Reshape to 4D => (b, 3, 3, 512)
        x = tf.reshape(x, [batch_size, 3, 3, 512])

        # ---- Now we can apply our conv2DTranspose layers ----
        x = self.conv1(x)       # => shape?
        x = self.dp1(self.bn1(x, training=training))
        x = tf.nn.leaky_relu(x)

        x = self.conv2(x)
        x = self.dp2(self.bn2(x, training=training))
        x = tf.nn.leaky_relu(x)

        x = self.conv3(x)
        x = self.dp3(self.bn3(x, training=training))
        x = tf.nn.leaky_relu(x)

        x = self.conv4(x)
        x = self.dp4(self.bn4(x, training=training))
        x = tf.nn.leaky_relu(x)

        x = self.conv5(x)
        # Optional dropout if needed
        # x = self.dp5(self.bn5(x, training=training))
        x = tf.nn.leaky_relu(x)

        # Flatten again => (b, ?)
        x = self.flatten(x)   
        # Final Dense => shape (b, output_size)
        x = self.fc_output(x)

        # Reshape to final output shape => (b, 1, 28, channels)
        x = self.output_layer(x)

        # Sigmoid output
        x = tf.sigmoid(x)

        return x

class GAIN_Discriminator_CNN(keras.Model):
    def __init__(self):
        super(GAIN_Discriminator_CNN, self).__init__()

        self.conv1 = layers.Conv2D(32, 1, 2, 'valid', activation="relu")
        self.conv2 = layers.Conv2D(64, 1, 2, 'valid', activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.dp2 = layers.Dropout(0.5)
        self.relu2 = layers.ReLU()

        self.conv3 = layers.Conv2D(128, 1, 2)
        self.bn3 = layers.BatchNormalization()
        self.dp3 = layers.Dropout(0.5)
        self.relu3 = layers.ReLU()

        self.conv4 = layers.Conv2D(256, 1, 2)
        self.bn4 = layers.BatchNormalization()
        self.dp4 = layers.Dropout(0.5)
        self.relu4 = layers.ReLU()

        self.conv5 = layers.Conv2D(512, 1, 2)
        self.bn5 = layers.BatchNormalization()
        self.dp5 = layers.Dropout(0.5)
        self.relu5 = layers.ReLU()

        # [b,h,w,3]= [b,-1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, hints, training=None, mask=None):

        x = tf.concat([inputs, hints], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dp2(self.bn2(x, training=training))
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.dp3(self.bn3(x, training=training))
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.dp4(self.bn4(x, training=training))
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.dp5(self.bn5(x, training=training))
        x = self.relu5(x)

        # [b,h,w,c]=>[b,-1]
        x = self.flatten(x)

        x=layers.Dense(inputs.shape[2]*inputs.shape[3])(x)

        x=layers.Reshape(target_shape=(inputs.shape[1],inputs.shape[2],inputs.shape[3]))(x)

        logits = tf.sigmoid(x)

        return logits