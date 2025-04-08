import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class New_GAN_Gennerator_CNN(keras.Model):
    def __init__(self, channel, output_size, input_shape):
        super(New_GAN_Gennerator_CNN, self).__init__()

        # z: [b,100]=>[b,3*3*512]=>[b,3,3,512]=>[b,64,64,3]=>

        self.fc = layers.Dense(3 * 3 * 512)
        self.conv1 = layers.Conv2DTranspose(256, 3, 2, 'valid')
        self.bn1 = layers.BatchNormalization()
        self.dp1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.dp2 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2DTranspose(64, 4, 2, 'valid')
        self.bn3 = layers.BatchNormalization()
        self.dp3 = layers.Dropout(0.5)

        self.conv4 = layers.Conv2DTranspose(32, 4, 2, 'valid')
        self.bn4 = layers.BatchNormalization()
        self.dp4 = layers.Dropout(0.5)

        self.conv5 = layers.Conv2DTranspose(channel, 4, 2, 'valid')
        self.bn5 = layers.BatchNormalization()
        self.dp5 = layers.Dropout(0.5)

        self.fl = layers.Flatten()
        self.fc_output = layers.Dense(output_size)
        self.output_layer = layers.Reshape(target_shape=input_shape)

    def call(self, inputs, training=None, mask=None, output_shape=None):
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        x = self.conv1(x)
        x = tf.nn.leaky_relu(self.dp1(self.bn1(x, training=training)))

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
        # x = self.dp5(self.bn5(x, training=training))
        x = tf.nn.leaky_relu(x)

        x = self.fl(x)
        x = self.fc_output(x)
        x = self.output_layer(x)

        x = tf.sigmoid(x)

        return x


class New_GAN_Discriminator_CNN(keras.Model):
    def __init__(self):
        super(New_GAN_Discriminator_CNN, self).__init__()
        # z: [b,1,9,9]=>[b,1]

        self.conv1 = layers.Conv2D(32, 1, 2, 'valid', activation="relu")
        self.conv2 = layers.Conv2D(64, 1, 2, 'valid', activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.dp2 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2D(128, 1, 2)
        self.bn3 = layers.BatchNormalization()
        self.dp3 = layers.Dropout(0.5)

        self.conv4 = layers.Conv2D(256, 1, 2)
        self.bn4 = layers.BatchNormalization()
        self.dp4 = layers.Dropout(0.5)

        self.conv5 = layers.Conv2D(512, 1, 2)
        self.bn5 = layers.BatchNormalization()
        self.dp5 = layers.Dropout(0.5)

        # [b,h,w,3]= [b,-1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)

        x = self.conv2(x)
        x = self.dp2(self.bn2(x, training=training))

        x = self.conv3(x)
        x = self.dp3(self.bn3(x, training=training))

        x = self.conv4(x)
        x = self.dp4(self.bn4(x, training=training))

        x = self.conv5(x)
        # x = self.dp5(self.bn5(x, training=training))

        # [b,h,w,c]=>[b,-1]
        x = self.flatten(x)
        logits = tf.sigmoid(self.fc(x))

        return logits
