import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband
#from data_generator import NpyDataGenerator
from custom_generator2 import NpyDataGenerator as custom_gen
# Define Hyperparameters
img_height = 128
img_width = 128
num_frames = 16
num_channels = 1  # Grayscale videos
num_classes = 4
input_shape=  (128, 128, 16, 1)
class C3DHyperModel(HyperModel):
    def build(self, hp):
        input_shape = (img_height, img_width, num_frames, num_channels)
        
        # Hyperparameters
        num_conv_layers = hp.Int('num_conv_layers', min_value=2, max_value=6, step=1)
        dense_units = hp.Choice('dense_units', [128, 256, 512, 1024])
        dropout_rate_dense = hp.Float('dropout_rate_dense', min_value=0.0, max_value=0.5, step=0.1)
        
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Convolutional layers
        for i in range(num_conv_layers):
            dropout_rate = hp.Float(f'dropout_rate_{i+1}', min_value=0.0, max_value=0.3, step=0.05)
            num_filters = hp.Choice(f'num_filters_layer_{i+1}', [32, 64, 128])
            kernel_size = hp.Choice(f'kernel_size_layer_{i+1}', [3, 5, 7])
            pool_size = 2  # Fixed pooling size
            x = layers.Conv3D(
                filters=num_filters,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                activation='relu',
                padding='same'
            )(x)
            x = layers.MaxPooling3D(
                pool_size=(pool_size, pool_size, pool_size),
                padding='same'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
                
        # Flatten and add dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate_dense)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create the model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        optimizer_name = hp.Choice('optimizer', ['adam', 'SGD'])
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        return model

# Custom Tuner
class MyHyperband(Hyperband):
    def __init__(self, *args, **kwargs):
        super(MyHyperband, self).__init__(*args, **kwargs)
        self.best_score = None  # Initialize best score to None
        self.best_trial_id = None  # To keep track of the best trial

    def run_trial(self, trial, *args, **kwargs):
        # Build the model
        model = self.hypermodel.build(trial.hyperparameters)
        print(model.summary())

        # Prepare the callbacks
        callbacks = kwargs.get('callbacks', [])
        
        # Update the kwargs for model.fit()
        fit_kwargs = kwargs.copy()
        fit_kwargs['callbacks'] = callbacks

        # Call model.fit() directly
        history = model.fit(*args, **fit_kwargs)

        # Get the best validation accuracy from this trial
        val_accuracy = max(history.history['val_accuracy'])

        # Report the metrics to the tuner
        self.oracle.update_trial(
            trial.trial_id,
            {'val_accuracy': val_accuracy}
        )

        # Check if this is the best score so far
        if self.best_score is None or val_accuracy > self.best_score:
            self.best_score = val_accuracy
            self.best_trial_id = trial.trial_id
            print(f"New best score: {self.best_score}, saving model.")
            # Save the model
            self.save_model(model)
        else:
            print(f"Trial {trial.trial_id} did not improve the best score.")

        return history

    def save_model(self, model):
        # Overwrite the best model each time a better one is found
        model_dir = '3D_CNN_best_model'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.keras')
        model.save(model_path)

# Initialize the tuner using the custom MyHyperband class
tuner = MyHyperband(
    C3DHyperModel(),
    objective='val_accuracy',
    max_epochs=100,  # Adjust as needed
    factor=3,
    directory='3DCNN_hyperband',
    project_name='3DCNN_optimization',
    overwrite=False  # Set to False to resume previous tuning
)

train_path = ""
test_path = ""

train_gen = custom_gen(npy_dir=train_path,
                        batch_size=8,
                        input_shape=input_shape,
                        shuffle=True,
                        augment=False)

# train_gen_n=
validation_gen = custom_gen(npy_dir=test_path,
                        batch_size=1,
                        input_shape=input_shape,
                        shuffle=False,
                        augment=False)

# Callbacks
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.01,
    patience=5,
    verbose=1,
    min_delta=1e-6
)

# Run the hyperparameter search
tuner.search(
    train_gen,
    validation_data=validation_gen,
    epochs=100,  # Adjust the number of epochs as needed
    callbacks=[earlystop, reduce_lr]
)

# Retrieve the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters: ", best_hyperparameters.values)

# Load the best model
best_model_path = os.path.join('3DCNN_best_model_Visual', '3DCNN_best_model_Visual.keras')
best_model = tf.keras.models.load_model(best_model_path)

# Evaluate the best model
loss, accuracy = best_model.evaluate(validation_gen)
print(f"Validation accuracy of the best model: {accuracy}")
