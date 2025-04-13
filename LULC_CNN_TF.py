import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

class LULC_CNN(models.Model):
    def __init__(self, input_shape, n_classes, learning_rate=0.0005, dropout_rate=0.3, epochs=75, batch_size=32):
        super(LULC_CNN, self).__init__()

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Build CNN model
        self.model = models.Sequential([
            layers.Input(shape=self.input_shape),

            # **First Convolutional Block**
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.002)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2),

            # **Second Convolutional Block**
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.002)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),

            # **Third Convolutional Block**
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.002)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),

            # **Flatten Layer**
            layers.Flatten(),

            # **Fully Connected Layers**
            layers.Dense(256, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.BatchNormalization(),

            layers.Dense(self.n_classes, activation='softmax')
        ])

        # Compile Model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, X_train, y_train, X_val=None, y_val=None, callbacks=None):
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                              epochs=self.epochs, batch_size=self.batch_size, verbose=1, 
                              callbacks=callbacks)

    def save_model(self, path):
        self.model.save(path)
