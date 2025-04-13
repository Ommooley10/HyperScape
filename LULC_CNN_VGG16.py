import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

class LULC_VGG16:
    def __init__(self, input_shape, n_classes, learning_rate=0.001, dropout_rate=0.3, epochs=50, batch_size=16):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Input Layer
        inputs = layers.Input(shape=self.input_shape)

        # Convert 50 spectral bands → 3 channels using a 1x1 Conv Layer
        x = layers.Conv2D(3, (1, 1), activation='relu')(inputs)

        # Load VGG16 with pre-trained ImageNet weights (excluding top layers)
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
        base_model.trainable = False  # Freeze initial layers

        # Pass transformed input through VGG16
        x = base_model(x, training=False)

        # Flatten and add custom classification layers
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)

        # Define the model
        self.model = models.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, X_train, y_train, X_val=None, y_val=None, callbacks=None):
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                              epochs=self.epochs, batch_size=self.batch_size,
                              verbose=1, callbacks=callbacks)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
