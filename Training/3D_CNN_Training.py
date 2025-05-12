# train_3dcnn.py

import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess_3dcnn import data  # assuming both scripts are in the same folder

# Extract data
X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]

# Model architecture
def build_3d_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),

        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train model
model = build_3d_cnn(X_train.shape[1:], y_train.shape[1])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# Save trained model
model.save("/content/drive/MyDrive/land-cover-classification-master/Models/3D CNN/3d_CNN_updated.h5")
