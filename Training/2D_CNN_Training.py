import tensorflow as tf
from LULC_CNN_TF import LULC_CNN

cnn_model = LULC_CNN(
    input_shape=(PATCH_SIZE, PATCH_SIZE, NUM_COMPONENTS),
    n_classes=NUM_CLASSES,
    learning_rate=0.0005,
    batch_size=32,
    epochs=75
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# Train Model
history = cnn_model.fit(X_train, y_train, X_val=X_val, y_val=y_val, callbacks=[early_stopping, reduce_lr])

# Save Model
cnn_model.save_model("/content/drive/MyDrive/land-cover-classification-master/2D_CNN_final_model.h5")
