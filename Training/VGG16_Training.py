# 📦 Imports
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from skimage.transform import resize

# ⚙️ Patch Generator with Augmentation & On-the-Fly Resizing
class PatchGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, augment=False, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(X))
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        ) if augment else None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = np.array([resize(self.X[i], (224, 224), preserve_range=True) for i in batch_indices])
        y_batch = self.y[batch_indices]
        if self.augment:
            X_batch = next(self.datagen.flow(X_batch, batch_size=len(X_batch), shuffle=False))
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# 📈 Model Definition using VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze base layers

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 🎯 Training
train_gen = PatchGenerator(X_train, y_train, batch_size=32, augment=True)
val_gen = PatchGenerator(X_val, y_val, batch_size=32, augment=False)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[early_stop]
)
model.save("/content/drive/MyDrive/land-cover-classification-master/vgg16_updated_model.h5")
