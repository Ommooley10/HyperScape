import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import pickle

# Load dataset
dataset_path = "/content/drive/MyDrive/land-cover-classification-master/data/19920612_AVIRIS_IndianPine_Site3.tif"
ground_truth_path = "/content/drive/MyDrive/land-cover-classification-master/data/19920612_AVIRIS_IndianPine_Site3_gr.tif"

with rasterio.open(dataset_path) as src:
    image = src.read()  # Shape: (220, H, W)

with rasterio.open(ground_truth_path) as src:
    labels = src.read(1)  # Shape: (H, W)

image = np.transpose(image, (1, 2, 0))  # (H, W, Bands)

# **Standardization Instead of Min-Max Normalization**
scaler = StandardScaler()
h, w, bands = image.shape
image = scaler.fit_transform(image.reshape(-1, bands)).reshape(h, w, bands)

# **PCA for Dimensionality Reduction**
NUM_COMPONENTS = 30  # Reduced from 50 to 30 for efficiency
pca = PCA(n_components=NUM_COMPONENTS)
image = pca.fit_transform(image.reshape(-1, bands)).reshape(h, w, NUM_COMPONENTS)

PATCH_SIZE = 5  # Adjusted patch size

# **Patch Extraction**
def extract_patches(image, labels, patch_size):
    half_patch = patch_size // 2
    h, w, bands = image.shape
    X, y = [], []

    for i in range(half_patch, h - half_patch):
        for j in range(half_patch, w - half_patch):
            if labels[i, j] > 0:  # Ignore unlabeled pixels
                patch = image[i - half_patch:i + half_patch + 1, j - half_patch:j + half_patch + 1, :]
                X.append(patch)
                y.append(labels[i, j] - 1)

    return np.array(X), np.array(y)

X, y = extract_patches(image, labels, PATCH_SIZE)

# Convert labels to categorical
NUM_CLASSES = 17
y = to_categorical(y, NUM_CLASSES)

# **Split into Train (70%), Validation (15%), and Test (15%)**
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# **Data Augmentation**
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),
])

# Augment training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# Save test data for later evaluation
with open("/content/drive/MyDrive/land-cover-classification-master/test_data.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")