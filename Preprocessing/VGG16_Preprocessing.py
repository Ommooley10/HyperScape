# 📦 Imports
import numpy as np
import rasterio
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from skimage.transform import resize
import os

# 📁 Paths
dataset_path = "/content/drive/MyDrive/land-cover-classification-master/data/19920612_AVIRIS_IndianPine_Site3.tif"
ground_truth_path = "/content/drive/MyDrive/land-cover-classification-master/data/19920612_AVIRIS_IndianPine_Site3_gr.tif"

# 📥 Load Hyperspectral Image
with rasterio.open(dataset_path) as src:
    image = src.read()  # (220, H, W)
image = np.transpose(image, (1, 2, 0))  # (H, W, 220)

# 📥 Load Labels
with rasterio.open(ground_truth_path) as src:
    labels = src.read(1)  # (H, W)

# ⚙️ Normalize
def normalize_image(image):
    h, w, bands = image.shape
    image_flat = image.reshape(-1, bands)
    image_scaled = MinMaxScaler().fit_transform(image_flat)
    return image_scaled.reshape(h, w, bands)

image = normalize_image(image)

# ⚙️ PCA: 220 → 3 channels for VGG16
def apply_pca(image, n_components=3):
    h, w, bands = image.shape
    image_reshaped = image.reshape(-1, bands)
    pca = PCA(n_components=n_components)
    image_pca = pca.fit_transform(image_reshaped)
    return image_pca.reshape(h, w, n_components)

image = apply_pca(image)

# 📦 Extract Patches
PATCH_SIZE = 15

def extract_patches(image, labels, patch_size):
    half = patch_size // 2
    h, w, bands = image.shape
    X, y, coords = [], [], []

    for i in range(half, h - half):
        for j in range(half, w - half):
            label = labels[i, j]
            if label > 0:
                patch = image[i - half:i + half + 1, j - half:j + half + 1, :]
                if patch.shape == (patch_size, patch_size, bands):
                    X.append(patch)
                    y.append(label - 1)
                    coords.append((i, j))

    print("✅ Total valid patches extracted:", len(X))
    return np.array(X), np.array(y), coords

X, y, coords = extract_patches(image, labels, PATCH_SIZE)

# 🔢 One-hot Encode Labels
NUM_CLASSES = len(np.unique(y))
y = to_categorical(y, NUM_CLASSES)

# 🔀 Train-Val-Test Split
X_temp, X_test, y_temp, y_test, coords_temp, coords_test = train_test_split(
    X, y, coords, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)
coords_train, coords_val = train_test_split(coords_temp, test_size=0.1765, random_state=42)

# 💾 Save test data
with open("/content/drive/MyDrive/land-cover-classification-master/test_data_vgg16.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)

with open("/content/drive/MyDrive/land-cover-classification-master/coords_test_vgg16.pkl", "wb") as f:
    pickle.dump(coords_test, f)