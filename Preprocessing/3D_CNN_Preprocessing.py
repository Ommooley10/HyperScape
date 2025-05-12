# preprocess_3dcnn.py

import numpy as np
import rasterio
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Paths
image_path = "/content/drive/MyDrive/land-cover-classification-master/data/19920612_AVIRIS_IndianPine_Site3.tif"
label_path = "/content/drive/MyDrive/land-cover-classification-master/data/19920612_AVIRIS_IndianPine_Site3_gr.tif"

# Load hyperspectral image
with rasterio.open(image_path) as src:
    image = src.read()
image = np.transpose(image, (1, 2, 0))  # (H, W, Bands)

# Load ground truth labels
with rasterio.open(label_path) as src:
    labels = src.read(1)  # (H, W)

# Standardization + PCA
h, w, b = image.shape
scaler = StandardScaler()
image_std = scaler.fit_transform(image.reshape(-1, b)).reshape(h, w, b)

NUM_COMPONENTS = 50
pca = PCA(n_components=NUM_COMPONENTS)
image_pca = pca.fit_transform(image_std.reshape(-1, b)).reshape(h, w, NUM_COMPONENTS)

# Patch Extraction
PATCH_SIZE = 5
half = PATCH_SIZE // 2
X, y, coords = [], [], []

for i in range(half, h - half):
    for j in range(half, w - half):
        if labels[i, j] > 0:
            patch = image_pca[i-half:i+half+1, j-half:j+half+1, :]
            X.append(patch)
            y.append(labels[i, j] - 1)
            coords.append((i, j))

X = np.array(X)[..., np.newaxis]  # Add channel dimension: (N, 5, 5, 50, 1)
y = to_categorical(np.array(y), num_classes=17)
coords = np.array(coords)

# Train/Val/Test Split
X_train, X_temp, y_train, y_temp, coords_train, coords_temp = train_test_split(
    X, y, coords, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test, coords_val, coords_test = train_test_split(
    X_temp, y_temp, coords_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Save only test data and coords
with open("/content/drive/MyDrive/land-cover-classification-master/test_data_3d_updated.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)

with open("/content/drive/MyDrive/land-cover-classification-master/test_coords_3d_updated.pkl", "wb") as f:
    pickle.dump(coords_test, f)

# Pass train and val data for training script
data = {
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val
}
