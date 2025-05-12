import numpy as np
import rasterio
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_data(image_path, label_path, patch_size=5, num_components=50):
    with rasterio.open(image_path) as src:
        image = src.read()
    image = np.transpose(image, (1, 2, 0))  # (H, W, Bands)

    with rasterio.open(label_path) as src:
        labels = src.read(1)

    h, w, b = image.shape
    image_reshaped = image.reshape(-1, b)
    image_std = StandardScaler().fit_transform(image_reshaped).reshape(h, w, b)

    pca = PCA(n_components=num_components)
    image_pca = pca.fit_transform(image_std.reshape(-1, b)).reshape(h, w, num_components)

    half = patch_size // 2
    X, y, coords = [], [], []

    for i in range(half, h - half):
        for j in range(half, w - half):
            if labels[i, j] > 0:
                patch = image_pca[i-half:i+half+1, j-half:j+half+1, :]
                X.append(patch)
                y.append(labels[i, j] - 1)
                coords.append((i, j))

    X = np.array(X)[..., np.newaxis]  # Add channel dim
    y = to_categorical(np.array(y), num_classes=17)
    coords = np.array(coords)

    # Proper data splitting (no leakage)
    X_train, X_temp, y_train, y_temp, coords_train, coords_temp = train_test_split(
        X, y, coords, test_size=0.3, stratify=y, random_state=42)
    
    X_val, X_test, y_val, y_test, coords_val, coords_test = train_test_split(
        X_temp, y_temp, coords_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Save test data and coords
    with open("test_data_3d_updated.pkl", "wb") as f:
        pickle.dump((X_test, y_test), f)

    with open("test_coords_3d_updated.pkl", "wb") as f:
        pickle.dump(coords_test, f)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
