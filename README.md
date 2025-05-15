# HyperScape 🚀  
**Hyperspectral Image Classification using Deep Learning**

HyperScape is a deep learning-based project that focuses on classifying hyperspectral images (HSI) by capturing both **spectral and spatial features**. The project explores multiple CNN architectures including a **custom 2D CNN**, **VGG16 with transfer learning**, and a **3D CNN** to achieve high accuracy and generalization across datasets like **Indian Pines** and **Salinas-A**.

---

## 📌 Features

- ✅ Custom-built 2D CNN architecture  
- ✅ Transfer learning with VGG16 using PCA-reduced HSI data  
- ✅ 3D CNN for volumetric spectral-spatial analysis  
- ✅ Preprocessing techniques: Band selection, patch extraction, normalization, PCA  
- ✅ Evaluation on Indian Pines; generalization tested on Salinas-A dataset  
- ✅ EarlyStopping and learning rate scheduling for stable training  
- ✅ H5 model and pickle file saving for future inference

---

## 🧠 Models Used

| Model         | Highlights                                                  |
|---------------|-------------------------------------------------------------|
| 2D CNN        | Built from scratch; flexible and efficient                  |
| VGG16         | Transfer learning with 3-channel PCA input                  |
| 3D CNN        | Captures detailed spatial + spectral features (Salinas-A)   |

---

## 📂 Datasets

- **Indian Pines**:  
  - Size: 145 × 145 pixels  
  - Bands: 224 (200 used)  
  - Classes: 16 land-cover types

- **Salinas-A** *(used for generalization)*:  
  - Size: 83 × 86 pixels  
  - Bands: 204  
  - Classes: 6 vegetation types

Both datasets were captured by the **AVIRIS** sensor.

---

## ⚙️ Preprocessing Steps

- Band removal (e.g., water absorption bands)
- Spectral normalization
- PCA for dimensionality reduction (for VGG)
- Patch extraction (e.g., 11×11 or 5×5×30)
- Data augmentation: rotation & flipping

---

## 🏗️ Training Details

- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Batch Size**: 32  
- **Epochs**: 75  
- **Callbacks**: EarlyStopping, ReduceLROnPlateau  
- **Model Output**: Saved as `.h5`  
- **Test Data**: Saved as `.pkl`

---

## 📊 Results

- **3D CNN** achieved the **highest accuracy**, generalizing well across unseen data.
- The **hybrid approach** (2D + 3D features) demonstrated strong potential for real-world application.
- Applied to **Hyperion satellite imagery** for urban land classification, supporting **smart city planning** and **environmental monitoring**.

---

| Model      | Accuracy (Indian Pines) | Accuracy (Salinas-A) | Notes                       |
| ---------- | ----------------------- | -------------------- | --------------------------- |
| 2D CNN     | \~93%                   | N/A                  | Good baseline performance   |
| VGG16 (TL) | \~90%                   | N/A                  | Fast training, PCA required |
| 3D CNN     | **\~95%**               | **\~96–97%**         | Best spatial-spectral model |

---
## 🔭 Future Work

- ✅ Feature fusion: Combine 2D and 3D features for robust classification  
- ✅ Apply on additional benchmark and real-world datasets  
- ✅ Scale to remote sensing applications in agriculture, city planning, and sustainability

---

## 🤝 Contributors

- [Ommooley10](https://github.com/Ommooley10) 
- [Vaishnavi Paswan](https://github.com/vaishnavipaswan)
- [Vedika Agrawal](https://github.com/vedikagrawal)  
- Laxmikant Dubey

---

## 📎 License

This project is open-source under the [MIT License](LICENSE).

---

## 📬 Contact

For queries, suggestions, or collaborations, feel free to reach out via GitHub Issues or email.

