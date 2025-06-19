# HyperScape 🚀  
**Hyperspectral Image Classification using Deep Learning**

HyperScape is a deep learning-based project that focuses on classifying hyperspectral images (HSI) by capturing both **spectral and spatial features**. The project explores multiple CNN architectures including a **custom 2D CNN**, **VGG16 with transfer learning**, and a **3D CNN** to achieve high accuracy and generalization across datasets like **Indian Pines** and **Salinas-A**.

---

## 📌 Features

-  Custom-built 2D CNN architecture  
-  Transfer learning with VGG16 using PCA-reduced HSI data  
-  3D CNN for volumetric spectral-spatial analysis  
-  Preprocessing techniques: Band selection, patch extraction, normalization, PCA  
-  Evaluation on Indian Pines; generalization tested on Salinas-A dataset  
-  EarlyStopping and learning rate scheduling for stable training  
-  H5 model and pickle file saving for future inference

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

## Deployment

This project is deployed on **Hugging Face Spaces** using **Gradio** as the frontend framework.

### 🔗 Live App

👉 [Click here to launch HyperScape](https://huggingface.co/spaces/Ommooley10/HyperScape)

### 📁 Files in Deployment

- `app.py` – Main Python script containing the Gradio interface and model inference logic.
- `requirements.txt` – Lists all required Python dependencies.
- `Models/` – Pretrained 2D and 3D CNN models (e.g., for Indian Pines, Salinas-A).
- `dataset/`- Containing the datasets and their respective RGB images.
- `Test_data/` – Test set data and coordinate points.
- `utils/` – Helper scripts for preprocessing, visualization, and prediction.

### Tech Stack

- **Frontend**: Gradio
- **Backend**: Python (TensorFlow/Keras, NumPy, SciPy, OpenCV, Matplotlib)
- **Platform**: Hugging Face Spaces

### How to Deploy on Hugging Face Spaces

To deploy this app on your own Hugging Face Space:

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Gradio** as the SDK.
3. Upload the following files to your Space:
   - `app.py`
   - `requirements.txt`
   - Any model/data files your app depends on
4. Click **"Commit"** and the app will build and launch automatically.

Once deployed, the app will be accessible via a public URL and automatically re-run on any code or file updates.

### Status

-  Public and live
-  Supports multiple datasets and models
-  Provides real-time predictions and visualization

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

Suggestions and improvements are welcome!! 🙂
For queries, suggestions, or collaborations, feel free to reach out via GitHub Issues or email.

