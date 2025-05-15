import gradio as gr
import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load class mappings
indian_class_df = pd.read_csv("dataset/AVIRIS_IndianPine_Site3_classes.csv")
indian_class_mapping = {row['class'] - 1: row['class_name'] for _, row in indian_class_df.iterrows() if row['class'] != 0}
salinas_class_mapping = {
    0: "Broccoli", 1: "Broccoli_Raab", 2: "Celery", 3: "Corn", 4: "Cotton",
    5: "Grapes", 6: "Peas", 7: "Potatoes", 8: "Soybeans", 9: "Squash",
    10: "Tomatoes", 11: "Vineyard", 12: "Wheat", 13: "Weeds", 14: "Cabbage",
    15: "Lettuce"
}

# Load models
model_2d = load_model("Models/2D_CNN/2D_CNN_final_model.h5")
model_3d = load_model("Models/3D_CNN/3d_CNN_updated.h5")
model_salinas = load_model("Models/3D_CNN/3d_CNN_fine_tuned_salinas.h5")

# Load Indian Pines test data
with open("Test_data/2D_CNN/test_data.pkl", "rb") as f:
    X_test_2d, y_test_2d = pickle.load(f)
with open("Test_data/2D_CNN/test_coords.pkl", "rb") as f:
    coords_2d = pickle.load(f)

with open("Test_data/3D_CNN/test_data_3d_updated.pkl", "rb") as f:
    X_test_3d, y_test_3d = pickle.load(f)
with open("Test_data/3D_CNN/test_coords_3d_updated.pkl", "rb") as f:
    coords_3d = pickle.load(f)

# Load Salinas test data (uploaded to /mnt/data/)
with open("Test_data/3D_CNN/test_data_3d_salinas.pkl", "rb") as f:
    X_test_salinas, y_test_salinas = pickle.load(f)
with open("Test_data/3D_CNN/test_coords_3d_salinas.pkl", "rb") as f:
    coords_salinas = pickle.load(f)

# Load Indian Pines full RGB image
full_rgb = cv2.imread("dataset/indian_pines_rgb_clean.png")
full_rgb = cv2.cvtColor(full_rgb, cv2.COLOR_BGR2RGB)

# Load Salinas-A full RGB image
salinas_rgb = cv2.imread("dataset/salinasA_rgb.png")
salinas_rgb = cv2.cvtColor(salinas_rgb, cv2.COLOR_BGR2RGB)

# Utility functions
def get_rgb_image(patch, size=100):
    try:
        if patch.ndim == 2:
            patch = np.expand_dims(patch, axis=-1)
        if patch.shape[2] < 3:
            patch = np.repeat(patch, 3, axis=2)
        rgb = patch[:, :, :3]
        rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-6)
        rgb = (rgb * 255).astype(np.uint8)
        return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print("get_rgb_image error:", e)
        return np.zeros((size, size, 3), dtype=np.uint8)

def mark_on_full_image(coord, image, marker_size=6, color=(255, 0, 0)):
    annotated = image.copy()
    x, y = coord[1], coord[0]
    cv2.drawMarker(annotated, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=2)
    return annotated

# Prediction logic
def predict(index, model_choice):
    try:
        if model_choice == "2D CNN":
            patch = X_test_2d[index]
            patch_input = np.expand_dims(patch, axis=0)
            pred = model_2d.predict(patch_input)
            true_class = int(np.argmax(y_test_2d[index]))
            coord = coords_2d[index]
            patch_image = get_rgb_image(patch)
            mapping = indian_class_mapping
        elif model_choice == "3D CNN":
            patch = X_test_3d[index]
            patch_input = np.expand_dims(patch, axis=0)
            pred = model_3d.predict(patch_input)
            true_class = int(np.argmax(y_test_3d[index]))
            coord = coords_3d[index]
            patch_image = get_rgb_image(patch[:, :, :, 0])
            mapping = indian_class_mapping
        else:
            raise ValueError("Invalid model selection.")

        pred_class = int(np.argmax(pred, axis=1)[0])
        marked_image = mark_on_full_image(coord, full_rgb)
        marked_image_resized = cv2.resize(marked_image, (300, 300), interpolation=cv2.INTER_AREA)

        return (
            patch_image,
            pred_class,
            mapping.get(pred_class, "Unknown"),
            true_class,
            mapping.get(true_class, "Unknown"),
            marked_image_resized
        )
    except Exception as e:
        print("Prediction error:", e)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        return (blank, -1, f"Error: {e}", -1, "Error", blank)

def predict_salinas(index):
    try:
        patch = X_test_salinas[index]
        patch_input = np.expand_dims(patch, axis=0)
        pred = model_salinas.predict(patch_input)
        true_class = int(np.argmax(y_test_salinas[index]))
        coord = coords_salinas[index]
        patch_image = get_rgb_image(patch[:, :, :, 0])
        pred_class = int(np.argmax(pred, axis=1)[0])

        # Mark the pointer on the Salinas RGB image
        if salinas_rgb is not None:
            marked_image_salinas = mark_on_full_image(coord, salinas_rgb)
        else:
            marked_image_salinas = np.zeros((300, 300, 3), dtype=np.uint8)

        marked_image_salinas_resized = cv2.resize(marked_image_salinas, (300, 300), interpolation=cv2.INTER_AREA)

        return (
            patch_image,
            pred_class,
            salinas_class_mapping.get(pred_class, "Unknown"),
            true_class,
            salinas_class_mapping.get(true_class, "Unknown"),
            marked_image_salinas_resized
        )
    except Exception as e:
        print("Salinas prediction error:", e)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        return (blank, -1, f"Error: {e}", -1, "Error", blank)

# UI
with gr.Blocks() as demo:
    gr.Markdown("# Hyperspectral Land Cover Classification")

    with gr.Tabs():
        with gr.Tab("Indian Pines (2D & 3D CNN)"):
            gr.Markdown("Select a patch index and model to view prediction and its location on the full image.")

            with gr.Row():
                with gr.Column(scale=1):
                    model_selector = gr.Dropdown(["2D CNN", "3D CNN"], label="Select Model", value="2D CNN")
                    index_slider = gr.Slider(minimum=0, maximum=len(X_test_2d)-1, step=1, label="Test Sample Index")
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", variant="primary")
                    full_image = gr.Image(label="Full Image with Patch Location")

                with gr.Column(scale=1.2):
                    patch_image = gr.Image(label="Input Patch (Simulated RGB)")
                    pred_idx = gr.Number(label="Predicted Class Index")
                    pred_class = gr.Text(label="Predicted Class")
                    true_idx = gr.Number(label="True Class Index")
                    true_class = gr.Text(label="True Class")

            submit_btn.click(fn=predict, inputs=[index_slider, model_selector], outputs=[
                patch_image, pred_idx, pred_class, true_idx, true_class, full_image
            ])

            clear_btn.click(lambda: [None]*6, outputs=[
                patch_image, pred_idx, pred_class, true_idx, true_class, full_image
            ])

            def update_slider(model_choice):
                return gr.update(maximum=(len(X_test_2d) - 1 if model_choice == "2D CNN" else len(X_test_3d) - 1))

            model_selector.change(fn=update_slider, inputs=model_selector, outputs=index_slider)

        with gr.Tab("Salinas-A (Fine-Tuned 3D CNN)"):
            gr.Markdown("Prediction using fine-tuned 3D CNN on Salinas-A dataset.")

            with gr.Row():
                with gr.Column(scale=1):
                    index_slider_salinas = gr.Slider(minimum=0, maximum=len(X_test_salinas)-1, step=1, label="Test Sample Index")
                    with gr.Row():
                        clear_btn_salinas = gr.Button("Clear")
                        submit_btn_salinas = gr.Button("Submit", variant="primary")
                    full_image_salinas = gr.Image(label="Full Image with Patch Location")

                with gr.Column(scale=1.2):
                    patch_image_salinas = gr.Image(label="Input Patch (Simulated RGB)")
                    pred_idx_salinas = gr.Number(label="Predicted Class Index")
                    pred_class_salinas = gr.Text(label="Predicted Class")
                    true_idx_salinas = gr.Number(label="True Class Index")
                    true_class_salinas = gr.Text(label="True Class")

            submit_btn_salinas.click(fn=predict_salinas, inputs=[index_slider_salinas], outputs=[
                patch_image_salinas, pred_idx_salinas, pred_class_salinas, true_idx_salinas, true_class_salinas, full_image_salinas
            ])

            clear_btn_salinas.click(lambda: [None]*6, outputs=[
                patch_image_salinas, pred_idx_salinas, pred_class_salinas, true_idx_salinas, true_class_salinas, full_image_salinas
            ])

demo.launch()
