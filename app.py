import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load model (adjust path if needed)
model_path = "house_detection_best.pt"
model = YOLO(model_path)

# Streamlit UI
st.set_page_config(page_title="🏡 House Style Segmentation", layout="centered")
st.title("🏡 House Detection & Segmentation")
st.write("Upload one or more images of houses to detect and segment architectural features.")

uploaded_files = st.file_uploader("Upload House Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Run prediction
        results = model.predict(image_np, conf=0.25)
        r = results[0]

        # Copy original for mask overlay
        masked_image = image_np.copy()

        if r.masks is not None:
            for mask in r.masks.data:
                mask = mask.cpu().numpy()
                mask_resized = cv2.resize(mask, (masked_image.shape[1], masked_image.shape[0]))
                binary_mask = mask_resized > 0.5
                masked_image[binary_mask] = [255, 0, 0]  # Red mask

        # Convert to PIL for display
        masked_pil = Image.fromarray(masked_image)

        # Display side-by-side
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_column_width=True)
        col2.image(masked_pil, caption="Predicted Mask Overlay", use_column_width=True)

        # Display confidence(s)
        if r.boxes.conf is not None:
            st.subheader("Confidence Scores")
            for i, conf in enumerate(r.boxes.conf.cpu().numpy()):
                st.write(f"Object {i+1}: **{conf:.2%}**")
        else:
            st.warning("No predictions made.")
