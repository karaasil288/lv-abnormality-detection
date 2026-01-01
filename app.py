import streamlit as st
import numpy as np
import cv2
from PIL import Image
from model_utils import load_model, preprocess_image, predict_mask, clean_mask, analyze_lv

# ===========================
# Page config
# ===========================
st.set_page_config(
    page_title="Fetal Brain Abnormality Detection",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Fetal Brain Abnormality Detection")
st.write("Upload an ultrasound image to analyze fetal brain structures and detect abnormalities.")

# ===========================
# Load UNet model
# ===========================
@st.cache_resource
def load_unet():
    return load_model("fetal_unet_model.h5")

model = load_unet()

# ===========================
# File upload
# ===========================
uploaded_file = st.file_uploader(
    "Upload ultrasound image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # ===========================
    # Preprocess and predict mask
    # ===========================
    input_image = preprocess_image(image_np)
    mask = predict_mask(model, input_image)
    mask = clean_mask(mask)

    # ===========================
    # Color mask and overlay
    # ===========================
    IMG_SIZE = mask.shape[0]
    COLORS = {1: (0, 0, 255), 2: (0, 255, 255), 3: (255, 0, 0)}
    mask_color = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    for cls, col in COLORS.items():
        mask_color[mask == cls] = col

    overlay = cv2.addWeighted(cv2.resize(image_np, (IMG_SIZE, IMG_SIZE)), 0.6, mask_color, 0.4, 0)

    # ===========================
    # Tabs for UI like Colab layout
    # ===========================
    tabs = st.tabs(["Segmentation Mask", "Overlay", "Medical Interpretation"])

    with tabs[0]:
        st.subheader("Predicted Segmentation Mask")
        st.image(mask_color, use_column_width=True)

    with tabs[1]:
        st.subheader("Overlay Image (Original + Mask)")
        st.image(overlay, use_column_width=True)

    with tabs[2]:
        st.subheader("🩺 Medical Interpretation")

        # LV and CSP pixels
        lv_pixels = np.sum(mask == 3)
        csp_pixels = np.sum(mask == 2)

        # Use your same Colab logic from analyze_lv
        diagnosis = analyze_lv(mask)

        # Display LV/CSP metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LV pixels", lv_pixels)
            st.metric("CSP pixels", csp_pixels)
        with col2:
            st.success(diagnosis)

        st.write("**Legend:**")
        st.markdown(
            """
            - 🟥 Red: Lateral Ventricles (LV)
            - 🟨 Yellow: CSP
            - 🟦 Blue: Brain structure / other
            """
        )
