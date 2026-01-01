import streamlit as st
import numpy as np
from PIL import Image

from model_utils import (
    load_model,
    preprocess_image,
    predict_mask,
    clean_mask,
    analyze_lv
)

st.set_page_config(
    page_title="Fetal Brain Abnormality Detection",
    layout="centered"
)

st.title("🧠 Fetal Brain Abnormality Detection")
st.write("Upload an ultrasound image to analyze fetal brain structures.")

@st.cache_resource
def load_unet():
    return load_model("fetal_unet_model.h5")

model = load_unet()

uploaded_file = st.file_uploader(
    "Upload ultrasound image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Original Image", use_container_width=True)

    input_image = preprocess_image(image_np)
    mask = predict_mask(model, input_image)
    mask = clean_mask(mask)

    st.subheader("Predicted Segmentation Mask")
    st.image(mask, use_container_width=True)

    diagnosis = analyze_lv(mask)

    st.subheader("🩺 Medical Interpretation")
    st.success(diagnosis)
