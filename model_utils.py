import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 256

# ================================
# Load trained UNet model
# ================================
def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


# ================================
# Preprocess image
# ================================
def preprocess_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ================================
# Predict segmentation mask
# ================================
def predict_mask(model, image):
    pred = model.predict(image)
    mask = np.argmax(pred[0], axis=-1)
    return mask


# ================================
# Simple post-processing
# ================================
def clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cleaned


# ================================
# Simple LV analysis
# ================================
def analyze_lv(mask):
    lv_pixels = np.sum(mask == 3)  # classe 3 = ventricule latéral
    if lv_pixels < 500:
        return "LV normal"
    else:
        return "Suspicion de ventriculomégalie"
