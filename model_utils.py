import numpy as np
import cv2
import tensorflow as tf

# ================================
# Load trained UNet model
# ================================
def load_model(fetal_unet_model.h5):
    model = tf.keras.models.load_model(fetal_unet_model.h5, compile=False)
    return model


# ================================
# Preprocess image
# ================================
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return np.expand_dims(image, axis=0)


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
# Dummy medical analysis (à enrichir)
# ================================
def analyze_lv(mask):
    lv_pixels = np.sum(mask == 2)  # exemple : classe 2 = ventricule
    if lv_pixels < 500:
        return "LV normal"
    else:
        return "Suspicion de ventriculomégalie"
