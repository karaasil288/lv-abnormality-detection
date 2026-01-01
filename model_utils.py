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
    # Resize
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    # Normalize
    image = image / 255.0
    # Expand dims
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# ================================
# Predict segmentation mask
# ================================
def predict_mask(model, image):
    pred = model.predict(image)
    # Multi-class
    if pred.shape[-1] > 1:
        mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
    else:  # Binary
        mask = (pred[0,:,:,0] > 0.5).astype(np.uint8)
    return mask

# ================================
# Simple LV analysis
# ================================
def analyze_lv(mask):
    lv_pixels = np.sum(mask == 3)
    csp_pixels = np.sum(mask == 2)

    if lv_pixels == 0:
        lv_text = "LV normal"
    elif lv_pixels < 500:
        lv_text = "Suspicion de ventriculomégalie légère"
    else:
        lv_text = "Suspicion de ventriculomégalie sévère"

    if csp_pixels == 0:
        csp_text = "CSP absent"
    else:
        csp_text = f"CSP détecté ({csp_pixels} pixels)"

    if lv_pixels >= 500 or csp_pixels == 0:
        return f"⚠️ Attention: {lv_text}, {csp_text}"
    else:
        return f"✅ Normal: {lv_text}, {csp_text}"
