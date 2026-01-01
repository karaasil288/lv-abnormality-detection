# ================================
# Fetal Brain Analysis Streamlit App
# ================================

import streamlit as st
import os
import math
import traceback
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# ================================
# PART 0: Load UNet model
# ================================

UNET_PATH = "fetal_unet_model.h5"  # adjust if needed
if not os.path.exists(UNET_PATH):
    st.error(f"Model not found at {UNET_PATH}. Upload your fetal_unet_model.h5 to this path or change UNET_PATH.")
    st.stop()

unet_model = tf.keras.models.load_model(UNET_PATH, compile=False)
st.success(f"✅ UNet model loaded from: {UNET_PATH}")

# ================================
# PART 1: Constants & Helpers
# ================================

IMG_SIZE = 256
COLORS = {1: (0, 0, 255), 2: (0, 255, 255), 3: (255, 0, 0)}

# Intergrowth HC table (shortened for brevity, add full table)
INTERGROWTH_HC = {
    14: (87.38, 88.69, 90.73, 97.88, 105.02, 107.06, 108.37),
    15: (99.22,100.61,102.78,110.37,117.97,120.13,121.53),
    16: (111.12,112.60,114.88,122.91,130.94,133.22,134.70),
    17: (123.04,124.59,127.00,135.44,143.87,146.28,147.83),
    18: (134.94,136.56,139.08,147.90,156.73,159.24,160.86),
    19: (146.77,148.46,151.08,160.26,169.45,172.07,173.76),
    20: (158.49,160.24,162.96,172.48,182.00,184.72,186.47),
    21: (170.06,171.87,174.67,184.50,194.34,197.14,198.95),
    22: (181.44,183.30,186.18,196.30,206.42,209.31,211.16),
    23: (192.59,194.50,197.46,207.84,218.22,221.18,223.08),
    24: (203.48,205.42,208.45,219.07,229.69,232.72,234.67),
    25: (214.05,216.04,219.13,229.97,240.81,243.90,245.89),
    26: (224.28,226.31,229.46,240.51,251.56,254.71,256.73),
    27: (234.13,236.20,239.40,250.65,261.89,265.10,267.16),
    28: (243.56,245.66,248.92,260.36,271.80,275.06,277.16),
    29: (252.52,254.66,257.98,269.61,281.25,284.56,286.70),
    30: (260.99,263.17,266.54,278.38,290.22,293.60,295.77),
    31: (268.92,271.14,274.58,286.64,298.71,302.15,304.36),
    32: (276.28,278.54,282.05,294.37,306.68,310.19,312.45),
    33: (283.02,285.34,288.93,301.53,314.13,317.72,320.03),
    34: (289.11,291.48,295.17,308.10,321.03,324.72,327.10),
    35: (294.50,296.95,300.75,314.07,327.39,331.18,333.63),
    36: (299.16,301.69,305.62,319.40,333.17,337.10,339.63),
    37: (303.05,305.68,309.76,324.07,338.39,342.47,345.10),
    38: (306.12,308.86,313.12,328.07,343.01,347.28,350.02),
    39: (308.33,311.21,315.68,331.37,347.05,351.52,354.40),
    40: (309.64,312.68,317.40,333.94,350.49,355.21,358.25)
}

def interp_intergrowth(ga_weeks):
    if ga_weeks < 14.0: ga_weeks = 14.0
    if ga_weeks > 40.0: ga_weeks = 40.0
    low, high = int(math.floor(ga_weeks)), int(math.ceil(ga_weeks))
    if low==high:
        vals = INTERGROWTH_HC[low]
    else:
        frac = ga_weeks - low
        lo, hi = INTERGROWTH_HC[low], INTERGROWTH_HC[high]
        vals = [lo[i]+(hi[i]-lo[i])*frac for i in range(len(lo))]
    return {'p3':vals[0],'p5':vals[1],'p10':vals[2],'p50':vals[3],'p90':vals[4],'p95':vals[5],'p97':vals[6]}

def estimate_sd_from_centiles(p3, p97): return (p97-p3)/3.761578
def normal_cdf(z): return 0.5*(1.0+math.erf(z/math.sqrt(2.0)))

def hc_to_z_percentile(hc_mm, ga_weeks):
    cent = interp_intergrowth(ga_weeks)
    sd = estimate_sd_from_centiles(cent['p3'], cent['p97'])
    z = (hc_mm-cent['p50'])/sd if sd>0 else 0
    pct = normal_cdf(z)*100
    return {'z':z,'pct':pct,'median':cent['p50'],'sd':sd,'centiles':cent}

# ================================
# PART 2: UNet Helpers
# ================================

IMG_SIZE = 256

def apply_unet_for_hc(image_bgr, pixel_size_mm):
    original_h, original_w = image_bgr.shape[:2]
    image_resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    image_norm = image_resized.astype(np.float32)/255.0
    pred = unet_model.predict(np.expand_dims(image_norm,0), verbose=0)[0]
    if pred.ndim==3 and pred.shape[2]>1:
        pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
    else:
        pr = pred[:,:,0] if pred.ndim==3 else pred
        pred_mask = (pr>0.5).astype(np.uint8)
    mask_color = np.zeros((IMG_SIZE,IMG_SIZE,3), np.uint8)
    for cls, col in COLORS.items(): mask_color[pred_mask==cls]=col
    brain_mask = (pred_mask==1).astype(np.uint8)
    contours,_ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hc_mm=0
    if contours:
        cnt = max(contours,key=cv2.contourArea)
        hc_pixels = cv2.arcLength(cnt, True)
        hc_mm = hc_pixels*pixel_size_mm
    overlay = cv2.addWeighted(image_resized,0.6,mask_color,0.4,0)
    unique_classes = np.unique(pred_mask)
    return image_resized, mask_color, overlay, pred_mask, hc_mm, unique_classes

def apply_unet_for_csp_ventricles(image_array):
    image_resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    image_norm = image_resized.astype(np.float32)/255.0
    pred_mask = unet_model.predict(np.expand_dims(image_norm,0), verbose=0)[0]
    pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)
    mask_color = np.zeros((IMG_SIZE,IMG_SIZE,3), np.uint8)
    for cls,col in COLORS.items(): mask_color[pred_mask==cls]=col
    overlay = cv2.addWeighted(image_resized,0.6,mask_color,0.4,0)
    lv_pixels = np.sum(pred_mask==3)
    csp_pixels = np.sum(pred_mask==2)
    return image_resized, mask_color, overlay, lv_pixels, csp_pixels

def analyse_anomalies(csp_pixels, lv_pixels, week, pixel_to_mm):
    pixel_size_mm = pixel_to_mm
    lv_area_mm2 = lv_pixels * (pixel_size_mm**2)
    lv_diameter_mm = 2*np.sqrt(lv_area_mm2/np.pi) if lv_pixels>0 else 0
    csp_area_mm2 = csp_pixels * (pixel_size_mm**2)
    csp_diameter_mm = 2*np.sqrt(csp_area_mm2/np.pi) if csp_pixels>0 else 0
    diagnostics=[]
    status="NORMAL"
    status_color="green"
    if csp_pixels==0: 
        status="ANORMAL"
        status_color="red"
        diagnostics.append(f"⚠️ CSP NON VISIBLE ({week}-{37} sem) → SUSPICION DE MALFORMATION")
        diagnostics.append("• Agénésie du corps calleux (ACC)")
        diagnostics.append("• Holoprosencéphalie (HPE)")
        diagnostics.append("• Dysplasie septo-optique (SOD)")
    if lv_pixels==0: diagnostics.append("ℹ️ Ventricules latéraux non détectés")
    return diagnostics, status, status_color, lv_diameter_mm, csp_diameter_mm

# ================================
# PART 3: Streamlit UI
# ================================

st.set_page_config(page_title="Fetal Brain Analysis", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4a6fa5;'>🧠 Fetal Brain Analysis System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>UNet Segmentation • HC Measurement • CSP/Ventricular Analysis</p>", unsafe_allow_html=True)

st.sidebar.header("Upload & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Ultrasound Image", type=['jpg','jpeg','png','bmp','tif'])
pixel_size = st.sidebar.number_input("Pixel size (mm)", value=0.219544094, step=0.0001)
ga_weeks = st.sidebar.number_input("Gestational age (weeks)", value=28.0, step=0.1)
process_btn = st.sidebar.button("Process & Analyze")

hc_col, csp_col = st.columns([2,3])

if process_btn:
    if uploaded_file is None:
        st.sidebar.error("❌ Please upload an image first!")
    else:
        try:
            pil = Image.open(uploaded_file).convert("RGB")
            img_rgb = np.array(pil)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # HC measurement
            img_resized_hc, mask_color_hc, overlay_hc, pred_mask_hc, hc_mm, unique_classes = apply_unet_for_hc(img_bgr, pixel_size)
            ig = hc_to_z_percentile(hc_mm, ga_weeks) if hc_mm>0 else None

            # CSP/LV analysis
            original_csp, mask_img_csp, overlay_csp, lv_pixels, csp_pixels = apply_unet_for_csp_ventricles(img_bgr)
            diagnostics, status, status_color, lv_diameter_mm, csp_diameter_mm = analyse_anomalies(csp_pixels, lv_pixels, ga_weeks, pixel_size)

            # HC Box
            hc_html=f"""
            <div style="border:2px solid #4a6fa5; padding:10px; border-radius:10px; background-color:#f0f8ff;">
            <h3>✅ HC MEASUREMENT RESULTS</h3>
            <p>Head Circumference: {hc_mm:.1f} mm</p>
            <p>Intergrowth median @ {ga_weeks:.2f} wks: {ig['median']:.1f} mm</p>
            <p>z-score: {ig['z']:.2f}</p>
            <p>percentile: {ig['pct']:.1f}th</p>
            <p>{"⚠ Macrocephaly (>= +2 SD)" if ig['z']>2 else ""}</p>
            <p>Segmentation classes seen: {list(unique_classes)}</p>
            </div>
            """
            st.markdown(hc_html, unsafe_allow_html=True)

            # CSP/LV Box
            diag_html=f"""
            <div style="border:2px solid {status_color}; padding:15px; border-radius:10px; background-color:#f8f9fa;">
            <h3 style="color:{status_color};">DIAGNOSTIC: {status}</h3>
            <p>Gestational age: {ga_weeks} weeks</p>
            <p>LV diameter: {lv_diameter_mm:.1f} mm</p>
            <p>CSP diameter: {csp_diameter_mm:.1f} mm</p>
            <ul>
            """
            for msg in diagnostics: diag_html+=f"<li>{msg}</li>"
            diag_html+="</ul></div>"
            st.markdown(diag_html, unsafe_allow_html=True)

            # Display overlays
            hc_col.image(cv2.cvtColor(overlay_hc, cv2.COLOR_BGR2RGB), caption="HC Segmentation Overlay", use_column_width=True)
            csp_col.image(cv2.cvtColor(overlay_csp, cv2.COLOR_BGR2RGB), caption="CSP & LV Overlay", use_column_width=True)

        except Exception as e:
            st.error("❌ Error during processing:")
            st.text(traceback.format_exc())
