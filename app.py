# ================================
# 0) Imports & Model Load
# ================================
import os
import io
import math
import traceback
import cv2
import numpy as np
import tensorflow as tf
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from PIL import Image, ImageDraw

# ================================
# 1) Load your UNet model
# ================================
UNET_PATH = "/content/drive/MyDrive/fetal_unet_model.h5"
if not os.path.exists(UNET_PATH):
    raise FileNotFoundError(f"Model not found at {UNET_PATH}.")

unet_model = tf.keras.models.load_model(UNET_PATH, compile=False)
print("✅ UNet model loaded from:", UNET_PATH)

# ================================
# 2) Constants & Helpers
# ================================
IMG_SIZE = 256
COLORS = {1: (0, 0, 255), 2: (0, 255, 255), 3: (255, 0, 0)}

# Intergrowth table (simplified, same as before)
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
    if ga_weeks < 14: ga_weeks = 14
    if ga_weeks > 40: ga_weeks = 40
    low, high = int(math.floor(ga_weeks)), int(math.ceil(ga_weeks))
    if low == high:
        vals = INTERGROWTH_HC[low]
        return {'p3':vals[0],'p5':vals[1],'p10':vals[2],'p50':vals[3],'p90':vals[4],'p95':vals[5],'p97':vals[6]}
    frac = ga_weeks - low
    lo, hi = INTERGROWTH_HC[low], INTERGROWTH_HC[high]
    interp = [ lo[i] + (hi[i]-lo[i])*frac for i in range(len(lo)) ]
    return {'p3':interp[0],'p5':interp[1],'p10':interp[2],'p50':interp[3],'p90':interp[4],'p95':interp[5],'p97':interp[6]}

def estimate_sd_from_centiles(p3, p97):
    return (p97 - p3) / 3.761578

def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def hc_to_z_percentile(hc_mm, ga_weeks):
    cent = interp_intergrowth(ga_weeks)
    p3, p50, p97 = cent['p3'], cent['p50'], cent['p97']
    sd = estimate_sd_from_centiles(p3, p97)
    if sd <= 0: return None
    z = (hc_mm - p50) / sd
    pct = normal_cdf(z) * 100.0
    return {'z': z, 'pct': pct, 'median': p50, 'sd': sd, 'centiles': cent}

# ================================
# 3) HC & CSP/LV Helpers
# ================================
def calculate_hc_from_segmentation(mask, pixel_size_mm, original_height=None):
    brain_mask = (mask == 1).astype(np.uint8)
    if np.sum(brain_mask) == 0: return 0.0, None, None
    kernel = np.ones((3,3), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return 0.0, None, None
    largest_contour = max(contours, key=cv2.contourArea)
    ellipse = None
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (center, axes, orientation) = ellipse
        major_axis, minor_axis = max(axes)/2.0, min(axes)/2.0
        h = ((major_axis - minor_axis) * 2) / ((major_axis + minor_axis) * 2 + 1e-10)
        circumference_pixels = math.pi * (major_axis + minor_axis) * (1 + (3*h)/(10 + math.sqrt(4-3*h)))
        perimeter_pixels = cv2.arcLength(largest_contour, True)
        hc_pixels = (circumference_pixels + perimeter_pixels)/2.0
    else:
        hc_pixels = cv2.arcLength(largest_contour, True)
    if original_height and original_height != IMG_SIZE:
        hc_pixels *= original_height / IMG_SIZE
    hc_mm = hc_pixels * pixel_size_mm
    return hc_mm, largest_contour, ellipse

def apply_unet_for_hc(image_bgr, pixel_size_mm):
    original_h, original_w = image_bgr.shape[:2]
    image_resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    image_norm = image_resized.astype(np.float32)/255.0
    pred = unet_model.predict(np.expand_dims(image_norm,0), verbose=0)[0]
    if pred.ndim == 3 and pred.shape[2] > 1:
        pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
    else:
        pr = pred[:,:,0] if pred.ndim==3 else pred
        pred_mask = (pr>0.5).astype(np.uint8)
    unique_classes = np.unique(pred_mask)
    mask_color = np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)
    for cls, col in COLORS.items(): mask_color[pred_mask==cls] = col
    hc_mm, contour, ellipse = calculate_hc_from_segmentation(pred_mask, pixel_size_mm, original_h)
    overlay = cv2.addWeighted(image_resized,0.6,mask_color,0.4,0)
    if contour is not None: cv2.drawContours(overlay,[contour],-1,(0,255,0),2)
    if ellipse is not None: cv2.ellipse(overlay, ellipse,(0,255,255),2)
    return image_resized, mask_color, overlay, pred_mask, hc_mm, unique_classes

def apply_unet_for_csp_ventricles(image_array):
    image_resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    image_norm = image_resized.astype(np.float32)/255.0
    pred_mask = unet_model.predict(np.expand_dims(image_norm,0), verbose=0)[0]
    pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)
    mask_color = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    for cls, col in COLORS.items(): mask_color[pred_mask==cls] = col
    overlay = cv2.addWeighted(image_resized,0.6,mask_color,0.4,0)
    lv_pixels = np.sum(pred_mask==3)
    csp_pixels = np.sum(pred_mask==2)
    return image_resized, mask_color, overlay, lv_pixels, csp_pixels

def analyse_anomalies(csp_pixels, lv_pixels, week, pixel_to_mm):
    # Logic same as original code
    pixel_size_mm = pixel_to_mm
    lv_area_mm2 = lv_pixels * (pixel_size_mm**2)
    lv_diameter_mm = 2*np.sqrt(lv_area_mm2/np.pi) if lv_pixels>0 else 0
    csp_area_mm2 = csp_pixels * (pixel_size_mm**2)
    csp_diameter_mm = 2*np.sqrt(csp_area_mm2/np.pi) if csp_pixels>0 else 0
    diagnostics = []
    status = "NORMAL"; status_color = "green"

    # CSP analysis
    if week < 16:
        diagnostics.append(f"ℹ️ Âge très précoce ({week} semaines)")
        diagnostics.append("→ CSP normalement NON visible avant 16-18 semaines")
        status = "NORMAL"
        status_color = "green"
    elif week>=16 and week<18:
        if csp_pixels==0:
            diagnostics.append(f"ℹ️ Âge gestationnel: {week} semaines")
            diagnostics.append("→ CSP peut être visible ou non à cet âge")
            diagnostics.append("→ Contrôle recommandé à 18-20 semaines")
            status="SUIVI RECOMMANDÉ"
            status_color="orange"
        else:
            diagnostics.append(f"✅ CSP visible précocement: {csp_diameter_mm:.1f} mm")
    elif week>=18 and week<=37:
        if csp_pixels==0:
            status="ANORMAL"
            status_color="red"
            diagnostics.append("❌ CSP NON VISIBLE (18-37 sem)")
            diagnostics.append("→ SUSPICION DE MALFORMATION")
            diagnostics.append("  • Agénésie du corps calleux (ACC)")
            diagnostics.append("  • Holoprosencéphalie (HPE)")
            diagnostics.append("  • Dysplasie septo-optique (SOD)")
        else:
            diagnostics.append(f"✅ CSP visible: {csp_diameter_mm:.1f} mm")
            if csp_diameter_mm<2 or csp_diameter_mm>8.5:
                diagnostics.append(f"⚠️ Taille CSP inhabituelle: {csp_diameter_mm:.1f} mm")
                if status!="ANORMAL": status="ATTENTION"; status_color="orange"
    else:
        diagnostics.append(f"✅ CSP encore visible: {csp_diameter_mm:.1f} mm")

    # LV analysis
    if lv_pixels>0:
        if lv_diameter_mm>=10:
            if status!="ANORMAL": status="ANORMAL"; status_color="red"
            diagnostics.append(f"⚠️ VENTRICULOMÉGALIE détectée")
            diagnostics.append(f"→ Diamètre ventriculaire: {lv_diameter_mm:.1f} mm (≥ 10 mm)")
            if lv_diameter_mm<13: diagnostics.append("→ Classification: Légère/Borderline (10-12 mm)")
            elif lv_diameter_mm<15: diagnostics.append("→ Classification: Modérée (13-15 mm)")
            else: diagnostics.append("→ Classification: Sévère (>15 mm)")
        else:
            diagnostics.append(f"✅ Ventricules latéraux normaux: {lv_diameter_mm:.1f} mm")
    else:
        diagnostics.append("ℹ️ Ventricules latéraux non détectés")

    if csp_pixels==0 and lv_diameter_mm>=10 and week>=18:
        diagnostics.append("📊 PATTERN COMBINÉ: CSP absent + Ventriculomégalie")
        diagnostics.append("→ Évoque fortement malformation cérébrale structurale")

    if week<18 and csp_pixels==0:
        diagnostics.append("📋 RECOMMANDATION: Contrôle à 18-20 semaines")

    return diagnostics, status, status_color, lv_diameter_mm, csp_diameter_mm

# ================================
# 4) UI Widgets
# ================================
image_uploader = widgets.FileUpload(description="Upload Ultrasound", accept='.jpg,.jpeg,.png,.bmp,.tif', multiple=False)
pixel_input = widgets.FloatText(description="Pixel size (mm):", value=0.219544094, step=0.0001)
ga_input = widgets.FloatText(description="Gest. age (wks):", value=28.0, step=0.1)
process_button = widgets.Button(description="Process & Analyze", button_style='success', icon='play')

hc_output = widgets.Output()
hc_image_display = widgets.Output()
csp_output = widgets.Output()
csp_diagnostic_display = widgets.Output()

# ================================
# 5) Processing callback
# ================================
def on_process(b):
    with hc_output: clear_output()
    with hc_image_display: clear_output()
    with csp_output: clear_output()
    with csp_diagnostic_display: clear_output()

    if not image_uploader.value:
        with hc_output: print("❌ No image selected"); return

    try:
        # Load image
        name = list(image_uploader.value.keys())[0]
        data = image_uploader.value[name]["content"]
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        pixel_size = float(pixel_input.value)
        ga_weeks = float(ga_input.value)

        # -------------------------------
        # HC Measurement
        # -------------------------------
        with hc_output:
            print(f"📄 Processing: {name}")
            print(f"   Pixel size (mm): {pixel_size:.6f}")
            print(f"   Gestational age (wks): {ga_weeks:.2f}")
            print("🔍 Running HC segmentation and measurements...")

        img_resized_hc, mask_color_hc, overlay_hc, pred_mask_hc, hc_mm, unique_classes = apply_unet_for_hc(img_bgr, pixel_size)
        ig = hc_to_z_percentile(hc_mm, ga_weeks) if hc_mm>0 else None

        # Display verbose Colab-style output
        with hc_output:
            print("\n" + "="*60)
            print("✅ HC MEASUREMENT RESULTS")
            print("="*60)
            print(f"Head Circumference: {hc_mm:.1f} mm" if hc_mm>0 else "Head Circumference: N/A")
            if ig:
                print(f"Intergrowth median @ {ga_weeks:.2f} wks: {ig['median']:.1f} mm")
                print(f"z-score: {ig['z']:.2f}")
                print(f"percentile: {ig['pct']:.1f}th")
                if ig['z']<=-3: print("⚠ Microcephaly (<= -3 SD)")
                elif ig['z']<=-2: print("⚠ Small head (<= -2 SD)")
                elif ig['z']>=2: print("⚠ Macrocephaly (>= +2 SD)")
                else: print("Within normal range")
            print(f"Segmentation classes seen: {unique_classes}")
            print("="*35)

        with hc_image_display:
            display(HTML(f"<h4>HC Segmentation Overlay</h4>"))
            display(Image.fromarray(cv2.cvtColor(overlay_hc, cv2.COLOR_BGR2RGB)))

        # -------------------------------
        # CSP & LV Analysis
        # -------------------------------
        img_resized_csp, mask_color_csp, overlay_csp, lv_pixels, csp_pixels = apply_unet_for_csp_ventricles(img_bgr)
        diagnostics, status, status_color, lv_diameter_mm, csp_diameter_mm = analyse_anomalies(csp_pixels, lv_pixels, ga_weeks, pixel_size)

        with csp_output:
            print("\n" + "="*60)
            print("✅ CSP & VENTRICULAR ANALYSIS RESULTS")
            print("="*60)
            print(f"STATUS: {status}")
            print(f"Ventricular diameter (LV): {lv_diameter_mm:.1f} mm")
            print(f"CSP diameter: {csp_diameter_mm:.1f} mm")
            print(f"Gestational age: {ga_weeks:.1f} weeks")
            print("\nDETAILS:")
            print("-"*40)
            for d in diagnostics: print("  ", d)
            print("="*60)

        with csp_diagnostic_display:
            display(HTML(f"<h4 style='color:{status_color}'>{status} Diagnostic</h4>"))
            display(Image.fromarray(cv2.cvtColor(overlay_csp, cv2.COLOR_BGR2RGB)))
            display(HTML("<b>Detailed analysis:</b><br>" + "<br>".join(diagnostics)))

    except Exception as e:
        with hc_output:
            print("❌ Error processing image:", e)
            traceback.print_exc()

process_button.on_click(on_process)

# ================================
# 6) Display UI
# ================================
display(widgets.VBox([
    widgets.HTML("<h2 style='color:green'>Fetal Head & Ventricular Analysis App</h2>"),
    image_uploader, pixel_input, ga_input, process_button,
    hc_output, hc_image_display, csp_output, csp_diagnostic_display
]))
