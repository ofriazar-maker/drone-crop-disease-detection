"""
Field Disease Detection — Streamlit App
========================================
Analyzes field images for plant disease (Potato or Sugar Beet).
Outputs: classification per plant zone + heatmap + summary stats.

Run:
    pip install streamlit opencv-python-headless matplotlib scikit-image numpy
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import tempfile
import os
import sys
import csv
import time
import zipfile
import pandas as pd

# ── Make sure the classifiers are importable from the same folder ──
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Field Disease Detector",
    page_icon="🌱",
    layout="wide",
)

# ─────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-box {
        background: #1c2333;
        border-radius: 10px;
        padding: 18px 14px;
        text-align: center;
        margin: 4px;
    }
    .metric-label { font-size: 0.85rem; color: #8b9dc3; margin-bottom: 4px; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .healthy-color  { color: #27ae60; }
    .suspicious-color { color: #f39c12; }
    .sick-color     { color: #e74c3c; }
    .alert-high     { background: #3d1515; border: 1px solid #e74c3c; border-radius: 8px; padding: 10px; }
    .alert-medium   { background: #3d2e10; border: 1px solid #f39c12; border-radius: 8px; padding: 10px; }
    .alert-low      { background: #0f2d1a; border: 1px solid #27ae60; border-radius: 8px; padding: 10px; }
    h1 { color: #e8eaf6 !important; }
    h2 { color: #cfd8dc !important; font-size: 1.1rem !important; }
    .stButton > button {
        background: #1565c0;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: #1976d2; }
    .quality-badge {
        border-radius: 8px; padding: 8px 14px; display: inline-block;
        font-weight: 700; font-size: 0.95rem;
    }
    .conf-bar { height: 14px; border-radius: 7px; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  Step 2 — Image Quality Check
# ─────────────────────────────────────────────────────────

def check_image_quality(img_bgr: np.ndarray) -> dict:
    """
    Laplacian sharpness, brightness distribution, exposure check.
    Returns: sharpness, brightness_mean, brightness_std, valid (bool), issues (list).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Sharpness via Laplacian variance
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Brightness stats
    brightness_mean = float(np.mean(gray))
    brightness_std  = float(np.std(gray))

    # Overexposed: more than 20% pixels near 255
    overexposed_pct = float(np.sum(gray > 240) / gray.size * 100)
    # Underexposed: more than 20% pixels near 0
    underexposed_pct = float(np.sum(gray < 15) / gray.size * 100)

    issues = []
    valid  = True

    if lap_var < 80:
        issues.append(f"Blurry (sharpness={lap_var:.0f} < 80)")
        valid = False
    if overexposed_pct > 20:
        issues.append(f"Overexposed ({overexposed_pct:.1f}% white pixels)")
        valid = False
    if underexposed_pct > 20:
        issues.append(f"Underexposed ({underexposed_pct:.1f}% black pixels)")
        valid = False
    if brightness_mean < 30:
        issues.append(f"Too dark (mean brightness={brightness_mean:.0f})")
        valid = False
    if brightness_mean > 230:
        issues.append(f"Too bright (mean brightness={brightness_mean:.0f})")
        valid = False

    return {
        "sharpness":         round(lap_var, 1),
        "brightness_mean":   round(brightness_mean, 1),
        "brightness_std":    round(brightness_std, 1),
        "overexposed_pct":   round(overexposed_pct, 1),
        "underexposed_pct":  round(underexposed_pct, 1),
        "valid":             valid,
        "issues":            issues,
    }


# ─────────────────────────────────────────────────────────
#  Statistical metrics per image
# ─────────────────────────────────────────────────────────

def compute_image_stats(img_bgr: np.ndarray) -> dict:
    """
    Compute statistical metrics for a single image:
    mean, std, entropy, yellow_pct, green_pct, dark_pct, brown_pct, total_pixels.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    total_pixels = int(img_bgr.shape[0] * img_bgr.shape[1])
    mean_val = float(np.mean(gray))
    std_val  = float(np.std(gray))

    # Shannon entropy from grayscale histogram
    hist, _  = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    hist     = hist.astype(float)
    hist_nz  = hist[hist > 0]
    prob     = hist_nz / hist_nz.sum()
    entropy_val = float(-np.sum(prob * np.log2(prob)))

    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch  = hsv[:, :, 0]
    s_ch  = hsv[:, :, 1]
    v_ch  = hsv[:, :, 2]

    yellow_pct = float(np.sum((h_ch >= 15) & (h_ch <= 35) & (s_ch > 60) & (v_ch > 80)) / total_pixels * 100)
    green_pct  = float(np.sum((h_ch >= 35) & (h_ch <= 85) & (s_ch > 40) & (v_ch > 50)) / total_pixels * 100)
    dark_pct   = float(np.sum(v_ch < 50) / total_pixels * 100)
    brown_pct  = float(np.sum((h_ch >= 5) & (h_ch <= 20) & (s_ch > 40) & (v_ch > 40)) / total_pixels * 100)

    return {
        "stat_mean":       round(mean_val, 2),
        "stat_std":        round(std_val, 2),
        "stat_entropy":    round(entropy_val, 4),
        "stat_yellow_pct": round(yellow_pct, 2),
        "stat_green_pct":  round(green_pct, 2),
        "stat_dark_pct":   round(dark_pct, 2),
        "stat_brown_pct":  round(brown_pct, 2),
        "total_pixels":    total_pixels,
    }


# ─────────────────────────────────────────────────────────
#  Step 11 — Histogram plots
# ─────────────────────────────────────────────────────────

def render_rgb_histogram(img_bgr: np.ndarray) -> bytes:
    """Plot R, G, B channel histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.patch.set_facecolor("#0d1117")

    channel_info = [
        (2, "Red",   "#e74c3c"),
        (1, "Green", "#27ae60"),
        (0, "Blue",  "#3498db"),
    ]
    for ax, (ch, name, color) in zip(axes, channel_info):
        ax.set_facecolor("#0d1117")
        hist = cv2.calcHist([img_bgr], [ch], None, [256], [0, 256])
        ax.fill_between(range(256), hist.ravel(), color=color, alpha=0.7)
        ax.plot(hist.ravel(), color=color, linewidth=1)
        ax.set_title(f"{name} Channel", color="white", fontsize=10, pad=4)
        ax.set_xlabel("Intensity", color="#aaa", fontsize=8)
        ax.set_ylabel("Pixel count", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")

    fig.suptitle("RGB Histogram", color="white", fontsize=12, y=1.01)
    plt.tight_layout(pad=0.8)
    return _fig_to_img(fig)


def render_hsv_histogram(img_bgr: np.ndarray) -> bytes:
    """Plot H, S, V channel histograms."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.patch.set_facecolor("#0d1117")

    channel_info = [
        (0, "Hue",        "#9b59b6", [0, 180]),
        (1, "Saturation", "#e67e22", [0, 256]),
        (2, "Value",      "#f1c40f", [0, 256]),
    ]
    for ax, (ch, name, color, rng) in zip(axes, channel_info):
        ax.set_facecolor("#0d1117")
        bins = 180 if ch == 0 else 256
        hist = cv2.calcHist([hsv], [ch], None, [bins], rng)
        ax.fill_between(range(bins), hist.ravel(), color=color, alpha=0.7)
        ax.plot(hist.ravel(), color=color, linewidth=1)
        ax.set_title(f"{name} Channel", color="white", fontsize=10, pad=4)
        ax.set_xlabel("Value", color="#aaa", fontsize=8)
        ax.set_ylabel("Pixel count", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")

    fig.suptitle("HSV Histogram", color="white", fontsize=12, y=1.01)
    plt.tight_layout(pad=0.8)
    return _fig_to_img(fig)


def render_hue_detail(img_bgr: np.ndarray, plant_mask: np.ndarray = None) -> bytes:
    """Hue histogram (vegetation pixels only if mask provided)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue_ch = hsv[:, :, 0]

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    if plant_mask is not None and plant_mask.sum() > 0:
        hue_vals = hue_ch[plant_mask > 0]
        title = "Hue Histogram (vegetation pixels)"
    else:
        hue_vals = hue_ch.ravel()
        title = "Hue Histogram (full image)"

    hist, edges = np.histogram(hue_vals, bins=36, range=(0, 180))
    centers = (edges[:-1] + edges[1:]) / 2

    # Color each bar by approximate hue color
    colors_hue = []
    for h in centers:
        rgb = cv2.cvtColor(np.uint8([[[int(h), 255, 200]]]), cv2.COLOR_HSV2RGB)[0][0]
        colors_hue.append(tuple(c / 255 for c in rgb))

    ax.bar(centers, hist, width=5, color=colors_hue, alpha=0.85, edgecolor="none")
    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.set_xlabel("Hue (0–180)", color="#aaa", fontsize=9)
    ax.set_ylabel("Pixel count", color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout(pad=0.5)
    return _fig_to_img(fig)


# ─────────────────────────────────────────────────────────
#  Step 15 — Confidence score
# ─────────────────────────────────────────────────────────

def compute_confidence(score: int, crop: str) -> dict:
    """
    Convert raw disease score to per-class probabilities.
    Uses sigmoid-like softmax over distance from each class center.
    """
    if crop == "Potato":
        # thresholds: sick>=5, suspicious>=2
        centers = {"Healthy": 0.5, "Suspicious": 3.0, "Sick": 6.0}
        max_score = 10
    else:
        # thresholds: sick>=5, suspicious>=3 (v8 algorithm)
        centers = {"Healthy": 1.0, "Suspicious": 4.0, "Sick": 7.0}
        max_score = 10

    # Compute negative squared distances from each class center
    raw = {cls: -((score - c) ** 2) for cls, c in centers.items()}

    # Softmax
    exp_vals = {k: np.exp(v / 3.0) for k, v in raw.items()}
    total    = sum(exp_vals.values())
    probs    = {k: round(v / total, 3) for k, v in exp_vals.items()}

    return probs


# ─────────────────────────────────────────────────────────
#  Step 20 — CSV row builder
# ─────────────────────────────────────────────────────────

def build_csv_row(
    image_name: str,
    crop: str,
    result: dict,
    quality: dict,
    confidence: dict,
    processing_time: float,
    experiment: str = "",
    ground_truth: str = "",
    img_stats: dict = None,
) -> dict:
    """Build one CSV row per the Step 20 spec."""
    label    = result.get("label", "")
    correct  = ""
    if ground_truth and label:
        correct = "1" if ground_truth.strip().lower() == label.strip().lower() else "0"

    gps_str = ""
    if result.get("latitude") and result.get("longitude"):
        gps_str = f"{result['latitude']:.6f},{result['longitude']:.6f}"

    stats = img_stats or {}
    return {
        "Experiment":      experiment,
        "Crop":            crop,
        "Image_ID":        image_name,
        "Ground_Truth":    ground_truth,
        "Mean_H":          result.get("mean_h", ""),
        "Mean_S":          result.get("sat_mean", result.get("mean_sat", "")),
        "Mean_V":          result.get("val_mean", ""),
        "Green_Percent":   result.get("green_ratio", result.get("green_pct", "")),
        "Yellow_Percent":  result.get("yellow_ratio", ""),
        "Brown_Percent":   result.get("brown_ratio", ""),
        "Dark_Percent":    result.get("dark_ratio", ""),
        "Vegetation_Pct":  result.get("veg_ratio", ""),
        "Prediction":      label,
        "Confidence":      confidence.get(label, ""),
        "Conf_Healthy":    confidence.get("Healthy", ""),
        "Conf_Suspicious": confidence.get("Suspicious", ""),
        "Conf_Sick":       confidence.get("Sick", ""),
        "Processing_Time": round(processing_time, 3),
        "GPS":             gps_str,
        "Sharpness":       quality.get("sharpness", ""),
        "Brightness":      quality.get("brightness_mean", ""),
        "Valid":           int(quality.get("valid", True)),
        "Correct":         correct,
        # ── Statistical metrics ──
        "Stat_Mean":       stats.get("stat_mean", ""),
        "Stat_Std":        stats.get("stat_std", ""),
        "Stat_Entropy":    stats.get("stat_entropy", ""),
        "Stat_Yellow_Pct": stats.get("stat_yellow_pct", ""),
        "Stat_Green_Pct":  stats.get("stat_green_pct", ""),
        "Stat_Dark_Pct":   stats.get("stat_dark_pct", ""),
        "Stat_Brown_Pct":  stats.get("stat_brown_pct", ""),
        "Total_Pixels":    stats.get("total_pixels", ""),
    }


def export_csv(rows: list) -> bytes:
    """Convert list of dicts to CSV bytes."""
    if not rows:
        return b""
    import io
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return sio.getvalue().encode("utf-8")


def compute_performance_metrics(rows: list) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1 and Confusion Matrix
    for rows that have both Ground_Truth and Prediction filled.
    Classes: Healthy, Suspicious, Sick.
    Returns dict with metrics and confusion matrix, or None if not enough labeled rows.
    """
    labeled = [r for r in rows if r.get("Ground_Truth") and r.get("Prediction")]
    if len(labeled) < 2:
        return None

    classes = ["Healthy", "Suspicious", "Sick"]
    # Confusion matrix: cm[true][pred]
    cm = {c: {p: 0 for p in classes} for c in classes}
    for r in labeled:
        gt  = r["Ground_Truth"].strip()
        pred = r["Prediction"].strip()
        if gt in cm and pred in classes:
            cm[gt][pred] += 1

    n = len(labeled)
    correct = sum(cm[c][c] for c in classes)
    accuracy = correct / n if n > 0 else 0.0

    precision, recall, f1 = {}, {}, {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in classes) - tp
        fn = sum(cm[c][p] for p in classes) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0
        precision[c] = round(p, 3)
        recall[c]    = round(r_, 3)
        f1[c]        = round(f, 3)

    macro_precision = round(sum(precision.values()) / len(classes), 3)
    macro_recall    = round(sum(recall.values()) / len(classes), 3)
    macro_f1        = round(sum(f1.values()) / len(classes), 3)

    return {
        "n_labeled": n,
        "accuracy":  round(accuracy, 3),
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
        "macro_f1":        macro_f1,
        "confusion_matrix": cm,
        "classes": classes,
    }


def export_csv_with_metrics(rows: list) -> bytes:
    """
    Export all rows as CSV, then append blank line + performance metrics section
    (Accuracy, Precision, Recall, F1, Confusion Matrix) if Ground_Truth is available.
    """
    if not rows:
        return b""
    import io
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    metrics = compute_performance_metrics(rows)
    if metrics:
        classes = metrics["classes"]
        sio.write("\n")
        sio.write("# ── Performance Metrics ──\n")
        sio.write(f"# N labeled,{metrics['n_labeled']}\n")
        sio.write(f"# Accuracy,{metrics['accuracy']}\n")
        sio.write("\n")
        sio.write("# Per-class Precision\n")
        sio.write("# Class," + ",".join(classes) + ",Macro\n")
        sio.write("# Precision," + ",".join(str(metrics['precision'][c]) for c in classes)
                  + f",{metrics['macro_precision']}\n")
        sio.write("# Recall,"    + ",".join(str(metrics['recall'][c])    for c in classes)
                  + f",{metrics['macro_recall']}\n")
        sio.write("# F1,"        + ",".join(str(metrics['f1'][c])        for c in classes)
                  + f",{metrics['macro_f1']}\n")
        sio.write("\n")
        sio.write("# Confusion Matrix (rows=True, cols=Predicted)\n")
        sio.write("# True\\Pred," + ",".join(classes) + "\n")
        for c in classes:
            sio.write(f"# {c}," + ",".join(str(metrics['confusion_matrix'][c][p]) for p in classes) + "\n")

    return sio.getvalue().encode("utf-8")


# ─────────────────────────────────────────────────────────
#  Step 21 — ZIP folder structure builder
# ─────────────────────────────────────────────────────────

def build_output_zip(
    image_name: str,
    img_bgr:    np.ndarray,
    img_proc:   np.ndarray,
    result:     dict,
    quality:    dict,
    confidence: dict,
    annotated:  bytes,
    rgb_hist:   bytes,
    hsv_hist:   bytes,
    hue_hist:   bytes,
    csv_bytes:  bytes,
) -> bytes:
    """
    Build a ZIP with the organized folder structure (Step 21):
    <image_name>/
      original/   gray/   preprocessed/
      h_channel/  s_channel/  v_channel/
      raw_mask/   clean_mask/ clipped/
      histograms/ overlay/    features/   result/
    """
    stem = os.path.splitext(image_name)[0]
    buf  = BytesIO()

    def _encode_bgr(img: np.ndarray) -> bytes:
        ok, enc = cv2.imencode(".png", img)
        return enc.tobytes() if ok else b""

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        base = f"{stem}/"

        # original
        zf.writestr(base + "original/original.png", _encode_bgr(img_bgr))

        # gray
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ok, enc = cv2.imencode(".png", gray)
        zf.writestr(base + "gray/gray.png", enc.tobytes() if ok else b"")

        # preprocessed
        zf.writestr(base + "preprocessed/preprocessed.png", _encode_bgr(img_proc))

        # HSV channels
        hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        ok_h, enc_h = cv2.imencode(".png", h_ch)
        ok_s, enc_s = cv2.imencode(".png", s_ch)
        ok_v, enc_v = cv2.imencode(".png", v_ch)
        zf.writestr(base + "h_channel/hue.png",        enc_h.tobytes() if ok_h else b"")
        zf.writestr(base + "s_channel/saturation.png", enc_s.tobytes() if ok_s else b"")
        zf.writestr(base + "v_channel/value.png",      enc_v.tobytes() if ok_v else b"")

        # masks — placeholder (actual mask computed inside classifiers, not exposed yet)
        zf.writestr(base + "raw_mask/.keep",   b"")
        zf.writestr(base + "clean_mask/.keep", b"")
        zf.writestr(base + "clipped/.keep",    b"")

        # histograms
        zf.writestr(base + "histograms/rgb_histogram.png", rgb_hist)
        zf.writestr(base + "histograms/hsv_histogram.png", hsv_hist)
        zf.writestr(base + "histograms/hue_detail.png",    hue_hist)

        # overlay (annotated image)
        zf.writestr(base + "overlay/annotated.png", annotated)

        # features (JSON-like text)
        feat_lines = [f"{k}: {v}" for k, v in result.items()]
        feat_lines += [f"confidence_{k}: {v}" for k, v in confidence.items()]
        feat_lines += [f"sharpness: {quality.get('sharpness', '')}"]
        feat_lines += [f"brightness_mean: {quality.get('brightness_mean', '')}"]
        zf.writestr(base + "features/features.txt", "\n".join(feat_lines).encode())

        # result
        label = result.get("label", "unknown")
        result_txt = (
            f"Classification: {label}\n"
            f"Score: {result.get('score', '')}\n"
            f"Confidence:\n"
            + "\n".join(f"  {k}: {v}" for k, v in confidence.items())
        )
        zf.writestr(base + "result/result.txt", result_txt.encode())

        # CSV
        if csv_bytes:
            zf.writestr(base + "result/data.csv", csv_bytes)

    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
#  Shared background-removal helpers
# ─────────────────────────────────────────────────────────

def _remove_bg_potato(hsv: np.ndarray, total: int) -> tuple:
    """
    Build a plant mask for potato that excludes bare soil / row gaps.
    Designed for aerial/overhead field shots where pinkish sandy soil
    sits between rows of green plants.

    Returns (plant_mask_u8, plant_area, veg_ratio, soil_ratio).
    """
    hh = hsv[:, :, 0]
    s  = hsv[:, :, 1]
    v  = hsv[:, :, 2]

    # ── Broad soil exclusion: orange-red-tan hue, LOW-MEDIUM saturation ──
    soil_mask = (
        ((hh >= 0)  & (hh <= 30)) &
        (s  >= 10)  & (s  <= 110) &
        (v  >= 40)
    ).astype(np.uint8) * 255

    # Also catch pale pinkish/beige sand (very low saturation, bright)
    soil_sand = (
        ((hh >= 0) & (hh <= 25)) &
        (s <= 80) & (v >= 100)
    ).astype(np.uint8) * 255
    soil_mask = cv2.bitwise_or(soil_mask, soil_sand)

    # Dilate soil slightly to suppress soil/leaf border pixels
    soil_mask = cv2.dilate(soil_mask, np.ones((3, 3), np.uint8), iterations=1)

    # ── Green plant tissue anchor — stricter saturation >= 50 ──
    green_seed = (
        (hh >= 20) & (hh <= 100) &
        (s  >= 50) &
        (v  >= 30)
    ).astype(np.uint8) * 255
    green_seed = cv2.bitwise_and(green_seed, cv2.bitwise_not(soil_mask))

    # Expand green zone to capture adjacent disease pixels
    expanded_green = cv2.dilate(green_seed, np.ones((7, 7), np.uint8), iterations=3)
    expanded_green = cv2.bitwise_and(expanded_green, cv2.bitwise_not(soil_mask))

    # Disease candidates ONLY inside expanded green zone
    # Higher saturation threshold for brown (>= 70) to avoid catching dry earth
    brown1 = cv2.inRange(hsv, np.array([5,  70, 40]), np.array([22, 255, 220]))
    brown2 = cv2.inRange(hsv, np.array([0,  70, 40]), np.array([5,  255, 220]))
    brown_in_leaf = cv2.bitwise_and(cv2.bitwise_or(brown1, brown2), expanded_green)

    yellow_in_leaf = cv2.bitwise_and(
        cv2.inRange(hsv, np.array([15, 60, 80]), np.array([38, 255, 255])),
        expanded_green
    )
    dark_raw = ((v >= 20) & (v <= 80) & (s >= 40)).astype(np.uint8) * 255
    dark_in_leaf = cv2.bitwise_and(
        cv2.bitwise_and(dark_raw, cv2.bitwise_not(soil_mask)),
        expanded_green
    )

    plant_mask = cv2.bitwise_or(green_seed,
                  cv2.bitwise_or(brown_in_leaf,
                  cv2.bitwise_or(yellow_in_leaf, dark_in_leaf)))

    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE,
                                   np.ones((7, 7), np.uint8), iterations=2)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8), iterations=1)

    soil_area  = int(np.sum(soil_mask > 0))
    plant_area = max(int(np.sum(plant_mask > 0)), 1)
    veg_ratio  = plant_area / total * 100
    soil_ratio = soil_area  / total * 100
    return plant_mask, plant_area, veg_ratio, soil_ratio


def _remove_bg_beet(hsv: np.ndarray, total: int) -> tuple:
    """
    Step 1 for beet: build a plant mask that includes ALL leaf tissue.
    Returns (plant_mask_u8, green_mask_u8, plant_area, veg_ratio).
    """
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ── Green seed (healthy leaf tissue) ──
    green_seed = ((hh >= 35) & (hh <= 85) & (s > 45) & (v > 40)).astype(np.uint8) * 255

    # ── Pale / bleached / dried leaf tissue ──
    pale_leaf  = ((s < 60) & (v > 90) & (v < 240)).astype(np.uint8) * 255

    # ── Dark red/purple beet leaf tissue ──
    dark_red   = ((hh >= 150) | (hh <= 15)) & (s > 40) & (v > 30) & (v < 180)
    dark_leaf  = (dark_red.astype(np.uint8)) * 255

    # ── Purple/blue-purple stressed/diseased leaf tissue ──
    # Hue 120-155 = blue-purple range: seen in phosphorus deficiency,
    # viral infection and anthocyanin accumulation under stress
    purple_leaf = ((hh >= 120) & (hh <= 155) & (s > 40) & (v > 40)).astype(np.uint8) * 255

    # ── Yellowing diseased tissue ──
    yellow_leaf = ((hh >= 18) & (hh <= 45) & (s > 50) & (v > 60)).astype(np.uint8) * 255

    # ── Soil / sand / gravel exclusion ──
    soil = ((hh >= 8) & (hh <= 32) & (s < 65) & (v > 80)).astype(np.uint8) * 255

    # ── Build plant mask ──
    expanded_green = cv2.dilate(green_seed, np.ones((15, 15), np.uint8), iterations=3)
    leaf_candidates = cv2.bitwise_or(
        cv2.bitwise_or(cv2.bitwise_or(pale_leaf, dark_leaf), yellow_leaf),
        purple_leaf
    )
    leaf_near_green = cv2.bitwise_and(leaf_candidates, expanded_green)
    combined = cv2.bitwise_or(green_seed, leaf_near_green)
    plant_mask = cv2.bitwise_and(combined, cv2.bitwise_not(soil))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE,
                                   np.ones((9, 9), np.uint8), iterations=2)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8), iterations=1)

    plant_area = max(int(np.sum(plant_mask > 0)), 1)
    veg_ratio  = plant_area / total * 100
    return plant_mask, green_seed, plant_area, veg_ratio


# ─────────────────────────────────────────────────────────
#  Potato classifier
# ─────────────────────────────────────────────────────────

def _analyze_cell_potato(cell_bgr: np.ndarray) -> dict:
    """Classify one grid cell: background removal → feature extraction → score."""
    if cell_bgr is None or cell_bgr.size == 0:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}
    h, w = cell_bgr.shape[:2]
    if h < 10 or w < 10:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}

    hsv   = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    gray  = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    total = h * w

    # ── Step 1: background removal ──
    plant_mask, plant_area, veg_ratio, soil_ratio = _remove_bg_potato(hsv, total)
    # A cell is "No Plant" only if very little vegetation detected (<10%)
    # Do NOT discard based on soil ratio alone — field images naturally have lots of soil
    if veg_ratio < 10:
        return {"label": "No Plant", "score": -1, "veg_ratio": round(veg_ratio, 1)}

    # ── Step 2: features on plant pixels only ──
    def _on_plant(mask):
        return cv2.bitwise_and(mask, plant_mask)

    b1 = cv2.inRange(hsv, np.array([10, 70, 50]), np.array([22, 255, 200]))
    b2 = cv2.inRange(hsv, np.array([0,  70, 50]), np.array([10, 255, 200]))
    brown_ratio  = np.sum(_on_plant(cv2.bitwise_or(b1, b2)) > 0) / plant_area * 100

    dark_raw  = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 80]))
    dark_px   = cv2.morphologyEx(_on_plant(dark_raw), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark_ratio = np.sum(dark_px > 0) / plant_area * 100

    yellow_px    = _on_plant(cv2.inRange(hsv, np.array([20, 60, 100]), np.array([35, 255, 255])))
    yellow_ratio = np.sum(yellow_px > 0) / plant_area * 100

    green_px    = _on_plant(cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255])))
    green_ratio = np.sum(green_px > 0) / plant_area * 100

    sat_vals = hsv[:, :, 1][plant_mask > 0]
    mean_sat = float(np.mean(sat_vals)) if len(sat_vals) > 0 else 0

    # Mean H, V on plant pixels
    hue_vals = hsv[:, :, 0][plant_mask > 0]
    val_vals = hsv[:, :, 2][plant_mask > 0]
    mean_h   = float(np.mean(hue_vals)) if len(hue_vals) > 0 else 0
    val_mean = float(np.mean(val_vals)) if len(val_vals) > 0 else 0

    # ── Step 3: score (on plant pixels only — soil already excluded from mask) ──
    score = 0
    # Brown lesions on leaf tissue — thresholds raised slightly since mask is cleaner
    if   brown_ratio > 20:  score += 3
    elif brown_ratio > 8:   score += 2
    elif brown_ratio > 3:   score += 1
    # Yellowing
    if   yellow_ratio > 15: score += 2
    elif yellow_ratio > 5:  score += 1
    # Low saturation across the leaf (stress / wilt signal)
    if   mean_sat < 70:     score += 2
    elif mean_sat < 85:     score += 1
    # Low green coverage relative to total plant area (leaf loss / necrosis)
    if   green_ratio < 45:  score += 2
    elif green_ratio < 60:  score += 1
    # Dark necrotic patches (only penalise if substantial)
    if dark_ratio > 5:      score += 1

    label = "Sick" if score >= 5 else "Suspicious" if score >= 2 else "Healthy"
    return {
        "label": label, "score": score, "veg_ratio": round(veg_ratio, 1),
        "brown_ratio": round(brown_ratio, 1), "yellow_ratio": round(yellow_ratio, 1),
        "green_ratio": round(green_ratio, 1), "dark_ratio": round(dark_ratio, 1),
        "mean_sat": round(mean_sat, 1), "mean_h": round(mean_h, 1),
        "val_mean": round(val_mean, 1),
    }


# ─────────────────────────────────────────────────────────
#  Beet classifier
# ─────────────────────────────────────────────────────────

def _analyze_cell_beet(cell_bgr: np.ndarray) -> dict:
    """
    Classify one beet cell — v8 algorithm.
    Key fix: bleach_cerc restricted to H=15-28 only (not H>28 = green range).
    true_yellow (H=20-30, S>80, V>100) is the real disease signal.
    Thresholds: Sick ≥5, Suspicious ≥3, Healthy <3.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}
    h, w = cell_bgr.shape[:2]
    if h < 10 or w < 10:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}

    hsv   = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray  = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    total = h * w
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    soil_mask  = (hh >= 5) & (hh <= 25) & (s > 20) & (v > 50)
    dark_mask  = v < 35
    plant_mask = ~soil_mask & ~dark_mask

    plant_area = int(np.sum(plant_mask))
    if plant_area == 0:
        return {"label": "No Plant", "score": -1, "veg_ratio": 0.0}

    veg_ratio = plant_area / total * 100
    if veg_ratio < 10:
        return {"label": "No Plant", "score": -1, "veg_ratio": round(veg_ratio, 1)}

    true_yellow = (hh >= 20) & (hh <= 30) & (s > 80) & (v > 100) & plant_mask
    ty_pct      = float(np.sum(true_yellow)) / plant_area * 100

    s_mean = float(np.mean(s[plant_mask]))

    # CRITICAL FIX: H ≤ 28 only — H>28 is green range, catches sun-lit healthy leaves
    bleach_cerc = (hh >= 15) & (hh <= 28) & (s >= 20) & (s < 60) & (v > 130) & plant_mask
    bleach_pct  = float(np.sum(bleach_cerc)) / plant_area * 100

    blur     = cv2.GaussianBlur(gray.astype(np.float32), (21, 21), 0)
    diff     = gray.astype(np.float32) - blur
    dark_sp  = float(np.sum((diff < -15) & plant_mask)) / plant_area * 100
    light_sp = float(np.sum((diff >  20) & plant_mask)) / plant_area * 100

    green_px  = plant_mask & (hh >= 35) & (hh <= 85) & (s > 40)
    green_pct = float(np.sum(green_px)) / total * 100

    yellow_ratio = float(np.sum(plant_mask & (hh >= 18) & (hh <= 38) & (s > 70) & (v > 80))) / plant_area * 100
    dark_ratio   = float(np.sum(plant_mask & (v < 60))) / plant_area * 100
    mean_h   = float(np.mean(hh[plant_mask]))
    val_mean = float(np.mean(v[plant_mask]))

    # ── Cercospora spot detection (v9 — contour-based, gated by s_mean) ──
    # Classic lesion: small round spot, bright/pale grey center (S<50, V>140)
    # Guard: only run when s_mean < 130 — high-sat leaves have water-drop reflections
    # that look identical. Low s_mean = desaturated/stressed leaf = real Cercospora.
    cercospora_count = 0
    cercospora_density = 0.0
    if plant_area > 200 and s_mean < 130:
        plant_u8 = plant_mask.astype(np.uint8) * 255
        # Lesion center: low-S, high-V blob on plant pixels
        lesion_mask = (
            (s < 50) & (v > 140) & plant_mask
        ).astype(np.uint8) * 255
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN,
                                       np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area_c = cv2.contourArea(c)
            if 8 < area_c < 800:
                perim = cv2.arcLength(c, True)
                if perim > 0 and (4 * np.pi * area_c / (perim * perim)) > 0.25:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        r_in  = max(2, int(np.sqrt(area_c / np.pi)))
                        r_out = r_in + 6
                        inner = np.zeros(gray.shape, np.uint8)
                        cv2.circle(inner, (cx, cy), r_in, 255, -1)
                        outer = np.zeros(gray.shape, np.uint8)
                        cv2.circle(outer, (cx, cy), r_out, 255, -1)
                        cv2.circle(outer, (cx, cy), r_in,  0,   -1)
                        outer = cv2.bitwise_and(outer, plant_u8)
                        iv = gray[inner > 0]
                        ov = gray[outer > 0]
                        # Center must be brighter than ring (pale center, dark ring)
                        if iv.size > 0 and ov.size > 0:
                            if float(iv.mean()) - float(ov.mean()) > 10:
                                cercospora_count += 1
        cercospora_density = cercospora_count / plant_area * 10000

    score = 0
    if   ty_pct > 0.30: score += 3
    elif ty_pct > 0.15: score += 2
    if s_mean < 80 and bleach_pct > 1.5:
        score += 2
    if s_mean > 175 and ty_pct > 0.10:
        score += 2
    if dark_sp > 4.0 and light_sp > 3.0:
        score += 1

    # Cercospora spots — weighted by inverse of s_mean
    # More weight when leaf is less saturated (typical of diseased/stressed beet)
    if cercospora_count > 0:
        spot_weight = max(0.3, 1.0 - (s_mean - 60) / 70.0)  # 1.0 at s_mean≤60, ~0.3 at s_mean≥130
        weighted = cercospora_count * spot_weight
        if   weighted >= 25: score += 4
        elif weighted >= 10: score += 3
        elif weighted >= 4:  score += 2
        elif weighted >= 1:  score += 1

    label = "Sick" if score >= 5 else "Suspicious" if score >= 3 else "Healthy"
    return {
        "label": label, "score": score, "veg_ratio": round(veg_ratio, 1),
        "ty_pct": round(ty_pct, 3), "s_mean": round(s_mean, 1),
        "bleach_pct": round(bleach_pct, 2), "dark_sp": round(dark_sp, 1),
        "light_sp": round(light_sp, 1),
        "yellow_ratio": round(yellow_ratio, 1), "sat_mean": round(s_mean, 1),
        "green_pct": round(green_pct, 1), "dark_ratio": round(dark_ratio, 1),
        "mean_h": round(mean_h, 1), "val_mean": round(val_mean, 1),
        "hue_entropy": 0.0,
        "cercospora_count": cercospora_count,
        "cercospora_density": round(cercospora_density, 2),
        "purple_ratio": 0.0,
    }


# REMOVED: old beet algorithm code below (v1-v7) replaced by v8 above
def _analyze_cell_beet_UNUSED(cell_bgr: np.ndarray) -> dict:
    """UNUSED — kept for reference only. Use _analyze_cell_beet (v8) instead."""
    if cell_bgr is None or cell_bgr.size == 0:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}
    h, w = cell_bgr.shape[:2]
    if h < 10 or w < 10:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}

    hsv   = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    total = h * w
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    return {"label": "Healthy", "score": 0, "veg_ratio": 0}


# ─────────────────────────────────────────────────────────
#  Grid analysis engine
# ─────────────────────────────────────────────────────────

def analyze_field(img_bgr: np.ndarray, crop: str, grid_rows: int, grid_cols: int):
    """
    Divide the image into a grid and classify every cell.
    Returns (results_matrix, score_matrix, label_matrix, summary).
    """
    H, W   = img_bgr.shape[:2]
    cell_h = H // grid_rows
    cell_w = W // grid_cols
    analyze = _analyze_cell_potato if crop == "Potato" else _analyze_cell_beet

    results_matrix = []
    score_matrix   = np.zeros((grid_rows, grid_cols))
    label_matrix   = np.full((grid_rows, grid_cols), "Healthy", dtype=object)

    progress = st.progress(0, text="Analyzing field...")
    for row in range(grid_rows):
        row_results = []
        for col in range(grid_cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            res    = analyze(img_bgr[y1:y2, x1:x2])
            res["row"] = row
            res["col"] = col
            row_results.append(res)
            score_matrix[row, col] = max(res["score"], 0)
            label_matrix[row, col] = res["label"]
        results_matrix.append(row_results)
        progress.progress((row + 1) / grid_rows,
                          text=f"Analyzing row {row + 1}/{grid_rows}...")

    progress.empty()

    counts = {"Healthy": 0, "Suspicious": 0, "Sick": 0, "No Plant": 0}
    for row in results_matrix:
        for r in row:
            counts[r["label"]] = counts.get(r["label"], 0) + 1

    plant = counts["Healthy"] + counts["Suspicious"] + counts["Sick"]
    summary = {
        "healthy":       counts["Healthy"],
        "suspicious":    counts["Suspicious"],
        "sick":          counts["Sick"],
        "no_plant":      counts.get("No Plant", 0),
        "plant_cells":   plant,
        "sick_pct":      round(counts["Sick"]       / max(plant, 1) * 100, 1),
        "suspicious_pct":round(counts["Suspicious"] / max(plant, 1) * 100, 1),
        "alert_level":   (
            "HIGH"   if counts["Sick"] > plant * 0.30 else
            "MEDIUM" if counts["Sick"] + counts["Suspicious"] > plant * 0.30
            else "LOW"
        ),
    }
    return results_matrix, score_matrix, label_matrix, summary


# ─────────────────────────────────────────────────────────
#  Map rendering
# ─────────────────────────────────────────────────────────

def render_overlay(img_bgr, results_matrix, grid_rows, grid_cols):
    """Original image with colored cell overlays."""
    H, W   = img_bgr.shape[:2]
    cell_h = H // grid_rows
    cell_w = W // grid_cols

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    rgba = {
        "Healthy":    (0.15, 0.75, 0.25, 0.20),
        "Suspicious": (1.00, 0.65, 0.00, 0.32),
        "Sick":       (0.85, 0.15, 0.15, 0.42),
        "No Plant":   (0.00, 0.00, 0.00, 0.00),
    }
    for row in range(grid_rows):
        for col in range(grid_cols):
            r = results_matrix[row][col]
            c = rgba.get(r["label"], (0, 0, 0, 0))
            ax.add_patch(plt.Rectangle(
                (col * cell_w, row * cell_h), cell_w, cell_h,
                linewidth=0.6, edgecolor="white",
                facecolor=c[:3], alpha=c[3]
            ))
            if r["label"] in ("Sick", "Suspicious"):
                ax.text(col * cell_w + cell_w / 2, row * cell_h + cell_h / 2,
                        str(r["score"]),
                        color="white", fontsize=max(5, min(9, 70 // grid_rows)),
                        ha="center", va="center", fontweight="bold")

    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title("Field — Classification Overlay", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


def render_heatmap(score_matrix, grid_rows, grid_cols):
    """Green → yellow → red disease score heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    cmap = LinearSegmentedColormap.from_list(
        "disease", ["#1a5c1a", "#f0c040", "#c0392b"], N=256
    )
    im = ax.imshow(score_matrix, cmap=cmap, vmin=0, vmax=8,
                   interpolation="bilinear", aspect="auto")

    for x in range(grid_cols + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.3, alpha=0.35)
    for y in range(grid_rows + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.3, alpha=0.35)

    for row in range(grid_rows):
        for col in range(grid_cols):
            s = score_matrix[row, col]
            if s > 0:
                ax.text(col, row, f"{int(s)}",
                        ha="center", va="center",
                        color="white" if s > 3 else "#111",
                        fontsize=max(5, min(9, 70 // grid_rows)),
                        fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Disease Score", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.set_xlabel("Column", color="#aaa", fontsize=9)
    ax.set_ylabel("Row",    color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.set_title("Disease Score Heatmap", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


def render_categorical(label_matrix, grid_rows, grid_cols, summary):
    """Categorical ✓ / ? / ✗ classification grid."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    lti  = {"Healthy": 0, "Suspicious": 1, "Sick": 2, "No Plant": -1}
    mat  = np.array([[lti.get(label_matrix[r, c], 0)
                      for c in range(grid_cols)]
                     for r in range(grid_rows)])
    cmap = LinearSegmentedColormap.from_list(
        "cls", ["#666666", "#1a7a1a", "#d4a017", "#c0392b"], N=4
    )
    ax.imshow(mat, cmap=cmap, vmin=-1, vmax=2,
              interpolation="nearest", aspect="auto")

    sym = {"Healthy": "✓", "Suspicious": "?", "Sick": "✗", "No Plant": "·"}
    for r in range(grid_rows):
        for c in range(grid_cols):
            ax.text(c, r, sym.get(label_matrix[r, c], ""),
                    ha="center", va="center", color="white",
                    fontsize=max(6, min(11, 70 // grid_rows)), fontweight="bold")

    for x in range(grid_cols + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.4, alpha=0.45)
    for y in range(grid_rows + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.4, alpha=0.45)

    legend_patches = [
        mpatches.Patch(color="#1a7a1a", label=f"✓ Healthy    ({summary['healthy']})"),
        mpatches.Patch(color="#d4a017", label=f"? Suspicious ({summary['suspicious']})"),
        mpatches.Patch(color="#c0392b", label=f"✗ Sick       ({summary['sick']})"),
        mpatches.Patch(color="#666666", label=f"· No Plant   ({summary['no_plant']})"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="white", fontsize=8, framealpha=0.85)

    ax.set_xlabel("Column", color="#aaa", fontsize=9)
    ax.set_ylabel("Row",    color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.set_title("Classification Map", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


def _fig_to_img(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
#  Single-plant / single-leaf mode
# ─────────────────────────────────────────────────────────

def detect_image_type(img_bgr: np.ndarray) -> str:
    """
    Decide whether an image shows a single close-up plant/leaf or a field/multi-plant scene.
    Returns 'single_plant' or 'field'.

    Primary signal: single-leaf lab images have a large uniform neutral background
    (gray/white), while field images do not.
    Secondary signals: soil presence, dark shadows, stem colour.
    """
    H, W = img_bgr.shape[:2]
    scale = min(800 / max(H, W), 1.0)
    img = cv2.resize(img_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    total = h * w

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ── PRIMARY: uniform neutral background (lab / studio shot) ──────────────
    # Neutral = low saturation + not very dark (not shadow)
    lab_bg = ((s < 25) & (v > 130)).astype(np.uint8) * 255
    lab_closed = cv2.morphologyEx(lab_bg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(lab_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_blob_pct = max((cv2.contourArea(c) for c in cnts), default=0) / total * 100
    # A single connected neutral blob >12% of the image = plain lab background
    if biggest_blob_pct > 12:
        return "single_plant"

    # ── SECONDARY scoring: field-specific signals ─────────────────────────────
    field_score = 0

    # Soil (sandy/brown) between plant rows
    soil = (((hh >= 5) & (hh <= 35)) & (s < 90) & (v > 50)).astype(np.uint8) * 255
    soil_pct = float(np.sum(soil > 0)) / total * 100
    if soil_pct > 8:
        field_score += 3
    elif soil_pct > 4:
        field_score += 2
    elif soil_pct > 2:
        field_score += 1

    # Shadow pixels between small leaves in dense field
    dark_pct = float(np.sum(v < 50)) / total * 100
    if dark_pct > 15:
        field_score += 2
    elif dark_pct > 8:
        field_score += 1

    # Potato/beet stem colour (reddish-brown stems visible from above)
    stem = (((hh >= 140) | (hh <= 10)) & (s > 60) & (v > 30) & (v < 180)).astype(np.uint8) * 255
    stem = cv2.morphologyEx(stem, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    stem_pct = float(np.sum(stem > 0)) / total * 100
    if stem_pct > 8:
        field_score += 2
    elif stem_pct > 4:
        field_score += 1

    # High green coverage fraction (field = dense canopy)
    green = ((hh >= 25) & (hh <= 90) & (s >= 40) & (v >= 35)).astype(np.uint8) * 255
    green_pct = float(np.sum(green > 0)) / total * 100
    if green_pct > 55:
        field_score += 1

    return "field" if field_score >= 3 else "single_plant"


def _remove_bg_potato_leaf(hsv: np.ndarray, total: int) -> tuple:
    """
    Plant mask for a single isolated potato leaf on a plain (gray/white/dark) background.
    Unlike the field version, there is NO soil to worry about — background is neutral gray.
    The key challenge: correctly capture diseased (yellowing, brown, necrotic) tissue.

    Returns (plant_mask_u8, plant_area, veg_ratio, disease_ratio).
    """
    hh = hsv[:, :, 0]
    s  = hsv[:, :, 1]
    v  = hsv[:, :, 2]

    # ── Gray/white/black background exclusion ──
    # Neutral background: very low saturation (near-achromatic), or very dark, or very bright
    gray_bg = (s < 30).astype(np.uint8) * 255  # low saturation = gray/white/black
    # Also exclude very bright white background
    white_bg = (v > 230).astype(np.uint8) * 255

    # ── Green healthy leaf tissue (broader than field — isolated leaf, no soil risk) ──
    green_seed = (
        (hh >= 20) & (hh <= 95) &
        (s  >= 35) &               # lower sat threshold: some leaf parts are pale green
        (v  >= 30)
    ).astype(np.uint8) * 255
    green_seed = cv2.bitwise_and(green_seed, cv2.bitwise_not(gray_bg))

    # ── Diseased tissue — brown, orange-brown lesions ──
    brown1 = cv2.inRange(hsv, np.array([5,  40,  40]), np.array([22, 255, 220]))
    brown2 = cv2.inRange(hsv, np.array([0,  40,  40]), np.array([5,  255, 220]))
    brown_mask = cv2.bitwise_or(brown1, brown2)
    brown_mask = cv2.bitwise_and(brown_mask, cv2.bitwise_not(gray_bg))

    # ── Yellowing diseased tissue ──
    yellow_mask = cv2.inRange(hsv, np.array([15, 40, 60]), np.array([38, 255, 255]))
    yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(gray_bg))

    # ── Dark necrotic patches (low brightness, any hue, some saturation) ──
    dark_mask = (
        (v  >= 20) & (v  <= 80) &
        (s  >= 20)              # not pure black background
    ).astype(np.uint8) * 255
    dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(gray_bg))

    # Combine all plant tissue
    all_tissue = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(green_seed, brown_mask), yellow_mask), dark_mask)

    # ── Grow the tissue mask to fill holes ──
    # Use larger dilation since no soil to worry about
    expanded = cv2.dilate(all_tissue, np.ones((11, 11), np.uint8), iterations=3)
    # But keep within the non-gray region
    expanded = cv2.bitwise_and(expanded, cv2.bitwise_not(gray_bg))
    expanded = cv2.bitwise_and(expanded, cv2.bitwise_not(white_bg))

    # Flood fill using convex hull of detected tissue to get a solid leaf shape
    # Morphological close to bridge gaps
    plant_mask = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE,
                                   np.ones((15, 15), np.uint8), iterations=3)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN,
                                   np.ones((5, 5), np.uint8), iterations=1)

    # Keep only the largest connected component (the leaf)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(plant_mask, connectivity=8)
    if n_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        plant_mask = ((labels == largest_label).astype(np.uint8)) * 255

    plant_area = max(int(np.sum(plant_mask > 0)), 1)
    veg_ratio  = plant_area / total * 100

    # Disease pixels within the leaf
    disease_pixels = cv2.bitwise_or(cv2.bitwise_or(brown_mask, yellow_mask), dark_mask)
    disease_in_leaf = cv2.bitwise_and(disease_pixels, plant_mask)
    disease_area   = int(np.sum(disease_in_leaf > 0))
    disease_ratio  = disease_area / plant_area * 100

    return plant_mask, plant_area, veg_ratio, disease_ratio


def _analyze_single_leaf_potato(cell_bgr: np.ndarray) -> dict:
    """
    Classify a single isolated potato leaf (lab / close-up image with neutral background).
    Uses more sensitive thresholds than the field-grid version.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}
    h, w = cell_bgr.shape[:2]
    if h < 20 or w < 20:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}

    hsv   = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    total = h * w

    plant_mask, plant_area, veg_ratio, disease_ratio = _remove_bg_potato_leaf(hsv, total)

    if veg_ratio < 5:
        return {"label": "No Plant", "score": -1, "veg_ratio": round(veg_ratio, 1)}

    hh = hsv[:, :, 0]
    s  = hsv[:, :, 1]
    v  = hsv[:, :, 2]

    def _on_plant(mask):
        return cv2.bitwise_and(mask, plant_mask)

    # Brown lesion ratio (sensitive thresholds for close-up leaves)
    b1 = cv2.inRange(hsv, np.array([5,  35, 35]), np.array([22, 255, 220]))
    b2 = cv2.inRange(hsv, np.array([0,  35, 35]), np.array([5,  255, 220]))
    brown_ratio = np.sum(_on_plant(cv2.bitwise_or(b1, b2)) > 0) / plant_area * 100

    # Yellow/chlorotic tissue ratio
    yellow_px    = _on_plant(cv2.inRange(hsv, np.array([15, 35, 60]), np.array([38, 255, 255])))
    yellow_ratio = np.sum(yellow_px > 0) / plant_area * 100

    # Dark necrotic patches
    dark_raw   = (v >= 20) & (v <= 80) & (s >= 20)
    dark_in    = _on_plant((dark_raw.astype(np.uint8)) * 255)
    dark_ratio = np.sum(dark_in > 0) / plant_area * 100

    # Green ratio (healthy tissue)
    green_px    = _on_plant(cv2.inRange(hsv, np.array([22, 35, 35]), np.array([90, 255, 255])))
    green_ratio = np.sum(green_px > 0) / plant_area * 100

    # Mean saturation on leaf pixels
    sat_vals = s[plant_mask > 0]
    mean_sat = float(np.mean(sat_vals)) if len(sat_vals) > 0 else 0.0

    hue_vals = hh[plant_mask > 0]
    val_vals = v[plant_mask > 0]
    mean_h   = float(np.mean(hue_vals)) if len(hue_vals) > 0 else 0.0
    val_mean = float(np.mean(val_vals)) if len(val_vals) > 0 else 0.0

    # ── Scoring — sensitive thresholds for isolated leaf images ──
    score = 0

    # Brown lesions — even 2% is suspicious on an isolated leaf
    if   brown_ratio > 15: score += 3
    elif brown_ratio > 5:  score += 2
    elif brown_ratio > 1.5: score += 1

    # Yellowing (chlorosis / mosaic)
    if   yellow_ratio > 20: score += 2
    elif yellow_ratio > 8:  score += 1

    # Dark necrosis
    if   dark_ratio > 10:  score += 2
    elif dark_ratio > 3:   score += 1

    # Low green coverage = significant diseased area
    if   green_ratio < 40:  score += 2
    elif green_ratio < 60:  score += 1

    # Low mean saturation = pale / stressed leaf
    if   mean_sat < 65:  score += 2
    elif mean_sat < 80:  score += 1

    # Precomputed overall disease pixel ratio (from mask step)
    if   disease_ratio > 25: score += 2
    elif disease_ratio > 10: score += 1

    label = "Sick" if score >= 5 else "Suspicious" if score >= 2 else "Healthy"
    return {
        "label": label, "score": score, "veg_ratio": round(veg_ratio, 1),
        "brown_ratio": round(brown_ratio, 1), "yellow_ratio": round(yellow_ratio, 1),
        "green_ratio": round(green_ratio, 1), "dark_ratio": round(dark_ratio, 1),
        "disease_ratio": round(disease_ratio, 1),
        "mean_sat": round(mean_sat, 1), "mean_h": round(mean_h, 1),
        "val_mean": round(val_mean, 1),
    }


def analyze_single_plant(img_bgr: np.ndarray, crop: str) -> dict:
    """Classify a single-plant / single-leaf image as one unit.
    Auto-detects field vs. single-leaf and uses the appropriate classifier."""
    if crop == "Potato":
        img_type = detect_image_type(img_bgr)
        if img_type == "field":
            return _analyze_cell_potato(img_bgr)
        else:
            return _analyze_single_leaf_potato(img_bgr)
    else:
        return _analyze_cell_beet(img_bgr)


def render_single_plant_result(img_bgr: np.ndarray, result: dict, crop: str) -> bytes:
    """Draw the original image annotated with vegetation mask + disease highlights."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H, W = img_bgr.shape[:2]

    if crop == "Potato":
        total = H * W
        img_type = detect_image_type(img_bgr)
        if img_type == "field":
            # Use field mask — excludes soil properly
            veg_mask, _, _, _ = _remove_bg_potato(hsv, total)
            # Disease = brown/yellow/dark ONLY within green-adjacent zone
            soil1 = (((hh>=0)&(hh<=30))&(s>=10)&(s<=110)&(v>=40)).astype(np.uint8)*255
            soil2 = (((hh>=0)&(hh<=25))&(s<=80)&(v>=100)).astype(np.uint8)*255
            soil  = cv2.bitwise_or(soil1, soil2)
            green = ((hh>=20)&(hh<=100)&(s>=50)&(v>=30)).astype(np.uint8)*255
            green = cv2.bitwise_and(green, cv2.bitwise_not(soil))
            exp   = cv2.dilate(green, np.ones((7,7),np.uint8), iterations=3)
            exp   = cv2.bitwise_and(exp, cv2.bitwise_not(soil))
            b1 = cv2.inRange(hsv, np.array([5,  70, 40]), np.array([22, 255, 220]))
            b2 = cv2.inRange(hsv, np.array([0,  70, 40]), np.array([5,  255, 220]))
            yellow_px = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([38, 255, 255]))
            disease_mask = cv2.bitwise_and(
                cv2.bitwise_or(cv2.bitwise_or(b1, b2), yellow_px), exp
            )
        else:
            # Single leaf — use leaf mask (no soil risk)
            veg_mask, _, _, _ = _remove_bg_potato_leaf(hsv, total)
            b1 = cv2.inRange(hsv, np.array([5,  35, 35]), np.array([22, 255, 220]))
            b2 = cv2.inRange(hsv, np.array([0,  35, 35]), np.array([5,  255, 220]))
            yellow_px = cv2.inRange(hsv, np.array([15, 35, 60]), np.array([38, 255, 255]))
            dark_px   = cv2.inRange(hsv, np.array([0,  20, 20]), np.array([180, 60, 80]))
            disease_mask = cv2.bitwise_and(
                cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(b1, b2), yellow_px), dark_px),
                veg_mask
            )
    else:
        veg_mask_bool = (hh >= 35) & (hh <= 85) & (s > 50) & (v > 50)
        veg_mask = (veg_mask_bool.astype(np.uint8)) * 255
        veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        plant_dilated = cv2.dilate(veg_mask, np.ones((7, 7), np.uint8), iterations=2)
        soil_mask_bool = (hh >= 10) & (hh <= 30) & (s < 60) & (v > 100)
        yellow_bool = (hh >= 18) & (hh <= 38) & (s > 70) & (v > 80) & (plant_dilated > 0) & ~soil_mask_bool
        disease_mask = (yellow_bool.astype(np.uint8)) * 255

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(rgb)

    contours, _ = cv2.findContours(veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            pts = cnt[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color="#00e676", linewidth=1.5, alpha=0.8)

    if disease_mask.sum() > 0:
        disease_rgba = np.zeros((H, W, 4), dtype=np.float32)
        disease_rgba[disease_mask > 0] = [1.0, 0.3, 0.0, 0.55]
        ax.imshow(disease_rgba)

    label  = result["label"]
    score  = result.get("score", 0)
    colors = {"Healthy": "#27ae60", "Suspicious": "#f39c12", "Sick": "#e74c3c", "No Plant": "#888"}
    badge_color = colors.get(label, "#888")
    icons = {"Healthy": "✓", "Suspicious": "?", "Sick": "✗", "No Plant": "·"}

    ax.text(0.02, 0.98,
            f"{icons.get(label, '')} {label}  (score: {score})",
            transform=ax.transAxes,
            color="white", fontsize=14, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color, alpha=0.88, edgecolor="white"))

    ax.axis("off")
    ax.set_title(f"Single Plant Analysis — {crop}", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


# ─────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────

def preprocess_img(img_bgr: np.ndarray) -> np.ndarray:
    """
    Pre-processing (Step 9.1.3.5):
    1. Resize to 800×600
    2. Gaussian blur (noise reduction)
    3. CLAHE on L-channel (adaptive contrast)
    4. Value-channel clipping (highlight suppression)
    """
    img = cv2.resize(img_bgr, (800, 600), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    v_clip = np.clip(v_ch, 0, int(np.percentile(v_ch, 99))).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h_ch, s_ch, v_clip]), cv2.COLOR_HSV2BGR)


def render_single_analysis_tab(img_bgr: np.ndarray, crop: str, image_name: str = "image"):
    """Full pipeline output for the 'Single Image Analysis' tab."""

    t_start = time.time()

    # ── Step 2: Image Quality Check ──
    quality = check_image_quality(img_bgr)

    # ── Step 3: Grayscale ──
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Step 4: Pre-processing ──
    with st.spinner("Pre-processing image..."):
        img_proc = preprocess_img(img_bgr)

    # ── Steps 5–9: Build plant mask + classify ──
    with st.spinner("Building plant mask & classifying..."):
        result   = analyze_single_plant(img_proc, crop)
        annotated = render_single_plant_result(img_proc, result, crop)

    t_end = time.time()
    proc_time = t_end - t_start

    # ── Statistical metrics ──
    img_stats = compute_image_stats(img_bgr)

    # ── Step 11: Histograms ──
    rgb_hist = render_rgb_histogram(img_proc)
    hsv_hist = render_hsv_histogram(img_proc)
    hue_hist = render_hue_detail(img_proc)

    # ── Step 15: Confidence ──
    confidence = compute_confidence(result.get("score", 0), crop)

    # ── Step 20: CSV row ──
    # Ground Truth is read from session_state (set by the selectbox widget before this run)
    ground_truth = st.session_state.get("ground_truth_input", "")
    csv_row = build_csv_row(image_name, crop, result, quality, confidence, proc_time,
                            ground_truth=ground_truth, img_stats=img_stats)
    # Store result dict in session so Ground Truth can be updated later without re-analysis
    st.session_state["last_result"] = {
        "image_name": image_name,
        "crop": crop,
        "result": result,
        "quality": quality,
        "confidence": confidence,
        "proc_time": proc_time,
        "img_stats": img_stats,
    }
    csv_bytes = export_csv([csv_row])
    # Accumulate rows — replace if same image was analyzed before
    st.session_state.csv_rows = [r for r in st.session_state.csv_rows
                                  if r.get("Image_ID") != image_name]
    st.session_state.csv_rows.append(csv_row)

    label = result["label"]
    score = result.get("score", 0)
    icons  = {"Healthy": "✓", "Suspicious": "?", "Sick": "✗", "No Plant": "·"}
    alert_cls = {"Healthy": "alert-low", "Suspicious": "alert-medium",
                 "Sick": "alert-high", "No Plant": "alert-low"}
    color_map  = {"Healthy": "#27ae60", "Suspicious": "#f39c12", "Sick": "#e74c3c"}
    badge_color = color_map.get(label, "#888")

    # ═══════════════════════════════════════════════════
    # ── Step 2 display: Quality banner ──
    # ═══════════════════════════════════════════════════
    with st.expander("📋 Image Quality Check (Step 2)", expanded=True):
        q_color = "#27ae60" if quality["valid"] else "#e74c3c"
        q_label = "✅ Valid — ready for analysis" if quality["valid"] else "⚠️ Quality Issues Detected"
        st.markdown(
            f"<div class='quality-badge' style='background:{q_color}22;border:1px solid {q_color};color:{q_color}'>"
            f"{q_label}</div>",
            unsafe_allow_html=True
        )
        qc1, qc2, qc3 = st.columns(3)
        qc1.metric("Sharpness (Laplacian)", f"{quality['sharpness']}", delta="≥80 OK")
        qc2.metric("Brightness mean",       f"{quality['brightness_mean']}",
                   delta=f"std: {quality['brightness_std']}")
        qc3.metric("Overexposed pixels",    f"{quality['overexposed_pct']}%",
                   delta=f"underexposed: {quality['underexposed_pct']}%")
        if quality["issues"]:
            for iss in quality["issues"]:
                st.warning(f"⚠️ {iss}")

    st.divider()

    # ═══════════════════════════════════════════════════
    # ── Step 3 + Annotated image side by side ──
    # ═══════════════════════════════════════════════════
    col_orig, col_gray, col_ann = st.columns(3)
    with col_orig:
        st.markdown("**Original**")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_gray:
        st.markdown("**Grayscale (Step 3)**")
        st.image(gray, use_container_width=True, clamp=True)
    with col_ann:
        st.markdown("**Annotated result (Step 16)**")
        st.image(annotated, use_container_width=True,
                 caption="Orange=disease | Green=plant boundary")

    st.markdown("")
    st.download_button("⬇ Download annotated image", data=annotated,
                       file_name=f"analysis_{label.lower()}.png", mime="image/png")

    st.divider()

    # ═══════════════════════════════════════════════════
    # ── Classification + Confidence ──
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
<div class="{alert_cls.get(label, 'alert-low')}">
  <b>{icons.get(label, '')} Classification: {label}</b>
  &nbsp;|&nbsp; Disease score: <b>{score}</b>
  &nbsp;|&nbsp; Processing time: <b>{proc_time:.2f}s</b>
</div>""", unsafe_allow_html=True)
    st.markdown("")

    cl1, cl2 = st.columns([1, 2])
    with cl1:
        st.markdown("**Classification Badge (Step 14)**")
        st.markdown(f"""
<div style='background:{badge_color};border-radius:8px;padding:14px 16px;text-align:center'>
  <span style='color:white;font-size:1.6rem;font-weight:700'>{icons.get(label, '')} {label}</span><br>
  <span style='color:rgba(255,255,255,0.8);font-size:0.9rem'>Score: {score}</span>
</div>""", unsafe_allow_html=True)

    with cl2:
        st.markdown("**Confidence Score (Step 15)**")
        conf_colors = {"Healthy": "#27ae60", "Suspicious": "#f39c12", "Sick": "#e74c3c"}
        for cls_name, prob in confidence.items():
            pct = int(prob * 100)
            c   = conf_colors.get(cls_name, "#888")
            st.markdown(
                f"<div style='margin-bottom:6px'>"
                f"<span style='color:#ccc;font-size:0.85rem;width:110px;display:inline-block'>{cls_name}</span>"
                f"<div class='conf-bar' style='width:{pct}%;background:{c};display:inline-block;vertical-align:middle;margin:0 8px'></div>"
                f"<span style='color:{c};font-weight:700'>{prob:.2%}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()

    # ═══════════════════════════════════════════════════
    # ── Feature vector ──
    # ═══════════════════════════════════════════════════
    with st.expander("📊 Feature Vector (Step 13)", expanded=False):
        if crop == "Sugar Beet":
            feats = [
                ("True Yellow %",    result.get("ty_pct", 0),        "%",  "#f1c40f"),
                ("Sat mean",         result.get("s_mean", result.get("sat_mean", 0)), "", "#3498db"),
                ("Bleach %",         result.get("bleach_pct", 0),    "%",  "#e67e22"),
                ("Cercospora spots", result.get("cercospora_count", 0), "", "#e74c3c"),
                ("Dark spots %",     result.get("dark_sp", 0),       "%",  "#888"),
                ("Green %",          result.get("green_pct", 0),     "%",  "#27ae60"),
                ("Mean Hue",         result.get("mean_h", 0),        "",   "#9b59b6"),
            ]
        else:
            feats = [
                ("Brown ratio",   result.get("brown_ratio", 0),   "%",  "#e67e22"),
                ("Yellow ratio",  result.get("yellow_ratio", 0),  "%",  "#f1c40f"),
                ("Green ratio",   result.get("green_ratio", 0),   "%",  "#27ae60"),
                ("Dark ratio",    result.get("dark_ratio", 0),    "%",  "#888"),
                ("Avg saturation",result.get("mean_sat", 0),      "",   "#3498db"),
                ("Mean Hue",      result.get("mean_h", 0),        "",   "#9b59b6"),
                ("Mean Value",    result.get("val_mean", 0),      "",   "#f1c40f"),
            ]
        feats.append(("Vegetation %", result.get("veg_ratio", 0), "%", "#8e44ad"))

        col_fa, col_fb = st.columns(2)
        for i, (name, val, unit, color) in enumerate(feats):
            col = col_fa if i % 2 == 0 else col_fb
            with col:
                ca, cb = st.columns([3, 2])
                ca.markdown(f"<span style='color:#aaa;font-size:0.88rem'>{name}</span>",
                            unsafe_allow_html=True)
                cb.markdown(f"<span style='color:{color};font-weight:700'>{val}{unit}</span>",
                            unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # ── Statistical Metrics ──
    # ═══════════════════════════════════════════════════
    with st.expander("📐 Statistical Metrics", expanded=False):
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Mean",         f"{img_stats['stat_mean']:.1f}")
        sm2.metric("Std Dev",      f"{img_stats['stat_std']:.1f}")
        sm3.metric("Entropy",      f"{img_stats['stat_entropy']:.3f}")
        sm4.metric("Total Pixels", f"{img_stats['total_pixels']:,}")
        sm5, sm6, sm7, sm8 = st.columns(4)
        sm5.metric("Yellow %",     f"{img_stats['stat_yellow_pct']:.1f}%")
        sm6.metric("Green %",      f"{img_stats['stat_green_pct']:.1f}%")
        sm7.metric("Dark %",       f"{img_stats['stat_dark_pct']:.1f}%")
        sm8.metric("Brown %",      f"{img_stats['stat_brown_pct']:.1f}%")
        st.caption(f"⏱ Processing time: {proc_time:.3f}s")

    # ═══════════════════════════════════════════════════
    # ── Histograms (Step 11) ──
    # ═══════════════════════════════════════════════════
    with st.expander("📈 Histograms (Step 11)", expanded=False):
        ht1, ht2, ht3 = st.tabs(["RGB", "HSV", "Hue Detail"])
        with ht1:
            st.image(rgb_hist, use_container_width=True)
        with ht2:
            st.image(hsv_hist, use_container_width=True)
        with ht3:
            st.image(hue_hist, use_container_width=True)

    # ═══════════════════════════════════════════════════
    # ── Pre-processing steps checklist ──
    # ═══════════════════════════════════════════════════
    with st.expander("⚙️ Pre-processing Steps (Step 4)", expanded=False):
        st.markdown("""
- ✅ **Resize** → 800 × 600
- ✅ **Gaussian blur** (σ=0.8, kernel 3×3) — noise reduction
- ✅ **CLAHE** on LAB L-channel (clipLimit=2.0, tile 8×8) — adaptive contrast
- ✅ **Brightness clipping** at 99th percentile — highlight suppression
        """)

    st.divider()

    # ═══════════════════════════════════════════════════
    # ── Export: CSV + ZIP (Steps 20–21) ──
    # ═══════════════════════════════════════════════════
    st.markdown("### 💾 Export (Steps 20–21)")
    exp_c1, exp_c2 = st.columns(2)

    with exp_c1:
        # Allow updating Ground Truth after seeing results, without re-running analysis
        st.markdown("**Ground Truth for this image**")
        gt_live = st.selectbox(
            "True label",
            options=["", "Healthy", "Suspicious", "Sick"],
            index=["", "Healthy", "Suspicious", "Sick"].index(ground_truth) if ground_truth in ["", "Healthy", "Suspicious", "Sick"] else 0,
            key="gt_live_select",
            label_visibility="collapsed",
        )
        if st.button("✅ Apply & save Ground Truth", key="apply_gt"):
            last = st.session_state.get("last_result", {})
            if last:
                updated_row = build_csv_row(
                    last["image_name"], last["crop"], last["result"],
                    last["quality"], last["confidence"], last["proc_time"],
                    ground_truth=gt_live, img_stats=last.get("img_stats"),
                )
                st.session_state.csv_rows = [r for r in st.session_state.csv_rows
                                              if r.get("Image_ID") != last["image_name"]]
                st.session_state.csv_rows.append(updated_row)
                csv_row = updated_row
                csv_bytes = export_csv([updated_row])
                st.success(f"Ground Truth set to: **{gt_live or '(empty)'}**")

        st.download_button(
            "📄 Download CSV (Step 20)",
            data=export_csv([csv_row]),
            file_name=f"results_{image_name}.csv",
            mime="text/csv",
            help="One row with all features, classification, confidence and metadata"
        )

    # Multi-image accumulated CSV with performance metrics
    all_rows = st.session_state.get("csv_rows", [])
    if len(all_rows) > 0:
        n_labeled = sum(1 for r in all_rows if r.get("Ground_Truth"))
        metrics_available = n_labeled >= 2
        metrics = compute_performance_metrics(all_rows) if metrics_available else None

        with st.expander(
            f"📊 Session CSV — {len(all_rows)} image(s)"
            + (f" · {n_labeled} labeled" if n_labeled > 0 else ""),
            expanded=False
        ):
            if metrics:
                classes = metrics["classes"]
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy",  f"{metrics['accuracy']:.1%}")
                mc2.metric("Precision", f"{metrics['macro_precision']:.1%}", help="Macro average")
                mc3.metric("Recall",    f"{metrics['macro_recall']:.1%}",    help="Macro average")
                mc4.metric("F1",        f"{metrics['macro_f1']:.1%}",        help="Macro average")

                st.markdown("**Per-class metrics**")
                tbl_rows = []
                for c in classes:
                    tbl_rows.append({
                        "Class":     c,
                        "Precision": f"{metrics['precision'][c]:.3f}",
                        "Recall":    f"{metrics['recall'][c]:.3f}",
                        "F1":        f"{metrics['f1'][c]:.3f}",
                    })
                st.dataframe(pd.DataFrame(tbl_rows), hide_index=True, use_container_width=True)

                st.markdown("**Confusion Matrix** (rows = True label, cols = Predicted)")
                cm_data = []
                for c in classes:
                    row_d = {"True \\ Predicted": c}
                    row_d.update({p: metrics["confusion_matrix"][c][p] for p in classes})
                    cm_data.append(row_d)
                st.dataframe(pd.DataFrame(cm_data), hide_index=True, use_container_width=True)
            elif n_labeled > 0:
                st.info(f"Need at least 2 labeled images for metrics. Currently: {n_labeled}.")
            else:
                st.info("Add Ground Truth labels in the sidebar to enable Accuracy / Precision / Recall / F1.")

            # Download full CSV with metrics appended
            full_csv = export_csv_with_metrics(all_rows)
            st.download_button(
                "⬇ Download full session CSV (with metrics)",
                data=full_csv,
                file_name="session_results_with_metrics.csv",
                mime="text/csv",
            )
            if st.button("🗑 Clear session history", key="clear_csv"):
                st.session_state.csv_rows = []
                st.rerun()

    with exp_c2:
        # Build ZIP on button click to avoid building it every render
        if st.button("📦 Prepare output ZIP (Step 21)", key="zip_btn"):
            with st.spinner("Building ZIP..."):
                zip_bytes = build_output_zip(
                    image_name=image_name,
                    img_bgr=img_bgr,
                    img_proc=img_proc,
                    result=result,
                    quality=quality,
                    confidence=confidence,
                    annotated=annotated,
                    rgb_hist=rgb_hist,
                    hsv_hist=hsv_hist,
                    hue_hist=hue_hist,
                    csv_bytes=csv_bytes,
                )
            st.download_button(
                "⬇ Download ZIP folder",
                data=zip_bytes,
                file_name=f"{os.path.splitext(image_name)[0]}_outputs.zip",
                mime="application/zip",
            )

    # ── Step 19: Summary Report ──
    with st.expander("📑 Summary Report (Step 19)", expanded=False):
        st.markdown(f"""
**Experiment Summary**

| Field              | Value |
|--------------------|-------|
| Image              | `{image_name}` |
| Crop               | {crop} |
| Quality            | {'✅ Valid' if quality['valid'] else '⚠️ Invalid'} |
| Sharpness          | {quality['sharpness']} |
| Brightness mean    | {quality['brightness_mean']} |
| Classification     | **{label}** |
| Disease score      | {score} |
| Confidence ({label}) | {confidence.get(label, 0):.2%} |
| Vegetation %       | {result.get('veg_ratio', '')}% |
| Processing time    | {proc_time:.3f}s |
        """)
        if quality["issues"]:
            st.markdown("**Quality issues:** " + "; ".join(quality["issues"]))


def main():
    # ── Session state: accumulated CSV rows ──
    if "csv_rows" not in st.session_state:
        st.session_state.csv_rows = []

    # ── Header ──
    st.markdown("# 🌱 Field Disease Detector")
    st.markdown("Upload a field photo and get a plant health map in seconds.")
    st.divider()

    # ── Sidebar controls ──
    with st.sidebar:
        st.markdown("## Settings")

        crop = st.radio(
            "Crop type",
            ["Potato", "Sugar Beet"],
            index=0,
            help="Selects the disease model: Early/Late Blight for Potato, Cercospora for Sugar Beet"
        )

        st.markdown("---")
        grid_size = st.select_slider(
            "Grid resolution",
            options=[5, 8, 10, 12, 15, 20],
            value=10,
            help="Number of rows and columns. Higher = more precise, slower."
        )
        st.caption(f"Grid: {grid_size} × {grid_size} = {grid_size**2} zones")

        st.markdown("---")
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            help="Field photo or single plant / leaf"
        )

        st.markdown("---")
        st.markdown("**Ground Truth (optional)**")
        ground_truth_input = st.selectbox(
            "True label for this image",
            options=["", "Healthy", "Suspicious", "Sick"],
            index=0,
            help="If you know the correct label, select it here. Used for Accuracy/Precision/Recall/F1 calculation.",
            key="ground_truth_input",
        )

        analyze_btn = st.button("Analyze", disabled=(uploaded is None))

        st.markdown("---")
        st.markdown("**Disease thresholds**")
        if crop == "Potato":
            st.markdown("- Score ≥ 5 → **Sick**\n- Score ≥ 2 → **Suspicious**\n- Score 0-1 → **Healthy**")
            st.markdown("*Signals: brown spots, dark necrosis, yellowing, low green coverage*")
        else:
            st.markdown("- Score ≥ 5 → **Sick**\n- Score ≥ 3 → **Suspicious**\n- Score 0-2 → **Healthy**")
            st.markdown("*Signals: true yellow (H=20-30, S>80), bleach spots (H=15-28), Cercospora pattern*")

    # ── Main area ──
    if uploaded is None:
        st.info("← Upload an image in the sidebar to get started.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**How it works:**
1. Select your crop type (Potato or Sugar Beet)
2. Upload a field photo
3. Choose grid resolution (10×10 recommended)
4. Click **Analyze**

**Output:**
- Quality check + grayscale view
- Full pipeline: preprocessing → mask → features → classification
- Confidence scores per class
- RGB / HSV / Hue histograms
- CSV export + organized output ZIP
            """)
        with col2:
            st.markdown("""
**Disease detection:**

🥔 **Potato** — Early Blight & Late Blight
- Detects brown spots, dark necrosis, yellowing
- Calibrated on 2,152 real leaf images

🌿 **Sugar Beet** — Cercospora Leaf Spot
- Detects yellowing halos, color stress, saturation shift
- Works on wide field shots with multiple leaves
            """)
        return

    # ── Load image ──
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the uploaded image. Please try a different file.")
        return

    # ── Cap resolution to 2000px on the long side for fast processing ──
    _MAX_DIM = 2000
    _h, _w = img_bgr.shape[:2]
    if max(_h, _w) > _MAX_DIM:
        _scale = _MAX_DIM / max(_h, _w)
        img_bgr = cv2.resize(img_bgr, (int(_w * _scale), int(_h * _scale)),
                             interpolation=cv2.INTER_AREA)

    H, W = img_bgr.shape[:2]

    # Show preview + metadata
    col_prev, col_meta = st.columns([2, 1])
    with col_prev:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                 caption=f"Uploaded: {uploaded.name}", use_container_width=True)
    with col_meta:
        st.markdown("**Image info**")
        st.write(f"- Size: {W} × {H} px")
        st.write(f"- Crop: **{crop}**")
        st.write(f"- Grid: **{grid_size} × {grid_size}**")
        st.write(f"- Cell size: {W // grid_size} × {H // grid_size} px")
        st.caption("🔍 Image type auto-detected after clicking Analyze")

    if not analyze_btn:
        st.caption("← Click **Analyze** in the sidebar when ready.")
        return

    # ── Detect image type ──
    st.markdown("---")
    st.markdown("### Analysis Results")

    with st.spinner("Detecting image type..."):
        img_type = detect_image_type(img_bgr)

    # ── Two tabs: single image analysis + field grid ──
    tab_single, tab_field = st.tabs(["🔬 Single Image Analysis", "🗺️ Field Grid Analysis"])

    with tab_single:
        st.caption("Full pipeline: quality check → grayscale → preprocessing → mask → features → classification")
        render_single_analysis_tab(img_bgr, crop, image_name=uploaded.name)

    with tab_field:
        if img_type == "single_plant":
            st.info("🔍 Auto-detected: **single plant image** — field grid shown anyway.")
        else:
            st.info("🌾 Auto-detected: **field image**.")

        with st.spinner("Processing field grid..."):
            results_matrix, score_matrix, label_matrix, summary = analyze_field(
                img_bgr, crop, grid_size, grid_size
            )

        # Summary metrics
        alert_class = f"alert-{summary['alert_level'].lower()}"
        st.markdown(f"""
<div class="{alert_class}">
  <b>Alert Level: {summary['alert_level']}</b> &nbsp;|&nbsp;
  Sick: <b>{summary['sick_pct']}%</b> &nbsp;|&nbsp;
  Suspicious: <b>{summary['suspicious_pct']}%</b> &nbsp;|&nbsp;
  Plant zones: <b>{summary['plant_cells']}</b>
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Healthy zones</div>
                <div class="metric-value healthy-color">{summary['healthy']}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Suspicious zones</div>
                <div class="metric-value suspicious-color">{summary['suspicious']}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Sick zones</div>
                <div class="metric-value sick-color">{summary['sick']}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">No vegetation</div>
                <div class="metric-value" style="color:#8b9dc3">{summary['no_plant']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Maps inside nested tabs
        mt1, mt2, mt3 = st.tabs(["📷 Overlay Map", "🌡️ Heatmap", "🗺️ Classification Grid"])
        with mt1:
            img_overlay = render_overlay(img_bgr, results_matrix, grid_size, grid_size)
            st.image(img_overlay, use_container_width=True)
            st.download_button("Download overlay map", data=img_overlay,
                               file_name=f"field_overlay_{crop.lower().replace(' ','_')}.png",
                               mime="image/png")
        with mt2:
            img_heat = render_heatmap(score_matrix, grid_size, grid_size)
            st.image(img_heat, use_container_width=True)
            st.download_button("Download heatmap", data=img_heat,
                               file_name=f"field_heatmap_{crop.lower().replace(' ','_')}.png",
                               mime="image/png")
        with mt3:
            img_cat = render_categorical(label_matrix, grid_size, grid_size, summary)
            st.image(img_cat, use_container_width=True)
            st.download_button("Download classification grid", data=img_cat,
                               file_name=f"field_grid_{crop.lower().replace(' ','_')}.png",
                               mime="image/png")

        # ── Sick zone details ──
        sick_zones = [
            (r["row"] + 1, r["col"] + 1, r["score"])
            for row in results_matrix for r in row
            if r["label"] == "Sick"
        ]
        if sick_zones:
            with st.expander(f"🔴 Sick zones detail ({len(sick_zones)} zones)", expanded=False):
                st.markdown("These grid positions need immediate attention:")
                cols = st.columns(4)
                for i, (row, col, score) in enumerate(sorted(sick_zones, key=lambda x: -x[2])):
                    cols[i % 4].markdown(
                        f"<div class='metric-box'>"
                        f"<div class='metric-label'>Row {row}, Col {col}</div>"
                        f"<div class='metric-value sick-color'>{score}</div>"
                        f"<div class='metric-label'>disease score</div></div>",
                        unsafe_allow_html=True
                    )

        susp_zones = [
            (r["row"] + 1, r["col"] + 1, r["score"])
            for row in results_matrix for r in row
            if r["label"] == "Suspicious"
        ]
        if susp_zones:
            with st.expander(f"🟡 Suspicious zones ({len(susp_zones)} zones)", expanded=False):
                st.markdown("These zones show early disease signals — monitor closely:")
                cols = st.columns(4)
                for i, (row, col, score) in enumerate(sorted(susp_zones, key=lambda x: -x[2])):
                    cols[i % 4].markdown(
                        f"<div class='metric-box'>"
                        f"<div class='metric-label'>Row {row}, Col {col}</div>"
                        f"<div class='metric-value suspicious-color'>{score}</div>"
                        f"<div class='metric-label'>disease score</div></div>",
                        unsafe_allow_html=True
                    )

        # ── Step 19: Field Summary Report ──
        with st.expander("📑 Field Summary Report (Step 19)", expanded=False):
            total_zones = grid_size * grid_size
            plant_zones = summary["plant_cells"]
            st.markdown(f"""
**Field Analysis Summary**

| Field              | Value |
|--------------------|-------|
| Crop               | {crop} |
| Total zones        | {total_zones} |
| Plant zones        | {plant_zones} ({round(plant_zones/total_zones*100,1)}%) |
| No-plant zones     | {summary['no_plant']} |
| ✅ Healthy          | {summary['healthy']} ({round(summary['healthy']/max(plant_zones,1)*100,1)}%) |
| 🟡 Suspicious      | {summary['suspicious']} ({summary['suspicious_pct']}%) |
| 🔴 Sick            | {summary['sick']} ({summary['sick_pct']}%) |
| Alert Level        | **{summary['alert_level']}** |
            """)


if __name__ == "__main__":
    main()
