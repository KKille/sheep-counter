import streamlit as st
import yolov5
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import math

st.set_page_config(page_title="Paddock Sheep Counter", page_icon="logo.png", layout="wide")

# --- Light styling pass: soft palette, rounded cards, tidier spacing ---
st.markdown("""
<style>
    .main {
        background-color: #FAF8F3;
    }
    h1 {
        color: #2F4A3C;
        font-weight: 700;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #6B7A6E;
        font-size: 1.05rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"], div[data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stDownloadButton button, .stButton button {
        border-radius: 8px;
        border: 1px solid #2F4A3C;
        color: #2F4A3C;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
    }
    .crop-caption {
        text-align: center;
        font-size: 0.8rem;
        color: #6B7A6E;
    }
</style>
""", unsafe_allow_html=True)

# --- Logo + title, centered ---
logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])
with logo_col2:
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=140)
    else:
        st.markdown("<div style='text-align:center; font-size:3rem;'>🐑</div>", unsafe_allow_html=True)

st.markdown("<h1>Paddock Sheep Counter</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an overhead drone photo to detect and count sheep</div>", unsafe_allow_html=True)

# --- Sidebar controls ---
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
    help="Lower = catches more sheep but more false positives. Higher = stricter, may miss some.")
iou_threshold = st.sidebar.slider("Overlap threshold (IOU)", 0.05, 0.95, 0.45, 0.05,
    help="Lower = more aggressive at splitting overlapping sheep into separate boxes.")
review_cutoff = st.sidebar.slider("Flag for review below", 0.05, 0.95, 0.50, 0.05,
    help="Detections below this confidence get flagged orange for a second look.")

# 1. Patch PyTorch's torch.load to bypass the strict security block for legacy models
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# 2. Load the fine-tuned aerial sheep model safely
@st.cache_resource
def load_aerial_model():
    torch.load = patched_torch_load
    try:
        model = yolov5.load('keremberke/yolov5m-aerial-sheep')
    finally:
        torch.load = _original_torch_load
    return model

model = load_aerial_model()
model.conf = conf_threshold
model.iou = iou_threshold


def compute_line_width(predictions):
    """Scale outline thickness to how big the sheep actually are in this photo."""
    if len(predictions) == 0:
        return 1
    boxes = predictions.tolist()
    avg_size = sum((b[2] - b[0] + b[3] - b[1]) / 2 for b in boxes) / len(boxes)
    return 1 if avg_size < 45 else (2 if avg_size < 90 else 3)


def draw_outline_only(image, predictions, review_cutoff):
    """Draw thin outlines only, no text — avoids label overlap in dense clusters."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    line_width = compute_line_width(predictions)

    detections = []
    for i, (*box, conf, cls) in enumerate(predictions.tolist(), start=1):
        x1, y1, x2, y2 = box
        conf = float(conf)
        needs_review = conf < review_cutoff
        color = (230, 126, 34) if needs_review else (39, 116, 79)  # orange vs green
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        detections.append({"#": i, "confidence": round(conf, 2), "review": "⚠️ check" if needs_review else "✅",
                            "box": (x1, y1, x2, y2)})

    return img, detections


def crop_detection(image, box, pad_ratio=0.6, min_size=60):
    """Crop a padded, isolated thumbnail around a single detection for the review gallery."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * pad_ratio, h * pad_ratio
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(image.width, int(x2 + pad_x))
    bottom = min(image.height, int(y2 + pad_y))
    crop = image.crop((left, top, right, bottom))
    if crop.width < min_size or crop.height < min_size:
        scale = min_size / max(crop.width, crop.height, 1)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
    return crop


# 3. File uploader & Processing
st.write("")
uploaded_file = st.file_uploader("Upload Drone Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    input_array = np.array(input_image)

    with st.spinner("Analyzing overhead image with aerial AI model..."):
        results = model(input_array, size=1280)
        predictions = results.pred[0]  # tensor: x1,y1,x2,y2,conf,cls
        sheep_count = len(predictions)

        annotated_image, detections = draw_outline_only(input_image, predictions, review_cutoff)
        flagged = [d for d in detections if d["review"] != "✅"]
        flagged.sort(key=lambda d: d["confidence"])

    st.write("")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Sheep detected", sheep_count)
    metric_col2.metric("Flagged for review", len(flagged))
    metric_col3.metric("Confidence threshold", f"{conf_threshold:.0%}")

    st.write("")
    st.image(annotated_image, caption=f"Processed Image ({sheep_count} Sheep Outlined) — orange = worth a second look",
              use_container_width=True)

    buf = io.BytesIO()
    annotated_image.save(buf, format="PNG")
    st.download_button("⬇️ Download annotated image (full resolution)", data=buf.getvalue(),
                        file_name="sheep_count_annotated.png", mime="image/png")
    st.caption("Download the full-res image to zoom in properly — the preview above is scaled down to fit the page.")

    st.divider()

    if flagged:
        st.subheader(f"⚠️ Review gallery — {len(flagged)} flagged detection(s)")
        st.caption("Each tile is a zoomed-in crop of one flagged detection, lowest confidence first. No overlapping labels — just look through and confirm.")

        show_all = st.checkbox("Show all flagged crops (can be a lot on dense photos)", value=len(flagged) <= 40)
        display_list = flagged if show_all else flagged[:40]
        if not show_all and len(flagged) > 40:
            st.caption(f"Showing the 40 lowest-confidence detections. Tick the box above to see all {len(flagged)}.")

        cols_per_row = 8
        for row_start in range(0, len(display_list), cols_per_row):
            row_items = display_list[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, d in zip(cols, row_items):
                crop = crop_detection(input_image, d["box"])
                with col:
                    st.image(crop, use_container_width=True)
                    st.markdown(f"<div class='crop-caption'>#{d['#']} · {d['confidence']:.0%}</div>", unsafe_allow_html=True)
    else:
        st.success("No detections flagged — everything's above your confidence threshold. 🎉")

    st.divider()
    final_count = st.number_input("Final count (adjust after reviewing)", min_value=0, value=sheep_count, step=1)
    st.caption(f"Model detected **{sheep_count}**, confirmed count: **{final_count}**")