import streamlit as st
import yolov5
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os

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


def get_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_clean_boxes(image, predictions, review_cutoff):
    """Draw boxes with small numbered tags instead of overlapping 'sheep 0.88' text."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    detections = []
    for i, (*box, conf, cls) in enumerate(predictions.tolist(), start=1):
        x1, y1, x2, y2 = box
        conf = float(conf)
        needs_review = conf < review_cutoff
        color = (230, 126, 34) if needs_review else (39, 116, 79)  # orange vs green

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = str(i)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        pad = 3
        tag_y = y1 - th - 2 * pad if y1 - th - 2 * pad > 0 else y1
        draw.rectangle([x1, tag_y, x1 + tw + 2 * pad, tag_y + th + 2 * pad], fill=color)
        draw.text((x1 + pad, tag_y + pad), label, fill="white", font=font)

        detections.append({"#": i, "confidence": round(conf, 2), "review": "⚠️ check" if needs_review else "✅"})

    return img, detections


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

        annotated_image, detections = draw_clean_boxes(input_image, predictions, review_cutoff)
        flagged_count = sum(1 for d in detections if d["review"] != "✅")

    st.write("")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Sheep detected", sheep_count)
    metric_col2.metric("Flagged for review", flagged_count)
    metric_col3.metric("Confidence threshold", f"{conf_threshold:.0%}")

    st.write("")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.image(annotated_image, caption=f"Processed Image ({sheep_count} Sheep Outlined)", use_container_width=True)

        buf = io.BytesIO()
        annotated_image.save(buf, format="PNG")
        st.download_button("⬇️ Download annotated image", data=buf.getvalue(),
                            file_name="sheep_count_annotated.png", mime="image/png")

    with col2:
        st.subheader("Review list")
        st.caption("Lowest confidence first — check these against the photo.")
        sorted_detections = sorted(detections, key=lambda d: d["confidence"])
        st.dataframe(sorted_detections, use_container_width=True, hide_index=True)

    st.divider()
    final_count = st.number_input("Final count (adjust after reviewing)", min_value=0, value=sheep_count, step=1)
    st.caption(f"Model detected **{sheep_count}**, confirmed count: **{final_count}**")