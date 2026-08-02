import streamlit as st
import yolov5
import torch
import numpy as np
from PIL import Image

st.set_page_config(page_title="Paddock Sheep Counter", page_icon="🐑")
st.title("🐑 Paddock Sheep Counter")
st.write("Upload an overhead drone photo to detect and count sheep.")

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

    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45   # Overlap threshold
    return model

model = load_aerial_model()

# 3. File uploader & Processing
uploaded_file = st.file_uploader("Upload Drone Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    # Force a writable array copy so yolov5's internal cv2 drawing doesn't
    # choke on a read-only np.asarray() view
    input_array = np.array(input_image)

    with st.spinner("Analyzing overhead image with aerial AI model..."):
        results = model(input_array, size=1280)

        results.render()
        output_image = Image.fromarray(results.ims[0])

        sheep_count = len(results.pred[0])

        st.success(f"🎯 Total Sheep Detected: {sheep_count}")
        st.image(output_image, caption=f"Processed Image ({sheep_count} Sheep Outlined)")