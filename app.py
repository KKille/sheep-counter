import streamlit as st
import yolov5
import torch
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
    # Apply patch specifically while loading the aerial weights
    torch.load = patched_torch_load
    try:
        model = yolov5.load('keremberke/yolov5m-aerial-sheep')
    finally:
        # Restore standard torch.load behavior
        torch.load = _original_torch_load
        
    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45   # Overlap threshold
    return model

model = load_aerial_model()

# 3. File uploader & Processing
uploaded_file = st.file_uploader("Upload Drone Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    with st.spinner("Analyzing overhead image with aerial AI model..."):
        # Run model inference on high-res input
        results = model(input_image, size=1280)
        
        # Draw bounding boxes directly on output image
        results.render()
        output_image = Image.fromarray(results.imgs[0])
        
        # Extract detected sheep count
        sheep_count = len(results.pred[0])
        
        # Display results
        st.success(f"🎯 Total Sheep Detected: {sheep_count}")
        st.image(output_image, caption=f"Processed Image ({sheep_count} Sheep Outlined)")