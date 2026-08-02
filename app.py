import streamlit as st
import yolov5
from PIL import Image

st.set_page_config(page_title="Sheep Counter", page_icon="🐑")
st.title("Glenfairy Sheep Counter")
st.write("Upload an overhead drone photo to count sheep")

# 1. Load the fine-tuned aerial sheep model from Hugging Face
@st.cache_resource
def load_aerial_model():
    model = yolov5.load('keremberke/yolov5m-aerial-sheep')
    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45   # Overlap threshold
    return model

model = load_aerial_model()

# 2. File uploader
uploaded_file = st.file_uploader("Upload Drone Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    with st.spinner("Analyzing overhead image with aerial AI model..."):
        # Run model inference on the high-res image
        results = model(input_image, size=1280)
        
        # Save output image with bounding boxes drawn around every sheep
        results.render()  # Draws boxes directly onto results.imgs
        output_image = Image.fromarray(results.imgs[0])
        
        # Calculate total count from detection tensor
        sheep_count = len(results.pred[0])
        
        # Show results
        st.success(f"Total Sheep Detected: {sheep_count}")
        st.image(output_image, caption=f"Processed Image ({sheep_count} Sheep Outlined)")