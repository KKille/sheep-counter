import streamlit as st
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image

st.title("🐑 Paddock Sheep Counter")
st.write("Upload a drone photo to automatically count sheep.")

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='yolov8x.pt',
        confidence_threshold=0.25,
        device='cpu'
    )

model = load_model()

uploaded_file = st.file_uploader("Choose a drone image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save("temp.jpg")
    
    with st.spinner("Counting sheep..."):
        result = get_sliced_prediction(
            "temp.jpg",
            model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        sheep_preds = [p for p in result.object_prediction_list if p.category.name == 'sheep']
        result.export_visuals(export_dir="./", file_name="out")
        
        st.success(f"Total Sheep Counted: {len(sheep_preds)}")
        st.image("out.png", caption="Processed Photo with Sheep Boxed")