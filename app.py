import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("🐾 Drag & Drop Animal Classifier")

model = load_model("animal_classifier_vgg16.h5")
labels = ['Bear','Bird','Cat','Cow','Deer','Dog','Dolphin','Elephant',
          'Giraffe','Horse','Kangaroo','Lion','Panda','Tiger','Zebra']

st.markdown(
    """
    <style>
    section[data-testid="stFileUploader"] div div div {
        border: 3px dashed #1abc9c;
        padding: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

file = st.file_uploader(
    "⬇️ Drag an image here or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Your image", width=300)
    img = img.resize((224, 224))
    arr = image.img_to_array(img)[np.newaxis] / 255.0
    pred = model.predict(arr)
    st.success(f"**{labels[np.argmax(pred)]}**")
