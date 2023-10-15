import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(page_title="NLB Maize Detection", page_icon="🌽", layout="wide")


# Define a function to preprocess an image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    return image


# Load the trained NLB detection model
model = load_model('nlb_detection_model.h5')

# Set the background color to dark blue
st.markdown(
    """
    <style>
    body {
        background-color: #03045e;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NLB Maize Detection")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True, output_format='JPEG', width=300)

    image = Image.open(uploaded_image)
    preprocessed_image = preprocess_image(image)

    # Perform NLB detection
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    result = 'Unhealthy' if prediction[0][0] > 0.5 else 'Healthy'

    if result == 'Unhealthy':
        st.warning("Your plants may be unhealthy. Consider spraying them with pesticide.")

    else:
        st.write(f"Prediction: {result}", width=300)

    st.write("")  # Add some spacing

st.write("Upload an image to detect NLB in maize leaves.")
