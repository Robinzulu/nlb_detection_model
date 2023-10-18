import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(page_title="NLB Maize Detection", page_icon="🌽", layout="centered")

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

# Add instructions
st.markdown("### Instructions")
st.write("This app is designed to detect Northern Leaf Blight (NLB) in maize plants. To use it, please upload an image of a maize leaf. We will analyze the image and provide you with a result.")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=False, output_format='JPEG', width=300)

    image = Image.open(uploaded_image)
    preprocessed_image = preprocess_image(image)

    # Perform NLB detection
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    prediction_probability = prediction[0][0]

    st.write(f"Prediction Probability: {prediction_probability:.2f}")

    if prediction_probability > 0.5:
        st.warning("Your plants may be unhealthy. Consider taking the following steps:")
        st.write("- Consult with an agricultural expert.")
        st.write("- Apply appropriate treatments.")
    else:
        st.success("Your maize plants appear to be healthy. Here are some tips for maintaining their well-being:")
        st.write("- Watering schedule.")
        st.write("- Fertilization recommendations.")

st.write("Upload an image to detect NLB in maize leaves.")
