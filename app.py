import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import VideoTransformer, webrtc_streamer

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

# Set the background color and text color
st.markdown(
    """
    <style>
    body {
        background-color: darkgreen; /* Dark blue background */
        color: darkgreen; /* White text */
    }
    h1 {
        color: darkgreen; /* Dark green title */
    }
    h2 {
        color: darkgreen; /* Orange for subheaders */
    }
    p {
        color: black; /* Yellow for paragraphs */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("NLB Maize Detection")
st.write("Detect Northern Leaf Blight (NLB) in maize plants from images.")

# Instructions with additional recommendations and photo requirements
st.header("Instructions")
st.markdown("1. Choose an option to provide an image for NLB detection:")
st.markdown("   - Option 1: Upload an image of a maize leaf.")
st.markdown("   - Option 2: Use your device's camera to capture an image.")
st.markdown("2. We will analyze the image and provide you with the result.")
st.markdown("3. For accurate results, make sure your photo meets these criteria:")
st.markdown("   - The photo should not contain too many leaves; focus on a single leaf or a few leaves.")
st.markdown("   - Ensure the photo is clear and well-lit.")
st.markdown("4. If your plants are healthy, we recommend some fertilizers for you.")
st.markdown("5. If your plants are unhealthy, we recommend taking the following steps:")
st.markdown("6. If your plants are unhealthy, we recommend taking the following steps:")

# Function to perform NLB detection on an image
def perform_nlb_detection(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    prediction_probability = prediction[0][0]
    return prediction_probability

# VideoTransformer class for webcam input
class VideoTransformer(VideoTransformer):
    def transform(self, frame):
        image = Image.fromarray(frame)
        prediction_probability = perform_nlb_detection(image)
        return f"Prediction Probability: {prediction_probability:.2f}"

# Option to upload an image
st.header("Option 1: Upload an Image")
uploaded_image = st.file_uploader("Upload an image of a maize leaf", type=["jpg", "jpeg", "png"])

# Option to capture an image from the camera
st.header("Option 2: Capture Image from Camera")
webrtc_streamer(
    key="camera",
    video_transformer_factory=VideoTransformer,
)

# Perform NLB detection based on user choice
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_image)
    st.header(perform_nlb_detection(image))
elif "camera" in st.session_state and st.session_state.camera is not None:
    if st.session_state.camera > 0.5:
        st.warning("Your plants may be unhealthy. Consider taking the following steps:")
        st.markdown("- Consult with an agricultural expert.")
        st.markdown("- Apply appropriate treatments.")
    else:
        st.success("Your maize plants appear to be healthy. Here are some tips for maintaining their well-being:")
        st.markdown("- Maintain a proper watering schedule.")
        st.markdown("- Follow fertilization recommendations.")
        st.header("Recommended Fertilizers:")
        st.markdown("- [Booster Foliar Fertilizer 1Ltr](https://cheapthings.co.ke/product/booster-foliar-fertilizer-1ltr/?gad=1&gclid=Cj0KCQjwhL6pBhDjARIsAGx8D59O3FXxJTZkvS9UTNG8iNWSBqVuQ6DNVfmrVQNTImX0ohgp80AX1qIaAvlJEALw_wcB)")
