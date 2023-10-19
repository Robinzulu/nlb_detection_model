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

# Set the background color and text color
st.markdown(
    """
    <style>
    body {
        background-color: #03045e; /* Dark blue background */
        color: #ffffff; /* White text */
    }
    h1 {
        color: darkgreen; /* Dark green title */
    }
    h2 {
        color: Darkgreen; /* e for subheaders */
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

# Instructions
st.header("Instructions")
st.markdown("1. Upload an image of a maize leaf.")
st.markdown("2. We will analyze the image and provide you with the result.")
st.markdown("3. If your plants are unhealthy, we recommend some fertilizers for you.")

# Upload image
uploaded_image = st.file_uploader("Upload an image of a maize leaf", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    image = Image.open(uploaded_image)
    preprocessed_image = preprocess_image(image)

    # Perform NLB detection
    with st.spinner("Analyzing..."):
        prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
        prediction_probability = prediction[0][0]

    st.header(f"Prediction Probability: {prediction_probability:.2f}")

    # Provide recommendations based on the prediction
    if prediction_probability > 0.5:
        st.warning("Your plants may be unhealthy. Consider taking the following steps:")
        st.markdown("- Consult with an agricultural expert.")
        st.markdown("- Apply appropriate treatments.")

    else:
        st.success("Your maize plants appear to be healthy. Here are some tips for maintaining their well-being:")
        st.markdown("- Maintain a proper watering schedule.")
        st.markdown("- Follow fertilization recommendations.")
        
        st.header("Recommended Fertilizers:")
        st.markdown("- [Booster Foliar Fertilizer 1Ltr](https://cheapthings.co.ke/product/booster-foliar-fertilizer-1ltr/?gad=1&gclid=Cj0KCQjwhL6pBhDjARIsAGx8D59O3FXxJTZkvS9UTNG8iNWSBqVuQ6DNVfmrVQNTImX0ohgp80AX1qIaAvlJEALw_wcB)")

#st.write("Upload an image to detect NLB in maize leaves.")
