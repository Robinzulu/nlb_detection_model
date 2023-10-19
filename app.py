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
        # Procedure to Cure NLB in Maize
        st.header("Procedure to Cure Northern Leaf Blight (NLB) in Maize:")

        # Step 1
        st.markdown("1. **Identification:**")
        st.markdown("   - Carefully examine the corn plants for symptoms of NLB. These symptoms include long, tan lesions with dark brown borders on the leaves, which often merge and result in large, irregularly shaped spots.")

        # Step 2
        st.markdown("2. **Isolation:**")
        st.markdown("   - As soon as you notice NLB symptoms, isolate the infected plants or areas from the healthy ones to prevent the disease from spreading.")
 
        #  Step 3
        st.markdown("3. **Pruning and Removal:**")
        st.markdown("   - Remove and destroy severely infected leaves and plants. This reduces the disease's severity and minimizes the spread of spores.")

        # Step 4
        st.markdown("4. **Fungicide Application:**")
        st.markdown("   - Consider applying fungicides that are effective against NLB. Consult with a local agricultural extension service or expert for guidance on the appropriate fungicide and application timing.")

        # Step 5
        st.markdown("5. **Crop Rotation:**")
        st.markdown("   - Implement a crop rotation plan, avoiding planting corn in the same field for at least two years after NLB infection. This helps break the disease cycle.")

        # Step 6
        st.markdown("6. **Resistant Varieties:**")
        st.markdown("   - Choose maize varieties known to be resistant to NLB for future plantings. Resistant varieties can significantly reduce the risk of disease.")

        # Step 7
        st.markdown("7. **Monitoring and Early Intervention:**")
        st.markdown("   - Regularly inspect your maize plants for signs of NLB throughout the growing season. Early detection allows for prompt intervention.")

        # Step 8
        st.markdown("8. **Timely Watering:**")
        st.markdown("   - Maintain a consistent and appropriate watering schedule. Overly wet conditions can promote NLB development.")

        # Step 9
        st.markdown("9. **Proper Fertilization:**")
        st.markdown("   - Follow recommended fertilization practices. Avoid excessive nitrogen, which can make plants more susceptible to NLB.")

        # Step 10
        st.markdown("10. **Improve Air Circulation:**")
        st.markdown("    - Ensure good air circulation between plants by planting them at appropriate distances. This reduces humidity and minimizes conditions favorable for disease development.")

        # Step 11
        st.markdown("11. **Mulching:**")
        st.markdown("    - Use mulch to prevent soil splashes on leaves, which can carry fungal spores. This minimizes the risk of NLB spread.")

        # Step 12
        st.markdown("12. **Hygiene and Sanitation:**")
        st.markdown("    - Practice good field hygiene by cleaning equipment, clothing, and shoes to prevent the introduction of spores from other areas.")

        #     Step 13
        st.markdown("13. **Post-Harvest Debris Removal:**")
        st.markdown("    - After harvesting, remove and destroy crop debris and residue to eliminate potential overwintering sites for NLB spores.")

        # Step 14
        st.markdown("14. **Regular Monitoring:**")
        st.markdown("    - Continue monitoring your fields for any resurgence of NLB in subsequent planting seasons and take necessary measures as needed.")

       # Step 15
        st.markdown("15. **Consult Experts:**")
        st.markdown("    - If NLB persists or worsens despite your efforts, seek advice from agricultural experts or extension services. They can provide tailored recommendations for your specific situation.")


    else:
        st.success("Your maize plants appear to be healthy. Here are some tips for maintaining their well-being:")
        st.markdown("- Maintain a proper watering schedule.")
        st.markdown("- Follow fertilization recommendations.")
        
        st.header("Recommended Fertilizers:")
        st.markdown("- [Booster Foliar Fertilizer 1Ltr](https://cheapthings.co.ke/product/booster-foliar-fertilizer-1ltr/?gad=1&gclid=Cj0KCQjwhL6pBhDjARIsAGx8D59O3FXxJTZkvS9UTNG8iNWSBqVuQ6DNVfmrVQNTImX0ohgp80AX1qIaAvlJEALw_wcB)")

#st.write("Upload an image to detect NLB in maize leaves.")
# Your additional recommendations
st.header("Recommendations for Healthy Maize Farming:")
st.markdown("1. **Soil Preparation:**")
st.markdown("   - Conduct a soil test before planting to determine the pH and nutrient levels in your soil. Adjust soil pH if necessary to fall within the optimal range for corn (around 6.0 to 6.8).")
st.markdown("   - Incorporate organic matter, such as compost or well-rotted manure, into the soil before planting to improve its structure and water-holding capacity.")

st.markdown("2. **Planting Depth and Spacing:**")
st.markdown("   - Plant corn seeds at the recommended depth, typically around 1.5 to 2 inches (4-5 cm) deep.")
st.markdown("   - Follow proper row spacing and plant spacing recommendations for the specific corn variety you are growing. Adequate spacing ensures good air circulation and prevents overcrowding.")

st.markdown("3. **Weed Control:**")
st.markdown("   - Implement a weed control strategy to minimize competition for nutrients, water, and sunlight. This can include using mulch, cultivating the soil, or using herbicides.")
