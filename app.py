# Instructions
st.subheader("Instructions")
st.write("1. Upload an image of a maize leaf OR click the button below to capture a photo with your camera.")
st.write("2. We will analyze the image and provide you with the result.")
st.write("3. If your plants are unhealthy, we recommend some fertilizers for you.")

# Capture image using the camera
if st.button("Capture Photo"):
    captured_image = st.camera(label="Capture a photo of a maize leaf")
    
    if captured_image is not None:
        # Display captured image
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        
        # Perform NLB detection on the captured image
        preprocessed_image = preprocess_image(captured_image)
        with st.spinner("Analyzing..."):
            prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
            prediction_probability = prediction[0][0]

        st.write(f"Prediction Probability: {prediction_probability:.2f}")

        # Provide recommendations based on the prediction
        if prediction_probability > 0.5:
            st.warning("Your plants may be unhealthy. Consider taking the following steps:")
            st.write("- Consult with an agricultural expert.")
            st.write("- Apply appropriate treatments.")

            # List of recommended fertilizers
            st.subheader("Recommended Fertilizers:")
            st.write("- [Booster Foliar Fertilizer 1Ltr](https://cheapthings.co.ke/product/booster-foliar-fertilizer-1ltr/?gad=1&gclid=Cj0KCQjwhL6pBhDjARIsAGx8D59O3FXxJTZkvS9UTNG8iNWSBqVuQ6DNVfmrVQNTImX0ohgp80AX1qIaAvlJEALw_wcB)")
            #st.write("- [Maize Pro-Gro Fertilizer 5Kg](https://cheapthings.co.ke/product/maize-pro-gro-fertilizer-5kg/?gad=1&gclid=Cj0KCQjwhL6pBhDjARIsAGx8D5ws7zqXY4_8ssJQqMuY-HOKJyoSBW96EO05Hh5uhQ8Fu8cVDOViJwcaAqVREALw_wcB)")
            # Add more recommended products and their links if desired

        else:
            st.success("Your maize plants appear to be healthy. Here are some tips for maintaining their well-being:")
            st.write("- Maintain a proper watering schedule.")
            st.write("- Follow fertilization recommendations.")
    else:
        st.warning("Photo capture was canceled or failed.")
