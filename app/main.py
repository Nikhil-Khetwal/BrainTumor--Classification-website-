import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import time
import requests
from streamlit.components.v1 import html

# Function to preprocess the image
def preprocess_image(image):
    target_size = (224, 224)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype('float32') / 255.0

# Function to predict the class of the image
def predict_image_class(model, image, class_indices):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

def main():
    st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="wide")

    # Function to fetch Google Analytics HTML code
    def fetch_google_analytics_code(google_analytics_url):
        response = requests.get(google_analytics_url)
        if response.status_code == 200:
            return response.text
        else:
            return None

    # Google Analytics HTML URL
    google_analytics_url = "https://github.com/Nikhil-Khetwal/BrainTumor--Classification-website-/raw/master/app/google_analytics.html"

    # Fetch Google Analytics HTML code
    google_analytics_code = fetch_google_analytics_code(google_analytics_url)

    if google_analytics_code:
        # Display Google Analytics tracking code
        st.components.v1.html(google_analytics_code, height=0)
    else:
        st.warning("Google Analytics HTML code not found or unable to fetch.")

    # Load the pre-trained model
    model_path = "https://github.com/Nikhil-Khetwal/BrainTumor--Classification-website-/raw/master/app/trained_model/BT.h5"
    model_filename = "BT.h5"

    try:
        # Download the model file locally
        with open(model_filename, "wb") as f:
            response = requests.get(model_path)
            f.write(response.content)

        # Load the model from the local file
        model = tf.keras.models.load_model(model_filename)

        # URL to the raw class indices file on GitHub
        class_indices_path = "https://github.com/Nikhil-Khetwal/BrainTumor--Classification-website-/raw/master/app/class_indices.json"

        # Load the class indices from the raw GitHub URL
        response = requests.get(class_indices_path)
        class_indices = json.loads(response.content)

    except Exception as e:
        st.error(f"Failed to load model or class indices: {e}")
        return

    # Dictionary containing information related to different tumor types
    tumor_info = {
        "Glioma": {
            "Description": "Gliomas are often treated with a combination of surgery, radiation therapy, and chemotherapy. The treatment plan depends on the size and location of the tumor.",
            "Symptoms": [
                "Headaches", "Seizures", "Nausea or vomiting", "Changes in vision",
                "Memory loss", "Weakness or paralysis"
            ]
        },
        "Meningioma": {
            "Description": "Meningiomas are usually slow-growing tumors that may not require immediate treatment. However, they can cause symptoms if they press against the brain or spinal cord.",
            "Symptoms": [
                "Headaches", "Weakness or numbness in arms or legs", "Changes in vision or hearing",
                "Seizures", "Personality changes"
            ]
        },
        "Pituitary": {
            "Description": "Pituitary tumors can affect hormone levels in the body, leading to a variety of symptoms. Treatment options include medication, surgery, and radiation therapy.",
            "Symptoms": [
                "Headaches", "Vision loss or changes", "Fatigue", "Mood changes",
                "Irregular menstrual periods", "Weight gain or loss", "Loss of libido"
            ]
        },
        "No Tumor": {
            "Description": "No tumor detected. Regular check-ups are recommended to monitor any changes in health.",
            "Symptoms": []
        }
    }

    # Center the title
    st.title("🧠 Brain Tumor Classifier")

    st.markdown("""
    Upload an MRI brain scan image (JPG, JPEG, or PNG) for classification.
    """)

    # Display warning message
    st.warning("Warning: This is a machine learning model and results should not be used for self-diagnosis. Always consult a medical professional for any health concerns.")

    # File uploader
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    st.divider()

    prediction = None  # Initialize prediction variable

    # Flag to track if an image is uploaded
    image_uploaded = False

    if uploaded_image is not None:
        image_uploaded = True
        image = Image.open(uploaded_image)

        st.write("")
        st.write("")

        # Split the page into two columns
        col1, col2 = st.columns([1, 1.5])

        # Display uploaded image and classify button in the first column
        with col1:
            st.write(""" ### ✔️ Image Uploaded""")
            st.image(image, width=250)
            # Empty space to center the button
            # Classification button
            if st.button("Classify"):
                with st.spinner("Classifying..."):
                    time.sleep(2)  # Simulate classification time
                    prediction = predict_image_class(model, image, class_indices)

        # Display prediction output and tumor information in the second column
        with col2:
            if prediction:
                st.success(f"Prediction: {prediction}")
                if prediction in tumor_info:
                    st.subheader("Tumor Information")
                    st.write(tumor_info[prediction]["Description"])
                    if prediction != "No Tumor":
                        st.subheader("Symptoms")
                        # Display symptoms for the predicted tumor type
                        for symptom in tumor_info[prediction]["Symptoms"]:
                            st.write(f"- {symptom}")
                        # Display warning message for predictions indicating a tumor
                        st.warning("If you experience any of these symptoms, it's important to consult a medical professional for further evaluation and guidance.")
    
    # Display image without rounded corners at the bottom if no image is uploaded
    if not image_uploaded:
        st.title("📚 Inforamtion on Brain Tumor")

        col1, col2 = st.columns([1, 1])

        # Display uploaded image and classify button in the first column
        with col1:
            st.write("")
            st.write("")
            st.write("")

            st.image("https://github.com/Nikhil-Khetwal/BrainTumor--Classification-website-/raw/master/app/img/2.png", width=300, output_format="JPEG", use_column_width=300)
            st.markdown("""
                ### What is a Tumor?

                - A tumor is an abnormal mass of tissue resulting from uncontrolled cell growth.
                - These growths can occur in various body parts, including the brain. 
                - Tumors are categorized as benign or malignant.
                - Benign tumors are non-cancerous and grow slowly.
                - while malignant tumors are cancerous and aggressive, potentially spreading to other areas of the body. 
                - Symptoms vary based on location and size, often including headaches and seizures.
                """)

        # Display markdown content in the second column
        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
          
            st.image("https://github.com/Nikhil-Khetwal/BrainTumor--Classification-website-/raw/master/app/img/3.png", width=300, output_format="JPEG", use_column_width=300)
            st.markdown("""
            ### Symptoms of Brain Tumors

            - Symptoms of brain tumors can vary depending on their size, location, and rate of growth. Common symptoms include:
                - Headaches and seizures
                - Nausea or vomiting
                - Changes in vision or hearing
                - Difficulty with balance or coordination
                - Changes in mood or personality
            """)

if __name__ == "__main__":
    main()
