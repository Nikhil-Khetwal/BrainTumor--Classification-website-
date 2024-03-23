import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import time

# Load the pre-trained model

def load_model():
    model_path =  "https://github.com/nikhilkhetwal/brain/raw/main/app/trained_model/Brain_Tumor.h5"
    return tf.keras.models.load_model(model_path)

# Load the class indices
def load_class_indices():
    class_indices_path = "https://raw.githubusercontent.com/nikhilkhetwal/brain/main/app/class_indices.json"
    with open(class_indices_path) as f:
        return json.load(f)


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

    # Load the pre-trained model and class indices
    model = load_model()
    class_indices = load_class_indices()


    # Dictionary containing information related to different tumor types
    tumor_info = {
        "Glioma": "Gliomas are often treated with a combination of surgery, radiation therapy, and chemotherapy. The treatment plan depends on the size and location of the tumor.",
        "Meningioma": "Meningiomas are usually slow-growing tumors that may not require immediate treatment. However, they can cause symptoms if they press against the brain or spinal cord.",
        "Pituitary Tumor": "Pituitary tumors can affect hormone levels in the body, leading to a variety of symptoms. Treatment options include medication, surgery, and radiation therapy.",
        "No Tumor": "No tumor detected. Regular check-ups are recommended to monitor any changes in health. If symptoms persist, consult with a medical professional for further evaluation and guidance."
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
        col1, col2 = st.columns([2, 7])

        # Display uploaded image and classify button in the first column
        with col1:
            st.write(""" ### ✔️ Image Uploaded""")
            st.image(image, width=300)
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
                    st.write(tumor_info[prediction])
            # Empty space to center the button
                    st.write("")
        
    # Display image without rounded corners at the bottom if no image is uploaded
    if not image_uploaded:
        st.title("📚 Inforamtion on Brain Tumor")

        col1, col2 = st.columns([1, 1])

        # Display uploaded image and classify button in the first column
        with col1:
            st.write("")
            st.write("")
            st.write("")

            st.image("C:/Users/Admin/Downloads/brain/app/2.jpg", width=300, output_format="JPEG", use_column_width=300)
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
          
            st.image("C:/Users/Admin/Downloads/brain/app/3.jpg", width=300, output_format="JPEG", use_column_width=300)
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
