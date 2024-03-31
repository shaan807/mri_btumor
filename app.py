

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import keras

# Load the trained model
model_path = os.path.join("models", "brain_tumor_classifier.h5")
model = keras.models.load_model(model_path)

# Define labels
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Create folders for each class if they don't exist
output_folders = {label: os.path.join("results", label.replace(" ", "_")) for label in labels}
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Function to preprocess the image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(255, 255))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Function to make predictions and save results
def make_prediction(image_file):
    img_array = preprocess_image(image_file)
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    return predicted_class

# Streamlit app
def main():
    st.title("Brain Tumor Classifier")
    st.write("Upload an MRI/CT image to classify the type of brain tumor.")

    uploaded_file = st.file_uploader("Choose an MRI/CT image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded MRI/CT Image', use_column_width=True)

        # Make predictions when button is clicked
        if st.button("Classify"):
            predicted_class = make_prediction(uploaded_file)
            if predicted_class == "notumor":
                output_folder = output_folders["notumor"]
            else:
                output_folder = output_folders[predicted_class]

            # Save the image in the appropriate folder
            with open(os.path.join(output_folder, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

            if predicted_class == "notumor":
               st.success(f"The MRI/CT image is classified as: {predicted_class}. Results saved in 'no_tumor' folder.")
            else:
               st.markdown(f'<div style="background-color:#FF0000; color:#FFFFFF; padding:10px; border-radius:5px;">'
                            f'The MRI/CT image is classified as: {predicted_class}. Results saved in corresponding folder.'
                            f'</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()





















# import streamlit as st
# import numpy as np
# import os
# from PIL import Image
# import keras

# # Load the deep learning model
# model_path = os.path.join("models", "trained.h5")
# model = keras.models.load_model(model_path)

# # Function to make predictions
# def make_predictions(image):
#     img = Image.open(image)
#     img_resized = img.resize((255, 255))
#     rgb_img = img_resized.convert('RGB')
#     img_array = np.array(rgb_img, dtype=np.float64)
#     img_array = img_array.reshape(-1, 255, 255, 3)
#     predictions = model.predict(img_array)
#     class_labels = ["glioma", "meningioma", "notumor", "pituitary"]
#     predicted_class = class_labels[np.argmax(predictions)]
#     return predicted_class

# # Streamlit UI
# def main():
#     st.title('Brain Tumor Prediction App')

#     uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png", "jfif"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded MRI Image', use_column_width=True)
#         st.write("")
#         st.write("Classifying...")

#         prediction = make_predictions(uploaded_file)
#         st.success(f'Prediction: {prediction}')

# # Run the app
# if __name__ == '__main__':
#     main()



