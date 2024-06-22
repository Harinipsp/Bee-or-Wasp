import streamlit as st
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

# Load models
custom_model = load_model('models/bee2.h5')
mobilenet_model = load_model('models/mobilenet_model.h5')
vgg16_model = load_model('models/vgg16_model.h5')
densenet_model = load_model('models/densenet_model.h5')

# Label map (same for all models)
label_map = {0: 'bee', 1: 'wasp'}

def preprocess_image(image, img_height, img_width):
    img = cv.resize(image, (img_height, img_width))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image_class(model, image, img_height, img_width):
    img = preprocess_image(image, img_height, img_width)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = label_map[predicted_class_index]
    accuracy = np.max(prediction)
    return predicted_class_name, accuracy

st.title("Bee or Wasp Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    custom_prediction, custom_accuracy = predict_image_class(custom_model, image, 25, 25)
    mobilenet_prediction, mobilenet_accuracy = predict_image_class(mobilenet_model, image, 32, 32)
    vgg16_prediction, vgg16_accuracy = predict_image_class(vgg16_model, image, 32, 32)
    densenet_prediction, densenet_accuracy = predict_image_class(densenet_model, image, 32, 32)

    st.write(f"Custom Model Prediction: {custom_prediction} (Accuracy: {custom_accuracy:.2f})")
    st.write(f"MobileNet Prediction: {mobilenet_prediction} (Accuracy: {mobilenet_accuracy:.2f})")
    st.write(f"VGG16 Prediction: {vgg16_prediction} (Accuracy: {vgg16_accuracy:.2f})")
    st.write(f"DenseNet Prediction: {densenet_prediction} (Accuracy: {densenet_accuracy:.2f})")

    # Accuracies for the graph
    accuracies = {
        'Custom Model': custom_accuracy,
        'MobileNet': mobilenet_accuracy,
        'VGG16': vgg16_accuracy,
        'DenseNet': densenet_accuracy
    }

    # Plotting the comparison graph
    plt.figure(figsize=(10, 5))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')

    # Save the plot to a PNG image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    st.image(buf, caption='Model Performance Comparison')
