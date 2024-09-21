import streamlit as st
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import seaborn as sns
from PIL import Image
import time

# Set page config
st.set_page_config(layout="wide", page_title="Scene Classification AI", page_icon="🌍")

# Load the trained model
@st.cache_resource
def load_classification_model():
    return load_model('scene_classification_model.h5')

model = load_classification_model()

# Set image size
IMG_SIZE = (224, 224)

# Class names
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
test_dir = './seg_test'

def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image_path, model, last_conv_layer_name='conv5_block3_out'):
    img_array = get_img_array(image_path, size=IMG_SIZE)
    preds = model.predict(img_array)
    pred_class_index = np.argmax(preds[0])
    pred_class = class_names[pred_class_index]
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    img = load_img(image_path)
    img = img_to_array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.uint8(superimposed_img)

    return img.astype('uint8'), superimposed_img, preds[0], pred_class

def handle_image_selection(img_path):
    start_time = time.time()
    original_image, gradcam_image, prediction_probs, pred_class = display_gradcam(img_path, model)
    prediction_time = time.time() - start_time
    confidence_percentage = int(np.max(prediction_probs) * 100)
    return original_image, gradcam_image, prediction_probs, pred_class, confidence_percentage, prediction_time

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;700&display=swap');
    * {
        font-family: 'Quicksand', sans-serif;
    }
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FFFFFF;
        color: #FF0000;
        font-weight: bold;
        border: 1px solid #FF0000;
        border-radius: 0.5rem;
        padding: 0.5rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFFFFF;
    }
    .stProgress .st-bo {
        background-color: #28a745;
    }
    h1 {
        color: #343a40;
    }
    h2, h3 {
        color: #495057;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
        padding: 1rem;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 14px;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    .custom-info-box {
        background-color: #e9ecef;
        border-left: 5px solid #007bff;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .custom-success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .img-container {
        display: block;
        text-align: center;
        margin: 0 auto;
    }
    .img-container img {
        max-width: 100%;
        height: auto;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Scene Classification")
    st.subheader("Input Options")
    option = st.radio("Choose Your Input Method", ["Upload Custom Image", "Random Image from Dataset"])

    img_path = None
    if option == "Upload Custom Image":
        uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img_path = f"./temp_{uploaded_file.name}"
            img.save(img_path)
            st.image(img, caption="Uploaded Image", use_column_width=True)
    else:
        if st.button("🎲 Select Random Image"):
            category = random.choice(class_names)
            folder_path = os.path.join(test_dir, category)
            img_path = os.path.join(folder_path, random.choice(os.listdir(folder_path)))
            img = Image.open(img_path)
            st.image(img, caption="Selected Random Image", use_column_width=True)
            st.write(f"Original Class: {category.capitalize()}")

# Main content
st.title("Scene Classification with Explainable AI")
st.markdown("""<div class="custom-info-box">
    Welcome to our AI-powered Scene Classification App! 🎉<br>
    Explore how our model classifies scenes and understand its decision-making process using Grad-CAM visualization.
</div>""", unsafe_allow_html=True)

if img_path:
    with st.spinner("Analyzing image..."):
        original_image, gradcam_image, prediction_probs, pred_class, confidence_percentage, prediction_time = handle_image_selection(img_path)

    st.markdown(f"""
    <div class="custom-success-box">
        <h3>Analysis Results</h3>
        <p><strong>Prediction:</strong> {pred_class.capitalize()}</p>
        <p><strong>Confidence:</strong> {confidence_percentage:.2f}%</p>
        <p><strong>Prediction Time:</strong> {prediction_time:.2f} seconds</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(Image.fromarray(original_image), use_column_width=True, caption='Original Image')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Grad-CAM Visualization")
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(Image.fromarray(gradcam_image), use_column_width=True, caption='Grad-CAM Image')
        st.markdown('</div>', unsafe_allow_html=True)
        st.info("The Grad-CAM heatmap highlights the regions of the image that were most important for the model's prediction. Warmer colors (red) indicate higher importance.")

    st.subheader("Confidence Levels")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=class_names, y=prediction_probs, ax=ax, palette="viridis")
    plt.xticks(rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Confidence")
    st.pyplot(fig)

    if option == "Upload Custom Image" and img_path:
        os.remove(img_path)
else:
    st.info("Please select an image to begin the analysis.")

# FAQ section
st.markdown("---")
st.header("❓ Frequently Asked Questions")

faqs = [
    {
        "question": "What is Grad-CAM?",
        "answer": "Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique for producing visual explanations for decisions from CNN-based models. It helps us understand which parts of an image are important for the model's prediction.",
    },
    {
        "question": "How accurate is this model?",
        "answer": "The model's accuracy can vary depending on the complexity and variety of the scenes it encounters. Always consider the confidence score provided with each prediction.",
    },
    {
        "question": "Can I use this for commercial purposes?",
        "answer": "This is a demonstration app. For commercial use, please consult with the model creators and check the licensing terms of the underlying technologies.",
    },
    {
        "question": "How can I improve the model's performance?",
        "answer": "Model performance can be improved by training on a larger and more diverse dataset, fine-tuning the model architecture, and using techniques like data augmentation.",
    },
]

for faq in faqs:
    with st.expander(f"🔍 {faq['question']}"):
        st.markdown(f"<div class='custom-info-box'>{faq['answer']}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p>👨‍💻 Developed using Streamlit | 🔧 Model: ResNet50 | 🎨 Visualization: Grad-CAM</p>
    <p>Developed by Nagendra Kumar K S</p>
</div>
""", unsafe_allow_html=True)
