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

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="🌍 Explainable Scene Classification 🌍", page_icon="🌍")

# Load the trained model
@st.cache
def load_classification_model():
    return load_model('scene_classification_model.h5')

model = load_classification_model()

# Set image size
IMG_SIZE = (224, 224)

# Get class names
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Directories
test_dir = './seg_test'

# Function to load and preprocess the image
def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

# Grad-CAM heatmap generation
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

# Function to generate Grad-CAM visualization
def display_gradcam(image_path, model, last_conv_layer_name='conv5_block3_out'):
    img_array = get_img_array(image_path, size=IMG_SIZE)
    
    # Get the model prediction
    preds = model.predict(img_array)
    pred_class_index = np.argmax(preds[0])  # Index of predicted class
    pred_class = class_names[pred_class_index]  # Predicted class name

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Load and preprocess the original image
    img = load_img(image_path)
    img = img_to_array(img)

    # Resize heatmap to match the image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Apply heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.uint8(superimposed_img)

    return img.astype('uint8'), superimposed_img, preds[0], pred_class

# Function to handle image selection and prediction
def handle_image_selection(img_path):
    original_image, gradcam_image, prediction_probs, pred_class = display_gradcam(img_path, model)
    
    # Convert confidence score to an integer percentage
    confidence_percentage = int(np.max(prediction_probs) * 100)
    
    return original_image, gradcam_image, prediction_probs, pred_class, confidence_percentage

# Custom CSS to improve the app's appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with a descriptive section
st.title("🌍 Explainable AI: Scene Classification with Grad-CAM")
st.markdown("""
Welcome to the **AI-powered Scene Classification App**! 🎉 
Explore how our model classifies scenes and understand its decision-making process using Grad-CAM visualization.

**How it works:**
1. Choose to upload your own image or select a random one from our dataset.
2. Our AI model will analyze the image and predict the scene category.
3. View the Grad-CAM heatmap to see which parts of the image influenced the model's decision.
4. Check the confidence scores for each possible category.

Let's dive in and explore the world of explainable AI! 🚀
""")

# Create two columns for the main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📸 Image Input")
    option = st.radio("Choose your input method:", ["Upload Custom Image", "Random Image from Dataset"])
    
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

with col2:
    st.header("🧠 AI Analysis")
    if img_path:
        with st.spinner("Analyzing image..."):
            original_image, gradcam_image, prediction_probs, pred_class, confidence_percentage = handle_image_selection(img_path)
        
        st.success("Analysis complete!")
        st.subheader(f"Prediction: {pred_class.capitalize()}")
        st.progress(confidence_percentage)
        st.write(f"Confidence: {confidence_percentage:.2f}%")

        tabs = st.tabs(["Original", "Grad-CAM", "Confidence Levels"])
        
        with tabs[0]:
            st.image(Image.fromarray(original_image), caption="Original Image", use_column_width=True)
        
        with tabs[1]:
            st.image(Image.fromarray(gradcam_image), caption="Grad-CAM Visualization", use_column_width=True)
            st.info("The Grad-CAM heatmap highlights the regions of the image that were most important for the model's prediction. Warmer colors (red) indicate higher importance.")
        
        with tabs[2]:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=class_names, y=prediction_probs, ax=ax, palette="viridis")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.xlabel("Classes")
            plt.ylabel("Confidence")
            st.pyplot(fig)
            st.info("This chart shows the model's confidence for each possible class. The highest bar corresponds to the predicted class.")

        # Explanation section
        st.subheader("📚 Understanding the Results")
        st.write(f"""
        - The model predicted this image to be a **{pred_class}** scene with {confidence_percentage:.2f}% confidence.
        - The Grad-CAM visualization highlights the areas of the image that were most influential in making this prediction.
        - The confidence levels chart shows how sure the model was about each possible class.
        
        Remember, while AI models can be very accurate, they're not perfect. Always use critical thinking when interpreting AI predictions!
        """)

        # Clean up the temporary file if it was uploaded
        if option == "Upload Custom Image" and img_path:
            os.remove(img_path)

    else:
        st.info("Please select an image to begin the analysis.")

# Add a FAQ section
st.markdown("---")
st.header("❓ Frequently Asked Questions")
faq = st.expander("Click to expand")
with faq:
    st.markdown("""
    **Q: What is Grad-CAM?**
    
    A: Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique for producing visual explanations for decisions from CNN-based models. It helps us understand which parts of an image are important for the model's prediction.

    **Q: How accurate is this model?**
    
    A: The model's accuracy can vary depending on the complexity and variety of the scenes it encounters. Always consider the confidence score provided with each prediction.

    **Q: Can I use this for commercial purposes?**
    
    A: This is a demonstration app. For commercial use, please consult with the model creators and check the licensing terms of the underlying technologies.

    **Q: How can I improve the model's performance?**
    
    A: Model performance can be improved by training on a larger and more diverse dataset, fine-tuning the model architecture, and using techniques like data augmentation.
    """)

# Add a footer
st.markdown("---")
st.markdown("👨‍💻 Developed with ❤️ using Streamlit | 🔧 Model: ResNet50 | 🎨 Visualization: Grad-CAM")
