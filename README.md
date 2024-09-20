# 🌍 Explainable AI: Scene Classification with Grad-CAM 🌍

This project demonstrates scene classification using a deep learning model (ResNet) and provides Grad-CAM visualizations to explain the model's decision-making. The project includes two main components:

- Jupyter Notebook: For training the scene classification model.
- Streamlit Web App: For interacting with the trained model to classify images and visualize Grad-CAM results.

## 🌟 Features

- Scene Classification: Predicts scenes such as buildings, forest, glacier, mountain, sea, and street.
- Grad-CAM Visualization: Shows which regions of the image influenced the model's prediction.
- Streamlit Interface: A simple, intuitive web interface for interacting with the model.

## 🚀 Getting Started

### 1. Clone the Repository

To begin, clone this repository to your local machine:

```
https://github.com/Nagendra2k00/Explainable-AI.git
```

### 2. Install Dependencies

Ensure you have Python installed, then install the required libraries using:

```
pip install -r requirements.txt
```

### 3. Train the Model

The scene classification model (scene_classification_model.h5) is not included in this repository. You need to train the model yourself by following these steps:

1. Open the `Scene_classification.ipynb` notebook in Jupyter or another notebook environment.
2. Run all the cells in the notebook. This will:
   - Load the dataset.
   - Train a ResNet model for scene classification.
   - Save the trained model as `scene_classification_model.h5` in the project directory.

**Important**: Ensure the `scene_classification_model.h5` file is saved in the same directory as `app.py` once the training is complete.

### 4. Running the Streamlit Web Application

After training the model and ensuring the `scene_classification_model.h5` file is in the project directory, you can launch the web application using Streamlit:

```
streamlit run app.py
```

This will open up a web interface where you can:
- Upload images or select random images from the dataset.
- View predictions along with Grad-CAM visualizations.

### 5. Using the App

- Upload or Select Image: Choose between uploading an image or selecting a random one from the provided dataset.
- Prediction & Confidence: The app will display the predicted scene and the model's confidence score.
- Grad-CAM Visualization: See which parts of the image influenced the model's decision with a Grad-CAM heatmap.

## 📂 Project Structure

```
.
├── app.py                         # Streamlit app for classification and Grad-CAM visualization
├── Scene_classification.ipynb     # Jupyter notebook for training the model
├── requirements.txt               # List of dependencies
├── README.md                      # Project documentation
└── scene_classification_model.h5  # (To be generated after training)
```

## 📸 Screenshots

### Data Flow
![Scene_classification](https://github.com/user-attachments/assets/273693b2-25f8-468d-a437-4d2f1eb2eeb5)

### Streamlit Interface
![image](https://github.com/user-attachments/assets/f5949f73-b34c-44d0-92a3-a36970b21065)

*The main interface of the Streamlit web application*

### Dataset
![image](https://github.com/user-attachments/assets/4b149673-34bf-462c-a489-4086ce69ab98)

*Example of a scene classification prediction*

### Grad-CAM Visualization
![s1](https://github.com/user-attachments/assets/a8fbd60d-4ec0-4dc2-a408-4ef0d5039564)

*Grad-CAM heatmap showing areas of importance for the classification*

### Input Image Example
![image](https://github.com/user-attachments/assets/b5f332ed-1235-4b18-a29a-418a3f0ff293)

*A sample input image for scene classification*

### Grad-CAM Result
![image](https://github.com/user-attachments/assets/63ac2cef-ffff-4fed-b9b4-83daf45273e4)

*Grad-CAM visualization overlaid on the input image*

### Confidence Level Display
![image](https://github.com/user-attachments/assets/415a1f63-dc72-40c7-874d-590f5c7f3bfa)

*Showing the model's confidence in its prediction*

### Understanding the Output
![image](https://github.com/user-attachments/assets/de48a0d5-9da3-4581-a2b5-b771e1227d2d)

*Explanation of how to interpret the model's output and Grad-CAM visualization*

## 📝 Notes

- The `scene_classification_model.h5` file is not included in this repository. You must run the `Scene_classification.ipynb` notebook to generate the model file before running the Streamlit app.
- The dataset is also not included in this repository due to its size. You need to download the dataset separately and place it in the appropriate directory as mentioned in the notebook.
- If you encounter issues with dependencies, you can manually install the necessary packages listed in `requirements.txt`.
