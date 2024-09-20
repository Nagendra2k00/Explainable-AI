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

![image](https://github.com/user-attachments/assets/6ce98dfc-dc6c-4149-994f-f6bdde57182e)
![image](https://github.com/user-attachments/assets/8d5e6e22-df73-411e-889f-b41b5a5a2064)
![image](https://github.com/user-attachments/assets/6ce98dfc-dc6c-4149-994f-f6bdde57182e)
![image](https://github.com/user-attachments/assets/6ce98dfc-dc6c-4149-994f-f6bdde57182e)

### Input Image:
![image](https://github.com/user-attachments/assets/6ce98dfc-dc6c-4149-994f-f6bdde57182e)

### Grad-CAM Image:
![image](https://github.com/user-attachments/assets/6ce98dfc-dc6c-4149-994f-f6bdde57182e)

### Confidence Level:
![image](https://github.com/user-attachments/assets/6ce98dfc-dc6c-4149-994f-f6bdde57182e)

### Understanding Output:
![image](https://github.com/user-attachments/assets/6ce98dfc-dc6c-4149-994f-f6bdde57182e)

## 📝 Notes

- The `scene_classification_model.h5` file is not included in this repository. You must run the `Scene_classification.ipynb` notebook to generate the model file before running the Streamlit app.
- The dataset is also not included in this repository due to its size. You need to download the dataset separately and place it in the appropriate directory as mentioned in the notebook.
- If you encounter issues with dependencies, you can manually install the necessary packages listed in `requirements.txt`.
