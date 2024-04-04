import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load precomputed feature vectors and corresponding filenames
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# Load pre-trained ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

# Create a sequential model with ResNet50 and GlobalMaxPooling2D layers
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Set the title of the Streamlit app
st.title('Fashion Recommender System')

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Function to extract features from an image using the model
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to recommend similar images based on extracted features
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File uploader widget
uploaded_file = st.file_uploader("Choose an image")

# If file is uploaded
if uploaded_file is not None:
    # If file is successfully saved
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, width=150)  # Set width to a medium value

        # Extract features from the uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Recommend similar images
        indices = recommend(features, feature_list)

        # Divide the screen into 3 columns for displaying images
        cols = st.columns(3)
        
        # Display recommended images
        for i in range(3):
            with cols[i]:
                st.image(filenames[indices[0][i]], width=200)  # Adjust the width here

    # If there's an error in file upload
    else:
        st.header("Some error occurred in file upload")




