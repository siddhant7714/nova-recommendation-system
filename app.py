import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load pre-trained ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a sequential model with ResNet50 and GlobalMaxPooling2D layers
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    """
    Function to extract features from an image using a pre-trained ResNet50 model.

    Parameters:
        img_path (str): Path to the image file.
        model (tensorflow.keras.Model): Pre-trained ResNet50 model.

    Returns:
        numpy.ndarray: Normalized feature vector extracted from the image.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Extract features using the model and flatten the output
    result = model.predict(preprocessed_img).flatten()

    # Normalize the feature vector
    normalized_result = result / norm(result)

    return normalized_result

# Get list of image filenames
filenames = [os.path.join('images', file) for file in os.listdir('images')]

# Extract features for each image and store in a list
feature_list = [extract_features(file, model) for file in tqdm(filenames)]

# Save the extracted features and filenames to pickle files
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))


