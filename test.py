import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

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

# Load and preprocess sample image
img = image.load_img('sample/DSC_0393.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Extract features from the sample image using the model
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Initialize NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

# Find nearest neighbors for the sample image features
distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

# Display the nearest neighbor images
for file in indices[0][1:4]:  # Skip the first index since it's the sample image itself
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512,512)))
    cv2.waitKey(0)
