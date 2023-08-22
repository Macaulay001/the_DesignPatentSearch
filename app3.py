import streamlit as st
import clip
import torch
import numpy as np
from PIL import Image
import pandas as pd

# Function to load metadata
@st.cache_data
def load_metadata(metadata_path):
    labels_df = pd.read_csv(metadata_path, sep='\t', header=None)
    return labels_df.iloc[:, 0].tolist()

# Function to load CLIP model
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-B/32', device=device)
    return model

# Function to find top 5 similar images
@st.cache_data
def find_similar_images(query_text, num_images=5):
    image_embeddings = np.load('image_embeddings.npy')
    text_inputs = clip.tokenize([query_text]).to(device)
    query_text_embedding = model.encode_text(text_inputs).cpu().detach().numpy()
    similarity_scores = np.matmul(query_text_embedding, image_embeddings.T).squeeze()
    top_indices = np.argsort(similarity_scores)[-num_images:][::-1]
    top_images = [convert_image(loaded_images[i]) for i in top_indices]
    top_labels = [metadata_labels[i] for i in top_indices]
    return top_images, top_labels


def convert_image(image_array):
    print(f"Converting image with shape {image_array.shape} and data type {image_array.dtype}") # Debugging line
    if image_array.ndim == 2:  # Grayscale image
        return Image.fromarray(image_array.astype('uint8'), mode='L')
    elif image_array.ndim == 3 and image_array.shape[2] == 3:  # RGB image
        return Image.fromarray(image_array.astype('uint8'), mode='RGB')
    else:
        raise ValueError(f"Unexpected image shape: {image_array.shape}")


device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_clip_model()

# Load all images from the numpy array
loaded_images = np.load('all_images.npy')

# Load metadata labels
metadata_path = 'metadata_2007.txt'
metadata_labels = load_metadata(metadata_path)

# Streamlit interface
st.title("Find Patent Images using CLIP")
query_text = st.text_input("Enter a description of the image you're looking for:")
num_images = st.slider("Select the number of images to be generated:", min_value=1, max_value=10, value=5)

if st.button("Find Images"):
    if query_text:
        st.write("Finding similar images...")
        similar_images, similar_labels = find_similar_images(query_text, num_images)

        # Display images in groups of 3
        for i in range(0, len(similar_images), 3):
            cols = st.columns(3) # Create 3 columns
            for j in range(3):
                if i + j < len(similar_images):
                    image = similar_images[i + j]
                    label = similar_labels[i + j]
                    cols[j].image(image, use_column_width=True) # Display image in each column
                    cols[j].write(label) # Display label in each column

        st.markdown("<center>Results of CLIP-based Semantic Search, Source: USPTO</center>", unsafe_allow_html=True)