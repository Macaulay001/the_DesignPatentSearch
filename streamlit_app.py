import streamlit as st
import numpy as np
import clip
import torch
import os
import pandas as pd
from PIL import Image

# Function to Read Metadata and Create Dictionary
@st.cache_data
def load_metadata(metadata_path):
    labels_df = pd.read_csv(metadata_path, sep='\t', header=None)
    return {row[0]: (row[0], row[2]) for _, row in labels_df.iterrows()}



# Function to Load CLIP Model and Embeddings
@st.cache_resource
def load_model_and_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-B/32', device=device)
    image_embeddings = np.load('image_embeddings.npy')
    return model, image_embeddings

# Function to Find Similar Images
@st.cache_data
def find_similar_images(query_text, num_images, image_embeddings):
    text_inputs = clip.tokenize([query_text])
    query_text_embedding = model.encode_text(text_inputs).cpu().detach().numpy()
    similarity_scores = np.matmul(query_text_embedding, image_embeddings.T).squeeze()
    top_indices = np.argsort(similarity_scores)[-num_images:][::-1]
    top_image_paths = [image_paths[i] for i in top_indices]
    top_descriptions = [descriptions[i] for i in top_indices]
    return top_image_paths, top_descriptions

metadata_path = 'design_patent_captions.tsv'
metadata_dict = load_metadata(metadata_path)

# Image paths and descriptions
image_paths = []
descriptions = []
folder_path = "design-patent-images"
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        identifier = filename.split('.')[0]
        if identifier in metadata_dict:
            image_paths.append(os.path.join(folder_path, filename))
            descriptions.append(metadata_dict[identifier])

model, image_embeddings = load_model_and_embeddings()

# Streamlit Interface
st.title("Find Patent Images using CLIP")

query_text = st.text_input("Enter a description of the image you're looking for:")
num_images = st.slider("Select the number of images to be generated:", min_value=1, max_value=10, value=3)

if st.button("Generate Images"):
    if query_text:
        st.write("Finding similar images...")
        similar_images, similar_descriptions = find_similar_images(query_text, num_images, image_embeddings)

        #st.markdown(f"<center>Top {num_images} similar images:</center>", unsafe_allow_html=True)

        for i in range(0, len(similar_images), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(similar_images):
                    image_path = similar_images[i + j]
                    description = similar_descriptions[i + j]
                    cols[j].write(f"End frame for a futon {description[0]}")
                    cols[j].image(image_path)
        st.markdown("<center>Results of CLIP-based Semantic Search, Source: USPTO</center>", unsafe_allow_html=True)

