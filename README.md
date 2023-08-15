
# Find Patent Images using CLIP

This repository contains code to perform semantic search on a collection of design patent images using OpenAI's CLIP (Contrastive Language–Image Pre-Training) model.

## Table of Contents

- [Introduction](#introduction)
- [Preprocessing](#preprocessing)
- [Main Code](#main-code)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

The code is designed to search for images based on textual descriptions by understanding and representing both text and images in a common embedding space using CLIP.

## Preprocessing

### Reading Labels

Load design patent captions from a TSV file.

```python
labels_path = 'design_patent_captions.tsv'
labels_df = pd.read_csv(labels_path, sep='	', header=None)
```

### Loading CLIP Model

Load the CLIP model and required transformations.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load('ViT-B/32', device=device)
```

### Image Paths Extraction

Identify the folder containing images and extract their paths.

```python
folder_path = "design-patent-images"
image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]
```

### Processing Images and Texts

Process images and texts to generate embeddings.

```python
def process_images_and_texts(image_paths, text_labels, batch_size=128):
    # Processing code here
```

### Saving Embeddings

Save generated embeddings for later use.

```python
np.save('image_embeddings.npy', image_embeddings)
np.save('text_embeddings.npy', text_embeddings)
```

## Main Code

### Function to Load Metadata

```python
@st.cache_data
def load_metadata(metadata_path):
    labels_df = pd.read_csv(metadata_path, sep='	', header=None)
    return {row[0]: (row[0], row[2]) for _, row in labels_df.iterrows()}
```

### Streamlit Interface

Create a user-friendly interface using Streamlit.

```python
st.title("Find Patent Images using CLIP")
query_text = st.text_input("Enter a description of the image you're looking for:")
num_images = st.slider("Select the number of images to be generated:", min_value=1, max_value=10, value=5)
```

### Finding Similar Images

Find similar images based on user input.

```python
if st.button("Generate Images"):
    similar_images, similar_descriptions = find_similar_images(query_text, num_images, image_embeddings)
    # Display results
```

## Usage

1. Clone the repository.
2. Install dependencies.
3. Run the preprocessing script to generate embeddings.
4. Run the main code to launch the Streamlit app.
5. Open the app in a web browser and enter a query.

## Conclusion

The code constitutes a semantic search system leveraging the CLIP model, enabling powerful search functionality for design patent images based on textual descriptions. Suitable for various applications such as intellectual property management and design exploration.
