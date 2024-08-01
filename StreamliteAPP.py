#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg16 import  preprocess_input


# In[3]:


# model = load_model('brain_tumor_detection_model.h5')
model = load_model('TF_brain_tumor_detection_model.h5')


# In[8]:


def resize_and_padding(image , target_size):
    image.thumbnail(target_size , Image.LANCZOS)
    new_image = Image.new("RGB" , target_size ,  (255, 255, 255))
    left = (target_size[0] - image.width) // 2
    top  = (target_size[1] - image.height) // 2
    
    new_image.paste(image , (left , top))
    
    return new_image

def rescale(image):
    return image / 255.0

def normalize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

def convert_to_array(image):
    return np.array(image)



# In[9]:


def predict(image):
    image = resize_and_padding(image, (224, 224))  # Ensure the target size matches your model's input
    image = convert_to_array(image)
    image = rescale(image)
    image = preprocess_input(image)  # Use VGG16 preprocessing
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return 'Tumor' if prediction[0] > 0.5 else 'No Tumor'


# In[10]:


def main():
    st.title("Brain Tumor Detection")
    
    st.write("Upload an MRI image for prediction.")
    
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Predict
        result = predict(image)
        
        # Show the result
        st.write(f"Prediction: {result}")
    else:
        st.write("ERRROR")
if __name__ == "__main__":
    main()






