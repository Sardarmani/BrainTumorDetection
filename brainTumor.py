#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
from PIL import Image
import  tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense ,Flatten , Dropout


# In[2]:


datadir = os.path.join('.','dataset','brain_tumor_dataset')


# In[3]:


yes_path = os.path.join(datadir ,  'yes')
no_path = os.path.join(datadir , 'no')


# In[4]:


def display_sample_img():
    tumor_image_name = os.path.join(yes_path , os.listdir(yes_path)[1])
    yes_sample = Image.open(tumor_image_name)
    plt.imshow(yes_sample)
    plt.axis('off')
    plt.title('Image With Tumor')
    plt.show()
    
    
    no_tumor_image = os.path.join(no_path , os.listdir(no_path)[0])
    no_sample = Image.open(no_tumor_image)
    plt.imshow(no_sample)
    plt.title('Image Without Tumor')
    plt.axis('off')
    plt.show()
    


# In[5]:


display_sample_img()


# In[6]:


def resize_and_padding(image , target_size):
    image.thumbnail(target_size , Image.LANCZOS)
    new_image = Image.new("RGB" , target_size ,  (255, 255, 255))
    left = (target_size[0] - image.width) // 2
    top  = (target_size[1] - image.height) // 2
    
    new_image.paste(image , (left , top))
    
    return new_image



# In[7]:


def rescale(image):
    return image / 255.0


# In[8]:


def normalize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std


# In[9]:


def convert_to_array(image):
    return np.array(image)


# In[10]:


def load_and_preprocess(datadir , target_size = (244,244)):
    images = []
    labels = []
    for label in ['yes' , 'no']:
        folder_path = os.path.join(datadir , label)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path , filename)
            image = Image.open(image_path).convert('RGB')
            image = resize_and_padding(image , target_size)
            image = convert_to_array(image)
            image = rescale(image)
            image = normalize_image(image)
            images.append(image)
            labels.append(1 if label == 'yes' else 0)
    images = np.array(images)
    labels = np.array(labels)
    return images , labels


# In[11]:


def split_data(images , labels ,test_size =0.2 , random_state = 42):
    return train_test_split(images , labels , test_size=test_size , random_state=random_state)


# In[12]:


def create_data_generator(train_data ,  val_data , img_height , img_width , batch_size):
    train_images , train_labels = train_data
    val_images , val_labels = val_data
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(
    train_images , train_labels,
        batch_size = batch_size
    )
    
    val_generator = val_datagen.flow(
    val_images , val_labels,
        batch_size = batch_size
    )
    
    return train_generator ,  val_generator


# In[21]:


def build_CNN(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
      
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[22]:


input_shape = (244, 244, 3)
model = build_CNN(input_shape)
model.summary()


# In[ ]:





# In[23]:


images , labels = load_and_preprocess(datadir)


# In[24]:


train_images, val_images, train_labels, val_labels = split_data(images, labels)


# In[25]:


img_height, img_width = 244, 244
batch_size = 32
train_data = (train_images, train_labels)
val_data = (val_images, val_labels)

train_generator, validation_generator = create_data_generator(
    train_data, val_data, img_height, img_width, batch_size
)


# In[26]:


input_shape = (244, 244, 3)
model = build_CNN(input_shape)
model.summary()


# In[ ]:





# In[28]:


history = model.fit(
        
        train_generator,
        steps_per_epoch=len(train_images) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(val_images) // batch_size,
        epochs=10   
)


# In[29]:


model.save('brain_tumor_detection_model.h5')


# In[1]:


def predict(image):
    image = resize_and_padding(image, (224, 224))  # Ensure the target size matches your model's input
    image = convert_to_array(image)
    image = rescale(image)
    image = normalize_image(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return 'Tumor' if prediction[0] > 0.5 else 'No Tumor'


# In[2]:


model = load_model('brain_tumor_detection_model.h5')


# In[ ]:




