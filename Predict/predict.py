import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from keras.src.layers import Dense, Flatten
from keras.src.applications.vgg16 import VGG16, preprocess_input

# Define the image size
IMAGE_SIZE = [224, 224]

# Paths to the training and test data directories
train_path = 'C:/Users/moham/OneDrive/Desktop/Pnemonia/Mlpnemo/Predict/static/Predict/img/chest_xray/train'
test_path = 'C:/Users/moham/OneDrive/Desktop/Pnemonia/Mlpnemo/Predict/static/Predict/img/chest_xray/test'

# Load the VGG16 model with pretrained weights
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the layers in the VGG16 model
for layer in vgg.layers:
    layer.trainable = False

# Build the model architecture
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(2, activation='softmax'))  # Output layer with 2 classes (normal and pneumonia)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Augment the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescale the test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of augmented data for training and rescaled data for testing
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5  # Increase the number of epochs
)

# Save the model
model.save('chest_xray.h5')
