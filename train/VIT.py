import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from transformers import TFAutoModelForImageClassification

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Paths to your train and validation data
train_path = "Dataset A/train"
val_path = "Dataset A/test"


# Generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),  # Size expected by ViT model
    batch_size=32,           # Batch size (adjust as needed)
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),  # Size expected by ViT model
    batch_size=32,           # Batch size (adjust as needed)
    class_mode='categorical')

from transformers import TFAutoModel
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the base Vision Transformer model
base_model = TFAutoModel.from_pretrained('google/vit-base-patch16-224')

# Define the correct input shape
input_layer = Input(shape=(224, 224, 3))

# Ensure the base model processes the input correctly
x = base_model(input_layer, training=False)[0]  # Correctly pass input tensor
x = GlobalAveragePooling2D()(x)
output_layer = Dense(4, activation='softmax')(x)  # Adjust for your number of classes

# Create the new model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,  # Number of epochs (adjust as needed)
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# Save the model
model.save('vit_base_model.h5')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from transformers import TFAutoModel, ViTFeatureExtractor
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Paths to your train and validation data
train_path = "Dataset A/train"
val_path = "Dataset A/test"

# Generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),  # Size expected by ViT model
    batch_size=32,           # Batch size (adjust as needed)
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),  # Size expected by ViT model
    batch_size=32,           # Batch size (adjust as needed)
    class_mode='categorical')

# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load the base Vision Transformer model
base_model = TFAutoModel.from_pretrained('google/vit-base-patch16-224')

# Define the correct input shape
input_layer = Input(shape=(224, 224, 3))

# Process input through the base model
x = base_model(input_layer)[0]
x = GlobalAveragePooling2D()(x)
output_layer = Dense(len(train_generator.class_indices), activation='softmax')(x)  # Dynamically set number of classes

# Create the new model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,  # Number of epochs (adjust as needed)
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# Save the model
model.save('vit_base_model.h5')

# # Import necessary libraries
# import os
# import numpy as np
# from transformers import ViTFeatureExtractor, TFViTForImageClassification, create_optimizer
# from tensorflow.keras.utils import image_dataset_from_directory
# import tensorflow as tf

# # Define paths and parameters
# data_dir = "Dataset A"  # Update with your dataset path
# train_dir = os.path.join(data_dir, 'train')
# val_dir = os.path.join(data_dir, 'test')
# image_size = (224, 224)
# batch_size = 32
# learning_rate = 2e-5
# num_epochs = 4
# model_name = 'google/vit-base-patch16-224'  # Choose the appropriate model

# # Initialize feature extractor
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# # Preprocess and load datasets
# def preprocess_images(image, label):
#     image = feature_extractor(images=image, return_tensors="tf").pixel_values
#     return tf.squeeze(image, axis=0), label

# train_dataset = image_dataset_from_directory(train_dir, label_mode='categorical', 
#                                              batch_size=batch_size, image_size=image_size)
# val_dataset = image_dataset_from_directory(val_dir, label_mode='categorical', 
#                                            batch_size=batch_size, image_size=image_size)

# train_dataset = train_dataset.map(preprocess_images)
# val_dataset = val_dataset.map(preprocess_images)

# # Load pre-trained ViT model
# num_labels = len(train_dataset.class_names)
# model = TFViTForImageClassification.from_pretrained(model_name, num_labels=num_labels)

# # Create optimizer with weight decay
# num_train_steps = len(train_dataset) * num_epochs
# optimizer, _ = create_optimizer(init_lr=learning_rate, num_train_steps=num_train_steps, 
#                                 weight_decay_rate=0.01, num_warmup_steps=0)

# # Compile the model
# model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
#               metrics=['accuracy'])

# # Train the model
# model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)

# # Save the model
# model.save_pretrained("vit.h5")  # Update with your save path
