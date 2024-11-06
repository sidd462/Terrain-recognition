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
base_model = TFAutoModel.from_pretrained('google/vit-large-patch16-224')

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
model.save('vit_large_model.h5')

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
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')

# Load the base Vision Transformer model
base_model = TFAutoModel.from_pretrained('google/vit-large-patch16-224')

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
model.save('vit_large_model.h5')
