"""
Training script for waste classification model using transfer learning.
This script uses MobileNetV2 as a base model with transfer learning.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4
WASTE_CATEGORIES = ['recyclable', 'organic', 'hazardous', 'non-recyclable']

def create_model():
    """Create a model using transfer learning with MobileNetV2"""
    
    # Load pre-trained MobileNetV2 without the top layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def train_with_data_augmentation():
    """Train the model with data augmentation (requires dataset)"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Check if dataset exists
    data_dir = 'dataset'
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found.")
        print("Please organize your dataset as follows:")
        print("dataset/")
        print("  ├── recyclable/")
        print("  ├── organic/")
        print("  ├── hazardous/")
        print("  └── non-recyclable/")
        print("\nEach folder should contain images of that waste category.")
        return None
    
    # Load and prepare data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )
    
    # Save the model
    model.save('waste_model.h5')
    print("Model saved as 'waste_model.h5'")
    
    return model

def create_simple_model():
    """Create a simple model for testing without a dataset"""
    print("Creating a simple model for testing purposes...")
    
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the untrained model (for testing structure)
    model.save('waste_model.h5')
    print("Simple model saved as 'waste_model.h5'")
    print("Note: This model is not trained. For production, use train_with_data_augmentation() with a proper dataset.")
    
    return model

if __name__ == '__main__':
    print("Waste Classification Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    if os.path.exists('dataset'):
        print("Dataset found. Training with data augmentation...")
        model = train_with_data_augmentation()
    else:
        print("No dataset found. Creating a simple model for testing...")
        print("To train a proper model, organize your images in a 'dataset' folder.")
        model = create_simple_model()
    
    if model:
        print("\nModel summary:")
        model.summary()
