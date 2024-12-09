import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import os
import random

def prepare_cnn_data(observations, base_path):
    """Prepare data for CNN with 5 channels: R, G, B, IR, landcover, altitude."""
    images = []
    labels = []
    filter_amount = 200
    total_images = len(observations)
    processed_count = 0

    for _, row in observations.iterrows():
        processed_count += 1
        
        if random.random() < 1 / filter_amount: 
            obs_id = str(row["observation_id"])
            
            # Chemins des fichiers
            rgb_path = os.path.join(base_path, obs_id[-2:], obs_id[-4:-2], f"{obs_id}_rgb.jpg")
            ir_path = os.path.join(base_path, obs_id[-2:], obs_id[-4:-2], f"{obs_id}_near_ir.jpg")
            landcover_path = os.path.join(base_path, obs_id[-2:], obs_id[-4:-2], f"{obs_id}_landcover.tif")
            altitude_path = os.path.join(base_path, obs_id[-2:], obs_id[-4:-2], f"{obs_id}_altitude.tif")
            
            if all(os.path.exists(path) for path in [rgb_path, ir_path, landcover_path, altitude_path]):
                # Charger les images
                rgb_image = load_img(rgb_path, target_size=(128, 128))
                ir_image = load_img(ir_path, color_mode="grayscale", target_size=(128, 128))
                landcover_image = load_img(landcover_path, color_mode="grayscale", target_size=(128, 128))
                altitude_image = load_img(altitude_path, color_mode="grayscale", target_size=(128, 128))
                
                # Convertir en tableaux numpy
                rgb_array = img_to_array(rgb_image)  # (128, 128, 3)
                ir_array = img_to_array(ir_image)[:, :, 0:1]  # Assure 1 seul canal
                landcover_array = img_to_array(landcover_image)[:, :, 0:1]
                altitude_array = img_to_array(altitude_image)[:, :, 0:1]
                
                # Combiner les canaux
                combined_image = np.concatenate([rgb_array, ir_array, landcover_array, altitude_array], axis=-1)  # (128, 128, 6)
                
                images.append(combined_image)
                labels.append(row["species_id"])
            else:
                print(f"Données manquantes pour {obs_id}")
        
        # Afficher la progression
        if processed_count % 100 == 0 or processed_count == total_images:
            print(f"Images traitées : {processed_count}/{total_images}")
    
    return np.array(images), np.array(labels)

def create_cnn_model(num_classes):
    """Create a CNN model with updated parameters."""
    model = Sequential([
        # Bloc 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 6), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Bloc 2
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Bloc 3
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Couches fully connected
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Optimiseur SGD avec learning rate initial
    optimizer = SGD(learning_rate=0.1, momentum=0.9)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def schedule(epoch, lr):
    """Learning rate scheduler."""
    if epoch in [25, 35, 41, 47, 50]:
        return lr * 0.1
    return lr

def train_cnn_model(model, train_images, train_labels, val_images, val_labels, epochs=50):
    """Train the CNN model with updated parameters."""
    # Callback pour ajuster le learning rate
    lr_scheduler = LearningRateScheduler(schedule)
    
    model.fit(
        train_images / 255.0,
        train_labels,
        validation_data=(val_images / 255.0, val_labels),
        epochs=epochs,
        batch_size=128,  # Taille des lots
        callbacks=[lr_scheduler]
    )
    return model
