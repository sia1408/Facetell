import pandas as pd
import os
import cv2
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight  # To calculate class weights

# Data loader and preprocessing function
def load_image(file_path, target_size=(224, 224)):
    image = cv2.imread(file_path)
    image_resized = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image

def load_data(csv_file, image_dir, target_size=(224, 224)):
    df = pd.read_csv(csv_file)
    
    X = []
    y = []
    
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row['videoname'].replace('.mp4', '.jpg'))  # Replace .mp4 with .jpg
        label = 1 if row['label'] == 'FAKE' else 0  # 1 for FAKE, 0 for REAL
        image = load_image(image_path, target_size)
        X.append(image)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, y

# Model setup using Xception
def build_xception_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load dataset
    X_train, X_val, y_train, y_val, y = load_data('data/metadata.csv', 'data/faces_224/')

    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f'Class weights: {class_weights_dict}')

    # Build the model
    model = build_xception_model()
    
    # Define a checkpoint callback to save the model
    checkpoint = ModelCheckpoint('models/cnn_model2.keras', monitor='val_loss', save_best_only=True)
    
    # Train the model with class weights
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, callbacks=[checkpoint], class_weight=class_weights_dict)

    print("Model training completed!")

if __name__ == "__main__":
    main()