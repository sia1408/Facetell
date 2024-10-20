import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load image helper function
def load_image(file_path, target_size=(224, 224)):
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise Exception(f"Failed to load image: {file_path}")
        image_resized = cv2.resize(image, target_size)
        image_resized = image_resized.astype("float32") / 255.0  # Normalize
        return np.expand_dims(image_resized, axis=0)  # Expand dims for model input
    except Exception as e:
        print(e)
        return None

# Load test dataset function
def load_test_data(csv_file, image_dir):
    df = pd.read_csv(csv_file)
    image_paths = []
    labels = []
    
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row['videoname'].replace('.mp4', '.jpg'))
        label = 1 if row['label'] == 'FAKE' else 0
        image_paths.append(image_path)
        labels.append(label)
    
    return image_paths, np.array(labels)

# Evaluate model on the test set
def evaluate_model():
    # Load the trained model
    model_path = 'models/xception_model.keras'
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}.")

    # Load test data
    csv_file = 'data/metadata.csv'  # Adjust as needed
    image_dir = 'data/faces_224/'
    image_paths, true_labels = load_test_data(csv_file, image_dir)
    
    # Prepare predictions
    predictions = []
    for image_path in image_paths:
        image = load_image(image_path)
        if image is not None:
            pred = model.predict(image)[0][0]
            predictions.append(1 if pred > 0.5 else 0)  # 1 for FAKE, 0 for REAL
        else:
            predictions.append(None)

    # Filter out None values
    predictions = [p for p in predictions if p is not None]

    # Compute accuracy
    accuracy = accuracy_score(true_labels[:len(predictions)], predictions)
    print(f'Test Accuracy: {accuracy}')

    # Generate a classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(true_labels[:len(predictions)], predictions, target_names=['REAL', 'FAKE']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels[:len(predictions)], predictions))

if __name__ == "__main__":
    evaluate_model()
