import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# DataGenerator class definition
class DataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=4, target_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(batch_image_paths, batch_labels)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_paths, batch_labels):
        X = np.empty((self.batch_size, *self.target_size, 3))
        y = np.empty((self.batch_size), dtype=int)
        for i, (img_path, label) in enumerate(zip(batch_image_paths, batch_labels)):
            image = self.load_image(img_path)
            if image is not None:
                X[i,] = image
            else:
                print(f"Warning: Could not load image {img_path}. Skipping.")
            y[i] = label
        return X, y

    def load_image(self, file_path):
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise Exception(f"Failed to load image: {file_path}")
            image_resized = cv2.resize(image, self.target_size)
            image_resized = img_to_array(image_resized) / 255.0
            return image_resized
        except Exception as e:
            print(e)
            return None

# Load metadata and split dataset
def load_metadata_and_split(csv_file, image_dir):
    df = pd.read_csv(csv_file)
    image_paths = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row['videoname'].replace('.mp4', '.jpg'))
        label = 1 if row['label'] == 'FAKE' else 0
        image_paths.append(image_path)
        labels.append(label)
    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# Xception model definition
def build_xception_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    csv_file = 'data/metadata.csv'
    image_dir = 'data/faces_224/'
    
    X_train, X_val, y_train, y_val = load_metadata_and_split(csv_file, image_dir)

    train_generator = DataGenerator(X_train, y_train, batch_size=4, target_size=(224, 224))
    val_generator = DataGenerator(X_val, y_val, batch_size=4, target_size=(224, 224))

    model = build_xception_model()

    checkpoint = ModelCheckpoint('models/xception_model.keras', monitor='val_loss', save_best_only=True)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[checkpoint]
    )

    print("Model training completed!")
    with open('logs/training_log.txt', 'w') as f:
        f.write(str(history.history))

if __name__ == "__main__":
    main()
