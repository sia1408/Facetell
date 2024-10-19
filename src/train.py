from tensorflow.keras.callbacks import ModelCheckpoint
from src.model import build_xception_model  # Updated import to use Xception model
import sys
import os
from src.data_loader import load_metadata_and_split, DataGenerator

def main():
    # Load metadata and split into train/val sets
    X_train, X_val, y_train, y_val = load_metadata_and_split('data/metadata.csv', 'data/faces_224/')

    # Create data generators
    train_generator = DataGenerator(X_train, y_train, batch_size=8, target_size=(224, 224))
    val_generator = DataGenerator(X_val, y_val, batch_size=8, target_size=(224, 224))

    # Build the Xception model
    model = build_xception_model()

    # Define checkpoint callback to save the best model
    checkpoint = ModelCheckpoint('models/cnn2_model.keras', monitor='val_loss', save_best_only=True)

    # Train the model using the data generators
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[checkpoint]
    )

    print("Model training completed!")

    # Save training logs if needed
    with open('logs/training_log.txt', 'w') as f:
        f.write(str(history.history))

if __name__ == "__main__":
    main()
