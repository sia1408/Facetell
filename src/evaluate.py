from tensorflow.keras.models import load_model
from src.data_loader import load_data  # Assuming your data loading logic is in a file
import numpy as np

# Load the trained model
model = load_model('models/cnn_model2.keras')
print("Model loaded successfully.")

# Load the test data
# Adjust paths and data_loader if needed
X_train, X_test, y_train, y_test = load_data('data/metadata.csv', 'data/faces_224/')  # Assuming your data loader returns train/test split

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")
print(f"Test loss: {test_loss}")

# Optionally, you can calculate more metrics or save the evaluation results