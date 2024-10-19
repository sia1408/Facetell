from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def build_xception_model():
    # Load the Xception model pre-trained on ImageNet, excluding the top layers
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Create the full model by adding custom layers on top of Xception
    model = Sequential([
        base_model,  # Pre-trained Xception model
        GlobalAveragePooling2D(),  # Global average pooling layer to reduce the feature map to a single vector
        Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
        Dropout(0.5),  # Dropout layer to reduce overfitting
        Dense(1, activation='sigmoid')  # Output layer for binary classification (REAL or FAKE)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model