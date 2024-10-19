from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_xception_model():
    # Load the Xception model pre-trained on ImageNet, exclude the top layers
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom classification layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling layer
    x = Dense(128, activation='relu')(x)  # Dense layer
    x = Dropout(0.5)(x)  # Dropout for regularization
    predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

    # Define the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers to prevent them from being trained
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
