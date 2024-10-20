from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_xception_model(config):
    # Load the Xception model pre-trained on ImageNet, exclude the top layers
    base_model = Xception(
        weights='imagenet', 
        include_top=False, 
        input_shape=tuple(config['model']['input_shape'])
    )

    # Add custom classification layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling layer
    x = Dense(config['model']['dense_units'], activation='relu')(x)  # Dense layer
    x = Dropout(config['model']['dropout_rate'])(x)  # Dropout for regularization
    predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

    # Define the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Optionally, freeze some layers of the base model to prevent overfitting
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model with an appropriate optimizer, loss function, and metrics
    model.compile(
        optimizer=Adam(learning_rate=config['model']['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
