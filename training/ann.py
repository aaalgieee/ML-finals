import numpy as np
import os
import cv2

# Basic environment config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Add constants for model files
MODEL_PATH = 'best_model.keras'
ENCODER_PATH = 'label_encoder.npy'

# Data loading functions
def load_data(image_dir, label_dir):
    images = []
    labels = []
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None and os.path.exists(label_path):
            try:
                with open(label_path, 'r') as file:
                    content = file.read().strip()
                    if content:  # Check if file is not empty
                        label_parts = content.split()
                        if label_parts:  # Check if split result is not empty
                            images.append(image)
                            labels.append(label_parts[0])
                        else:
                            print(f"Warning: Empty content in {label_file}")
                    else:
                        print(f"Warning: Empty file {label_file}")
            except Exception as e:
                print(f"Error reading {label_file}: {str(e)}")
                continue
    
    if not images or not labels:
        raise ValueError("No valid image-label pairs found")
    
    return np.array(images), np.array(labels)

def preprocess_data(images, labels, label_encoder=None, target_size=(64, 64)):  # Increased resolution
    # Preprocess images
    processed_images = []
    for image in images:
        resized_image = cv2.resize(image, (target_size[0], target_size[1]))
        processed_images.append(resized_image / 255.0)
    
    X = np.array(processed_images)
    X = X.reshape(X.shape[0], -1)
    
    # Convert labels to integers if they're not already
    labels = labels.astype(int)
    
    # Encode labels
    if label_encoder is None:
        le = LabelEncoder()
        y = le.fit_transform(labels)
    else:
        le = label_encoder
        y = le.transform(labels)
    
    y = to_categorical(y)
    
    return X, y, le

def save_label_encoder(label_encoder, filename=ENCODER_PATH):
    """Save the label encoder to a file"""
    np.save(filename, label_encoder.classes_)

def load_label_encoder(filename=ENCODER_PATH):
    """Load the label encoder from a file"""
    le = LabelEncoder()
    le.classes_ = np.load(filename)
    return le

def load_model():
    """Load the trained model"""
    return tf.keras.models.load_model(MODEL_PATH)

def predict_single_image(image, model=None, label_encoder=None, target_size=(64, 64)):
    """Predict class for a single image"""
    if model is None:
        model = load_model()
    if label_encoder is None:
        label_encoder = load_label_encoder()
        
    # Preprocess image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, target_size)
    processed = (resized / 255.0).reshape(1, -1)
    
    # Predict
    prediction = model.predict(processed)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = float(np.max(prediction))
    
    return {
        'class': int(predicted_class[0]),
        'confidence': confidence
    }

# Build ANN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(2048, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    return model

# Main execution
if __name__ == "__main__":
    # Set paths for train and test directories
    train_image_dir = "ann-data/train/images"
    train_label_dir = "ann-data/train/labels"
    test_image_dir = "ann-data/test/images"
    test_label_dir = "ann-data/test/labels"
    
    # Load training and test data
    train_images, train_labels = load_data(train_image_dir, train_label_dir)
    test_images, test_labels = load_data(test_image_dir, test_label_dir)
    
    # Preprocess training and test data
    X_train, y_train, label_encoder = preprocess_data(train_images, train_labels)
    X_test, y_test, _ = preprocess_data(test_images, test_labels, label_encoder=label_encoder)
    
    # Create and compile model with custom learning rate
    model = create_model(X_train.shape[1], len(label_encoder.classes_))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Add callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        min_delta=0.001
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_model.keras',  # Changed extension from .h5 to .keras
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train model with callbacks (CPU-optimized batch size)
    history = model.fit(X_train, y_train,
                       epochs=100,
                       batch_size=32,  # Reduced for CPU
                       validation_split=0.2,
                       callbacks=[early_stopping, model_checkpoint])
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Print training history
    print("\nTraining History:")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

    # After training completes, save the label encoder
    save_label_encoder(label_encoder)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Label encoder saved to {ENCODER_PATH}")
    
    # Test loading and prediction
    test_model = load_model()
    test_encoder = load_label_encoder()
    print("Model and encoder can be loaded successfully")