import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 50

def load_data(images_dir, labels_dir):
    images = []
    labels = []
    
    # Get list of all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            try:
                # Read label from txt file (YOLO format)
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # Check if file is not empty
                        # Get the first number which is the class ID
                        label = int(content.split()[0])
                        
                        # Load and preprocess image
                        img_path = os.path.join(images_dir, image_file)
                        img = tf.keras.preprocessing.image.load_img(
                            img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = img_array / 255.0  # Normalize
                        images.append(img_array)
                        labels.append(label)
                    else:
                        print(f"Warning: Empty label file found: {label_path}")
            except (ValueError, IndexError) as e:
                print(f"Warning: Invalid label format in {label_path}: {str(e)}")
    
    if not images:
        raise ValueError(f"No valid images and labels found in {images_dir} and {labels_dir}")
    
    print(f"Loaded {len(images)} images with valid labels")
    return np.array(images), np.array(labels)

def create_model():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')  # 3 classes: fire, smoke, no fire/smoke
    ])
    return model

def plot_training_history(history):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_data('dataset/train/images', 'dataset/train/labels')
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    
    # Load validation data
    print("Loading validation data...")
    X_val, y_val = load_data('dataset/valid/images', 'dataset/valid/labels')
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('fire_smoke_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Print model size
    model_size = os.path.getsize('fire_smoke_model.tflite') / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")
    
    # Evaluate final model on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_accuracy:.2f}")

if __name__ == "__main__":
    main() 