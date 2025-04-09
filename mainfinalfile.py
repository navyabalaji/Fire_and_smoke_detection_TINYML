import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time

# Constants
IMG_HEIGHT = 64
IMG_WIDTH = 64
CLASS_NAMES = ['Fire', 'Smoke', 'No Fire/Smoke']
# CONFIDENCE_THRESHOLD = 0.80

def load_and_preprocess_image(image_path):
    # Load image
    img = Image.open(image_path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img)
    
    # Normalize
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def main():
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="fire_smoke_model.tflite")
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Process all images in TESTIMAGES folder
    test_dir = "TESTIMAGES"
    total_images = 0
    correct_predictions = 0
    
    print("Processing test images...")
    
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, filename)
            
            # Preprocess image
            input_data = load_and_preprocess_image(image_path)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            start_time = time.time()
            interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Get prediction
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data[0])
            confidence = output_data[0][predicted_class]
            
            # Apply confidence threshold
            # if confidence < CONFIDENCE_THRESHOLD:
            #     predicted_class = 2  # Change to No Fire/Smoke
            #     confidence = output_data[0][2]  # Get confidence for No Fire/Smoke class
            
            # Print results
            print(f"\nImage: {filename}")
            print(f"Prediction: {CLASS_NAMES[predicted_class]}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Inference time: {inference_time:.3f} seconds")
            
            total_images += 1
            
            # If you have ground truth labels, you can calculate accuracy here
            # For now, we'll just count the total processed images
    
    print(f"\nProcessed {total_images} images")

if __name__ == "__main__":
    main() 