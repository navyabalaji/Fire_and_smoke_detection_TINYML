# Fire_and_smoke_detection_TINYML

## Overview

This project trains a machine learning model to classify images into one of three categories:
- Fire
- Smoke
- No Fire/Smoke

The model is built using a Convolutional Neural Network (CNN) and optimized using TensorFlow Lite (TFLite) for deployment on edge devices. For detailed information about the model architecture and training process, refer to the accompanying PowerPoint presentation.

## Prerequisites

Ensure the following are installed:

- Python 3.7+ (recommended: below 3.10)
- TensorFlow and TensorFlow Lite
- Pillow
- NumPy

Install dependencies using:

```
pip install tensorflow pillow numpy
```

## Running the Model

### On Windows

1. Open Command Prompt  
2. Navigate to the project directory:

```
cd path/to/project
```

3. Run the script:

```
python test_fire_smoke.py
```

### On Linux/macOS

1. Open Terminal  
2. Navigate to the project directory:

```
cd path/to/project
```

3. Run the script:

```
python3 test_fire_smoke.py
```

## Dataset

The dataset consists of manually collected images of fire, smoke, and normal (no fire/smoke) scenarios. It has been preprocessed using techniques like rotation, zoom, and brightness adjustments.

You can also use additional fire and smoke image datasets from sources like Kaggle or other public repositories. The dataset used for this project will be uploaded soon.

## Input Images

Place all test images inside the `test/` directory.  
Supported formats include `.jpg`, `.jpeg`, and `.png`.

The script will process each image and output the predicted class along with a confidence score.

## Model Details

- Model file: `fire_smoke_model.tflite`
- Input: RGB images resized to 64x64
- Output: One of three classes with confidence score
- Optimized for deployment using TFLite

For more detailed insights on architecture and training, refer to the PowerPoint presentation.

## Troubleshooting

- Ensure that `fire_smoke_model.tflite` is in the project directory.
- Verify that test images are in the correct format and resolution.
- If you encounter missing package errors, run:

```
pip install --upgrade tensorflow pillow numpy
```

## License

This project is intended for research and educational purposes.  
You are free to modify and use the code as needed.
