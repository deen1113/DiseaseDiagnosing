# X-Ray Disease Diagnosing Project

A machine learning project to classify lung X-ray images into four categories: **COVID**, **Lung Opacity**, **Normal**, and **Viral Pneumonia** using transfer learning with ResNet50.

## About This Project

As my first machine learning project, I wanted to create something that while teaching me how to use tools such as Tensorflow/Keras, would still have the ability to have an impact on the world. After searching through kaggle databases, I found these lung x-rays and decided that I would create a model that would be able to identify and diagnose what was wrong with a given lung x-ray, with the intention of the model eventually being able to diagnose more than just the current four classes.

## Features
- Transfer learning with ResNet50 base model
- Data augmentation and preprocessing pipeline
- Hyperparameter tuning with Keras Tuner
- Model evaluation with precision, recall, and confusion matrix
- Flask API for predictions (optional)

## Installation

### Requirements
- Kaggle Covid-19 Radiography Database
    - Link: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Python 3.9+
- TensorFlow 2.x
- Required packages:
```bash
pip install tensorflow keras-tuner scikit-learn numpy pandas matplotlib seaborn opencv-python
```

### File Formatting

DiseaseDiagnosing/\
├── processed_data/\
│   ├── train/\
│   ├── val/\
│   └── test/\
├── models/\
│   └── best_model.h5\
├── main/\
│   ├── model.py\
|   ├── evaluation.py\
|   ├── matrix.py\
|   ├── callbacks.py\
│   ├── testing.py\
│   ├── preprocessing.py\
│   └── app.py\
└── README.md

## Training
Use the model.py script to begin training.
- Uses ResNet50 with custom dense layers
- Includes early stopping and model checkpointing
- Uses Keras Tuner for optimal parameter search
    - Tunes learning rate, dropout rate, and dense units
- Automatically saves best model to models/best_model.h5

## Evaluation
Use the evaluation.py script to test the model after training. Uses best model saved from training.
- Test accuracy, precision, and recall
- Confusion matrix visualization
- Classification report

## Predictions
To have the model make predictions, first run app.py to start the server, then run predict.py with an image filepath.
```bash
python main/app.py
python main/predict.py file/path/to/img.png
```
The model will give a prediction of the diagnoses of the xray from the 4 classes.
