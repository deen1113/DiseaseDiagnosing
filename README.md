# X-ray Disease Diagnosing Project

A machine learning project to classify lung X-ray images into four categories: **COVID**, **Lung Opacity**, **Normal**, and **Viral Pneumonia** using transfer learning with ResNet50.

## Features
- Transfer learning with ResNet50 base model
- Data augmentation and preprocessing pipeline
- Hyperparameter tuning with Keras Tuner
- Model evaluation with precision, recall, and confusion matrix
- Flask API for predictions (optional)

## Installation

### Requirements
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

## Acknowledgments
Dataset from COVID-19 Radiography Database

Inspired by COVID-19 detection research papers

TensorFlow/Keras documentation