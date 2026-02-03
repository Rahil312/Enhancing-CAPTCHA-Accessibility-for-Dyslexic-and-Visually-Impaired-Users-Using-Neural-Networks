# CAPTCHA Recognition using CNN

A deep learning project that uses Convolutional Neural Networks (CNN) to automatically recognize and decode CAPTCHA images.

## Project Overview

This project implements a CNN-based solution for recognizing 6-character CAPTCHAs containing lowercase letters (a-z) and digits (0-9). The model processes grayscale images of size 50x200 pixels and predicts each character position independently.

## Features

- **CNN Architecture**: Uses convolutional layers, max pooling, and batch normalization
- **Multi-output Model**: 6 separate dense layers for each character position
- **Character Set**: Supports 36 characters (26 letters + 10 digits)
- **High Accuracy**: Achieves good performance on CAPTCHA recognition tasks

## Model Architecture

- **Input**: 50x200x1 grayscale images
- **Conv Layers**: 3 convolutional layers (16, 32, 32 filters)
- **Pooling**: Max pooling for dimensionality reduction
- **Batch Normalization**: For stable training
- **Output**: 6 dense layers with softmax activation (36 classes each)

## Files

- `Captch_Final_Code_1.ipynb` - Main notebook with complete implementation
- `Proj.pdf` - Project documentation
- `1.png`, `2.png`, `3.png`, `4.png` - Project images/examples

## Requirements

```
tensorflow
numpy
pandas
opencv-python
matplotlib
scikit-learn
```

## Usage

1. Open the Jupyter notebook `Captch_Final_Code_1.ipynb`
2. Update the data path to your CAPTCHA dataset
3. Run all cells to train the model
4. Use the `predict()` function to test on new CAPTCHA images

## Training

- **Loss Function**: Categorical crossentropy
- **Optimizer**: Adam
- **Epochs**: 12
- **Batch Size**: 32
- **Train/Validation Split**: 90/10

## Results

The model outputs predictions for each character position and calculates overall accuracy by comparing predicted vs. actual CAPTCHA text.

---

*Neural Networks Final Project - Semester 2*