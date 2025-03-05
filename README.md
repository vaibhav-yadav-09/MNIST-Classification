# MNIST Digit Classification using TensorFlow

This project implements a simple neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## 📌 Project Overview
- Uses **TensorFlow** and **Keras** to build a deep learning model.
- Trains a **fully connected neural network (MLP)** to classify digits (0-9).
- **MNIST dataset** (28x28 grayscale images) is used for training and testing.
- Achieves high accuracy with **ReLU activation** and **softmax output**.
```bash

pip install tensorflow numpy matplotlib scikit-learn
python mnist_classifier.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model definition
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
