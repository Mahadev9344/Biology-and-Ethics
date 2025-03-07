# Code for building the FNN model

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, ELU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Nadam
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

# File Paths
encodedwithoutTsai_location = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\encoded8x23withoutTsai.pkl"
guideseq_location = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\guideseq8x23.pkl"
flpath = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project"

def load_data(file_path):
    """Load data from pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded: {file_path}")
    print("Keys:", data.keys())
    return data

def transform_images(x, y):
    """Normalize and reshape input images."""
    x = x.reshape(x.shape[0], -1).astype('float32') / 255
    return x, y

def combine_datasets(data1, data2):
    """Combine two datasets into one."""
    x_combined = np.concatenate((data1['images'], data2['images']), axis=0)
    y_combined = np.concatenate((data1['target'], data2['target']), axis=0)
    
    print(f"Combined Dataset Shape: {x_combined.shape}, Labels: {y_combined.shape}")
    return x_combined, y_combined

def build_fnn(input_shape):
    """Build the optimized FNN model."""
    model = Sequential()
    model.add(Dense(256, input_dim=input_shape))
    model.add(ELU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.05))
    
    model.add(Dense(128))
    model.add(ELU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.05))
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Nadam(learning_rate=0.0008), loss=binary_crossentropy, metrics=['accuracy'])
    return model

def train_fnn(xtrain, ytrain, xtest, ytest, batch_size, epochs):
    """Train the FNN model with early stopping."""
    model = build_fnn(xtrain.shape[1])
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

    history = model.fit(
        xtrain, ytrain,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(xtest, ytest),
        callbacks=callbacks,
        verbose=1
    )

    score = model.evaluate(xtest, ytest, verbose=1)
    print("Final Test Loss:", score[0])
    print("Final Test Accuracy:", score[1])

    model.save(os.path.join(flpath, 'fnn_best_model.h5'))
    print(f"✅ Best Model saved at: {os.path.join(flpath, 'fnn_best_model.h5')}")

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Final Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load Datasets
    encodedwithoutTsai = load_data(encodedwithoutTsai_location)
    guideseq = load_data(guideseq_location)

    # Combine Both Datasets
    x_combined, y_combined = combine_datasets(encodedwithoutTsai, guideseq)
    x_combined, y_combined = transform_images(x_combined, y_combined)

    # Train Model
    train_fnn(x_combined, y_combined, x_combined, y_combined, batch_size=16, epochs=10)
