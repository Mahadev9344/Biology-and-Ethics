#Code For FNN MOdel
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns

# File Paths
encodedwithoutTsai_location = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\encoded8x23withoutTsai.pkl"
guideseq_location = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\guideseq8x23.pkl"
flpath = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\model"

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


from tensorflow.keras.regularizers import l2

def build_fnn_best(input_shape):
    """Build the optimized FNN model with regularization and increased dropout."""
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, kernel_regularizer=l2(0.01)))
    model.add(ELU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))  # Increased dropout
    
    model.add(Dense(64, kernel_regularizer=l2(0.01)))
    model.add(ELU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))  # Increased dropout
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Nadam(learning_rate=0.0008), loss=binary_crossentropy, metrics=['accuracy'])
    return model

def train_fnn_best(xtrain, ytrain, xtest, ytest, batch_size, epochs):
    """Train the FNN model with early stopping and learning rate scheduler."""
    model = build_fnn_best(xtrain.shape[1])
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    
    history = model.fit(
        xtrain, ytrain,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(xtest, ytest),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    score = model.evaluate(xtest, ytest, verbose=1)
    print("Final Test Loss:", score[0])
    print("Final Test Accuracy:", score[1])

    model.save(os.path.join(flpath, 'fnn_best_model.keras'))  # Save in .keras format
    print(f"✅ Best Model saved at: {os.path.join(flpath, 'fnn_best_model.keras')}")

    # Plot learning curves
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Final Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model

def generate_confusion_matrix(ytrue, ypred):
    """Generate Confusion Matrix"""
    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print("\nClassification Report:\n", classification_report(ytrue, ypred))

def generate_roc_curve(ytrue, ypred_proba):
    """Generate ROC Curve"""
    fpr, tpr, _ = roc_curve(ytrue, ypred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
    print(f"AUC Score: {roc_auc:.2f}")

def get_roc_data(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

fnn_fpr, fnn_tpr, fnn_auc = get_roc_data(ytest, y_pred_proba)


if __name__ == "__main__":
    # Load Datasets
    encodedwithoutTsai = load_data(encodedwithoutTsai_location)
    guideseq = load_data(guideseq_location)

    # Combine Both Datasets
    x_combined, y_combined = combine_datasets(encodedwithoutTsai, guideseq)
    x_combined, y_combined = transform_images(x_combined, y_combined)

    # Split Data into Training and Testing Sets
    xtrain, xtest, ytrain, ytest = train_test_split(x_combined, y_combined, test_size=0.2, random_state=42)

    # Train Model
    model = train_fnn_best(xtrain, ytrain, xtest, ytest, batch_size=16, epochs=20)

    # Get model predictions
    y_pred_proba = model.predict(xtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Generate Evaluation Metrics
    generate_confusion_matrix(ytest, y_pred)
    generate_roc_curve(ytest, y_pred_proba)
