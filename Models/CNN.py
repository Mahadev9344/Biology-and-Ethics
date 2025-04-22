# Code for CNN Model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Nadam
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns

# Paths
encoded_path = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\encoded8x23withoutTsai.pkl"
guideseq_path = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\guideseq8x23.pkl"
model_save_path = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\cnn_model.keras"

def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

def prepare_data():
    data1 = load_pickle(encoded_path)
    data2 = load_pickle(guideseq_path)

    x = np.concatenate((data1['images'], data2['images']), axis=0).astype('float32') / 255.0
    y = np.concatenate((data1['target'], data2['target']), axis=0)

    x = x.reshape(-1, 23, 8)  # 23x8 sequence format
    return train_test_split(x, y, test_size=0.2, random_state=42)

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Nadam(0.0008), loss=binary_crossentropy, metrics=['accuracy'])
    return model

def plot_confusion_matrix(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print("\nClassification Report:\n", classification_report(ytrue, ypred))

def plot_roc(ytrue, ypred_prob):
    fpr, tpr, _ = roc_curve(ytrue, ypred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def get_roc_data(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def train_and_evaluate():
    xtrain, xtest, ytrain, ytest = prepare_data()
    model = build_cnn(xtrain.shape[1:])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6)
    ]

    history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=20, batch_size=16, callbacks=callbacks)
    model.save(model_save_path)
    print(f"✅ CNN Model saved at: {model_save_path}")

    score = model.evaluate(xtest, ytest)
    print("Test Loss:", score[0])
    print("Test Accuracy:", score[1])

    y_pred_prob = model.predict(xtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    plot_confusion_matrix(ytest, y_pred)
    plot_roc(ytest, y_pred_prob)

    return ytest, y_pred_prob  # return for global ROC access if needed

if __name__ == "__main__":
    ytest, y_pred_prob = train_and_evaluate()
    cnn_fpr, cnn_tpr, cnn_auc = get_roc_data(ytest, y_pred_prob)
    print(f"✅ CNN AUC Score: {cnn_auc:.4f}")
