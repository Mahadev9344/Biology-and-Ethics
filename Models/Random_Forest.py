# Code For Random Forest Model
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# File paths
encoded_path = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\encoded8x23withoutTsai.pkl"
guideseq_path = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\guideseq8x23.pkl"
model_save_path = r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\rf_model.joblib"

# Load pickle
def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

# Prepare data
def prepare_data():
    data1 = load_pickle(encoded_path)
    data2 = load_pickle(guideseq_path)

    x = np.concatenate((data1['images'], data2['images']), axis=0).astype('float32') / 255.0
    y = np.concatenate((data1['target'], data2['target']), axis=0)

    x = x.reshape((x.shape[0], -1))  # Flatten to (samples, 184)
    return train_test_split(x, y, test_size=0.2, random_state=42)

# Confusion Matrix
def plot_confusion_matrix(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ROC Curve
def plot_roc(ytrue, ypred_prob):
    fpr, tpr, _ = roc_curve(ytrue, ypred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

# Get ROC data
def get_roc_data(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

# Train Random Forest
def train_rf():
    xtrain, xtest, ytrain, ytest = prepare_data()

    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')
    clf.fit(xtrain, ytrain)

    joblib.dump(clf, model_save_path)
    print(f"✅ Random Forest model saved at: {model_save_path}")

    y_pred = clf.predict(xtest)
    y_pred_prob = clf.predict_proba(xtest)[:, 1]

    print("\nClassification Report:\n", classification_report(ytest, y_pred))
    plot_confusion_matrix(ytest, y_pred)
    plot_roc(ytest, y_pred_prob)

    return get_roc_data(ytest, y_pred_prob)

# Main
if __name__ == "__main__":
    rf_fpr, rf_tpr, rf_auc = train_rf()
