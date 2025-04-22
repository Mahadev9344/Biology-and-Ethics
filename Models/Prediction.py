# Code for prediction on new DNA sequence
import numpy as np
import tensorflow as tf

# --- 8-bit Encoding for DNA bases ---
base_encoding = {
    'A': [1,0,0,0,0,0,0,0],
    'C': [0,1,0,0,0,0,0,0],
    'G': [0,0,1,0,0,0,0,0],
    'T': [0,0,0,1,0,0,0,0],
    'N': [0,0,0,0,0,0,0,1]  # optional for ambiguous bases
}

# --- Encode sequence to 23x8 matrix ---
def encode_sequence(seq):
    if len(seq) != 23:
        raise ValueError("Sequence must be 23 bases long")
    return np.array([base_encoding.get(base.upper(), [0]*8) for base in seq])

# --- Load FNN Model ---
fnn_model = tf.keras.models.load_model(r"C:\Users\mbm54\OneDrive\Desktop\CRISPR Cas-9\My Project\model\fnn_best_model.keras")

# --- Prediction Function ---
def predict_off_target(seq):
    encoded = encode_sequence(seq).astype('float32')  # shape: (23, 8)
    input_flat = encoded.flatten().reshape(1, -1)    # shape: (1, 184)

    # Predict
    fnn_pred = float(fnn_model.predict(input_flat, verbose=0)[0][0])

    # Output confidence score and target label
    print(f"Prediction: {'Off-target' if fnn_pred > 0.5 else 'On-target'}")
    print(f"Confidence Score: {fnn_pred:.4f}")
    

# --- Input Sequence from User ---
if __name__ == "__main__":
    # Input the sequence from the user
    new_sequence = input("Enter the 23-base DNA sequence: ")  # Make sure sequence is 23 bases long
    if len(new_sequence) == 23:
        predict_off_target(new_sequence)
    else:
        print("Error: The sequence must be exactly 23 bases long.")
