# Code for LSTM Model
def build_lstm(input_shape):
    model = Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Nadam(0.0008), loss=binary_crossentropy, metrics=['accuracy'])
    return model

def get_roc_data(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def train_lstm():
    xtrain, xtest, ytrain, ytest = prepare_data()
    model = build_lstm(xtrain.shape[1:])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6)
    ]

    history = model.fit(
        xtrain, ytrain,
        validation_data=(xtest, ytest),
        epochs=20,
        batch_size=16,
        callbacks=callbacks
    )

    save_path = model_save_path.replace('cnn_model', 'lstm_model')
    model.save(save_path)
    print(f"✅ LSTM Model saved at: {save_path}")

    score = model.evaluate(xtest, ytest)
    print("Test Loss:", score[0])
    print("Test Accuracy:", score[1])

    y_pred_prob = model.predict(xtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    plot_confusion_matrix(ytest, y_pred)
    plot_roc(ytest, y_pred_prob)

    # Get ROC data (optional)
    fpr, tpr, roc_auc = get_roc_data(ytest, y_pred_prob)
    return fpr, tpr, roc_auc

if __name__ == "__main__":
    lstm_fpr, lstm_tpr, lstm_auc = train_lstm()
