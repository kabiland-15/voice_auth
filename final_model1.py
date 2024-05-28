import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the dataset
data = pd.read_csv("DATASET-balanced.csv")

# Separate features (X) and labels (y)
X = data.drop(columns=['LABEL'])  # Assuming 'label' is the name of the target column
y = data['LABEL']

# Convert features and labels to numpy arrays
X = X.values
y = y.values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the input data for compatibility with CNN
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(np.shape(X_reshaped))
# Define a custom callback to save the model with the highest training accuracy
class SaveBestModelCallback(callbacks.Callback):
    def __init__(self, file_prefix):
        super().__init__()
        self.file_prefix = file_prefix
        self.best_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        train_accuracy = logs.get('accuracy')  # Use 'accuracy' instead of 'val_accuracy'
        if train_accuracy is not None and train_accuracy > self.best_accuracy:
            self.best_accuracy = train_accuracy
            model_path = self.file_prefix + '_best_accuracy.h5'
            self.model.save(model_path)
            print(f"Model with highest training accuracy ({train_accuracy:.4f}) saved.")



# Build the CRNN model
model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_reshaped.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(128),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the callback to save the best model based on validation accuracy
save_callback = SaveBestModelCallback(file_prefix='crnn')

# Train the model with the custom callback
history = model.fit(X_reshaped, y_encoded, epochs=40, batch_size=32, callbacks=[save_callback])

# Evaluate the best model
best_model = models.load_model('crnn_best_accuracy.h5')
loss, accuracy = best_model.evaluate(X_reshaped, y_encoded)
print(f'Best Model - Test Loss: {loss}, Test Accuracy: {accuracy}')
