import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load the dataset
data = pd.read_csv("DATASET-balanced.csv")

# Separate features (X) and labels (y)
X = data.drop(columns=['LABEL'])  # Assuming 'label' is the name of the target column
y = data['LABEL']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert features and labels to numpy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data for compatibility with CNN
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# Define a custom callback to save the model with the highest validation accuracy
class SaveBestModelCallback(callbacks.Callback):
    def __init__(self, file_prefix):
        super().__init__()
        self.file_prefix = file_prefix
        self.best_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            model_path = self.file_prefix + '_best_accuracy.h5'
            self.model.save(model_path)
            print(f"Model with highest validation accuracy ({val_accuracy:.4f}) saved.")

# Build the CRNN model
model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
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
history = model.fit(X_train_reshaped, y_train_encoded, epochs=1000, batch_size=32,
                    validation_data=(X_test_reshaped, y_test_encoded), callbacks=[save_callback])

# Evaluate the best model
best_model = models.load_model('crnn_best_accuracy.h5')
loss, accuracy = best_model.evaluate(X_test_reshaped, y_test_encoded)
print(f'Best Model - Test Loss: {loss}, Test Accuracy: {accuracy}')
