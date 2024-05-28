import matplotlib.pyplot as plt
import pickle

# Load the dictionary containing metrics
with open('crnn_epoch_20_history.pkl', 'rb') as file:
    history_dict = pickle.load(file)

# Extract the data for each metric
epochs = list(range(2, 21))  # Epochs from 2 to 20
loss = history_dict['loss']
accuracy = history_dict['accuracy']
val_loss = history_dict['val_loss']
val_accuracy = history_dict['val_accuracy']

# Plotting
plt.figure(figsize=(12, 8))

# Training Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, loss, marker='o', linestyle='-')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Training Accuracy
plt.subplot(2, 2, 2)
plt.plot(epochs, accuracy, marker='o', linestyle='-')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Validation Loss
plt.subplot(2, 2, 3)
plt.plot(epochs, val_loss, marker='o', linestyle='-')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Validation Accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, val_accuracy, marker='o', linestyle='-')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
