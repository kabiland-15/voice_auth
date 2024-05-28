import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Load the model from the file
model = tf.keras.models.load_model(r'D:\projects\voice_detection\model\crnn_epoch_20.h5')

# Function to compute gradients
def compute_gradients(model, input_data):
    with tf.GradientTape() as tape:
        outputs = model(input_data)
    gradients = tape.gradient(outputs, model.trainable_variables)
    return gradients

# Function to visualize weights of a specific layer
def visualize_weights(layer, layer_index):
    if isinstance(layer, tf.keras.layers.Conv1D):
        weights = layer.get_weights()[0]
        plt.figure(figsize=(10, 5))
        plt.hist(weights.flatten(), bins=50, color='blue', alpha=0.7, rwidth=0.8)
        plt.title(f'Conv1D Weights Distribution - {layer.name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    elif isinstance(layer, tf.keras.layers.LSTM):
        weights = layer.get_weights()
        if len(weights) > 0:
            for i, weight_matrix in enumerate(weights):
                plt.figure(figsize=(10, 5))
                plt.hist(weight_matrix.flatten(), bins=50, color='blue', alpha=0.7, rwidth=0.8)
                plt.title(f'LSTM Layer {layer_index} Weights Distribution')
                plt.xlabel('Weight Value')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.show()
        else:
            print(f"No weights available for LSTM layer: {layer.name}")
    elif isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()[0]
        plt.figure(figsize=(10, 5))
        plt.hist(weights.flatten(), bins=50, color='blue', alpha=0.7, rwidth=0.8)
        plt.title(f'Dense Weights Distribution - {layer.name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# Example usage: Compute gradients and visualize weights for all layers
input_data = np.array([0.338055, 0.027948, 2842.948867, 4322.916759, 6570.586186, 0.04105, -462.169586, 90.311272, 19.073769, 24.046888, -0.092606, 5.963933, -12.073119, -1.526925, -6.735845, -9.344831, -14.181895, -6.686564, 0.902086, -7.251551, -1.198342, 4.747403, -4.986279, 0.953935, -5.013138, -6.77906])
input_data = input_data.reshape(1, -1, 1)

gradients = compute_gradients(model, input_data)

for layer, grad in zip(model.layers, gradients):
    if grad is not None:
        print(f"Layer {layer.name}: Gradient Norm = {tf.norm(grad)}")

for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.LSTM):
        visualize_weights(layer, i + 1)
