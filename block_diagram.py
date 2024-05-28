from graphviz import Digraph
from tensorflow.keras import layers


# Create a Digraph object
dot = Digraph()

# Add nodes for each layer
input_shape = "Input\n(Length, Channels)"
dot.node("Input", label=input_shape, shape="parallelogram", style="filled", fillcolor="lightblue")

# Conv1D layers
dot.node("Conv1D_1", label="Conv1D\nFilters: 64\nKernel Size: 3\nActivation: ReLU")
dot.node("Conv1D_2", label="Conv1D\nFilters: 128\nKernel Size: 3\nActivation: ReLU")

# MaxPooling1D layers
dot.node("MaxPooling1D_1", label="MaxPooling1D\nPool Size: 2")
dot.node("MaxPooling1D_2", label="MaxPooling1D\nPool Size: 2")

# Dropout layers
dot.node("Dropout_1", label="Dropout\nRate: 0.3")
dot.node("Dropout_2", label="Dropout\nRate: 0.3")

# LSTM layers
dot.node("LSTM_1", label="LSTM\nUnits: 64\nReturn Sequences: True")
dot.node("LSTM_2", label="LSTM\nUnits: 64")

# Dense layers
dot.node("Dense_1", label="Dense\nUnits: 64\nActivation: ReLU")
dot.node("Dense_2", label="Dense\nUnits: 1\nActivation: Sigmoid")

# Add edges between layers
layers = [
    ("Input", "Conv1D_1"),
    ("Conv1D_1", "MaxPooling1D_1"),
    ("MaxPooling1D_1", "Dropout_1"),
    ("Dropout_1", "Conv1D_2"),
    ("Conv1D_2", "MaxPooling1D_2"),
    ("MaxPooling1D_2", "Dropout_2"),
    ("Dropout_2", "LSTM_1"),
    ("LSTM_1", "LSTM_2"),
    ("LSTM_2", "Dense_1"),
    ("Dense_1", "Dense_2"),
    # Feedback connections from LSTM layers
    ("LSTM_1", "LSTM_1"),
    ("LSTM_2", "LSTM_1")
]

for edge in layers:
    dot.edge(edge[0], edge[1])

# Save the diagram as a PDF file
file_name = "network_diagram_with_feedback"
dot.render(file_name, format='pdf', cleanup=True)

print("Block diagram generated successfully!")
