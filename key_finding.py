# Load history file for the current epoch
import pickle
history_path = f'crnn_epoch_20_history.pkl'
with open(history_path, 'rb') as file:
    history_dict = pickle.load(file)

# Inspect the keys in the history dictionary
print(history_dict)
