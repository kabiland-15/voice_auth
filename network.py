from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
def predict(x):
    x = pd.DataFrame([x])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_reshaped = x_scaled.reshape(x_scaled.shape[0], x_scaled.shape[1], 1)
    shape = np.shape(x_reshaped)
    model = load_model('crnn_best_accuracy.h5')
    predictions = model.predict(x_reshaped)
    print('Shape of the input: ', shape)
    print(predictions)

input_data = [435534543452345, 43719,3563.81031,3719.747211,6948.646497,0.099258,-356.692535,88.206108,-25.650661,16.694134,-4.345403,15.386952,-22.00659,0.705341,-16.617756,-2.169059,-9.152279,-0.383366,2.15471,-9.367912,-3.818097,1.608914,-6.248724,3.591924,-3.741955,-5.954679]
predict(input_data)
