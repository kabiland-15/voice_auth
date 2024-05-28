import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data from CSV file
data = pd.read_csv('DATASET-balanced.csv')

# Extract features (all columns except the last one) and labels (last column)
X = data.drop(['LABEL'], axis=1)
y = data['LABEL']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=20000, random_state=42)
rfc.fit(X_train, y_train)

# Predict on test set
y_pred = rfc.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save model
import joblib
joblib.dump(rfc, 'rfc_model.pkl')
print('Model saved successfully.')
