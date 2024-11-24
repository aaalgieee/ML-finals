import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load and prepare the data
df = pd.read_csv('dt_maternal.csv')

# Convert RiskLevel to numeric values
risk_map = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
df['RiskLevel'] = df['RiskLevel'].map(risk_map)

# Prepare features (X) and target (y)
X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = df['RiskLevel']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model with optimized parameters
dt_classifier = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)
dt_classifier.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = dt_classifier.score(X_train_scaled, y_train)
test_score = dt_classifier.score(X_test_scaled, y_test)

print(f'Training accuracy: {train_score:.2f}')
print(f'Testing accuracy: {test_score:.2f}')
print('\nClassification Report:')
print(classification_report(y_test, dt_classifier.predict(X_test_scaled)))

# Save the model and scaler
with open('./dt_model.pkl', 'wb') as f:
    pickle.dump(dt_classifier, f)

with open('./scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
