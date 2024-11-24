import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the data
df = pd.read_csv('dt_maternal.csv')

# Prepare features (X) and target (y)
X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)

# Evaluate the model
train_score = dt_classifier.score(X_train, y_train)
test_score = dt_classifier.score(X_test, y_test)

print(f'Training accuracy: {train_score:.2f}')
print(f'Testing accuracy: {test_score:.2f}')

# Save the model and label encoder for Flask
with open('./dt_model.pkl', 'wb') as f:
    pickle.dump(dt_classifier, f)

with open('./label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
