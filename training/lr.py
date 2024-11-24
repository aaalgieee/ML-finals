import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('lr.csv')

# Select features based on CSV columns
features = [
    'age', 'male', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 'totChol', 
    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
]
target = 'Risk'

# Handle missing values
df = df.dropna() 

# Prepare X (features) and y (target)
X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Training Score: {train_score:.4f}")
print(f"Testing Score: {test_score:.4f}")

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Coefficient', ascending=False))

print("\nModel Features:", features)
print("Number of features:", len(features))
print("Number of samples:", len(df))

# Save the model and scaler
with open('hypertension_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
