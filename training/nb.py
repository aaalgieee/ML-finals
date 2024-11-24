import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import pickle

class HeartDiseasePredictor:
    """
    Heart Disease Prediction using Naive Bayes
    
    Features:
    - age: Age in years
    - sex: Gender (0=male, 1=female)
    - cp: Chest pain type (0-3)
    - trestbps: Resting blood pressure (mm Hg)
    - chol: Serum cholesterol (mg/dl)
    - fbs: Fasting blood sugar > 120 mg/dl (1=true, 0=false)
    - restecg: Resting ECG results (0-2)
    - thalach: Maximum heart rate achieved
    - exang: Exercise induced angina (1=yes, 0=no)
    - oldpeak: ST depression induced by exercise
    - slope: Slope of peak exercise ST segment (0-2)
    - ca: Number of major vessels colored by fluoroscopy (0-4)
    - thal: Thalium stress test result (0-3)
    """
    
    def __init__(self):
        self.model = GaussianNB()
        self.scaler = StandardScaler()
        
    def validate_input(self, data):
        """Validate input data ranges"""
        validators = {
            'age': lambda x: 0 <= x <= 120,
            'sex': lambda x: x in [0, 1],
            'cp': lambda x: 0 <= x <= 3,
            'trestbps': lambda x: 0 <= x <= 300,
            'chol': lambda x: 0 <= x <= 600,
            'fbs': lambda x: x in [0, 1],
            'restecg': lambda x: 0 <= x <= 2,
            'thalach': lambda x: 0 <= x <= 250,
            'exang': lambda x: x in [0, 1],
            'oldpeak': lambda x: 0 <= x <= 10,
            'slope': lambda x: 0 <= x <= 2,
            'ca': lambda x: 0 <= x <= 4,
            'thal': lambda x: 0 <= x <= 3
        }
        
        for column, validator in validators.items():
            if not all(data[column].apply(validator)):
                raise ValueError(f"Invalid values in column {column}")

    def train(self, dataset_path):
        # Load and prepare data
        df = pd.read_csv(dataset_path)
        self.validate_input(df)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate and print detailed metrics
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f"Training Accuracy: {train_accuracy:.2f}")
        print(f"Testing Accuracy: {test_accuracy:.2f}")
        
        # Verify predictions distribution
        y_pred = self.model.predict(X_test)
        unique, counts = np.unique(y_pred, return_counts=True)
        print("\nPrediction Distribution:")
        print(dict(zip(unique, counts)))
        
        return test_accuracy
    
    def predict(self, input_data):
        """
        Predict heart disease status
        Args:
            input_data: DataFrame with patient features
        Returns:
            predictions: 0 (no disease) or 1 (disease present)
        """
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        self.validate_input(input_data)
        # Scale input data
        scaled_data = self.scaler.transform(input_data)
        # Make prediction
        return self.model.predict(scaled_data)
    
    def predict_proba(self, input_data):
        """
        Get prediction probabilities
        Args:
            input_data: DataFrame with patient features
        Returns:
            probabilities: Array of probabilities [no_disease_prob, disease_prob]
        """
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        self.validate_input(input_data)
        scaled_data = self.scaler.transform(input_data)
        return self.model.predict_proba(scaled_data)
    
    def export_model(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        """Export model and scaler using pickle"""
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)

if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    accuracy = predictor.train('nb.csv')  # Make sure you're using the correct dataset file
    print(f"\nFinal Model Accuracy: {accuracy:.2f}")
    
    # Test prediction on sample data
    sample_data = pd.DataFrame([{
        'age': 55,
        'sex': 1,
        'cp': 2,
        'trestbps': 150,
        'chol': 250,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.0,
        'slope': 1,
        'ca': 0,
        'thal': 2
    }])
    
    prediction = predictor.predict(sample_data)
    proba = predictor.predict_proba(sample_data)
    print("\nSample Prediction Test:")
    print(f"Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
    print(f"Probabilities: Low Risk: {proba[0][0]:.2f}, High Risk: {proba[0][1]:.2f}")
    
    predictor.export_model()
    print("\nModel exported successfully")
