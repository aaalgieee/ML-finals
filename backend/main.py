from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import logging
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import cv2
from skimage.feature import peak_local_max
from scipy import ndimage
from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)
CORS(app)


# SVM Model
def load_svm_model_and_data():
    # Load the trained SVM model and preprocessed data
    clf = pickle.load(open('./svm/pneumonia_svm_model.pkl', 'rb'))
    X_train = np.load('./svm/pneumonia_X_train.npy')
    y_train = np.load('./svm/pneumonia_y_train.npy')
    X_test = np.load('./svm/pneumonia_X_test.npy')
    y_test = np.load('./svm/pneumonia_y_test.npy')

    # Flatten the 3D arrays to 2D
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return clf

# Initialize SVM model
svm_model = load_svm_model_and_data()

@app.route('/api/svm/predict', methods=['POST'])
def predict_svm():
    try:
        if 'image' not in request.files:
            app.logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            app.logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400

        img = Image.open(file.stream).convert('L')
        img = img.resize((32, 32), resample=Image.BICUBIC)
        X = np.array(img) / 255.0
        X = X.reshape(1, -1)

        # Get prediction without probability
        prediction = svm_model.predict(X)[0]
        
        # Use decision_function as a confidence measure instead of probability
        confidence_score = float(abs(svm_model.decision_function(X)[0]))
        # Normalize confidence score to a 0-100 scale
        confidence = min(100, confidence_score * 20)  # Adjust multiplier as needed
        
        prediction_value = 1 if str(prediction).upper() == 'PNEUMONIA' else 0
        prediction_message = 'Pneumonia' if prediction_value == 1 else 'Normal'
        
        return jsonify({
            'success': True,
            'prediction': prediction_value,
            'message': prediction_message,
            'confidence': confidence,
            'raw_prediction': str(prediction)
        })

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500



# KNN Model
def load_knn_model_and_data():
    model = joblib.load('knn/knn_model.pkl')
    label_encoders = joblib.load('knn/label_encoders.pkl')
    scaler = joblib.load('knn/scaler.pkl')
    return model, label_encoders, scaler

# Initialize KNN model
knn_model, label_encoders, scaler = load_knn_model_and_data()

@app.route('/api/knn/predict', methods=['POST'])
def predict_knn():
    try:
        data = request.get_json(force=True)
        app.logger.debug(f"Received data: {data}")
        
        input_data = pd.DataFrame([data['features']])
        app.logger.debug(f"Input data: {input_data}")
        
        # Debug available encoders and their keys
        app.logger.debug(f"Label encoder keys: {list(label_encoders.keys())}")
        app.logger.debug(f"Available gender classes: {label_encoders['gender'].classes_}")
        
        # Rename columns to match the label encoder keys
        column_mapping = {
            'gender': 'gender',
            'age': 'age',
            'hypertension': 'hypertension',
            'heart_disease': 'heart_disease',
            'smoking_history': 'smoking_history',
            'bmi': 'bmi',
            'HbA1c_level': 'HbA1c_level',
            'blood_glucose_level': 'blood_glucose_level'
        }
        input_data = input_data.rename(columns=column_mapping)
        
        # Handle string transformations with proper casing
        input_data['gender'] = input_data['gender'].str.capitalize()
        # Map smoking history values to match training data
        smoking_map = {
            'no info': 'No Info',
            'not current': 'not current',
            'current': 'current',
            'former': 'former',
            'never': 'never',
            'ever': 'ever'
        }
        input_data['smoking_history'] = input_data['smoking_history'].str.lower().map(smoking_map)
        
        app.logger.debug(f"After string transformation: {input_data}")
        
        # Transform categorical variables
        categorical_columns = ['gender', 'smoking_history']
        for column in categorical_columns:
            if column in label_encoders:
                input_data[column] = label_encoders[column].transform([input_data[column].iloc[0]])[0]
                app.logger.debug(f"Transformed {column}: {input_data[column]}")
        
        # Convert numeric values
        numeric_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        for column in numeric_columns:
            input_data[column] = pd.to_numeric(input_data[column])
        
        # Ensure boolean values are 1 and 0
        input_data['hypertension'] = input_data['hypertension'].astype(int)
        input_data['heart_disease'] = input_data['heart_disease'].astype(int)
        
        app.logger.debug(f"Preprocessed data: {input_data}")
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        app.logger.debug(f"Scaled input: {input_scaled}")
        
        # Make prediction
        prediction = knn_model.predict(input_scaled)
        app.logger.debug(f"Prediction: {prediction}")
        
        prediction_result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        
        # Generate scatter plot data with only the prediction point
        scatter_plot_data = [{
            'x': float(row['age']),
            'y': float(row['bmi']),
            'label': prediction_result
        } for _, row in input_data.iterrows()]
        
        app.logger.debug(f"Scatter plot data: {scatter_plot_data}")
        
        return jsonify({
            'prediction': prediction_result,
            'scatter_plot_data': scatter_plot_data
        })
    
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


#Linear Regression Model
try:
    lr_model = joblib.load('./lr/hypertension_model.pkl')
    lr_scaler = joblib.load('./lr/scaler.pkl')  # Add scaler if available
except Exception as e:
    app.logger.error(f"Error loading Linear Regression model: {str(e)}")
    lr_model = None
    lr_scaler = None

@app.route('/api/lr/predict', methods=['POST'])
def predict_lr():
    try:
        if lr_model is None:
            raise ValueError("Model not properly loaded")

        data = request.get_json(force=True)
        features = data['features']
        
        # Create DataFrame with features in exact training order
        input_data = pd.DataFrame([
            [
                features['age'],
                features['male'],
                features['currentSmoker'],
                features['cigsPerDay'],
                features['BPMeds'],
                features['diabetes'],
                features['totChol'],
                features['sysBP'],
                features['diaBP'],
                features['BMI'],
                features['heartRate'],
                features['glucose']
            ]
        ], columns=[
            'age', 'male', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 
            'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
        ])

        app.logger.debug(f"Input data with training order: {input_data.columns.tolist()}")

        # Scale the input data if scaler is available
        if lr_scaler is not None:
            input_scaled = lr_scaler.transform(input_data)
        else:
            input_scaled = input_data

        app.logger.debug(f"Scaled input data: {input_scaled}")

        # Make prediction and apply sigmoid transformation
        raw_prediction = lr_model.predict(input_scaled)[0]
        risk_percentage = 100 / (1 + np.exp(-raw_prediction))  # Sigmoid transformation
        risk_percentage = float(min(100, max(0, risk_percentage)))  # Clamp between 0 and 100
        
        # Updated risk thresholds and messages
        if risk_percentage < 25:
            risk_level = "Low Risk"
            message = "Your cardiovascular disease risk appears to be low. Maintain a healthy lifestyle."
        elif risk_percentage < 50:
            risk_level = "Moderate Risk"
            message = "You have a moderate risk of cardiovascular disease. Consider lifestyle improvements."
        elif risk_percentage < 75:
            risk_level = "High Risk"
            message = "You have an elevated risk of cardiovascular disease. Consult with a healthcare provider."
        else:
            risk_level = "Very High Risk"
            message = "Your cardiovascular disease risk is significantly elevated. Immediate medical consultation recommended."

        # Add risk factors analysis
        risk_factors = []
        if features['age'] > 55:
            risk_factors.append("Age above 55")
        if features['sysBP'] > 140:
            risk_factors.append("High blood pressure")
        if features['totChol'] > 200:
            risk_factors.append("High cholesterol")
        if features['currentSmoker'] == 1:
            risk_factors.append("Current smoker")
        if features['diabetes'] == 1:
            risk_factors.append("Diabetes")
        
        return jsonify({
            'success': True,
            'prediction': risk_percentage,
            'risk_level': risk_level,
            'message': message,
            'risk_factors': risk_factors
        })

    except ValueError as ve:
        app.logger.error(f"Validation error: {str(ve)}")
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ANN Model
def load_label_encoder(filepath):
    try:
        # Load the numpy array containing the label encoder data
        data = np.load(filepath, allow_pickle=True)
        # If the data is a 0-dimensional array containing an object (the encoder),
        # access it using [()] instead of .item()
        return data[()]
    except Exception as e:
        app.logger.error(f"Error loading label encoder: {str(e)}")
        raise

def load_ann_model():
    try:
        model = load_model('./ann/best_model.keras')
        label_encoder = load_label_encoder('./ann/label_encoder.npy')  # Note: changed extension to .npy
        return model, label_encoder
    except Exception as e:
        app.logger.error(f"Error loading ANN model: {str(e)}")
        raise

# Initialize ANN model
ann_model, ann_encoder = load_ann_model()

@app.route('/api/ann/predict', methods=['POST'])
def predict_ann():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read and preprocess image
        img_array = np.frombuffer(file.read(), np.uint8)
        original_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if (original_img is None):
            return jsonify({'error': 'Invalid image file'}), 400

        # Resize and normalize for prediction
        img = cv2.resize(original_img, (64, 64))
        img = img / 255.0
        img = img.reshape(1, -1)

        # Make prediction
        prediction = ann_model.predict(img)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction) * 100)

        class_labels = {
            0: "Glioma",
            1: "Meningioma",
            2: "No Tumor",
            3: "Pituitary"
        }
        prediction_message = class_labels.get(predicted_class, f"Class {predicted_class}")

        # Only perform tumor detection if a tumor is predicted
        tumor_region = None
        if prediction_message != "No Tumor":
            # Image processing for tumor detection
            blurred = cv2.GaussianBlur(original_img, (5, 5), 0)
            thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (assumed to be the tumor)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Convert to percentage coordinates for frontend scaling
                height, width = original_img.shape
                tumor_region = {
                    'x': (x / width) * 100,
                    'y': (y / height) * 100,
                    'width': (w / width) * 100,
                    'height': (h / height) * 100
                }

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'message': prediction_message,
            'tumor_region': tumor_region
        })

    except Exception as e:
        app.logger.error(f"Error processing ANN request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Decision Tree Model
# Define risk levels and messages with correct mapping
risk_levels = {
    0: "Low Risk",
    1: "Mid Risk",  # Changed from "Moderate Risk" to "Mid Risk" to match CSV
    2: "High Risk"
}

risk_messages = {
    "Low Risk": "Normal pregnancy indicators. Continue regular check-ups.",
    "Mid Risk": "Some concerning indicators. Increased monitoring recommended.",  # Updated message
    "High Risk": "Critical indicators detected. Immediate medical attention required."
}

def load_dt_model():
    try:
        model = joblib.load('./dt/dt_model.pkl')
        scaler = joblib.load('./dt/scaler.pkl')
        return model, scaler
    except Exception as e:
        app.logger.error(f"Error loading decision tree model: {str(e)}")
        raise

# Initialize Decision Tree model and scaler
dt_model, dt_scaler = load_dt_model()

@app.route('/api/dt/predict', methods=['POST'])
def predict_dt():
    try:
        data = request.get_json(force=True)
        features = data['features']
        
        # Create input DataFrame with proper types
        input_data = pd.DataFrame([{
            'Age': int(features['age']),
            'SystolicBP': int(features['systolicBP']),
            'DiastolicBP': int(features['diastolicBP']),
            'BS': float(features['bs']),
            'BodyTemp': float(features['bodyTemp']),
            'HeartRate': int(features['heartRate'])
        }])

        # Validate inputs with updated ranges based on CSV data
        validation_ranges = {
            'Age': (13, 70),
            'SystolicBP': (70, 180),
            'DiastolicBP': (40, 120),
            'BS': (6.0, 19.0),
            'BodyTemp': (97.0, 103.0),
            'HeartRate': (40, 200)
        }

        for column, (min_val, max_val) in validation_ranges.items():
            if not (min_val <= input_data[column].iloc[0] <= max_val):
                raise ValueError(f"{column} must be between {min_val} and {max_val}")

        # Scale features
        scaled_features = dt_scaler.transform(input_data)
        
        # Make prediction
        prediction = dt_model.predict(scaled_features)[0]
        prediction_proba = dt_model.predict_proba(scaled_features)[0]

        # Get risk level and message
        risk_level = risk_levels[prediction]
        message = risk_messages[risk_level]

        # Updated thresholds based on CSV data analysis
        risk_factors = []
        thresholds = {
            'SystolicBP': 140,  # High BP threshold
            'DiastolicBP': 90,  # High BP threshold
            'BS': 11.0,  # High blood sugar threshold
            'BodyTemp': 100.4,  # Fever threshold
            'HeartRate': 100  # High heart rate threshold
        }

        # Analyze risk factors
        for param, threshold in thresholds.items():
            value = input_data[param].iloc[0]
            if value > threshold:
                if param == 'BS':
                    risk_factors.append("High Blood Sugar")
                elif param == 'BodyTemp':
                    risk_factors.append("Fever")
                elif param == 'HeartRate':
                    risk_factors.append("Elevated Heart Rate")
                else:
                    risk_factors.append(f"High {param}")

        # Add age-related risk factor
        if not (18 <= input_data['Age'].iloc[0] <= 35):
            risk_factors.append("Age-related risk factor")

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'risk_level': risk_level,
            'message': message,
            'confidence': float(max(prediction_proba) * 100),
            'risk_factors': risk_factors,
            'probabilities': {
                'low_risk': float(prediction_proba[0]),
                'mid_risk': float(prediction_proba[1]),
                'high_risk': float(prediction_proba[2])
            }
        })

    except ValueError as ve:
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Error processing decision tree request: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# Load Naive Bayes model
def load_nb_model():
    try:
        with open('./nb/model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('./nb/scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        app.logger.error(f"Error loading Naive Bayes model: {str(e)}")
        raise

# Initialize Naive Bayes model
nb_model, nb_scaler = load_nb_model()

@app.route('/api/nb/predict', methods=['POST'])
def predict_nb():
    try:
        data = request.get_json(force=True)
        app.logger.debug(f"Received data for NB prediction: {data}")
        
        features = data['features']
        input_data = pd.DataFrame([{
            'age': features['age'],
            'sex': features['sex'],
            'cp': features['cp'],
            'trestbps': features['trestbps'],
            'chol': features['chol'],
            'fbs': features['fbs'],
            'restecg': features['restecg'],
            'thalach': features['thalach'],
            'exang': features['exang'],
            'oldpeak': features['oldpeak'],
            'slope': features['slope'],
            'ca': features['ca'],
            'thal': features['thal']
        }])
        
        app.logger.debug(f"Input data before scaling: {input_data}")
        
        # Scale features
        scaled_data = nb_scaler.transform(input_data)
        
        # Make prediction and get probabilities
        prediction = nb_model.predict(scaled_data)[0]
        prediction_proba = nb_model.predict_proba(scaled_data)[0]
        
        # Calculate confidence and risk level
        high_risk_prob = prediction_proba[1]
        confidence = float(max(prediction_proba) * 100)
        
        # Define risk levels based on probability
        if high_risk_prob >= 0.75:
            risk_level = "Very High"
            message = "Immediate medical attention recommended."
        elif high_risk_prob >= 0.5:
            risk_level = "High"
            message = "Consultation with a healthcare provider is strongly advised."
        elif high_risk_prob >= 0.25:
            risk_level = "Moderate"
            message = "Regular monitoring and lifestyle modifications recommended."
        else:
            risk_level = "Low"
            message = "Continue maintaining healthy lifestyle habits."

        # Create risk factors analysis
        risk_factors = []
        if input_data['age'].iloc[0] > 60:
            risk_factors.append("Age above 60")
        if input_data['trestbps'].iloc[0] > 140:
            risk_factors.append("High blood pressure")
        if input_data['chol'].iloc[0] > 240:
            risk_factors.append("High cholesterol")
        if input_data['thalach'].iloc[0] < 120:
            risk_factors.append("Low maximum heart rate")
        
        app.logger.debug(f"Prediction: {prediction}, Probability: {prediction_proba}")
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'risk_level': risk_level,
            'message': message,
            'confidence': confidence,
            'probabilities': {
                'low_risk': float(prediction_proba[0]),
                'high_risk': float(prediction_proba[1])
            },
            'risk_factors': risk_factors,
            'details': {
                'systolic_bp': int(input_data['trestbps'].iloc[0]),
                'cholesterol': int(input_data['chol'].iloc[0]),
                'max_heart_rate': int(input_data['thalach'].iloc[0])
            }
        })

    except Exception as e:
        app.logger.error(f"Error processing Naive Bayes request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)