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
lr_model = joblib.load('./lr/lr.pkl')

@app.route('/api/lr/predict', methods=['POST'])
def predict_lr():
    try:
        data = request.get_json(force=True)
        app.logger.debug(f"Received data: {data}")
        
        features = data['features']
        input_data = pd.DataFrame([{
            'male': 1 if features['male'] == 1 else 0,
            'age': features['age'],
            'currentSmoker': features['currentSmoker'],
            'cigsPerDay': features['cigsPerDay'],
            'BPMeds': features['BPMeds'],
            'diabetes': features['diabetes'],
            'totChol': features['totChol'],
            'sysBP': features['sysBP'],
            'diaBP': features['diaBP'],
            'BMI': features['BMI'],
            'heartRate': features['heartRate'],
            'glucose': features['glucose']
        }])

        # Make prediction
        prediction = lr_model.predict(input_data)[0]
        risk_percentage = float(prediction * 100)
        
        # Create a friendly risk assessment
        if risk_percentage < 25:
            risk_level = "Low Risk"
            message = "Your cardiovascular disease risk appears to be low."
        elif risk_percentage < 50:
            risk_level = "Moderate Risk"
            message = "You have a moderate risk of cardiovascular disease."
        elif risk_percentage < 75:
            risk_level = "High Risk"
            message = "You have an elevated risk of cardiovascular disease."
        else:
            risk_level = "Very High Risk"
            message = "Your cardiovascular disease risk is significantly elevated."
        
        return jsonify({
            'success': True,
            'prediction': risk_percentage,
            'risk_level': risk_level,
            'message': message
        })

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
def load_dt_model():
    try:
        model = joblib.load('./dt/dt_model.pkl')
        label_encoder = joblib.load('./dt/label_encoder.pkl')  # Fixed: using label_encoder
        return model, label_encoder
    except Exception as e:
        app.logger.error(f"Error loading decision tree model: {str(e)}")
        raise

# Initialize Decision Tree model
dt_model, dt_label_encoder = load_dt_model()  # Now correctly unpacking model and label_encoder

@app.route('/api/dt/predict', methods=['POST'])
def predict_dt():
    try:
        data = request.get_json(force=True)
        app.logger.debug(f"Received data for DT prediction: {data}")
        
        features = data['features']
        
        # Create input DataFrame with correct column names matching training data
        input_data = pd.DataFrame([{
            'Age': features['age'],
            'SystolicBP': features['systolicBP'],
            'DiastolicBP': features['diastolicBP'],
            'BS': features['bs'],
            'BodyTemp': features['bodyTemp'],
            'HeartRate': features['heartRate']
        }])
        
        app.logger.debug(f"Input data shape: {input_data.shape}")
        app.logger.debug(f"Input data: {input_data}")

        # Input validation with updated column names
        if not (13 <= input_data['Age'].iloc[0] <= 70):
            raise ValueError("Age must be between 13 and 70")
        if not (70 <= input_data['SystolicBP'].iloc[0] <= 180):
            raise ValueError("Systolic BP must be between 70 and 180 mmHg")
        if not (40 <= input_data['DiastolicBP'].iloc[0] <= 120):
            raise ValueError("Diastolic BP must be between 40 and 120 mmHg")
        if not (30 <= input_data['BS'].iloc[0] <= 300):
            raise ValueError("Blood sugar must be between 30 and 300 mg/dL")
        if not (35 <= input_data['BodyTemp'].iloc[0] <= 42):
            raise ValueError("Body temperature must be between 35°C and 42°C")
        if not (40 <= input_data['HeartRate'].iloc[0] <= 200):
            raise ValueError("Heart rate must be between 40 and 200 bpm")

        try:
            # Make prediction with correct column names
            prediction = dt_model.predict(input_data)[0]
            prediction_proba = dt_model.predict_proba(input_data)[0]
            confidence = float(max(prediction_proba) * 100)
        except Exception as pred_error:
            app.logger.error(f"Prediction error: {pred_error}")
            raise ValueError("Error in making prediction. Model may need retraining.")

        # Map risk levels
        risk_levels = {
            0: "Low Risk",
            1: "Mid Risk",
            2: "High Risk"
        }
        
        risk_messages = {
            "Low Risk": "Normal pregnancy indicators. Continue regular check-ups.",
            "Mid Risk": "Some concerning indicators. Increased monitoring recommended.",
            "High Risk": "Critical indicators detected. Immediate medical attention required."
        }
        
        risk_level = risk_levels.get(prediction, "Unknown Risk")
        message = risk_messages.get(risk_level, "Unable to determine risk level")

        app.logger.debug(f"Prediction successful: {risk_level} with {confidence}% confidence")

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'risk_level': risk_level,
            'message': message,
            'confidence': confidence
        })

    except ValueError as ve:
        app.logger.warning(f"Validation error: {str(ve)}")
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        app.logger.error(f"Error processing decision tree request: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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