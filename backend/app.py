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

app = Flask(__name__)
CORS(app)


# SVM Model
def load_model_and_data():
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

# Initialize model
model = load_model_and_data()

@app.route('/api/svm/predict', methods=['POST'])
def predict_svm():
    try:
        #app.logger.debug(f"Received request files: {request.files}")
        #app.logger.debug(f"Received request form: {request.form}")
        
        if 'image' not in request.files:
            app.logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            app.logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400

        # Load and preprocess the image
        img = Image.open(file.stream).convert('L')
        img = img.resize((32, 32), resample=Image.BICUBIC)
        X = np.array(img) / 255.0
        X = X.reshape(1, -1)

        # Make prediction
        prediction = model.predict(X)[0]
        
        # Convert string prediction to numeric value if needed
        prediction_value = 1 if str(prediction).upper() == 'PNEUMONIA' else 0
        prediction_message = 'Pneumonia' if prediction_value == 1 else 'Normal'
        
        return jsonify({
            'success': True,
            'prediction': prediction_value,
            'message': prediction_message,
            'raw_prediction': str(prediction)
        })

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Load saved objects
model = joblib.load('knn/knn_model.pkl')
label_encoders = joblib.load('knn/label_encoders.pkl')
scaler = joblib.load('knn/scaler.pkl')

# Print available classes for label encoders
for key, encoder in label_encoders.items():
    print(f"{key} classes:", encoder.classes_)

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
        prediction = model.predict(input_scaled)
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


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)