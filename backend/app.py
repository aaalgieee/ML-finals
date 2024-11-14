from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import logging

app = Flask(__name__)
CORS(app)
#logging.basicConfig(level=logging.DEBUG)

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
def predict():
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

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)