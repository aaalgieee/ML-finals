Documentation
Backend (main.py)
Imports: The script starts by importing necessary libraries such as Flask, numpy, PIL, pickle, sklearn, logging, pandas, joblib, tensorflow, cv2, and scipy. These libraries are used for web server functionality, data manipulation, machine learning, and image processing.

Flask App Initialization: The Flask app is initialized and CORS is enabled to allow cross-origin requests.

SVM Model:

Function load_svm_model_and_data: Loads the SVM model and preprocessed data from files.
Route /api/svm/predict: Handles POST requests to predict pneumonia from an uploaded image. The image is processed, resized, and normalized before being fed into the SVM model for prediction. The confidence score is calculated and returned along with the prediction.
KNN Model:

Function load_knn_model_and_data: Loads the KNN model, label encoders, and scaler from files.
Route /api/knn/predict: Handles POST requests to predict diabetes risk based on input features. The input data is preprocessed, scaled, and fed into the KNN model for prediction. The prediction and scatter plot data are returned.
Linear Regression Model:

Route /api/lr/predict: Handles POST requests to predict cardiovascular disease risk based on input features. The input data is preprocessed and fed into the linear regression model for prediction. The risk level and message are returned based on the prediction.
ANN Model:

Function load_label_encoder: Loads the label encoder from a file.
Function load_ann_model: Loads the ANN model and label encoder from files.
Route /api/ann/predict: Handles POST requests to predict brain tumor from an uploaded MRI image. The image is processed, resized, and normalized before being fed into the ANN model for prediction. If a tumor is detected, the tumor region is identified and returned along with the prediction.
Decision Tree Model:

Function load_dt_model: Loads the decision tree model and label encoder from files.
Route /api/dt/predict: Handles POST requests to predict maternal health risk based on input features. The input data is preprocessed and fed into the decision tree model for prediction. The risk level and message are returned based on the prediction.
Naive Bayes Model:

Function load_nb_model: Loads the Naive Bayes model and scaler from files.
Route /api/nb/predict: Handles POST requests to predict heart disease risk based on input features. The input data is preprocessed, scaled, and fed into the Naive Bayes model for prediction. The risk level, message, and risk factors are returned based on the prediction.
Health Check Route:

Route /api/health: Handles GET requests to check the health status of the server.
Main Execution: The Flask app is configured to run on a specified port.

Frontend (home-page.tsx and algorithm-test.tsx)
Imports: The script imports necessary libraries and components such as React, React Router, UI components, icons, and Chart.js.

Algorithm List: Defines a list of algorithms with their respective IDs, names, icons, descriptions, and input fields.

Prediction Response Type: Defines the structure of the prediction response object.

AlgorithmTest Component:

State Initialization: Initializes state variables for the selected algorithm, input values, result, loading status, image file, and image preview.
Effect Hook: Updates the selected algorithm and input fields based on the URL parameter.
Dropzone Configuration: Configures the dropzone for image upload.
Input Change Handler: Updates the input values based on user input.
Form Submission Handler: Handles form submission to send input data or image to the backend API for prediction. The response is processed and displayed to the user.
Rendering: Renders the algorithm test form with input fields or image upload based on the selected algorithm. Displays the prediction result after form submission.

Training Files
SVM Training (svm.ipynb):

Imports: Imports necessary libraries for data manipulation, image processing, and machine learning.
Data Loading: Loads and preprocesses training and test data from image files.
Model Training: Trains an SVM model on the preprocessed data.
Model Evaluation: Evaluates the model's accuracy on the test data.
Model Saving: Saves the trained model and data to files.
Naive Bayes Training (nb.py):

Imports: Imports necessary libraries for data manipulation and machine learning.
HeartDiseasePredictor Class: Defines a class for heart disease prediction using Naive Bayes. Includes methods for data validation, training, prediction, and model export.
Main Execution: Trains the model on a dataset and tests prediction on sample data.
Linear Regression Training (lr.ipynb):

Imports: Imports necessary libraries for data manipulation, machine learning, and visualization.
Data Loading: Loads and preprocesses the dataset.
Model Training: Trains a linear regression model on the preprocessed data.
Model Evaluation: Evaluates the model's performance on the test data.
Model Saving: Saves the trained model to a file.
KNN Training (knn.ipynb):

Imports: Imports necessary libraries for data manipulation and machine learning.
Data Loading: Loads and preprocesses the dataset.
Model Training: Trains a KNN model on the preprocessed data.
Model Evaluation: Evaluates the model's accuracy on the test data.
Model Saving: Saves the trained model and preprocessing objects to files.
Decision Tree Training (dt.py):

Imports: Imports necessary libraries for data manipulation and machine learning.
Data Loading: Loads and preprocesses the dataset.
Model Training: Trains a decision tree model on the preprocessed data.
Model Evaluation: Evaluates the model's accuracy on the test data.
Model Saving: Saves the trained model and label encoder to files.
ANN Training (ann.py):

Imports: Imports necessary libraries for data manipulation, image processing, and deep learning.
Data Loading: Defines functions to load and preprocess image data.
Model Creation: Defines a function to create an ANN model.
Main Execution: Loads and preprocesses training and test data, creates and trains the ANN model, evaluates the model, and saves the trained model and label encoder to files.

