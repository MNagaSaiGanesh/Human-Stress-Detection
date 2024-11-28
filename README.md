Human Stress Detection and Prediction Using ANN
Project Overview
This project aims to build a system that detects human stress levels during sleep using physiological data. An Artificial Neural Network (ANN) is trained to classify sleep data into "Stressed" or "Not Stressed."

Key Features:

Use of physiological data (heart rate, respiratory rate, body movements) to detect stress levels.
The system utilizes an Artificial Neural Network (ANN) to make binary predictions: 1 for stressed and 0 for not stressed.
Model performance metrics include accuracy, precision, recall, and F1-score.
The trained model can be deployed as a Flask-based web application for real-time stress detection.
Modules and Features
1. Data Preprocessing and Visualization
The dataset includes physiological features (e.g., heart rate, respiratory rate, body movements).
Features are visualized to understand distributions and class balance (stressed vs. not stressed).
2. Model Architecture
The ANN consists of multiple layers:
Input Layer: Takes physiological features as input.
Hidden Layers: Dense layers with ReLU activation.
Output Layer: Sigmoid activation for binary classification.
3. Training and Evaluation
The model is trained with binary cross-entropy loss and Adam optimizer.
Performance metrics (accuracy, precision, recall, F1-score) are calculated.
Visualizations for training progress (accuracy and loss over epochs).
4. Model Deployment
The trained model is deployed as a Flask app.
Flask API exposes an endpoint to make predictions on new sleep data.
The system supports input scaling using a pre-saved scaler.pkl file.
Technologies Used
TensorFlow/Keras: For building and training the ANN model.
Flask: For creating the web application.
NumPy: For data manipulation and model input preparation.
Scikit-learn: For scaling the input features.
Joblib: For saving/loading the scaler.
Setup Instructions
1. Clone the repository
First, clone the repository to your local machine:

bash
Copy code
git clone <repository-url>
cd <project-folder>
2. Install dependencies
Install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
Create a requirements.txt file with the following dependencies:

makefile
Copy code
Flask==2.0.3
tensorflow==2.8.0
scikit-learn==1.0.2
joblib==1.1.0
numpy==1.21.2
3. Model and Scaler Files
Ensure that the following files are in your project directory:

stress_detection_model.h5: The trained ANN model.
scaler.pkl: The scaler used during training to normalize input data.
4. Run the Flask Application
Start the Flask app using:

bash
Copy code
python app.py
This will start the server on http://127.0.0.1:5000/.

Usage
1. Prediction Endpoint
The Flask app exposes a /predict endpoint for predicting stress levels based on input features.

Request Format: Send a POST request to the /predict endpoint with a JSON payload containing the physiological features as follows:

json
Copy code
{
  "features": [70, 16, 15, 98, 95, 45]
}
The features list should contain the values for the following physiological data:

Heart Rate
Respiratory Rate
Body Movements
(Add any other features used in the dataset)
Response Format: The response will be in JSON format with the predicted stress level:

json
Copy code
{
  "stress_level": 0  // or 1
}
2. Example Using Python requests Library
python
Copy code
import requests

url = 'http://127.0.0.1:5000/predict'
data = {'features': [70, 16, 15, 98, 95, 45]}  # Example feature values
response = requests.post(url, json=data)

print(response.json())  # Outputs predicted stress level
Model Performance Metrics
During training, the following metrics are evaluated:

Accuracy: Percentage of correct predictions.
Precision: Proportion of true positives among the predicted positives.
Recall: Proportion of true positives among the actual positives.
F1-Score: Harmonic mean of precision and recall.
Troubleshooting
1. Model Compilation Warning
If you see a warning like:

ruby
Copy code
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built...
This warning is safe to ignore unless you need to evaluate the model again. The model will still make predictions successfully.

2. Missing Files (Scaler or Model)
Make sure you have both stress_detection_model.h5 and scaler.pkl files in your project folder. If they are missing, you won't be able to load the model or scale the input features.

Future Improvements
Implement real-time data collection from wearables for stress prediction.
Add more features to improve the model's accuracy, such as EEG data.
Deploy the Flask app to production using a WSGI server like Gunicorn.
