from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def index():
    return "Welcome to the Fake Bills Detector API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['height_left'], data['height_right'], data['margin_low'], data['margin_up'], data['length']]])
    
    prediction = model.predict(features)
    
    return jsonify({
        'prediction': 'Fake Bill' if prediction[0] == 1 else 'Genuine Bill'
    })

if __name__ == '__main__':
    app.run(debug=True)
