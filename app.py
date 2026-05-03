from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model (update path if needed)
# model = joblib.load('model.joblib')  # Uncomment when you add model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')
        image = image.resize((128, 128))
        
        # Dummy prediction (replace with your model)
        prediction = np.random.choice(['Normal', 'Mild Cognitive Impairment', 'Alzheimer'])
        confidence = np.random.uniform(0.85, 0.98)
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence:.1%}",
            'message': f'Patient shows signs of {prediction}' if prediction != 'Normal' else 'No Alzheimer\'s detected'
        })
    except:
        return jsonify({'error': 'Upload failed'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
