from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load the pre-trained model (will be trained/setup later)
model = None
WASTE_CATEGORIES = ['recyclable', 'organic', 'hazardous', 'non-recyclable']

def load_model():
    global model
    try:
        # Try to load a pre-trained model
        model_path = os.path.join(os.path.dirname(__file__), 'waste_model.h5')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        else:
            print("Model file not found, using simulation mode")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/api/classify', methods=['POST'])
def classify_waste():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to match model input
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        if model is not None:
            # Use actual model prediction
            predictions = model.predict(image_array)
            confidence = float(np.max(predictions))
            predicted_class = WASTE_CATEGORIES[np.argmax(predictions)]
        else:
            # Simulation mode (for MVP)
            import random
            predicted_class = random.choice(WASTE_CATEGORIES)
            confidence = round(random.uniform(0.7, 0.95), 2)
        
        return jsonify({
            'category': predicted_class,
            'confidence': confidence,
            'all_categories': WASTE_CATEGORIES
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    return jsonify({'categories': WASTE_CATEGORIES})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
