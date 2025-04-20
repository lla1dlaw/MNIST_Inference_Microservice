"""
Author: Liam Laidlaw
Filename: MNISTPredictor.py
Purpose: Flask microservice for predicting handwritten digits.
"""

import os
from flask import  request, jsonify
from flask import Flask
from Model_Loader import Loader

app = Flask(__name__)
app.debug = True
models = Loader(os.path.join("MNISTPredictor", "model_dicts"))


@app.route('/infer', methods=["POST"])
def get_inference():
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Invalid Format... Expected JSON',
                'status': 'error'
            }), 400
        
        # pull the model type and input image from the request body
        data = request.get_json()
        model_key = data['model']
        image = data['image']

        prediction = models.infer(model_key, image)
        activations = models.get_activations(model_key)

        print(f"Prediction: {prediction}")
        print(f"Activations: \n{activations}")
        return jsonify({
            'prediction': str(prediction),
            'activation': activations
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
    

@app.route('/get_available_models', methods=['GET'])
def get_available_models():
    try:
        return jsonify({
            "available_models": models.get_available_models()
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


def main():
    print("Starting MNSIT Microservice...")
    app.run(port=8001)
    

if __name__ == "__main__":
    main()