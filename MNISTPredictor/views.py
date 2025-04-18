"""
Author: Liam Laidlaw
Filename: MNISTPredictor.py
Purpose: Flask microservice for predicting handwritten digits.
"""

import json
from flask import  request, jsonify
from MNISTPredictor import app, models
from flask_restplus import Api, Resource


@app.route('/get_inference', methods=["POST"])
def get_inference():
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Invalid Format... Expected JSON',
                'status': 'error'
            }), 400
        
        # pull the model type and input image from the request body
        model_key = request.form['model']
        image = request.form['image']

        print(model_key)
        print(image)
        # if not "cnn" in model_key: # ensure that the model is expecting flat data (convert to 2d for conv NN)
        #     res = models[model_key]()
        # else:
        #     ...

    # except json.JSONDecoder:
    #     return jsonify({
    #         'error': 'Invalid JSON Format',
    #         'status': 'error'
    #     }), 500
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


def main():
    print("Starting MNSIT Microservice...")
    app.run(port=8000)
    

if __name__ == "__main__":
    main()