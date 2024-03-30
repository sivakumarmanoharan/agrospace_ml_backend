from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from flask_cors import CORS
import json

app= Flask(__name__)
CORS(app)
rfc=joblib.load("model.pkl")
print("Model Loaded")
rfc_columns=joblib.load("model_columns.pkl")
print("Model Columns Loaded")

@app.route('/crop-predict', methods=["POST"])
def crop_predict():
    if rfc:
        try:
            json_ = request.get_json(force=True)
            features = [[float(entry['nitrogen']), float(entry['phosphorous']), float(entry['potassium']), 
                   float(entry['temperature']), float(entry['humidity']),float(entry['ph']), float(entry['rainfall'])] for entry in json_]
            prediction = list(rfc.predict(features))
            prediction=list(prediction[0])
            try:
                value=prediction.index(True)
            except ValueError:
                value=-1
            if value!=-1:
                predicted_crop=list(rfc_columns)[value]
                output_string=""
                str_list = predicted_crop.split("label_")
                for element in str_list:
                    output_string += element
                return jsonify({'prediction': output_string})
            else:
                output_string="NO_CROP"
                return jsonify({'prediction':output_string})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return("No data available for training")
    

if __name__=="__main__":
    
    app.run(debug=True, port=4000)
