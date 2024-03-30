from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

app= Flask(__name__)
rfc=joblib.load("model.pkl")
print("Model Loaded")
rfc_columns=joblib.load("model_columns.pkl")
print("Model Columns Loaded")

@app.route('/crop-predict', methods=["POST"])
def crop_predict():
    if rfc:
        try:
            json_ = request.get_json(force=True)
            features = [[entry['Nitrogen'], entry['Phosporous'], entry['Potassium'], 
                   entry['temperature'], entry['humidity'], entry['ph'], entry['rainfall']] for entry in json_]
            prediction = list(rfc.predict(features))
            prediction=list(prediction[0])
            true_index=prediction.index(True)
            if true_index==-1:
                return ("Nothing will grow in this condition")
            else:
                predicted_value=rfc_columns[true_index].lstrip("label_")
                return jsonify({'prediction': predicted_value})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return("No data available for training")
    

if __name__=="__main__":
    
    app.run(debug=True, port=4000)
