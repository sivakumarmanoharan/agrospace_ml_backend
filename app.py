from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from flask_cors import CORS
import json
import pickle

app= Flask(__name__)
CORS(app)
rfc=joblib.load("model.pkl")
print("Model Loaded")
rfc_columns=joblib.load("model_columns.pkl")
print("Model Columns Loaded")
model = pickle.load(open('classifier.pkl','rb'))
ferti = pickle.load(open('fertilizer.pkl','rb'))

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
        return({"prediction":"No data available for training"})

@app.route('/predict-fertilizer',methods=['POST'])
def predict():
    if model and ferti:
        crop_dict={"Barley":0,"Cotton":1,"Ground Nuts":2,"Maize":3,"Millets":4,"Oil_seeds":5,"Paddy":6,"Pulses":7,"Sugarcane":8,"Tobacco":9,"Wheat":10}
        soil_dict={"Black":0,"Clayey":1,"Loamy":2,"Red":3,"Sandy":4}
        json_ = request.get_json(force=True)
        soil_type=json_[0]["soil_type"]
        soil_type_val=soil_dict[soil_type]
        crop_type=json_[0]["crop"]
        crop_type_val=crop_dict[crop_type]
        features = [[int(entry['temperature']), int(entry['humidity']), int(entry['moisture']), 
                   soil_type_val, crop_type_val,int(entry['nitrogen']), int(entry['potassium']),int(entry['phosphorous'])] for entry in json_]
        res=ferti.classes_[model.predict(features)]
        return jsonify({'prediction':res[0]})
    else:
        return({"prediction":"No data available for training"})

ensemble_RFSVMCNN_model =  joblib.load('ensembler_RFSVMCNN.pkl')
ensemble_DTCCNN_models = joblib.load('ensemble_DTCCNN.pkl')  
@app.route('/predict-yield', methods =["POST"])
def predict_yield():
        json_ = request.get_json(force=True)
        user_input_area=json_[0]["area"]
        user_input_crop=json_[0]["crop"]
        user_input_rain=float(json_[0]["rainfall"])
        user_input_pesticides=float(json_[0]["pesticides"])
        user_input_temp=float(json_[0]["temperature"])
        res = pd.read_csv("ml_models/model_results.csv")
        yield_df = pd.read_csv('ml_models/Data_Preprocessed.csv')
        yield_df_onehot = pd.get_dummies(yield_df, columns=['Area',"Item"], prefix = ['Country',"Item"])
        features=yield_df_onehot.loc[:]
        label=yield_df['hg/ha_yield']
        features = features.drop(['Year'], axis=1)
        features = features.drop(['hg/ha_yield'], axis=1)
        features = features.iloc[[0]]
        features.loc[0, 'average_rain_fall_mm_per_year'] = user_input_rain
        features.loc[0, 'pesticides_tonnes'] = user_input_pesticides
        features.loc[0, 'avg_temp'] = user_input_temp
        features_columns = features.shape[1]
        for i in range(features_columns):
            column_name = features.columns[i]
            substring = column_name.split("_")[1]
            if substring == user_input_area:
                features.loc[0, column_name] = 1
            if substring == user_input_crop:
                features.loc[0, column_name] = 1
        features_array = features.values
        features_array = features_array.reshape(-1,2)
        features_array
        res["hg/ha yield"] = [ ensemble_RFSVMCNN_model.predict(features_array)[0],ensemble_DTCCNN_models.predict(features_array)[0]]
        return jsonify({'prediction':res.get('hg/ha yield')[0]})


    

if __name__=="__main__":
    
    app.run(debug=True, port=4000)
