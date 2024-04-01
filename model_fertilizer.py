import pickle
import sklearn as sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import joblib

def training_fert_data(input_values):
    # Loading the dataset
    fertilizer_data=pd.read_csv("ml_models/Fertilizer_Prediction.csv")
    encode_soil = LabelEncoder()
    fertilizer_data.Soil_Type = encode_soil.fit_transform(fertilizer_data.Soil_Type)
    Soil_Type = pd.DataFrame(zip(encode_soil.classes_,encode_soil.transform(encode_soil.classes_)),columns=['Original','Encoded'])
    Soil_Type = Soil_Type.set_index('Original')
    encode_crop = LabelEncoder()
    fertilizer_data.Crop_Type = encode_crop.fit_transform(fertilizer_data.Crop_Type)
    Crop_Type = pd.DataFrame(zip(encode_crop.classes_,encode_crop.transform(encode_crop.classes_)),columns=['Original','Encoded'])
    Crop_Type = Crop_Type.set_index('Original')
    encode_ferti = LabelEncoder()
    fertilizer_data.Fertilizer = encode_ferti.fit_transform(fertilizer_data.Fertilizer_Name)
    Fertilizer = pd.DataFrame(zip(encode_ferti.classes_,encode_ferti.transform(encode_ferti.classes_)),columns=['Original','Encoded'])
    Fertilizer = Fertilizer.set_index('Original')
    x_train, x_test, y_train, y_test = train_test_split(fertilizer_data.drop('Fertilizer_Name',axis=1),fertilizer_data.Fertilizer,test_size=0.2,random_state=1)
    params = {
    'n_estimators':[350,400,450],
    'max_depth':[2,3,4,5,6,7],
    'min_samples_split':[2,5,8]
    }
    rand=RandomForestClassifier()
    grid_rand = GridSearchCV(rand,params,cv=3,verbose=3,n_jobs=-1)
    grid_rand.fit(fertilizer_data.drop('Fertilizer_Name',axis=1),fertilizer_data.Fertilizer)
    pickle_out_class = open('classifier.pkl','wb')
    pickle.dump(grid_rand,pickle_out_class)
    pickle_out_class.close()
    model = pickle.load(open('classifier.pkl','rb'))
    pickle_out_fert = open('fertilizer.pkl','wb')
    pickle.dump(encode_ferti,pickle_out_fert)
    pickle_out_fert.close()
    ferti = pickle.load(open('fertilizer.pkl','rb'))
    predict=ferti.classes_[model.predict(input_values)]
    print(predict)
    print(Crop_Type)
    print(Soil_Type)
    print(Fertilizer)
    
if __name__=="__main__":
    training_fert_data([[94,70,48,25.1,84.8,6.2,91.4,60]])
