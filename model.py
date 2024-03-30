
import sklearn as sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def training_prod_data(input_values):
    # Loading the dataset
    crop_data=pd.read_csv("ml_models/Crop_recommendation.csv")
    columns=list(crop_data.columns)
    crop_data_mod=crop_data[columns]
    categoricals = []
    for col, col_type in crop_data_mod.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            crop_data_mod.loc[:, col] = crop_data_mod[col].fillna(0)
    crop_data_ohe=pd.get_dummies(crop_data_mod, columns=categoricals,dummy_na=True)
    cols_predict=list(crop_data_ohe.columns)
    x=crop_data_ohe[cols_predict[0:7]]
    y=crop_data_ohe[cols_predict[7:len(cols_predict)]]
    SEED=42
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=SEED)
    rfc = RandomForestClassifier(n_estimators=900, 
                             max_depth=7,
                             random_state=SEED)
    rfc.fit(X_train.values, y_train)
    prediction=rfc.predict(input_values)
    value=list(prediction[0])
    value=value.index(True)
    predicted_crop=list(crop_data_ohe)[value].lstrip("label_")
    joblib.dump(rfc,'model.pkl')
    print("Model Dumped")
    rfc=joblib.load("model.pkl")
    model_columns=list(y.columns)
    joblib.dump(model_columns,"model_columns.pkl")
    print("Model Columns dumped")
    print(list(crop_data_ohe))
    

if __name__=="__main__":
    training_prod_data([[101,17,45,30.6,65.5,7,30.0]])



