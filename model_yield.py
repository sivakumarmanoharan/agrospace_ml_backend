import numpy as np 
import pandas as pd
import warnings

# ignore all warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd 
import warnings

# ignore all warnings
warnings.filterwarnings("ignore")
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer #to vectorize the text document
from sklearn.model_selection import train_test_split #to train the texting data
from sklearn.linear_model import LogisticRegression #to perform logistic regression on the data

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay  #different evaluation model for evaluation
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder




import joblib
ensemble_RFSVMCNN_model =  joblib.load('ensembler_RFSVMCNN.pkl')
ensemble_DTCCNN_models = joblib.load('ensemble_DTCCNN.pkl')


 
# A decorator used to tell the application
# which URL is associated function

def predictyield(user_input_area,user_input_crop,user_input_rain,user_input_pesticides,user_input_temp):
        res = pd.read_csv("ml_models/model_results.csv")
    # user_input_area = input("Enter the Area: ")
    # user_input_crop = input("Enter the Crop: ")
    # user_input_rain = float(input("Enter the  Average Rainfall: "))
    # user_input_pesticides = float(input("Enter the Pesticides: "))
    # user_input_temp = float(input("Enter the  Average Temperature: "))


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
        print(res.get('hg/ha yield')[0]) 

if __name__=="__main__":
     predictyield("India","Soybeans",1083,40093.69,23.51)