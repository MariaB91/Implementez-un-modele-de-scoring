# Load librairies
import pandas as pd
import sklearn
import joblib
from flask import Flask, jsonify, request
import json
import numpy as np
import lightgbm


# Load the data
#--------------
# processed data for applying the scoring model
data_processed = pd.read_csv("X_sample.csv", index_col='SK_ID_CURR', encoding ='utf-8')
# original data for displaying personal data
data_original = pd.read_csv("merged_data.csv", index_col='SK_ID_CURR')
# features description
features_desc = pd.read_csv("features_descriptions.csv", index_col=0, encoding= 'unicode_escape')

data_processed.drop('TARGET', axis = 1, inplace = True)



###############################################################
app = Flask(__name__)

def load_model():   
    '''loading the trained model'''
    pickle_in = open("lgbmclassifier.joblib", 'rb') 
    model = joblib.load(pickle_in)
    return model

model = load_model()

@app.route("/")
def loaded():
    return "API, models and data loaded…"

@app.route('/api/sk_ids/')
# Test : http://127.0.0.1:5000/api/sk_ids/
#serv : https://api-3nes.onrender.com/api/sk_ids/?sk_ids=245991
def sk_ids():
    # Extract list of 'SK_ID_CURR' from the DataFrame
    sk_ids = list(data_original.index)[:50]

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': sk_ids
     })


@app.route('/api/prediction/')
# Test : http://127.0.0.1:5000/api/prediction?SK_ID_CURR=245991
# serv : https://api-3nes.onrender.com/api/prediction?SK_ID_CURR=245991
def load_prediction():
        SK_ID_CURR = int(request.args.get('SK_ID_CURR'))
        applicant_data = data_processed.iloc[SK_ID_CURR:SK_ID_CURR]
        applicant_score = model.predict(applicant_data)[0][1]
        
        return applicant_score



@app.route('/api/personal_data/')
# Test : http://127.0.0.1:5000/api/personal_data?SK_ID_CURR=245991
# serv : https://api-3nes.onrender.com/api/prediction?SK_ID_CURR=245991
def personal_data():
    # Parsing the http request to get arguments (applicant ID)
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))

    # Getting the personal data for the applicant (pd.Series)
    personal_data = data_original.loc[SK_ID_CURR, :]

    # Converting the pd.Series to JSON
    personal_data_json = json.loads(personal_data.to_json())
    
    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': personal_data_json
     })


@app.route('/api/features_desc/')
# Test : http://127.0.0.1:5000/api/features_desc
# serv : https://api-3nes.onrender.com/api/features_desc
def send_features_descriptions():

    # Converting the pd.Series to JSON
    features_desc_json = json.loads(features_desc.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': features_desc_json
     })






#################################################
if __name__ == "__main__":
    app.run(debug=False)
