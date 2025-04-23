import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load

# reading all the datasets from csv files
df = pd.read_csv('./datasets/dataset.csv')
df1 = pd.read_csv('./datasets/Symptom-severity.csv')
description = pd.read_csv('./datasets/symptom_Description.csv')
precaution = pd.read_csv('./datasets/symptom_precaution.csv')

# pre-processing datas
df.isna().sum()
df.isnull().sum()

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)

df = df.fillna(0)
df.head()

vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)

d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination',0)
df = d.replace('foul_smell_of urine',0)
df.head()

(df[cols] == 0).all()

df['Disease'].value_counts()

df['Disease'].unique()

data = df.iloc[:,1:].values
labels = df['Disease'].values


# loading the trained (pickled) model
# model = pickle.load(open('./model/model.sav', 'rb'))

# running flask app
app = Flask(__name__)
CORS(app)

rf_model = load('./model/random_forest.joblib')

def predict_random_forest(symptoms_list):
    psymptoms = symptoms_list.copy()
    
    # Convert symptoms to weights
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j] == a[k]:
                psymptoms[j] = b[k]
    
    # Pad to 17 symptoms
    psymptoms += [0] * (17 - len(psymptoms))
    input_vector = [psymptoms]
    
    # Make prediction
    pred = rf_model.predict(input_vector)
    disease = pred[0]

    # Fetch description
    disp = description[description['Disease'] == disease]['Description'].item()
    
    # Fetch all precautions
    row = precaution[precaution['Disease'] == disease]
    precaution_list = []
    for i in range(1, len(row.columns)):
        val = row.iloc[0, i]
        if pd.notna(val):
            precaution_list.append(val)

    return {
        "disease": disease,
        "description": disp,
        "precautions": precaution_list
    }


# function to predict disease based on symptoms
def SVM(symptoms):
    
    sym = np.array(df1["Symptom"])
    wei = np.array(df1["weight"])

    for j in range(len(symptoms)):
        for k in range(len(sym)):
            if symptoms[j]==sym[k]:
                symptoms[j]=wei[k]

    total_length = 17
    zeros_required = total_length - len(symptoms)
    
    nulls_required = [0] * zeros_required

    total_symptoms = [symptoms + nulls_required]

    pred2 = model.predict(total_symptoms)

    return list(pred2)


@app.route('/', methods = ['POST'])
def index():

    print(request)
    # read the request body for symptoms and location
    symptoms = request.json['symptoms']
    # location = request.json['location']

    # # predict the disease
    # res = SVM(symptoms)
    # res = res[0]

    # # get the description and precautions for the predicted disease
    # des = description[description.Disease == res]['Description'].item()
    # prec = precaution[precaution.Disease == res]['Precaution_1'].item()
    # des = str(des)
    # prec = str(prec)
    
    # # return the result
    # return jsonify(
    #     disease=res,
    #     description=des,
    #     precaution=prec
    # ), 200

    result = predict_random_forest(symptoms)
    return jsonify(result), 200

@app.route('/disease')
def disease():

    # all the possible symptoms
    dis =  ["fatigue", "yellowish_skin", "loss_of_appetite", "yellowing_of_eyes", 'family_history',"stomach_pain", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"]
    return jsonify(
        response=dis
    ), 200

@app.route('/location')
def location():

    # all the possible locations
    loc =  ["New Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]
    return jsonify(
        response=loc
    ), 200


if __name__ == '__main__':
    app.debug = True
    app.run()