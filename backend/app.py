import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# running flask app
app = Flask(__name__)
CORS(app)

description = pd.read_csv('./datasets/symptom_Description.csv')
precaution = pd.read_csv('./datasets/symptom_precaution.csv')

df = pd.read_csv('./datasets/dataset.csv')
df1 = pd.read_csv('./datasets/Symptom-severity.csv')
for col in df.columns:
    df[col] = df[col].str.replace('_',' ')

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)
df = df.fillna(0)

df1['Symptom'] = df1['Symptom'].str.replace('_',' ')

vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)
d = d.replace('dischromic  patches', 0)
d = d.replace('spotting  urination',0)
df = d.replace('foul smell of urine',0)

data = df.iloc[:,1:].values
labels = df['Disease'].values

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

def predict_top_diseases_rf(symptoms_list, top_k=3):
    psymptoms = symptoms_list.copy()

    # Convert symptom names to weights
    symptom_names = np.array(df1["Symptom"])
    symptom_weights = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(symptom_names)):
            if psymptoms[j] == symptom_names[k]:
                psymptoms[j] = symptom_weights[k]

    # Pad to 17 symptoms
    psymptoms += [0] * (17 - len(psymptoms))
    input_vector = [psymptoms]

    # Get probability predictions
    proba = rf_model.predict_proba(input_vector)[0]
    top_indices = np.argsort(proba)[::-1][:top_k]

    top_diseases = []
    for idx in top_indices:
        disease_name = rf_model.classes_[idx]
        confidence = round(proba[idx] * 100, 2)

        entry = {
            "disease": disease_name,
            "confidence": confidence
        }

        try:
            desc = description[description['Disease'] == disease_name]['Description'].item()
        except:
            desc = "No description available."

        row = precaution[precaution['Disease'] == disease_name]
        precaution_list = []
        if not row.empty:
            for i in range(1, len(row.columns)):
                val = row.iloc[0, i]
                if pd.notna(val):
                    precaution_list.append(val)

        entry["description"] = desc
        entry["precautions"] = precaution_list

        top_diseases.append(entry)

    return top_diseases


@app.route('/', methods = ['POST'])
def index():

    # print(request)
    # read the request body for symptoms and location
    symptoms = request.json['symptoms']
    result = predict_top_diseases_rf(symptoms)
    print(result)
    return jsonify(result), 200

@app.route('/disease')
def disease():

    # all the possible symptoms
    dis =  ["fatigue", "vomiting", "high fever", "loss of appetite", "nausea", "headache", "abdominal pain", "yellowish skin", "yellowing of eyes", "chills", "skin rash", "malaise", "chest pain", "joint pain", "itching", "sweating", "dark urine", "cough", "diarrhoea", "irritability", "muscle pain", "excessive hunger", "weight loss", "lethargy", "breathlessness", "phlegm", "mild fever", "swelled lymph nodes", "loss of balance", "blurred and distorted vision"]
    print(dis)
    return jsonify(
        response=dis
    ), 200


if __name__ == '__main__':
    app.debug = True
    app.run()