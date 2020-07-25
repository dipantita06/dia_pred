from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

model = pickle.load(open('random_forest_diabetes_prediction.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl', 'rb')


@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    if request.method=="POST":
        Name=request.form['name']
        Age=int(request.form['age'])
        Pregnancies=int(request.form['pregnancies'])
        Glucose=int(request.form['glucose'])
        BloodPressure=int(request.form['bloodpresure'])
        SkinThickness=int(request.form['skinthickness'])
        Insulin=int(request.form['insulin'])
        BMI=float(request.form['BMI'])
        DiabetesPedigreeFunction =float(request.form['diabetesprdi'])
        data=[[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        data=scaler.transform(data)
        output=model.predict(data)[0]
        if output==1:
            msg='Hey {} .Sorry You Have Diabetes'.format(Name)
            return render_template("index.html",msg=msg)
        else:
            msg='Good News {} !.You dont have diabetes.'.format(Name)
            return render_template("index.html", msg=msg)


if __name__=="__main__":
    app.run(debug=True)
