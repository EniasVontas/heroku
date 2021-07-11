# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 13:06:33 2021

@author: Enias
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__) #Initialize the flask App
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    model = pickle.load(open('model.pkl', 'rb'))
    age = request.args.get('age')
    bmi = request.args.get('bmi')
    children = request.args.get('children')
    sex = request.args.get('sex_male')
    smoker = request.args.get('smoker_yes')
    
    test_df = pd.DataFrame({'Age':[age],'BMI':[bmi],"No of Children":[children],"Sex":[sex],"Smoker":[smoker]})
    pred_price = int(model.predict(test_df))

    return jsonify(pred_price)

if __name__ == "__main__":
    app.run(debug=True)
