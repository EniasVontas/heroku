# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 13:06:33 2021

@author: Enias
"""

import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__) #Initialize the flask App
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods = ["GET", "POST"])
def home():
    if(request.method == "GET"):
        data = "hello world"
        return jsonify({'data':data})

@app.route('/predict')
def predict():
    model = pickle.load(open('model.pkl', 'rb'))
    age = request.args.get('age')
    bmi = request.args.get('bmi')
    children = request.args.get('children')
    sex = request.args.get('sex_male')
    smoker = request.args.get('smoker_yes')
    
    test_df = pd.DataFrame({'Age':[age],'BMI':[bmi],"No of Children":[children],"Sex":[sex],"Smoker":[smoker]})
    pred_price = model.predict(test_df)

    return jsonify({'Insurance':str(pred_price)})

if __name__ == "__main__":
    app.run(debug=True)