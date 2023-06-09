import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
rfrmodel = pickle.load(open('RFReg.pkl', 'rb'))
scaler = pickle.load(open("scaled.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = rfrmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input = scaler.transform(np.array(data).reshape(1,-1))
    print(input)
    output = rfrmodel.predict(input)
    return render_template("home.html", prediction_text="The predicted house rent is Rs.{} per month".format(int(output[0])))

if __name__ =="__main__":
    app.run(debug=True)