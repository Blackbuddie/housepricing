import pickle
from flask import Flask, request,app, jsonify, render_template, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
app = Flask(__name__)#
#loading the model
model=pickle.load(open('linear_regression_model.pkl','rb'))
# load scaler
scaler = pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data =request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data =scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])
@app.route('/predict', methods=['POST'])
def predict():

    crim = float(request.form['crim'])
    zn = float(request.form['zn'])
    indus = float(request.form['indus'])
    chas = float(request.form['chas'])
    nox = float(request.form['nox'])
    rm = float(request.form['rm'])
    age = float(request.form['age'])
    dis = float(request.form['dis'])
    rad = float(request.form['rad'])
    tax = float(request.form['tax'])
    ptratio = float(request.form['ptratio'])
    b = float(request.form['b'])
    lstat = float(request.form['lstat'])

    data = [[
        crim, zn, indus, chas, nox,
        rm, age, dis, rad, tax,
        ptratio, b, lstat
    ]]

    final_input = scaler.transform(data)

    prediction = model.predict(final_input)[0]

    return render_template(
        'home.html',
        prediction_text=f"Estimated House Price: ${prediction*1000:,.0f}"
    )

if __name__=="__main__":
    app.run(debug=True)
