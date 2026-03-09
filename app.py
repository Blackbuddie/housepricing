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
@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template('home.html', prediction_text='The predicted House price is {}'.format(output))


if __name__=="__main__":
    app.run(debug=True)
