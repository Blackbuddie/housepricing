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

if __name__=="__main__":
    app.run(debug=True)




    import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

model = pickle.load(open('linear_regression_model.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))   # <-- load fitted scaler

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    X = np.array(list(data.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)
    output = model.predict(X_scaled)
    return jsonify(float(output[0]))