import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        features = [
            float(request.form.get('Temperature')),
            float(request.form.get('RH')),
            float(request.form.get('Ws')),
            float(request.form.get('Rain')),
            float(request.form.get('FFMC')),
            float(request.form.get('DMC')),
            float(request.form.get('ISI')),
            float(request.form.get('Classes')),
            float(request.form.get('Region'))
        ]
        new_data_scaled = standard_scaler.transform([features])
        result = ridge_model.predict(new_data_scaled)[0]
    return render_template('home.html', result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
