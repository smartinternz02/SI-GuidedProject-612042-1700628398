from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('floods.save')
sc = load('transform.save')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/flood_predict', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/data_predict', methods=['GET','POST'])
def predict():
    try:
        input_data = [float(request.form['cloudCover']),
                      float(request.form['annualRainfall']),
                      float(request.form['janFebRainfall']),
                      float(request.form['marchMayRainfall']),
                      float(request.form['juneSeptRainfall'])]

        print("Received input data:")
        for key, value in request.form.items():
            print(f"{key}: {value}")

        data = [input_data]
        print("Data for prediction:")
        print(data)

        transformed_data = sc.transform(data)
        prediction = model.predict(transformed_data)
        output = prediction[0]

        if (output == 1).all():
         return render_template('nochance.html', prediction='No Possibility of Severe Flood')
        else:
          return render_template('chance.html', prediction='Possibility of Severe Flood')

    except (KeyError, ValueError) as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return render_template('error.html', error_message=error_message)

        

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)