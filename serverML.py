#Python Libraries 
from flask import Flask, request, jsonify, render_template
import numpy as np
from load import joblib
import os 
from werkzeug.utils import secure_filename

#Load model
dt = joblib.load('dt1_ml.joblib')
#Create Flask App
server = Flask(__name__)
#Define route
@server.route('/predictjson', methods=['POST'])
def predictjson():
    data= request.json
    print(data)
    inputData= np.array([
        data["pH"],
        data["sulphates"],
        data["alcohol"]
    ])
    result= dt.predict(inputData.reshape(1,-1))
    return jsonify({"Prediction": str(result[0])})
if __name__ == "__main__":
    server.run(debug= False , host ="0.0.0.0",port=8080)
