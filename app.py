import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


app = Flask(__name__)

model2 = pickle.load(open('./model.pkl', 'rb'))


@app.route('/')
def world():
    return render_template('index.html')





@app.route('/', methods=['POST'])
def predictD():
    ini_feature = [int(y) for y in request.form.values()]
    final_features = [np.array(ini_feature)]
    pred = model.predict_proba(final_features)
    output = '{0:.{1}f}'.format(pred[0][1], 2)
    if output > str(0.5):
        return render_template('index.html', output='Expected Diabetic')
    else:
        return render_template('index.html', output='Not Diabetic')

if __name__ == "__main__":
    app.run(debug=True)
