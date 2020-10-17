from flask import Flask, render_template, flash, request
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib, numpy as np

def char_tokenizer(input):
    characters = []
    for i in input:
        characters.append(i)
    return characters
app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/main/', methods=['GET', 'POST'])
def mainpage():
    if request.method == "POST":
        enteredPassword =str(request.form['password'])

        # Load the algorithm models
        DecisionTree_model = joblib.load('DecisionTree_model.joblib')
        LogisticRegression_model = joblib.load('LogisticRegression_model.joblib')
        NaiveBayes_model = joblib.load('NaiveBayes_model.joblib')
        RandomForest_model = joblib.load('RandomForest_model.joblib')
        
        Password = [[enteredPassword]]
        # Predict the strength
        DecisionTree_Test = DecisionTree_model.predict(Password)
        LogisticRegression_Test = LogisticRegression_model.predict(Password)
        NaiveBayes_Test = NaiveBayes_model.predict(Password)
        RandomForest_Test = RandomForest_model.predict(Password)
        

        return render_template("main.html", DecisionTree=DecisionTree_Test[0],
                                            LogReg=LogisticRegression_Test[0], 
                                            NaiveBayes=NaiveBayes_Test[0],
                                            RandomForest=RandomForest_Test[0],
                                            )
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)