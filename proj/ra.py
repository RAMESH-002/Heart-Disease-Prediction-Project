from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_pymongo import PyMongo
import pandas as pd

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/heardiseasedb'
mongo = PyMongo(app)
app.secret_key = 'your_secret_key'

# Function to load and preprocess data
def preprocess_data():
    # Load and preprocess data
    data = pd.read_csv('C:\\Users\\Ramesh\\OneDrive\\Desktop\\heart2.csv')

    # Define columns based on your DataFrame
    columns = list(data.columns)
    columns.remove('target')

    X = data.drop('target', axis=1)
    y = data['target']

    return X, y, columns

if __name__ == '__main__':
    # Call the preprocessing function when needed
    X, y, columns = preprocess_data()
    app.run(debug=True)
