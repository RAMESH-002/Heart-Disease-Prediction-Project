from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('C:\\Users\\Ramesh\\Desktop\\heart2.csv')
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier().fit(X_train, y_train)
# After training your classifier
y_pred = clf.predict(X_test)  # This line was missing in your code
print("Accuracy:", accuracy_score(y_test, y_pred))


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    
    if request.method == 'POST':
        details = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        prediction = clf.predict([details])[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
