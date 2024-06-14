import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:\\Users\\Ramesh\\Desktop\\heart2.csv')

# Split the data
X = data.drop('target', axis=1)
y = data['target']

# Scale the data since logistic regression benefits from feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)

# Output the results
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

def predict_user_input(details):
    details_scaled = scaler.transform([details])
    prediction = clf_lr.predict(details_scaled)
    return prediction[0]

# Collect user input
details = [
    float(input("Enter age: ")),
    float(input("Enter gender (e.g., 0 for female, 1 for male): ")),
    float(input("Enter cp: ")),
    float(input("Enter trestbps: ")),
    float(input("Enter chol: ")),
    float(input("Enter fbs: ")),
    float(input("Enter restecg: ")),
    float(input("Enter thalach: ")),
    float(input("Enter exang: ")),
    float(input("Enter oldpeak: ")),
    float(input("Enter slope: ")),
    float(input("Enter ca: ")),
    float(input("Enter thal: "))
]

prediction = predict_user_input(details)
print("Prediction based on the provided details:", prediction)
