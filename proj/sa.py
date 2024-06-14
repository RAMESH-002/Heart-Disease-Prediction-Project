import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Let's say you've saved your data to a CSV named 'heart_disease_dataset.csv'
data = pd.read_csv('C:\\Users\\Ramesh\\Desktop\\heart2.csv')
# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Train a decision tree
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save the trained model for use in your Flask app
joblib.dump(clf, 'heart_disease_model.pkl')
