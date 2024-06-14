import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from imblearn.combine import SMOTEENN
# Load the heart disease dataset
heart_disease_data = pd.read_csv('C:\\Users\\Ramesh\\Desktop\\heart2.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(heart_disease_data.drop('target', axis=1),
                                                    heart_disease_data['target'],
                                                    test_size=0.25,
                                                    random_state=42)

# Create a pipeline for feature engineering and classification
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('clf', RandomForestClassifier(n_estimators=1000, max_depth=10))
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test data
score = pipeline.score(X_test, y_test)
print('Accuracy:', score)

# Make predictions on new data
new_user_features = [58, 0, 1, 1, 120, 0, 0, 0, 0, 0, 1, 0, 1]
prediction = pipeline.predict_proba(np.array(new_user_features).reshape(1, -1))
print('Probability of heart disease:', prediction[0][1])

# Perform cross-validation to estimate the generalization performance of the model
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print('Cross-validation scores:', cv_scores)
print('Mean cross-validation score:', np.mean(cv_scores))
