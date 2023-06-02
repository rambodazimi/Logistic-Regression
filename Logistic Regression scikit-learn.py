"""
Author: Rambod Azimi
This Python code will make use of numpy and scikit-learn libraries to implement and utilize the Logistic Regression
machine learning algorithm.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

# defining our training set examples
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]) # 6 training examples with 2 features
y_train = np.array([0, 0, 0, 1, 1, 1]) # targets

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train) # fit the model on the training data by caliing fit function

prediction = logistic_regression_model.predict(X_train)

for i in range(X_train.shape[0]):
    print(f"Prediction value: {prediction[i]} \t Actual value: {y_train[i]}")

print(f"Accuracy of the model: {logistic_regression_model.score(X_train, y_train)* 100}%")