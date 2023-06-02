"""
Author: Rambod Azimi
This Python code will make use of numpy library to implement and utilize the cost function for logistic regression
It gets a data set with multiple features as well as the target values (outputs)
Then, it computes the cost function, which is a measure of how well the parameters fit the training set, using Loss function
"""

import numpy as np
import math

# sample training examples (features and targets)
X_train = np.array([[0.5, 1.5], [2, 2], [0.76, 2], [0, 0.02], [4, 0.75], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]) # 6 training examples with 2 features
y_train = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1]) # binary classification

# compute the cost of a logistic regression model using loss function
def compute_cost(w, b, x, y):
    sum = 0.0 # store the result in this variable
    for i in range(x.shape[0]):
        z = np.dot(w, x[i]) + b
        f = 1 / (1 + math.exp(-z)) # sigmoid function
        sum += -y[i]*np.log(f) - (1-y[i])*np.log(1-f)
    sum /= x.shape[0]
    return sum


cost = compute_cost(np.array([1, 1]), -3, X_train, y_train)
print(f"The cost function is {cost:0.3f}")