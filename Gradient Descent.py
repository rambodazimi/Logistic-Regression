"""
Author: Rambod Azimi
This Python code will make use of numpy library to implement and utilize the gradient descent for logistic regression
It gets a data set with multiple features as well as the target values (outputs)
Then, it computes the gradient descent
"""

import numpy as np
import copy
import math

# this method computes the gradient for logistic regression
def compute_gradient(x, y, w, b):

    m = x.shape[0]
    n = x.shape[1]
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    e = math.e
    for i in range(m):
        f = sigmoid(np.dot(w, x[i]) + b)
        error = f - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i,j]
        dj_db += error
    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

# this method performs batch gradient descent
def gradient_descent(x, y, wi, bi, alpha, iterations):

    w = copy.deepcopy(wi)
    b = bi
    
    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b

def sigmoid(z):
    z = np.clip( z, -500, 500 ) # protect against overflow
    g = 1.0 / (1.0 + np.exp(-z))
    return g

# Now, let's run the algorithm

# sample training examples (features and targets)
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]) # 6 training examples with 2 features
y_train = np.array([0, 0, 0, 1, 1, 1]) # binary classification

w_initial = np.zeros_like(X_train[0])
b_initial = 0.0
alpha = 0.1
iterations = 10000

w, b = gradient_descent(X_train, y_train, w_initial, b_initial, alpha, iterations)

print(f"w = {w}, b = {b}")
print(f"Boundary: f(x) = {w[0]}x1 + {w[1]}x2 + {b}")
