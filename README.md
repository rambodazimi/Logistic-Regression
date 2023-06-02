# Logistic Regression

### Cost Function
This Python code demonstrates the implementation and utilization of the cost function for logistic regression using the numpy library. Logistic regression is a widely used classification algorithm that predicts binary outcomes based on given features. The cost function measures how well the parameters fit the training set by evaluating the difference between predicted and actual values.

The code starts by defining a sample training dataset consisting of multiple features (X_train) and their corresponding target values (y_train). These features represent different characteristics or attributes of the data points, while the target values indicate the desired classification (0 or 1) for each example.

The compute_cost function takes in the weight vector (w), bias (b), input features (x), and target values (y) as parameters. It iterates over each training example, calculates the logistic regression function using the sigmoid activation function, and computes the cost using the binary cross-entropy loss formula.

Finally, the code calls the compute_cost function with predefined weight and bias values, along with the provided training dataset, to obtain the cost of the logistic regression model. The cost is then printed to the console for evaluation.

This code provides a basic understanding of how the cost function is computed for logistic regression and can serve as a starting point for more complex implementations and optimizations.
