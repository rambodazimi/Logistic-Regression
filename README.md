# Logistic Regression

### Cost Function
This Python code demonstrates the implementation and utilization of the cost function for logistic regression using the numpy library. Logistic regression is a widely used classification algorithm that predicts binary outcomes based on given features. The cost function measures how well the parameters fit the training set by evaluating the difference between predicted and actual values.

The code starts by defining a sample training dataset consisting of multiple features (X_train) and their corresponding target values (y_train). These features represent different characteristics or attributes of the data points, while the target values indicate the desired classification (0 or 1) for each example.

The compute_cost function takes in the weight vector (w), bias (b), input features (x), and target values (y) as parameters. It iterates over each training example, calculates the logistic regression function using the sigmoid activation function, and computes the cost using the binary cross-entropy loss formula.

Finally, the code calls the compute_cost function with predefined weight and bias values, along with the provided training dataset, to obtain the cost of the logistic regression model. The cost is then printed to the console for evaluation.

This code provides a basic understanding of how the cost function is computed for logistic regression and can serve as a starting point for more complex implementations and optimizations.

### Gradient Descent
This Python code leverages the NumPy library to implement and utilize gradient descent for logistic regression. The purpose of the code is to compute the gradient descent on a dataset consisting of multiple features and corresponding target values (outputs).

The code begins by defining two essential methods: "compute_gradient" and "gradient_descent." The "compute_gradient" method calculates the gradient for logistic regression using the given input features (x), target values (y), weight coefficients (w), and bias term (b). It iterates through the dataset, computing the error between the predicted output (obtained using the sigmoid function) and the actual target value. The gradients with respect to each feature and the bias term are accumulated and divided by the total number of samples (m) to obtain the average gradients.

The "gradient_descent" method performs batch gradient descent, updating the weight coefficients and bias term iteratively. It takes in the input features (x), target values (y), initial weights (wi), initial bias term (bi), learning rate (alpha), and the number of iterations. Inside the loop, it calls the "compute_gradient" method to obtain the gradients for the current weight coefficients and bias term. The gradients are then used to update the weights and bias term by subtracting the product of the learning rate and gradients. This process is repeated for the specified number of iterations.

The code also includes a "sigmoid" function that applies the sigmoid activation function to the given input. It ensures that the input is within a safe range to avoid overflow.

To demonstrate the functionality of the code, a sample training dataset is provided. It consists of six training examples, each with two features. The target values are binary (0 or 1) for binary classification. The initial weights and bias term are set to zeros. The learning rate is defined as 0.1, and the number of iterations is set to 10,000.

Finally, the code executes the "gradient_descent" method using the sample training dataset and displays the learned weights (w) and bias term (b). Additionally, it prints the equation of the decision boundary in the form of f(x) = w[0]x1 + w[1]x2 + b, where x1 and x2 represent the features.

Overall, this Python code provides a concise implementation of gradient descent for logistic regression, allowing for efficient training and decision boundary calculation on datasets with multiple features and binary target values.
