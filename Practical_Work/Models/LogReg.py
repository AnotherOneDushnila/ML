import numpy as np
from Utils.ClassificationLoss import Loss_functions

class LogisticRegression:

    def __init__(self, learning_rate = 0.1, alpha = float, mode = int,fit_intercept = True) -> None:
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.mode = mode
        self.fit_intercept = fit_intercept
        self.Loss = Loss_functions()
        self.weight_list = []

    def fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        n_samples, n_features = X.shape
        self.weight_list = np.zeros(n_features)

        # gradient decent
        for _ in range(n_features):
            prediction = X.dot(self.weight_list)
            prediction = 1 / (1+np.exp(-prediction))
            error = prediction - y 
            grad_w = 2/n_samples * X.T.dot(error)  
            self.weight_list -= self.learning_rate * grad_w

        # Lasso
        if self.mode == 1:
            pass

        # Пока не понятно, что делать т.к общего аналитического решения не нахожу

        # Ridge
        elif self.mode == 2:
            if self.alpha != 0:
                lambdaI = self.alpha * np.eye(X.shape[1])
                if self.fit_intercept:
                    lambdaI[-1, -1] = 0

                    self.weight_list = np.linalg.inv(X.T @ X + lambdaI) @ X.T @ y


    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        prediction = X.dot(self.weight_list)
        prediction = 1 / (1 + np.exp(-prediction))
        classes = [0 if i < 0.5 else 1 for i in prediction]
        return np.array(classes)
    
    