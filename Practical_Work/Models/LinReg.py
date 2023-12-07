from Utils.RegressionLoss import Loss_Functions
import numpy as np



class LinearRegression:
    
    def __init__(self, learning_rate = 0.1) -> None:
        self.learning_rate = learning_rate
        self.Loss = Loss_Functions()
        self.weight_list = []
        self.score_list = []
    
    def fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        n_samples, n_features = X.shape
        self.weight_list = np.zeros(n_features)

        for _ in range(n_features):
            prediction = X.dot(self.weight_list)
            mse = self.Loss.MSE(prediction, y)
            self.score_list.append(mse)
            grad_w = 2/len(X)*np.dot(X.T,((prediction - y)))
            self.weight_list -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        return X.dot(self.weight_list)
    
    def get_weights(self):
        return self.weight_list
    
class RidgeRegression(LinearRegression):
    
    def __init__(self, learning_rate = 0.01, fit_intercept = True, alpha = float) -> None:
        super().__init__(learning_rate)
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        
    def fit(self, X, y):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        n_samples, n_features = X.shape
        self.weight_list = np.zeros(n_features)

        
        lambdaI = self.alpha * np.eye(X.shape[1])
        if self.fit_intercept:
            lambdaI[-1, -1] = 0

        self.weight_list = np.linalg.inv(X.T @ X + lambdaI) @ X.T @ y

    
    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        return X.dot(self.weight_list)

    def get_weights(self):
        return self.weight_list

class LassoRegression(RidgeRegression):

    def __init__(self, learning_rate=0.01, fit_intercept=True, alpha=float) -> None:
        super().__init__(learning_rate, fit_intercept, alpha)

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        n_samples, n_features = X.shape
        self.weight_list = np.zeros(n_features)

        # Пока не понятно, что делать т.к общего аналитического решения не нахожу

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        return X.dot(self.weight_list)

    def get_weights(self):
        return self.weight_list