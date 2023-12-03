from Utils.Loss import Loss_Functions
from Utils.Regularization import L1_Regularization, L2_Regularization
import numpy as np

class LinearRegression():
    
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
    
class RegularizedLinearRegression(LinearRegression):
    
    def __init__(self, learning_rate = 0.1, mode = int, alpha = float) -> None:
        super().__init__(learning_rate)
        self.mode = mode
        self.alpha = alpha

        
    def fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        self.n_samples, self.n_features = X.shape
        self.weight_list = np.zeros(self.n_features)

        if self.mode == 1:
            self.regularization = L1_Regularization(alpha=self.alpha)
        elif self.mode == 2:
            self.regularization = L2_Regularization(alpha=self.alpha)

        for _ in range(self.n_samples):
            prediction = X.dot(self.weight_list)
            mse = self.Loss.MSE(prediction, y) + self.regularization(self.weight_list)
            self.score_list.append(mse)
            grad_w = 2/len(X)*np.dot(X.T,((prediction - y)))
            self.weight_list -= self.learning_rate * grad_w
    
    
    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        return X.dot(self.weight_list)
    
    def get_weights(self):
        return self.weight_list
    
