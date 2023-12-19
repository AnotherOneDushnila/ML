from Utils.RegressionLoss import Loss_Functions
import abc
import numpy as np


# Short description for classes and methods
class LinRegDoc(abc.ABC):

    @abc.abstractmethod
    def __init__():
        """Method which basic hyperparameters are set.

        Parameters
        --------------------------------------------------
        `learning_rate : float`
         Model learning speed

        `fit_intercept : bool`
         Adding a free weight to the sample

        `Loss : class object`
         RegressionLoss class object

        `weight_list : List[float]`
         Empty list for weights

        `Returns : None`
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def fit():
        """Method which builds and optimizes the weights.

        Parameters
        ----------------------------------------------------
        `X : Any`
         Training features

        `y : Any`
         Training targets

        `Returns : final weights`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict():
        """Method that makes the prediction.

        Parameters
        -----------------------------------------------------
        `X : Any`
         Test features
        
        `Returns : prediction`
        """
        raise NotImplementedError
class RidgeRegDoc(abc.ABC):

    @abc.abstractmethod
    def __init__():
        """Method which basic hyperparameters are set.

        Parameters
        ------------------------------------------------------
        `learning_rate : float`
         Model learning speed

        `fit_intercept : bool`
         Adding a free weight to the sample

        `alpha : float`
         Regularization coefficient

        `Loss : class object`
         RegressionLoss class object

        `weight_list : List[float]`
         Empty list for weights

        `Returns : None`
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def fit():
        """Method which builds and optimizes and regularizes the weights with L2 Regularization.
        
        L2 regularization loss function: 

        1/n_samples*||y - Xw||^2_2 + alpha*||w||^2_2  --> min
       
        Parameters
        ---------------------------------------------------------
        `X : Any`
         Training features

        `y : Any`
         Training targets

        `Returns : final weights`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict():
        """Method which builds and optimizes the weights.

        Parameters
        ----------------------------------------------------------
        `X : Any`
         Test features
        
        `Returns : prediction`
        """
        raise NotImplementedError
class LassoRegDoc(abc.ABC):

    @abc.abstractmethod
    def __init__():
        """Method which basic hyperparameters are set.

        Parameters
        -----------------------------------------------------------
        `learning_rate : float`
         Model learning speed

        `fit_intercept : bool`
         Adding a free weight to the sample

        `alpha : float`
         Regularization coefficient

        `Loss : class object`
         RegressionLoss class object

        `weight_list : List[float]`
         Empty list for weights

        `Returns : None`
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def fit():
        """Method which builds and optimizes and regularizes the weights with L1 Regularization.

        L1 regularization loss function:

        1/n_samples*||y - Xw||^2_2 + alpha*|w|_1  --> min        
        
        Parameters
        -------------------------------------------------------------
        `X : Any`
         Training features

        `y : Any`
         Training targets

        `Returns : final weights`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict():
        """Method which builds and optimizes the weights.

        Parameters
        -------------------------------------------------------------
        `X : Any`
         Test features
        
        `Returns : prediction`
        """
        raise NotImplementedError
# Short description for classes and methods


class LinearRegression(LinRegDoc):
    
    def __init__(self, learning_rate = 0.1, fit_intercept = True) -> None:
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.Loss = Loss_Functions()
        self.weight_list = []
    
    def fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        self.n_samples, self.n_features = X.shape
        self.weight_list = np.zeros(self.n_features)

        for _ in range(self.n_features):
            prediction = X.dot(self.weight_list)
            grad_w = 2/len(X)*np.dot(X.T,(prediction - y))
            self.weight_list -= self.learning_rate * grad_w
        

    def predict(self, X):
        assert hasattr(self, "weight_list"), "Linear regression must be fitted first"
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        return X.dot(self.weight_list)
    
    def get_weights(self):
        return np.array(self.weight_list)
    
    

class RidgeRegression(RidgeRegDoc):
    
    def __init__(self, learning_rate = 0.1, fit_intercept = True, alpha = float) -> None:
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.Loss = Loss_Functions()
        self.weight_list = []

        
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
        assert hasattr(self, "weight_list"), "Ridge regression must be fitted first"
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        return X.dot(self.weight_list)

    def get_weights(self):
        return np.array(self.weight_list)



class LassoRegression(LassoRegDoc):

    def __init__(self, learning_rate = 0.1, fit_intercept = True, alpha = float) -> None:
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.Loss = Loss_Functions()
        self.weight_list = []

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        n_samples, n_features = X.shape
        self.weight_list = np.zeros(n_features)

        # Пока не понятно, что делать т.к общего аналитического решения не нахожу

    def predict(self, X):
        assert hasattr(self, "weight_list"), "Lasso regression must be fitted first"
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        return X.dot(self.weight_list)

    def get_weights(self):
        return np.array(self.weight_list)