import numpy as np
import abc



# Short description for class and methods
class BaseLogRegDoc(abc.ABC):

    @abc.abstractmethod
    def __init__():
        """Method which basic hyperparameters are set.

        Parameters
        --------------------------------------------------
        `learning_rate : float`
         Model learning speed

        `fit_intercept : bool`
         Adding a free weight to the sample

        `regularization : bool`
         Choose a mode of regularization:

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
# Short description for class and methods


class LogisticRegression(BaseLogRegDoc):

    def __init__(self, learning_rate : float = 0.1, alpha : float = None, regularization : bool = False, fit_intercept : bool = True) -> None:
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.weight_list = []

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        n_samples, n_features = X.shape
        self.weight_list = np.zeros(n_features)

        # gradient decent
        for _ in range(n_features):
            prediction = X.dot(self.weight_list)
            prediction = 1 / (1 + np.exp(-prediction))
            error = prediction - y 
            grad_w = 2/n_samples * X.T.dot(error)
            self.weight_list -= self.learning_rate * grad_w

        # Ridge
        if self.regularization:
            if self.alpha != 0:
                lambdaI = self.alpha * np.eye(X.shape[1])
                if self.fit_intercept:
                    lambdaI[-1, -1] = 0

                    self.weight_list = np.linalg.inv(X.T @ X + lambdaI) @ X.T @ y


    def predict(self, X):
        assert hasattr(self, "weight_list"), "Logistic regression must be fitted first"
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        prediction = X.dot(self.weight_list)
        prediction = 1 / (1 + np.exp(-prediction))
        classes = [0 if i < 0.5 else 1 for i in prediction]
        return np.array(classes)
    
    