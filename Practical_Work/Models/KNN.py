from Utils.misc import euclidian_distance
import abc
import numpy as np


# Short description for class and methods
# ᕕ(╭ರ╭ ͟ʖ╮•́)⊃¤=(————
class BaseKNNClassifierDoc(abc.ABC):

    @abc.abstractmethod
    def __init__():
        """Method which basic hyperparameters are set.

        Parameters
        --------------------------------------------------
        `k : int`
         Number of neighbors"""
    
        raise NotImplementedError

    @abc.abstractmethod
    def fit():
        """Method that remembers the data.

        Parameters
        ----------------------------------------------------
        `X : Any`
         Training features

        `y : Any`
         Training targets

        `Returns: final weights`
        """

        raise NotImplementedError
    
    @abc.abstractmethod
    def predict():
        """Method that makes the prediction.

        Parameters
        ----------------------------------------------------
        `X : Any`
         Test objects

        `Returns: prediction`
        """
        raise NotImplementedError
class BaseKNNRegressorDoc(abc.ABC):

    @abc.abstractmethod
    def __init__():
        """Method which basic hyperparameters are set.

        Parameters
        --------------------------------------------------
        `k : int`
         Number of neighbors"""
    
        raise NotImplementedError

    @abc.abstractmethod
    def fit():
        """Method that remembers the data.

        Parameters
        ----------------------------------------------------
        `X : Any`
         Training features

        `y : Any`
         Training targets

        `Returns: final weights`
        """

        raise NotImplementedError
    
    @abc.abstractmethod
    def predict():
        """Method that makes the prediction.

        Parameters
        ----------------------------------------------------
        `X : Any`
         Test objects

        `Returns: prediction`
        """
        raise NotImplementedError
# ᕕ(╭ರ╭ ͟ʖ╮•́)⊃¤=(————
# Short description for class and methods


class KNN_Classifier(BaseKNNClassifierDoc):

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def __most_common(self, y):
        """Method that finds the most common class.

        Parameters
        ----------------------------------------------------
        `y : Any`
         Training targets

        `Returns: most common class`
        """
        labels = np.unique(y)
        count = [list(y).count(i) for i in labels]
        return labels[np.argmax(count)]
    
    def __find_labels(self, x):
        """Method that finds k nearest objects.

        Parameters
        ----------------------------------------------------
        `x : Any`
         Training targets

        `Returns : most common object`
        """
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        k_nearest = np.argsort(distances)[:self.k] 
        labels = [self.y_train[i] for i in k_nearest]
        return self.__most_common(labels)

    def predict(self, X_test):
        labels = [self.__find_labels(x) for x in X_test]
        return np.array(labels)


class KNN_Regressor(BaseKNNRegressorDoc):

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def __most_common(self, y):
        """Method that finds the most common object.

        Parameters
        ----------------------------------------------------
        `y : Any`
         Training targets

        `Returns: most common object`
        """
        return np.sum(y)/len(y)
    
    def __find_labels(self, x):
        """Method that finds k nearest objects.

        Parameters
        ----------------------------------------------------
        `x : Any`
         Training targets

        `Returns : most common object`
        """
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        k_nearest = np.argsort(distances)[:self.k] 
        labels = [self.y_train[i] for i in k_nearest]
        return self.__most_common(labels)
    
    def predict(self, X_test):
        labels = [self.__find_labels(x) for x in X_test]
        return np.array(labels)
