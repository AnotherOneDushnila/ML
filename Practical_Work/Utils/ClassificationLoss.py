import numpy as np

class Loss_functions:

    def log_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=0) / len(y_true)
    
    
    def Entropy(self, y):
        unique_labels = np.unique(y)
        entropy = 0
        for label in unique_labels:
            count = len(y[y == label])
            p = count / len(y)
            entropy += -p * np.log2(p)
        return entropy
    

    def Gini(self, y):
        counts = np.unique(y, return_counts = True)
        ans = 0
        for i in counts:
            p = i/len(y)
            ans += p * (1-p)
        return ans
    

    def Impurity_function(self, y) -> np.float64:
        one = []
        zero = []
        for i in y:
            if i == 1:
                one.append(i)
            else:
                zero.append(i)
        return self.Gini(y) - (len(one)/len(y)) * self.Gini(one) - (len(zero)/len(y)) * self.Gini(zero)
    

    def Accuracy_score(self, y_pred, y_true):
        return np.sum((y_pred > 0.5) == y_true) / len(y_pred)