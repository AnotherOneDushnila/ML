import numpy as np

class Loss_functions:

    def log_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return abs(np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    
    
    def Entropy(self, obj : list):
        one = []
        zero = []
        for i in obj:
            if i == 1:
                one.append(i)
            else:
                zero.append(i)
        on = len(one)/len(obj)
        zer = len(zero)/len(obj)
        ent = -on * np.log(on) - zer * np.log(zer) 
        return ent
    

    def Gini(self, obj : list):
        counts = np.unique(obj, return_counts = True)
        ans = 0
        for i in counts:
            p = i/len(obj)
            ans += p * (1-p)
        return ans
    

    def Impurity_function(self, obj, one, zero) -> np.float64:
        return self.Gini(obj) - (len(one)/len(obj)) * self.Gini(one) - (len(zero)/len(obj)) * self.Gini(zero)
    

    def Accuracy_score(self, y_pred, y_true):
        return np.sum((y_pred > 0.5) == y_true) / len(y_pred)