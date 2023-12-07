import numpy as np

class Loss_Functions:

    def MSE(self, y_pred, y) -> float:
        return np.square(y_pred - y).mean()
    
    def RMSE(self, y_pred, y) -> float:
        return np.sqrt(np.square(y_pred - y).mean())
    
    def MAE(self, y_pred, y) -> float:
       return abs(y_pred - y).mean()

    def R2(self, y_pred, y) -> float:
        return 1 - np.sum((y_pred - y)**2) / np.sum((y.mean() - y)**2)
    
    def Huber_Loss(self, delta, y_pred, y) -> float:
        huber_mse = 0.5*(y - y_pred)**2
        huber_mae = delta * (np.abs(y - y_pred) - 0.5 * delta)
        huber_loss = (np.where(np.abs(y - y_pred) <= delta, huber_mse, huber_mae)).mean()
        return huber_loss