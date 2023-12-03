import numpy as np
from typing import List


class L1_Regularization:
    
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def __call__(self, weights : List[float]) -> float:
        self.weights = weights

        return sum(abs(self.weights))
    
class L2_Regularization:

    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def __call__(self, weights : List[float]) -> float:
        self.weights = weights

        return np.sqrt(sum(np.square(self.weights)))
