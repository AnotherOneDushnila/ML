import numpy as np


def euclidian_distance(x1, x2) -> float:
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)

    return np.sqrt(distance)

