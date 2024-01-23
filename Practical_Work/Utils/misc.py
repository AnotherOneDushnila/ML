import numpy as np
from typing import List, Tuple, Any, Iterable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm


class Plot:
    
    def correlation_heatmap(
            self, data : pd.DataFrame, 
            sides : Tuple[int], px : float, annot_size : int,
            only_numeric : bool
            ) -> sns.heatmap:



        if only_numeric:

            numeric_col = list(data.select_dtypes(include=['int64']).columns)+list(data.select_dtypes(include=['float64']).columns)

            corr_matrix = data.loc[:,numeric_col].corr()
            plt.figure(figsize=sides, dpi=px)
            sns.heatmap(corr_matrix, annot=True, annot_kws={"size": annot_size})
        else:
            corr_matrix = data.corr()
            plt.figure(figsize=sides, dpi=px)
            sns.heatmap(corr_matrix, annot=True, annot_kws={"size": annot_size})

        plt.show()



    def plot_3d(self,
            x_data : Any, y_data : Any, z_data : Any,
            x_title : str, y_title : str, z_title : str,
            sides : Tuple[int], px : float, trace = None, mode : {'scatter', 'heatmap', 'contour_lines_plot'} = 'scatter',
            colored : bool = False,
            color1 : str = None, color2 : str = None, color3 : str = None,
            range1 = None, range2 = None, range3 = None
            ) -> "plot_3d":
        


        fig = plt.figure(figsize=sides, dpi=px)
        ax = fig.add_subplot(111, projection="3d")


        if mode == 'heatmap':
            surf = ax.plot_surface(x_data, y_data, z_data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)
        

        elif mode == 'scatter':
            if colored:
                if range1:
                    colors = [color1 for i in range(range1)]
                    if range2:
                        colors = np.concatenate((colors, [color2 for i in range(range2)]))
                        if range3:
                            colors = np.concatenate((colors, [color3 for i in range(range3)]))

            ax.scatter(x_data, y_data, z_data, c = colors)
        

        elif mode == 'contour_lines_plot':
            ax.plot_surface(x_data, y_data, z_data, alpha=0.6)
            ax.contour(x_data, y_data, z_data, offset=z_data.min(), cmap=cm.coolwarm)

            if trace:
                ax.plot(trace[:, 0], trace[:, 1], y_data, "o-")
                ax.set_xlim(x_data.min(), x_data.max())
                ax.set_ylim(y_data.min(), y_data.max())
                ax.set_zlim(z_data.min(), z_data.max())


        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_zlabel(z_title)


        plt.show()

    def descent_map(X : Any, y : Any, weights : Iterable, sides : Tuple[int], px : float, Title : str,  x_title : str, y_title : str) -> "descent_map":
        A, B = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
        weights = np.array(weights)

        levels = np.empty_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                w_tmp = np.array([A[i, j], B[i, j]])
                levels[i, j] = np.mean(np.power(np.dot(X, w_tmp) - y, 2))

        plt.figure(figsize=sides, dpi=px)
        plt.title(Title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.xlim(weights[:, 0].min() - 0.2, weights[:, 0].max() + 0.2)
        plt.ylim(weights[:, 1].min() - 0.2, weights[:, 1].max() + 0.2)
        plt.gca().set_aspect('equal')

        CS = plt.contour(A, B, levels, levels = np.logspace(0, 1, num=20), cmap = plt.cm.rainbow_r)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')

        plt.scatter(weights[:, 0], weights[:, 1])
        plt.plot(weights[:, 0], weights[:, 1])

        plt.show()



def euclidian_distance(x1, x2) -> float:
    """ Euclidian distance of a pair of points """
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)

    return np.sqrt(distance)
