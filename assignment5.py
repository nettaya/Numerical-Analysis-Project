"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
from sampleFunctions import *
from sklearn.cluster import KMeans
from functionUtils import AbstractShape

class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, area):
         self._area = area

    def area(self) -> np.float32:
        return self._area
    pass


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        cordinates = contour(3000)
        # convert the points into numpy array
        cordinates_mat = np.array(cordinates)
        # get the x-coordinates
        x = cordinates_mat[:, 0]
        # get the y-coordinates
        y = cordinates_mat[:, 1]
        # calculate the area using the shoelace formula
        res = np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
        return np.float32(res)

    
    def fit_shape(self, sample: callable, maxtime: float) -> MyShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """
        # multiply maxtime with 3500 to get the number of iterations
        maxtime *= 3500
        x_y_points= []
        # get the data points from the sample function
        for i in range(int(maxtime/2)):
            x, y = sample()
            x_y_points.append([x, y])

        # use KMeans to cluster the data points
        kmeans_points = KMeans(n_clusters=34,n_init="auto")
        kmeans_points.fit(x_y_points)
        # get the cluster centers
        cluster_centers = kmeans_points.cluster_centers_
        # convert the cluster centers into a list of lists
        center_list = [[center[0], center[1]] for center in cluster_centers]

        sorted_c_list = sorted(center_list)

        first_center = sorted_c_list[0]

        sorted_centers = [first_center]

        def iterative_sort(sort_centers, sorted_c_list, first_center):
            """
             function is a helper function that is used to sort the cluster centers by their order on the shape contour.
             It takes as input the current sorted list of centers, the original list of centers sorted by x-coordinate,
             the first center in the sorted list, and the distance between centers.
            :param sort_centers: sorted centers
            :param sorted_c_list: original list of centers sorted by x-coordinate
            :param first_center: the first center in the sorted list
            :return: the distance between centers
            """
            def distance_auc(point):
                return (first_center[0] - point[0]) ** 2 + (first_center[1] - point[1]) ** 2

            while len(sorted_c_list) > 2:
                center = sorted(sorted_c_list, key=distance_auc)[1]
                sort_centers.append(center)
                sorted_c_list.remove(first_center)
                first_center = center
            sort_centers.append(sorted_c_list[1])
            sort_centers.append(sort_centers[0])
            return sort_centers

        centers = iterative_sort(sorted_centers, sorted_c_list, first_center)

        area = sum([0.5 * (centers[i][0] * centers[i + 1][1] - centers[i + 1][0] * centers[i][1]) for i in
                    range(len(centers) - 1)])
        return MyShape(abs(area))


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #
    #     def sample():
    #         time.sleep(7)
    #         return circ()
    #
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=sample, maxtime=5)
    #     T = time.time() - T
    #     self.assertTrue(isinstance(shape, AbstractShape))
    #     self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
