"""
In this assignment you should interpolate the given function.
"""
import time
import random

import numpy as np
from sampleFunctions import bezier3


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

        pass
    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        This function uses the Bézier interpolation method to estimate the value of the function f at any point within the interval [a, b].
        The method first divides the interval into n equal parts, and then calculates the value of the function f at each of these points.
        These values are then used as control points for the Bézier interpolation.
        :param f: callable function that takes in a single float and returns a single float.
        :param a:a float value representing the lower limit of the interval on which the function will be interpolated.
        :param b:a float value representing the upper limit of the interval on which the function will be interpolated.
        :param n:an int value representing the number of points on the interval [a, b] that will be used to interpolate the function.
        :return:This function returns
        """

        if n == 1:
            def g(x):
                return f((a + b) / 2)
            return g

        def diff(x, i):
            """
            Calculate the derivative of the function at a point x, using a small value of h.

            Args:
            - x: A float value representing the point at which the derivative is to be calculated.
            - i: An int value representing the index of the point x in the list of x values.

            Returns:
            - The derivative of the input function f at the point x, calculated using a small value of h.

            """
            h = 0.000000001
            return (f(x + h) - y_val[i]) / h

        # Calculate the x values for the given interval
        x_val = np.linspace(a, b, n // 2)
        # Calculate the y values of the input function at the x values
        y_val = np.array([f(x) for x in x_val])
        # Calculate the derivative of the input function at the x values
        diff_val = np.array([diff(x, i) for i, x in enumerate(x_val)])
        # Calculate the step size between each pair of adjacent x values
        bezier_curves = []
        d = abs(x_val[1] - x_val[0])

        # Calculate the control points for each segment of the Bézier curve
        for i in range(len(x_val) - 1):
            p0 = (x_val[i], y_val[i])
            p3 = (x_val[i + 1], y_val[i + 1])
            p1 = (x_val[i] + d / 3, y_val[i] + diff_val[i] * (d / 3))
            p2 = (x_val[i + 1] - d / 3, y_val[i + 1] - diff_val[i + 1] * (d / 3))
            bezier_curves.append(bezier3(p0, p1, p2, p3))

        def g(x):
            pos = int(((x - a) / (b - a)) * (n // 2 - 1))
            t = (x - a - pos * d) / d
            if pos == len(bezier_curves):
                return bezier_curves[pos - 1](1)[1]
            return bezier_curves[pos](t)[1]

        return g




##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
