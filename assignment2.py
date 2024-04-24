"""
In this assignment you should find the intersection points for two functions.
"""
import math

import numpy
import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Initialize the Assignment2 class.
        This method can be used to store any values that are common to all functions
        and need to be computed only once.
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
          Finds as many intersection points as possible between two given functions `f1` and `f2` in the given range [a, b].
          The functions are expected to have at least two intersection points, one with a positive x and one with a negative x.

          Note: This method may not work correctly if there is an infinite number of intersection points.

          Parameters
          ----------
          f1 : callable
              The first given function.
          f2 : callable
              The second given function.
          a : float
              The start of the interpolation range.
          b : float
              The end of the interpolation range.
          maxerr : float, optional (default=0.001)
              An upper bound on the difference between the function values at the approximate intersection points.

          Returns
          -------
          X : iterable of float
              An iterable of approximate intersection Xs such that for each x in X, `|f1(x) - f2(x)| <= maxerr`.
              The values in X are sorted in increasing order.

          """

        def secant(f, a, b, maxerr):
            """
            Finds a root of the function `f` between `a` and `b` using the secant method, with the given maximum error `maxerr`.
            """
            x = b - f(b) * (b - a) / (f(b) - f(a))
            while np.abs(f(x)) > maxerr:
                a, b = b, x
                x = b - f(b) * (b - a) / (f(b) - f(a))
            return x

        def bisection(f, a, b, maxerr):
            """
            Finds a root of the function `f` between `a` and `b` using the bisection method, with the given maximum error `maxerr`.
            """
            c = (a + b) / 2
            while np.abs(f(c)) > maxerr:
                if f(a) * f(c) < 0:
                    b = c
                else:
                    a = c
                c = (a + b) / 2
            return c

        f_x = lambda x: f1(x) - f2(x)
        points = np.linspace(a, b, 200)  # 200 evenly spaced points between a and b
        roots = []  # list to store the roots found
        for i in range(len(points) - 1):
            if f_x(points[i]) * f_x(points[i + 1]) <= 0:
                # if the function changes sign between points[i] and points[i+1],
                # it means there is a root in this interval
                try:
                    roots.append((secant(f_x, points[i], points[i + 1], maxerr)))
                except:
                    roots.append((bisection(f_x, points[i], points[i + 1], maxerr)))
        return sorted(roots)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

if __name__ == "__main__":
    unittest.main()
