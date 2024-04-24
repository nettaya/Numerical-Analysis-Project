"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""
import assignment2
import numpy as np
import time
import random


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the integration error.
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions.

        Integration error will be measured compared to the actual value of the
        definite integral.

        Note: It is forbidden to call f more than n times.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        # If n is equal to 0, return 0
        if n == 0:
            return np.float32(0)

        # If n is equal to 1, use the midpoint rule to estimate the definite integral
        if n == 1:
            h = b - a
            x0 = (a + b) / 2
            return np.float32(h * f(x0))

        # If n is equal to 2, use the trapezoidal rule to estimate the definite integral
        elif n == 2:
            h = b - a
            return np.float32((h / 2) * (f(a) + f(b)))

        # If n is greater than or equal to 3, use the Simpson's rule
        n -= 1
        if n % 2 == 1:
            n -= 1
        h = (b - a) / n
        # Generate an array of x values
        x_points = np.linspace(a, b, n + 1)
        # Evaluate the function at each x value and store the results in an array
        y = [f(x) for x in x_points]
        # Apply Simpson's rule formula to the array of function values
        res = h / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])

        # Convert the result to a float32 and return it
        return np.float32(res)
    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        intersection = assignment2.Assignment2()
        int_pts = intersection.intersections(f1, f2, 1, 100, maxerr=0.001)
        list_int_pts = sorted(list(int_pts))

        if len(list_int_pts) < 2:
            return None

        area = np.float32(0)
        for i in range(0, len(list_int_pts) - 1):
            first = list_int_pts[i]
            second = list_int_pts[i + 1]
            num_in_range = first + (second - first) / 2
            if f1(num_in_range) > f2(num_in_range):
                func = lambda x: f1(x) - f2(x)
            else:
                func = lambda x: f2(x) - f1(x)
            area += np.float32(abs(self.integrate(func, list_int_pts[i], list_int_pts[i + 1], 200)))
        return area







##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
