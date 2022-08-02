import numpy as np
import matlab
import matlab_interface

"""
Interval matrix arithmetic module.
For multiplication of two interval matrices, a call is made to intMatMul present in intMatMul.m using
a matlab engine interface.

Note: Represent all interval vectors as np.darrays with shape (1, n) or (n, 1) depending on 
whether it is row or column vector. Basically represent vectors as matrices and not arrays.
"""


class Imatrix:
    def __init__(self, lower, upper):
        """Expected type: numpy d-array"""
        self.lower = lower
        self.upper = upper

    def __neg__(self):
        return Imatrix(-1*self.upper, -1*self.lower)

    def __add__(self, other):
        res_lower = self.lower + other.lower
        res_upper = self.upper + other.upper

        return Imatrix(res_lower, res_upper)

    def __sub__(self, other):
        res_lower = self.lower - other.upper
        res_upper = self.upper - other.lower

        return Imatrix(res_lower, res_upper)

    def __rmul__(self, other):
        if type(other) not in [int, float]:
            raise TypeError(f"Multiplication for type {type(other)} with type {type(self)} undefined")
        else:
            if other >= 0:
                return Imatrix(other * self.lower, other * self.upper)
            else:
                return Imatrix(other * self.upper, other * self.lower)

    def __mul__(self, other):
        if type(other) in [int, float]:  # Multiplication of scalar and interval matrix
            if other >= 0:
                return Imatrix(other * self.lower, other * self.upper)
            else:
                return Imatrix(other * self.upper, other * self.lower)

        elif type(other) == type(self):  # Multiplication of 2 Interval matrices
            X1 = matlab.double(self.lower)
            X2 = matlab.double(self.upper)
            X3 = matlab.double(other.lower)
            X4 = matlab.double(other.upper)

            [Y1, Y2] = matlab_interface.ENG.intMatMul(X1, X2, X3, X4, nargout=2)

            output = Imatrix(np.array(Y1), np.array(Y2))

            return output

        else:
            raise TypeError(f"Multiplication for type {type(other)} with type {type(self)} undefined")

    def __str__(self):
        s = f"Lower limit is: \n {str(self.lower)}\n Upper Limit is: \n {str(self.upper)}"

        return s
