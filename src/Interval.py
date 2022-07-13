from interval import interval
import numpy as np
import time

"""
Temporary Interval Matrix class.(Inefficient data structure, replace with python wrapper to INTLAB)
Supports addition and multiplication of interval matrices.
Uses 'list' data structure to store the intervals.
Intervals are represented using 'interval' objects from pyinterval package.
Create empty Imatrix -> use create_matrix by passing numpy arrays to populate the Imatrix
Access the 'ij'th element by using A[i, j] and not A[i][j]
"""


class Imatrix:
    def __init__(self):
        self.shape = []
        self.matrix = []

    def create_matrix(self, lower, upper):
        if not lower.shape == upper.shape:
            raise TypeError("Lower and Upper matrices must have same shape")
        else:
            self.shape = lower.shape
            self.matrix = [[interval((lower[i][j], upper[i][j])) for j in range(self.shape[1])] for i in
                           range(self.shape[0])]

    def __getitem__(self, items):
        print(items)
        i, j = items
        return self.matrix[i][j]

    def __str__(self):
        return str(self.matrix)

    def __repr__(self):
        return str(self.matrix)

    def __len__(self):
        return len(self.matrix)

    def __add__(self, other):
        if not self.shape == other.shape:
            raise TypeError("Both matrices should have the same shape")
        else:
            output = Imatrix()
            output.shape = self.shape

            rows = self.shape[0]
            cols = self.shape[1]

            output.matrix = [[self.matrix[r][c] + other.matrix[r][c] for c in range(cols)] for r in range(rows)]

            return output

    def __mul__(self, other):
        if not self.shape[1] == other.shape[0]:
            raise TypeError("Incompatible shapes for matrix multiplication")
        else:
            rows_of_self = self.shape[0]
            cols_of_other = other.shape[1]

            output = Imatrix()
            output.shape = (rows_of_self, cols_of_other)

            for i in range(rows_of_self):
                out_row = []
                for j in range(cols_of_other):
                    temp = interval(0)
                    for k in range(self.shape[1]):
                        temp = temp + (self.matrix[i][k] * other.matrix[k][j])

                    out_row.append(temp)
                output.matrix.append(out_row)

            return output

    def get_lower(self, index):
        a = [self.matrix[index][j][0].inf for j in range(len(self.matrix[index]))]

        return np.array(a)

    def get_upper(self, index):
        b = [self.matrix[index][j][0].sup for j in range(len(self.matrix[index]))]

        return np.array(b)


def to_Imatrix(some_list):
    """
    some_list has to be a proper list representation of a 2d matrix
    """
    output = Imatrix()

    rows = len(some_list)
    cols = len(some_list[0])

    output.shape = (rows, cols)
    output.matrix = []
    for r in range(rows):
        output.matrix.append([interval(some_list[r][c]) for c in range(cols)])

    return output
