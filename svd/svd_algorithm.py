import numpy as np
from numpy import linalg as LA
import time
import math

class SVDAlgorithm:

    def __init__(self):
        pass

    def eigen_decomposition(self, M):
        """
        Returns Eigen values and corresponding eigen vectors arranged in descending order.

        @params:
        M: Input numpy matrix

        Output:
        Returns list - sorted_eigen_values, sorted_eigen_vectors
        sorted_eigen_values - list of sorted eigen_values
        sorted_eigen_vectors - numpy matrix containing eigen vectors
        """
        eigen_values, eigen_vectors = LA.eig(M)
        eigen_values = eigen_values.real  # Considering real parts only
        eigen_vectors = eigen_vectors.real  # Considering real parts only
        for i in range(len(list(eigen_values))):
            eigen_values[i] = round(eigen_values[i], 2)  # Rounding values of 2 digits
        for i in range(eigen_vectors.shape[0]):
            for j in range(eigen_vectors.shape[1]):
                eigen_vectors[i][j] = round(eigen_vectors[i][j], 2)  # Rounding values of 2 digits

        eigen = dict()
        for i in range(len(eigen_values)):
            if eigen_values[i] != 0:
                eigen[eigen_values[i]] = eigen_vectors[:, i]  # Removing zeros

        sorted_eigen_values = sorted(list(eigen.keys()), reverse=True)
        sorted_eigen_vectors = np.zeros_like(eigen_vectors)
        for i in range(len(sorted_eigen_values)):
            sorted_eigen_vectors[:, i] = eigen[sorted_eigen_values[i]]

        sorted_eigen_vectors = sorted_eigen_vectors[:, :len(sorted_eigen_values)]  # Removing zeroed eigen vectors

        return sorted_eigen_values, sorted_eigen_vectors

    def svd(self, M, dimension_reduction=1.0):
        """
        Applies Singular Value Decomposition to input matrix M - minimum reconstruction
        error of M expressed as U, sigma and V such that M = U * sigma * V

        Supports dimensionality reduction where least values of sigma are removed along with
        their corresponding U columns and V rows.

        @params:
        M : Input numpy matrix M
        dimension_reduction: Reduce the dimensions. Recommended range: 0.8 - 1.0

        Output:
        Returns list - U, sigma, V
        sigma - singular values of M
        """
        try:
            assert dimension_reduction <= 1.0 or dimension_reduction == None
        except AssertionError as ae:
            return "Wrong dimension_reduction value"

        try:
            assert type(M) == np.ndarray
        except AssertionError as ae:
            return "Wrong Matrix type. (numpy.ndarray) required."

        eigen_values_u, U = self.eigen_decomposition(np.dot(M, M.T))
        eigen_values_v, V = self.eigen_decomposition(np.dot(M.T, M))

        V = V.T

        sigma = np.diag([i**0.5 for i in eigen_values_u])
        if dimension_reduction == 1.0 or dimension_reduction == None:
            return U, sigma, V
        else:
            total_sigma = np.sum(sigma ** 0.5)
            for i in range(sigma.shape[0]):
                sigma_sum = np.sum(sigma[:i+1, :i+1])
                if sigma_sum > dimension_reduction * total_sigma:
                    sigma = sigma[:i, :i]
                    U = U[:, :i]
                    V = V[:i, :]
                    return U, sigma, V
