import numpy as np


def rmse(M, M_p):
    """
    Computes Root Mean Square Error.

    Input:
    @M - Actual numpy array
    @M_p - Predicted numpy array

    Returns: Root Mean square error - float
    """
    x_len = M.shape[0]
    y_len = M.shape[1]
    error = 0
    N = x_len * y_len
    for x in range(x_len):
        for y in range(y_len):
            error += ((M[x][y] - M_p[x][y]) ** 2) / N
    error = error ** 0.5
    return error



def top_k(k, M, M_p, ignore=True):
    """
    Returns precision of predicted results in top k ratings.

    Input:
    @M - Actual numpy array.
    @M_p - Predicted numpy array.
    @ignore - Ignores already rated values.

    Returns:
    Precision of predictions in top K - float
    """
    x_len = M.shape[0]
    y_len = M.shape[1]
    precision = []
    for i in range(x_len):
        sorted_M = sorted(M[i], reverse=True)[:k]
        for j in range(y_len):
            precision_count = 0
            if ignore:
                if M[i][j] == 0:
                    try:
                        if sorted_M.index(M_p[i][j]) > -1:
                            precision_count += 1
                    except ValueError as ve:
                        pass
            else:
                try:
                    if sorted_M.index(M_p[i][j]) > -1:
                        precision_count += 1
                except ValueError as ve:
                    pass
        precision.append(precision_count)

    average_precision = 0
    p_len = len(precision)
    for p in precision:
        average_precision += p / p_len
    return average_precision

def spearman_correlation(M, M_p):
    """
    Returns Spearman score for the prediction.
    Formula: 1 - [sum(diff(predicted - actual)^2) / n((n^2)-1)]

    Input:
    @M - Actual numpy array.
    @M_p - Predicted numpy array.

    Returns:
    Spearman score - float
    """
    x_len = M.shape[0]
    y_len = M.shape[1]
    N = 0
    sum = 0
    for i in range(x_len):
        for j in range(y_len):
            if M[i][j] != 0:
                N += 1
                sum += (M[i][j] - M_p[i][j]) ** 2
    N = (N*(N**2 - 1))
    sum = 1 - (sum/N)
    return sum
