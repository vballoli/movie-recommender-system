import numpy as np
import random

from svd.svd_algorithm import SVDAlgorithm

def cur(M, c, r, dim_red = None, repeat=None):
    """
    CUR function returns C,U,R

    Input:
    @M: input numpy array
    @c: Number of column selections
    @r: Number of row selections
    @repeat: Repetition allowed
    """
    m_square_sum = np.sum(M ** 2)
    M_col, cols_sel = column_selection(M, m_square_sum, c, repeat_allowed=repeat)
    M_row, rows_sel = column_selection(M.T, m_square_sum, r, repeat_allowed=repeat)
    W = M_col[rows_sel, :]
    if dim_red == None or dim_red == 1:
        W_u, W, W_v = SVDAlgorithm().svd(W)
    else:
        W_u, W, W_v = SVDAlgorithm().svd(W, dimension_reduction=dim_red)
    for i in range(W.shape[0]):
        W[i][i] = 1 / W[i][i]
    U = np.dot(np.dot(W_v.T, W ** 2), W_u.T)
    M_p = np.dot(np.dot(M_col,  U), M_row.T)
    return M_p


def column_selection(M, m_square_sum, c, repeat_allowed=False):
    """
    Column selection algorithm

    Input:
    @M: Input numpy matrix M
    @m_square_sum: Sum of squares of elements of M
    @c: number of columns to select
    @repeat: Repetition allowed
    """
    column_probabilities = list()
    for i in range(M.T.shape[0]):
        column_probability = np.sum(M.T[i] ** 2) / m_square_sum
        column_probabilities.append(column_probability)
    column_selections = list()
    for i in range(c):
        column_selections.append(random.randint(0, c-1))
    C = np.zeros([M.shape[0], c])
    if not repeat_allowed:
        column_selections_set = list(set(column_selections))
        C = np.zeros([M.shape[0], len(column_selections_set)])

    c_count = 0
    if not repeat_allowed:
        for col in column_selections_set:
            prob = column_probabilities[col]
            prob = (c * prob) ** 0.5
            C[:, c_count] = (M[:, col] / prob) * column_selections.count(col)
            c_count += 1
        return C, column_selections_set
    else:
        for col in column_selections:
            prob = column_probabilities[col]
            prob = (c * prob) ** 0.5
            C[:, c_count] = M[:, col] / prob
            c_count += 1
        return C, column_selections
