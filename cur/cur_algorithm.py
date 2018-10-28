import numpy as np
import random

from svd.svd_algorithm import SVDAlgorithm

def cur(M, c, r, repeat=None, retain=0.8):
    """
    CUR function returns C,U,R

    Input:
    @M: input numpy array
    @c: Number of column selections
    @r: Number of row selections
    @repeat: Repetition allowed
    """
    m_square_sum = np.sum(M ** 2)
    print("m_square_sum " + str(m_square_sum))
    M_col, cols_sel = column_selection(M, m_square_sum, c, repeat=repeat)
    M_row, rows_sel = row_selection(M, m_square_sum, r, repeat=repeat)
    print(rows_sel)
    print(cols_sel)
    W = M[rows_sel, :]
    W = W[:, cols_sel]
    _, W, _ = SVDAlgorithm().svd(W)
    print(W.shape)
    for i in range(W.shape[0]):
        if W[i][0] != 0:
            W[i] = 1 / W[i]
    if retain != 1.0:
        W_sum = np.sum(W ** 2)
        retention_index = 0
        for j in range(W.shape[0]):
            if np.sum((W ** 2)[:j]) >= retain * W_sum:
                W = W[:j]
                retention_index = j
                M_col = M_col[:, :retention_index]
                M_row = M_row[:retention_index, :]
                break
    W = np.diag(W)
    print(M_col.shape)
    print(W.shape)
    print(M_row.shape)
    M_p = np.dot(np.dot(M_col,  W), M_row)
    print("M_p")
    print(M_p)
    return M_p


def column_selection(M, m_square_sum, c, repeat=None):
    """
    Column selection algorithm

    Input:
    @M: Input numpy matrix M
    @m_square_sum: Sum of squares of elements of M
    @c: number of columns to select
    @repeat: Repetition allowed
    """
    m_prob = np.zeros_like(M, dtype=np.float32)
    col_value = 0
    for i in range(M.shape[1]):
        col_value =np.sum(M.T[i] ** 2)
        col_value = float(col_value) / float(m_square_sum)
        m_prob.T[i] = col_value
    if repeat:
        column_selections = set()
        while len(column_selections) < c:
            column_selections.add(random.randint(0, M.shape[1]-1))
        column_selections = list(column_selections)
    else:
        column_selections = list()
        for i in range(c):
            column_selections.append(random.randint(0, M.shape[1]-1))
    m_prob = m_prob[:, column_selections]
    M = M[:, column_selections]
    for i in range(M.shape[1]):
        M.T[i] = M.T[i] / ((c*m_prob[0][i])**0.5)
    return [M * m_prob, column_selections]

def row_selection(M, m_square_sum, r, repeat=None):
    """
    Row selection algorithm

    Input:
    @M: Input numpy matrix M
    @m_square_sum: Sum of squares of elements of M
    @r: number of rows to select
    @repeat: Repetition allowed
    """
    m_prob = np.zeros_like(M, dtype=np.float32)
    row_value = 0
    for i in range(M.shape[0]):
        row_value =np.sum(M[i] ** 2)
        row_value = float(row_value) / float(m_square_sum)
        m_prob[i] = row_value
    if repeat:
        row_selections = set()
        while len(row_selections) < r:
            row_selections.add(random.randint(0, M.shape[0]-1))
        row_selections = list(row_selections)
    else:
        row_selections = list()
        for i in range(r):
            row_selections.append(random.randint(0, M.shape[0]-1))
    m_prob = m_prob[row_selections, :]
    M = M[row_selections, :]
    for i in range(M.shape[0]):
        M[i] = M[i] / ((r*m_prob[i][0])**0.5)
    return M * m_prob, row_selections
