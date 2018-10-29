import numpy as np

def pearson_sim(M, x, y):
    """
    Pearson correlation coefficient of two rows M(x) and M(y)

    Input:
    M (numpy.ndarray): Input Matrix
    x (int): Index of first item
    y (int): Index of second item
    """
    x_mean = sum(M[x])/np.count_nonzero(M[x])
    y_mean = sum(M[y])/np.count_nonzero(M[y])

    numerator = 0
    denom_x = 0
    denom_y = 0

    for i in range(len(M[x])):
        if M[x][i] != 0 and M[y][i] != 0:
            numerator += (M[x][i] - x_mean)*(M[y][i] - y_mean)
            denom_x += (M[x][i] - x_mean)**2
            denom_y += (M[y][i] - y_mean)**2
        elif M[x][i] == 0 and M[y][i] != 0:
            denom_y += (M[y][i] - y_mean)**2
        elif M[x][i] != 0 and M[y][i] == 0:
            denom_x += (M[x][i] - x_mean)**2

    return numerator/np.sqrt(denom_x * denom_y)
