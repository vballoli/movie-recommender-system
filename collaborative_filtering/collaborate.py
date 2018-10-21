import time
import numpy as np
from similarities import pearson_sim as sim
from constants import INT_MIN

class Collaborate:
    """
    Class to perform collaborative filtering with and without baseline approach
    """
    def __init__(self, M):
        """
        Initialize utility (ratings) matrix
        Note: Matrix needs to have items as rows and users as columns
        """
        self.M = M

    def estimate(self, user, item, k=2, baseline=False):
        """
        Estimate rating for a given input user and item.
        k nearest neighbours taken based on similarity (k = 2 by default).
        """
        # Ratings matrix
        r = self.M
        # Mean baseline deviation
        mu = 0
        # User baseline deviation
        b_user = 0
        # Item baseline deviation
        b_item = 0
        # With baseline deviation considered
        if baseline is True:
            mu = sum(sum(r))/np.count_nonzero(r)
            b_user = sum(r[:, user])/np.count_nonzero(r[:, user]) - mu
            b_item = sum(r[item])/np.count_nonzero(r[item]) - mu
        # Overall baseline deviation
        b = mu + b_user + b_item
        ### Rating estimation ###
        # Calculate similarities
        S = np.zeros(r.shape[0])
        for i in r[:, user]:
            S[i] = sim(r, item, i)
        S = S[1:]
        # Estimate the rating
        numerator = 0
        denominator = 0
        for _ in range(k):
            max_idx = np.argmax(S)
            numerator += (r[max_idx, user] - b_user)*S[max_idx]
            denominator += S[max_idx]
            S[max_idx] = INT_MIN

        rating = b + (numerator/denominator)
        if np.isnan(rating):
            rating = 0

        return rating

    def fill(self, k=2, baseline=False):
        """
        Fills gaps in utility matrix using CF estimates
        """
        # Start time
        t0 = time.time()
        # Complete the matrix
        filled = np.zeros(self.M.shape)
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.M[i, j] == 0:
                    filled[i, j] = self.estimate(j, i, k=k, baseline=baseline)
                else:
                    filled[i, j] = self.M[i, j]

        print("CF estimation time: ", time.time() - t0)
        return filled
################ TEST BLOCK #################
if __name__ == "__main__":
    input_mat = np.array([[1,0,3,0,0,5,0,0,5,0,4,0],
                   [0,0,5,4,0,0,4,0,0,2,1,3],
                   [2,4,0,1,2,0,3,0,4,3,5,0],
                   [0,2,4,0,5,0,0,4,0,0,2,0],
                   [0,0,4,3,4,2,0,0,0,0,2,5],
                   [1,0,3,0,3,0,0,2,0,0,4,0]])

    cf = Collaborate(input_mat)
    output_mat = cf.fill()
    print(output_mat)
##############################################
