import time

import pandas as pd
import numpy as np
import pickle
from preprocessing.constants import USER_ID, MOVIE_ID, RATINGS

class CleanData:
    """
    Helper class to structure dataset. Save's final matrix into a .npy file
    """

    def __init__(self, filename=None):
        """
        Class initialized with 135359 * 220970 given in the dataset documentation.
        Link: http://files.grouplens.org/datasets/movielens/ml-100k.zip
        """
        self.filename = filename
        self.user_number = 943
        self.movie_number = 1682

    def read_data(self, filename):
        """
        Returns a pandas dataframe of the dataset with columns labelled as 0,1,2.
        """
        df = pd.read_csv(filename, '\t', header=None)
        return df

    def process(self, limit_users=None):
        """
        Initializes output matrix and fills the matrix with ratings according to the dataset.

        Input:
        @limit - number of entries in the dataset to be considered.

        Output:
        Dataframe and numpy array saved as 'data_df.csv' and 'data_np.npy' respectively.
        """
        clean_start_time = time.time()
        print("Formatting dataset")
        df = self.read_data(self.filename)
        data = np.zeros([self.user_number, self.movie_number])
        for i in range(df.shape[0]):
            data[df.iloc[i][USER_ID]-1][df.iloc[i][MOVIE_ID]-1] = df.iloc[i][RATINGS]
        np.save('data.npy', data)
        print("Formatted dataset.")
        print("Time to format dataset: " + str(time.time() - clean_start_time))


if __name__=="__main__":
    cleaner = CleanData('ml-100k/u.data')
    cleaner.process()
