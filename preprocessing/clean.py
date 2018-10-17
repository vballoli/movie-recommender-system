import time

import pandas as pd
import numpy as np
from preprocessing.constants import USER_ID, PROFILE_ID, RATINGS

class CleanData:
    """
    Helper class to structure dataset. Save's final matrix into a .npy file
    """

    def __init__(self, filename=None):
        """
        Class initialized with 135359 * 220970 given in the dataset documentation.
        Link: http://www.occamslab.com/petricek/data/
        """
        self.filename = filename
        self.number_unique_users = 135359
        self.number_profiles = 220970

    def read_data(self, filename):
        """
        Returns a pandas dataframe of the dataset with columns labelled as 0,1,2.
        """
        df = pd.read_csv(filename, header=None)
        return df

    def process(self):
        """
        Initializes output matrix and fills the matrix with ratings according to the dataset.
        """
        clean_start_time = time.time()
        print("Start time: " + str(clean_start_time))
        df = self.read_data(self.filename)
        final_matrix = np.zeros([self.number_unique_users , self.number_profiles])
        for i in range(len(df[USER_ID])):
            if i%1000 == 0:
                print("Reached " + str(i))
            final_matrix[df.iloc[i][USER_ID]-1][df.iloc[i][PROFILE_ID]-1] = df.iloc[i][RATINGS]
        np.save('../data.npy', final_matrix)
        print("Clean time: " + str(time.time() - clean_start_time))

if __name__=="__main__":
    cleaner = CleanData('ratings.dat')
    cleaner.process()
