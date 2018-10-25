import time

import pandas as pd
import numpy as np
import pickle
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

    def process(self, limit_users):
        """
        Initializes output matrix and fills the matrix with ratings according to the dataset.

        Input:
        @limit - number of entries in the dataset to be considered.

        Output:
        Dataframe and numpy array saved as 'data_df.csv' and 'data_np.npy' respectively.
        """
        clean_start_time = time.time()
        print("Formatting dataset")
        df = self.read_data(filename=self.filename)
        profile_columns = sorted(list(set(list(df[PROFILE_ID]))))
        user_rows = [i for i in range(1,limit_users+1)]
        final_df = pd.DataFrame(index=user_rows, columns=profile_columns)
        for i in range(len(df[USER_ID])):
            if df.iloc[i][USER_ID] <= limit_users:
                print("USER " + str(df.iloc[i][USER_ID]))
                print("PROFILE " + str(df.iloc[i][PROFILE_ID]))
                print("Ratings: " + str(df.iloc[i][RATINGS]))
                final_df.loc[df.iloc[i][USER_ID]][df.loc[i][PROFILE_ID]] = df.iloc[i][RATINGS]
            else:
                break
        final_df= final_df.dropna(how='all')
        final_df = final_df.fillna(0)
        final_df.to_csv('data_df.csv')
        np.save('data_np.npy', final_df.values)
        print("Format time: " + str(time.time() - clean_start_time))
        print("Formatted dataset.")


if __name__=="__main__":
    cleaner = CleanData('ratings.dat')
    cleaner.process()
