import pandas as pd
import numpy as np

class CleanData:
    """
    Helper class to structure dataset. Save's final matrix into a .npy file
    """

    def __init__(self, filename=None):
        self.filename = filename
        self.number_unique_users = 135359
        self.number_profiles = 220970

    def read_data(self, filename):
        df = pd.read_csv(filename, headers=['user_id', 'profile_id', 'ratings'])
        return df

    def process(self):
        """
        Initializes output matrix and fills the matrix with ratings according to the dataset.
        """
        df = read_data(self.filename)
        final_matrix = np.zeros([self.number_unique_users, self.number_profiles])
        for i in range(len(df['user'])):
            final_matrix[df.iloc[i]['user_id']][df.iloc[i]['profile_id']] = df.iloc[i]['ratings']
        np.save('../data.npy', final_matrix)

if __name__=="__main__":
    pass
