import os

import numpy as np
import pandas as pd

from preprocessing.clean import CleanData

def format_dataset():
    cleaner = CleanData('preprocessing/ratings.dat')
    cleaner.process()


if __name__=="__main__":
    formated_dataset = False
    for files in os.listdir('.'):
        if str(files).endswith('.npy'):
            formated_dataset = True
            
            break
    if formated_dataset is False:
        format_dataset()
