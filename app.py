import os

import numpy as np
import pandas as pd

from preprocessing.clean import CleanData

def format_dataset():
    for file in os.listdir('preprocessing/'):
        if str(file).endswith('.dat'):
            print("Dataset exists.")
            cleaner = CleanData('preprocessing/ratings.dat')
            cleaner.process(limit_users=100)
            break
        elif os.listdir('preprocessing/').index(file) == len(os.listdir('preprocessing/')) - 1:
            print("Dataset doesn't exist. Rerun run.sh again.")


if __name__=="__main__":
    formated_dataset = False
    for files in os.listdir('.'):
        if str(files).endswith('.npy') or str(files).endswith('.csv'):
            print("Formated dataset already exists.")
            formated_dataset = True
            break
    if formated_dataset is False:
        format_dataset()
