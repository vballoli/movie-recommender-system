import os

import numpy as np
import pandas as pd

from preprocessing.clean import CleanData
from svd.svd_algorithm import SVDAlgorithm
from error_measures.measures import *
from cur.cur_algorithm import *

def format_dataset():
    for file in os.listdir('preprocessing/'):
        if str(file).endswith('ml-100k'):
            print("Dataset exists.")
            cleaner = CleanData('preprocessing/ml-100k/ml-100k/u.data')
            cleaner.process()
            break
        elif os.listdir('preprocessing/').index(file) == len(os.listdir('preprocessing/')) - 1:
            print("Dataset doesn't exist. Rerun run.sh again.")


def run_collaborative_filtering(M):
    pass

def run_svd(M):
    s = SVDAlgorithm()
    print(M.shape)
    U, sigma, V = s.svd(M, dimension_reduction=0.9)
    M_p = np.dot(np.dot(U, sigma), V)
    print(M_p.shape)
    print(rmse(M, M_p))

def run_cur(M):
    M_p = cur(M, 80, 80)
    print(rmse(M, M_p))


if __name__=="__main__":
    formated_dataset = False
    for files in os.listdir('.'):
        if str(files).endswith('.npy') or str(files).endswith('.csv'):
            print("Formatted dataset already exists.")
            formated_dataset = True
            break
    if formated_dataset is False:
        format_dataset()
    M = np.load('data.npy')
    M = M[1:, :]
    run_svd(M)
    run_collaborative_filtering(M)
    run_cur(M)
