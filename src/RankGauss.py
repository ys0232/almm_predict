import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

def rank_gauss(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2  # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x)  # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x


if __name__=="__main__":
    x = np.random.rand(500)
    print(x)
    # histogram test: the histogram of rank_gauss should be gauss-liked and centered
    pd.Series(x).hist()
    plt.show()
    pd.Series(rank_gauss(x)).hist()
    plt.show()
