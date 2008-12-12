#!/usr/bin/python

__all__ = ['congruence', 'kurtosis', 'skewness']

import numpy as np
from scipy.stats import skewness,
from scipy.stats import kurtosis

def congruence(x,y):
    tx = np.array(x).flatten()
    ty = np.array(y).flatten()
    tmp = np.sum(tx*ty)/np.sqrt(np.sum(tx*tx)*np.sum(ty*ty))

    return tmp
