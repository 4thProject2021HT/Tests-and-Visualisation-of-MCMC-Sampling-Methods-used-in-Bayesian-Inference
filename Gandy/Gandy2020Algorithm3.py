#alpha: the overall desired false rejection rate
#k: the maximum number of sequential steps
#delta: the factor by which to multiple the sample size after the first iteration.

from Gandy2020Algorithm1 import Algorithm1
import numpy as np
import pints
import pints.toy as toy
import matplotlib.pyplot as plt
from scipy.stats import binom
from datetime import datetime
from scipy.stats import chisquare

def Algorithm3(alpha, k, delta, L, N, model, log_prior, log_prior_used, times, noise, noise_used, MCMCmethod):
    beta0 = alpha/k 
    gamma = pow(beta0, 1/k)
    betaArray = np.empty(k+1, dtype=object)
    betaArray[0] = beta0
    for i in range(k):
        print(i)
        #vector of p-values from one of the algorithms(sample size n)
        #q = min(p/d)
        average_p, average_p_theta, average_p_y, _ = Algorithm1(L, N, model, log_prior, log_prior_used, times, noise, noise_used, MCMCmethod)
        q = average_p
        if q <= betaArray[i]:
            return 'FAIL'
        if q > gamma + betaArray[i]:
            break
        betaArray[i+1] = betaArray[i]/gamma
        if i == 0:
            N = delta * N
    return "OK"

