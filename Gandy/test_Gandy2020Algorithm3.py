import time
import matplotlib.pyplot as plt
import numpy as np
import pints
import pints.plot
import pints.toy as toy
import pandas as pd
import math
from scipy.stats import binom
from scipy.stats import ks_2samp
from datetime import datetime
from ndtest import ks2d2s
from scipy import stats
from scipy.stats import ks_2samp
from Gandy2020Algorithm3 import Algorithm3

time_start=time.time()
class Model(pints.ForwardModel):
    def __init__(self):
        self.model = toy.LogisticModel()
    def simulate(self, x, times):
        return self.model.simulate([0.01, x[0]], times)
    def simulateS1(self, x, times):
        values, gradient = self.model.simulateS1([0.01, x[0]], times)
        gradient = gradient[:, 0]
        return values, gradient
    def n_parameters(self):
        return 1

model = Model()
#log_prior = pints.GaussianLogPrior(400, 1)

log_prior = pints.UniformLogPrior(
    [400],
    [600]
)

#log_prior_incorrect = pints.GaussianLogPrior(400, 10)

log_prior_incorrect = pints.UniformLogPrior(
    [200],
    [800]
)

model = Model()
alpha = pow(10,-5)
k = 7
delta = 4
L = 50
N = 1000
times = np.linspace(1, 1000, 5)
noise = 10
MCMCmethod = pints.HaarioACMC

print(Algorithm3(alpha, k, delta, L, N, model, log_prior, log_prior_incorrect, times, noise, noise, MCMCmethod))
time_end=time.time()
duration = time_end-time_start
print('duration: ', duration)