import time
time_start=time.time()
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
from Gandy2020Algorithm1 import *

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
#log_prior = pints.GaussianLogPrior(400, 10)

log_prior = pints.UniformLogPrior(
    [400],
    [600]
)

#log_prior_used = pints.GaussianLogPrior(400, 10)

log_prior_used = pints.UniformLogPrior(
    [200],
    [800]
)

L = 10
N = 1000
times = np.linspace(1, 1000, 50)
noise = 10
noise_used = 10
MCMCmethod = pints.HaarioACMC
average_p, average_p_theta, average_p_y, duration, thetatildaArray, thetaArray, ytildaArray, yArray = Algorithm1(L, N, model, log_prior, log_prior_used, times, noise, noise_used, MCMCmethod)
hist(thetatildaArray, thetaArray, ytildaArray, yArray)

