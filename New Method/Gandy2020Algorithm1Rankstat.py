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

def rank_statistic(thetalist,thetai):
    count = 0
    for elem in thetalist:
        if elem < thetai:
            count += 1
    return count



def Algorithm1Rankstat(N, L, Ldashdash, model, log_prior, log_prior_used, times, noise, noise_used, MCMCmethod, param=0):
    time_start=time.time()
    sum_p = 0
    sum_p_theta = 0
    sum_p_y = 0
    

    rankstats = []
    
    d = 0
    for n in range(N):
        print(n)
        thetaArray = np.empty(L, dtype=float)
        thetatilde = log_prior.sample(n=1)[0]
        org_values = model.simulate(thetatilde,times)
        ytilde_n = org_values + np.random.normal(0, noise, org_values.shape)
        problem = pints.SingleOutputProblem(model, times, ytilde_n)
        log_likelihood_used = pints.GaussianKnownSigmaLogLikelihood(problem, [noise_used])
        log_posterior = pints.LogPosterior(log_likelihood_used, log_prior_used)
        for l in range(L):
            #Run Markov chain L steps from thetatilde
            xs = [thetatilde]
            mcmc = pints.MCMCController(log_posterior, 1, xs, method=MCMCmethod)
            # Add stopping criterion
            sample_size = Ldashdash+1
            mcmc.set_max_iterations(sample_size)

            # Start adapting after sample_size//4 iterations
            mcmc.set_initial_phase_iterations(sample_size//4)

            # Disable logging mode
            mcmc.set_log_to_screen(False)

            chain = mcmc.run()[0]
            theta_l  = chain[Ldashdash]

            thetaArray[l] = theta_l[param]
    

        rankstat = rank_statistic(thetaArray, thetatilde)
        rankstats.append(rankstat)

    current_date_and_time = datetime.now()
    current_date_and_time_string = str(current_date_and_time)

    plt.hist(rankstats, bins = range(N+2),align='left')
    plt.axhline(y=c/(N+1), color='r', linestyle='-')
    plt.axhline(y=binom.ppf(0.005, N, 1/(L+1)), color='b')
    plt.axhline(y=binom.ppf(0.995, N, 1/(L+1)), color='b')
    plt.savefig('./Gandy2020AlgorithmRankstat'+current_date_and_time_string+'.png',dpi=500,bbox_inches = 'tight')
    time_end=time.time()
    print('total running time',time_end-time_start)


    plt.show()






