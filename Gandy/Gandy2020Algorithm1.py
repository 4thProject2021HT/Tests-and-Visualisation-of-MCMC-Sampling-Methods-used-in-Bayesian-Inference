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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.stats import ks_2samp
import matplotlib
matplotlib.rcParams['text.usetex'] = True

def Algorithm1(L, N, model, log_prior, log_prior_used, times, noise, noise_used, MCMCmethod, param=0):
    time_start=time.time()
    sum_p = 0
    sum_p_theta = 0
    sum_p_y = 0
    c = 1
    for i in range(c):
        print(i)
        thetatildeArray = []
        ytildeArray = []

        d = 0
        for n in range(N):
            print(n)
            thetatilde = log_prior.sample(n=1)[0]
            org_values = model.simulate(thetatilde,times)
            ytilde_n = org_values + np.random.normal(0, noise, org_values.shape)
            problem = pints.SingleOutputProblem(model, times, ytilde_n)
            log_likelihood_used = pints.GaussianKnownSigmaLogLikelihood(problem, [noise_used])
            log_posterior = pints.LogPosterior(log_likelihood_used, log_prior_used)
            #Start from thetatilde
            xs = [thetatilde]
            '''xs = [
                    thetatilde,
                    thetatilde*1.01,
                    thetatilde*0.99
                ]
            isinf=False
            for x in xs:
                #print(x)
                if (math.isinf(log_posterior.evaluateS1(x)[0])):
                    isinf = True
                    d+=1
                    break
            if (isinf==True):
                print('isinf:', isinf)
                continue
            #Run Markov chain L steps from thetatilde'''
            mcmc = pints.MCMCController(log_posterior, len(xs), xs, method=MCMCmethod)
            # Add stopping criterion
            sample_size = 1000

            mcmc.set_max_iterations(sample_size)

            # Start adapting after sample_size//4 iterations
            mcmc.set_initial_phase_iterations(sample_size//4)

            # Disable logging mode
            mcmc.set_log_to_screen(False)
            '''chains = mcmc.run()
            s = sample_size//4+1
            b = False
            while s < sample_size:
                chains_cut = chains[:,sample_size//4:s+1]
                #HMC: chains_cut = chains[:,0:s+1]
                rhat = pints.rhat(chains_cut)
                s+=1
                if rhat[0] < 1.05:
                    print('converge')
                    b = True
                    break
            if b == False:
               d += 1
               continue'''

            chain = mcmc.run()[0]


            #print(s)
            thetatilde_n  = chain[L]
            #thetatilde_n = chains[0][(s+sample_size)//2-1]
            #print(thetatilde)
            thetatildeArray.append(thetatilde_n[param])
            ytildeArray.append(ytilde_n[param])
            
        thetaArray = np.empty(N-d, dtype=float)
        yArray = np.empty(N-d, dtype=float)

        for n in range(N-d):
            theta_n = log_prior.sample(n=1)[0]
            org_values = model.simulate(theta_n,times)
            y_n = org_values + np.random.normal(0, noise, org_values.shape)
            thetaArray[n] = theta_n[param]
            yArray[n] = y_n[param]
            res2.append((theta_n[param], y_n[param]))
        
        p = ks2d2s(thetatildeArray, ytildeArray, thetaArray, yArray)
        statistic_theta, p_theta = ks_2samp(thetatildeArray, thetaArray)
        statistic_y, p_y = ks_2samp(ytildeArray, yArray)
        sum_p += p
        sum_p_theta += p_theta
        sum_p_y += p_y
    time_end=time.time()
    duration = time_end-time_start
    
    average_p = sum_p/c
    average_p_theta = sum_p_theta/c
    average_p_y = sum_p_y/c
    print('average_p:', average_p)
    print('average_p_theta:', average_p_theta)
    print('average_p_y:', average_p_y)
    return average_p, average_p_theta, average_p_y, duration, thetatildeArray, thetaArray, ytildeArray, yArray


def hist(thetatildeArray, thetaArray, ytildeArray, yArray):
    current_date_and_time_string = datetime.now()
    #bins=np.histogram(np.hstack((ytildeArray,yArray)), bins=40)[1] 
    #plt.hist(ytildeArray,bins, alpha=0.5, label=r'$\widetilde y$')
    #plt.hist(yArray,bins, alpha=0.5, label=r'$y$')
    #plt.title(r'$y, \widetilde y$')
    #plt.legend(loc='upper right')
    #plt.savefig('ytildeArray_yArray'+str(current_date_and_time_string)+'.png')
    plt.clf()

    bins=np.histogram(np.hstack((thetatildeArray,thetaArray)), bins=40)[1] 
    plt.figure(figsize=(12,4),dpi=500)
    plt.xlabel('r')
    plt.hist(thetaArray,bins, alpha=0.5, label=r'$\theta$',color='green')
    plt.hist(thetatildeArray,bins, alpha=0.5, label=r'$\widetilde\theta$',color='pink')


    plt.title(r'$\theta, \widetilde\theta$')
    plt.legend(loc='upper right')
    plt.savefig('thetaArray_thetatildeArray'+str(current_date_and_time_string)+'.png')

    df = pd.DataFrame(columns=['thetatilde','theta','ytilde','y'])
    df['thetatilde'] = thetatildeArray
    df['theta'] = thetaArray
    df['ytilde'] = yArray
    df['y'] = yArray
    df.to_csv('arrays'+str(current_date_and_time_string)+'.csv')
