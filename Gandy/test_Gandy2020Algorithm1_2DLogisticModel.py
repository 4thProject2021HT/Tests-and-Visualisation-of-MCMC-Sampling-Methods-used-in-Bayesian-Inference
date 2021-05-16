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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.stats import ks_2samp
from Gandy2020Algorithm1 import *
from Gandy2020Algorithm1WithConvergence import *

model = pints.toy.LogisticModel()
log_prior = pints.UniformLogPrior(
    [0.015, 400],
    [0.017, 600]
)
log_prior_used = pints.UniformLogPrior(
    [0.015, 200],
    [0.017, 800]
)

L = 50
N = 1000
times = np.linspace(1, 1000, 50)
noise = 10
noise_used = 1
MCMCmethod = pints.HaarioACMC

average_p, average_p_theta, average_p_y, duration, thetatildeArray, thetaArray, ytildeArray, yArray = Algorithm1WithConvergence(L, N, model, log_prior, log_prior_used, times, noise, noise_used, MCMCmethod,param=0)
hist(thetatildeArray, thetaArray, ytildeArray, yArray)


'''data1 = {'Theta': thetatildeArray, 'Y': ytildeArray, 'Class': [1]*N1}
df1 = pd.DataFrame(data1, index = range(N1))
data2 = {'Theta': thetatildeArray, 'Y': ytildeArray, 'Class': [0]*N2}
df2 = pd.DataFrame(data2, index = range(N1, N1+N2))
df = pd.concat([df1, df2])
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)


RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(train[['Theta', 'Y']], train['Class'])
c_pred = RF.predict(test[['Theta', 'Y']])
cm = confusion_matrix(test['Class'], c_pred)
tn, fp, fn, tp = cm.ravel()
tpr = tp/(tp+fn)
# Specificity or true negative rate
tnr = tn/(tn+fp) 
print(cm)
print('True Positive Rate:', tpr)
print('True Negative Rate:', tnr)

import sklearn as sk
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(train[['Theta', 'Y']], train['Class'])
c_pred = NN.predict(test[['Theta', 'Y']])
cm = confusion_matrix(test['Class'], c_pred)
tn, fp, fn, tp = cm.ravel()
tpr = tp/(tp+fn)
# Specificity or true negative rate
tnr = tn/(tn+fp) 
print(cm)
print('True Positive Rate:', tpr)
print('True Negative Rate:', tnr)'''