import pints
import pints.toy as toy
import pints.plot
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import matplotlib
matplotlib.rcParams['text.usetex'] = True


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


# Load a forward model
def run(model, real_parameters, noise_used, log_prior_used):
    # Create some toy data
    
    times = np.linspace(1, 1000, 50)
    org_values = model.simulate(real_parameters, times)

    # Add noise
    noise = 10
    values = org_values + np.random.normal(0, noise, org_values.shape)
    real_parameters = np.array(real_parameters)


    # Create an object with links to the model and time series
    problem = pints.SingleOutputProblem(model, times, values)

    # Create a log-likelihood function (adds an extra parameter!)
    log_likelihood_used = pints.GaussianKnownSigmaLogLikelihood(problem, [noise_used])

    # Create a uniform prior over both the parameters and the new noise variable

    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood_used, log_prior_used)

    # Choose starting points for 3 mcmc chains
    xs = [
        real_parameters,
        real_parameters * 1.01,
        real_parameters * 0.99,
    ]

    # Create mcmc routine with four chains
    mcmc = pints.MCMCController(log_posterior, 3, xs, method=pints.HaarioACMC)
    
    sample_size = 4000
    # Add stopping criterion
    mcmc.set_max_iterations(sample_size)

    # Start adapting after 1000 iterations
    mcmc.set_initial_phase_iterations(sample_size//4)

    # Disable logging mode
    mcmc.set_log_to_screen(False)

    # Run!
    print('Running...')
    chains = mcmc.run()
    print('Done!')
    s = sample_size//4+1
    #HMC: s = 1
    b = False
    while s < sample_size:
        chains_cut = chains[:,sample_size//4:s+1]
        rhat = pints.rhat(chains_cut)
        s+=1
        if rhat[0] < 1.05:
            b = True
            break
    print(s)
    return chains[0][s:][:, 0]

model = toy.LogisticModel()
real_parameters = [0.016, 500]
chain1 = run(model, real_parameters, 10,pints.UniformLogPrior([0.015, 400],[0.017, 600]))
chain2 = run(model, real_parameters, 1,pints.UniformLogPrior([0.015, 400],[0.017, 600]))
current_date_and_time_string = datetime.now()
bins=np.histogram(np.hstack((chain1,chain2)), bins=40)[1] 
plt.figure(figsize=(12,4),dpi=500)
plt.xlabel('r')
plt.hist(chain1,bins, alpha=0.5, label=r'posterior samples generated with correct likelihood $N(f(t),10^2)$')
plt.hist(chain2,bins, alpha=0.5, label=r'posterior samples generated with incorrect likelihood $N(f(t),1^2)$')
plt.title(r'Compare posterior samples generated with correct likelihood $N(f(t),10^2)$ and with incorrect likelihood $N(f(t),1^2)$ (Time series of length 50)')
plt.legend(loc='upper right')
plt.savefig('compare'+str(current_date_and_time_string)+'.png')


'''model = Model()
real_parameters = [500]
chain1 = run(model, real_parameters, 10,pints.UniformLogPrior([400],[600]))
chain2 = run(model, real_parameters, 10,pints.UniformLogPrior([200],[800]))
current_date_and_time_string = datetime.now()
bins=np.histogram(np.hstack((chain1,chain2)), bins=40)[1] 
plt.figure(figsize=(12,4),dpi=500)
plt.xlabel('k')
plt.hist(chain1,bins, alpha=0.5, label=r'posterior samples generated with correct prior $U([400,600])$')
plt.hist(chain2,bins, alpha=0.5, label=r'posterior samples generated with incorrect prior $U([200,800])$')
plt.title(r'Compare posterior samples generated with correct prior $U([400,600])$ and with incorrect prior $U([200,800])$ (Time series of length 50)')
plt.legend(loc='upper right')
plt.savefig('compare'+str(current_date_and_time_string)+'.png')'''




'''model = Model()
real_parameters = [400]
chain1 = run(model, real_parameters, 10,pints.GaussianLogPrior(400, 10))
chain2 = run(model, real_parameters, 10,pints.GaussianLogPrior(400, 1))
current_date_and_time_string = datetime.now()
bins=np.histogram(np.hstack((chain1,chain2)), bins=40)[1] 
plt.figure(figsize=(12,4),dpi=500)
plt.xlabel('k')
plt.hist(chain1,bins, alpha=0.5, label=r'posterior samples generated with correct prior $N(400,10^2)$')
plt.hist(chain2,bins, alpha=0.5, label=r'posterior samples generated with incorrect prior $N(400,1^2)$')
plt.title(r'Compare posterior samples generated with correct prior $N(400,10^2)$ and with incorrect prior $N(400,1^2)$ (Time series of length 50)')
plt.legend(loc='upper right')
plt.savefig('compare'+str(current_date_and_time_string)+'.png')'''