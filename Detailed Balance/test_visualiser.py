import matplotlib.pyplot as plt
import numpy as np
import pints
import pints.plot
import pints.toy as toy
import numpy as np
import matplotlib.pyplot as plt
from VisualisationController import MCMCVisualiser 

def detailedBalanceMetricplot(log_pdf, xs, start,end,step,MCMCmethod):
    metriclist = []
    for i in range(start,end,step):
        print(i)
        mcmc = MCMCVisualiser(log_pdf, len(xs), xs, method=MCMCmethod)
        mcmc.set_max_iterations(i)
        #for sampler in mcmc.samplers():
            #sampler.set_leapfrog_step_size(0.5)
        metric = mcmc.detailedBalanceMatrix_1D_metric(10,xs)
        metriclist.append(metric)
    plt.figure(figsize=(8,4),dpi=500)
    plt.ylabel('Detailed Balance metric')
    plt.xlabel('Sample size')
    plt.plot(list(range(start,end,step)),metriclist)
    plt.savefig('metric over sample size.png')


# Load a forward model
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
model=Model()

# Create some toy data
real_parameters = np.array([500])
times = np.linspace(0, 1000, 1000)
org_values = model.simulate(real_parameters, times)

# Add noise
noise = 10
values = org_values + np.random.normal(0, noise, org_values.shape)

# Create an object with links to the model and time series
problem = pints.SingleOutputProblem(model, times, values)

# Create a log-likelihood function
log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, noise)

# Create a uniform prior over the parameters
log_prior = pints.UniformLogPrior(
    [400],
    [600]
)

# Create a posterior log-likelihood (log(likelihood * prior))
log_posterior = pints.LogPosterior(log_likelihood, log_prior)

# Choose starting points for 3 mcmc chains
xs = [
    real_parameters * 1.1,
    real_parameters * 0.9,
    real_parameters * 1.15,
]

# Choose a covariance matrix for the proposal step
#sigma0 = np.abs(real_parameters) * 5e-4

# Create mcmc routine
mcmc = MCMCVisualiser(log_posterior, 3, xs, method=pints.MetropolisRandomWalkMCMC)

# Add stopping criterion
mcmc.set_max_iterations(20000)

mcmc.detailedBalanceMatrix_1D_result(10,xs)

detailedBalanceMetricplot(log_posterior, [real_parameters * 1.1], 200,60000,1000,pints.MetropolisRandomWalkMCMC)
