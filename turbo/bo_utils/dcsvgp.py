import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
# from gpytorch.variational import VariationalStrategyDecoupledConditionals
from .variational_strategy_decoupled_conditionals import VariationalStrategyDecoupledConditionals
from .variational_strategy_decoupled_conditionals_v2 import VariationalStrategyDecoupledConditionalsV2
# from torch.utils.data import TensorDataset, DataLoader
from botorch.posteriors.gpytorch import GPyTorchPosterior

class DCSVGP(ApproximateGP):
    def __init__(self, inducing_points):
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))

        covar_module_mean = gpytorch.kernels.RBFKernel()
        variational_strategy = VariationalStrategyDecoupledConditionals(self, inducing_points, 
                                                 variational_distribution, covar_module_mean)
        super(DCSVGP, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.covar_module = gpytorch.kernels.RBFKernel()
    
    def posterior(
        self, X, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        self.likelihood.eval()
        dist = self.forward(X)
        if observation_noise:
            dist = self.likelihood(dist, *args, **kwargs)

        return GPyTorchPosterior(mvn=dist)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# DCSVGP_V2 (TRAIN WITH BETA 2'S)
class DCSVGP_V2(ApproximateGP):
    def __init__(self, inducing_points):
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))

        covar_module_mean = gpytorch.kernels.RBFKernel()
        variational_strategy = VariationalStrategyDecoupledConditionalsV2(self, inducing_points, 
                                                 variational_distribution, covar_module_mean)
        super(DCSVGP_V2, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.covar_module = gpytorch.kernels.RBFKernel()
    
    def posterior(
        self, X, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        self.likelihood.eval()
        dist = self.forward(X, **kwargs)
        if observation_noise:
            dist = self.likelihood(dist, *args, **kwargs)

        return GPyTorchPosterior(mvn=dist)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Baseline SVGP (OR PPGPR) Model
class BaselineSVGP(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, 
                                                   variational_distribution)
        super(BaselineSVGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.covar_module = gpytorch.kernels.RBFKernel()
    
    def posterior(
        self, X, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        self.likelihood.eval()
        dist = self.forward(X)
        if observation_noise:
            dist = self.likelihood(dist, *args, **kwargs)

        return GPyTorchPosterior(mvn=dist)
         
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# # SVGP:
# if mll_type == "ELBO":
#     mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_y.size(0))
# # PPGPR: 
# elif mll_type == "PLL":
#     mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=train_y.size(0))