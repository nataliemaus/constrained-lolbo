import sys
import time
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from lolbo.utils.bo_utils.dcsvgp_dkl.variational_strategy_nn_mean_predictor import VariationalStrategyNNMeanPredictor
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

class NNSVGP(ApproximateGP):
    def __init__(self, inducing_points, kernel_type='se', 
        learn_inducing_locations=True, ard_num_dims=None,
        hidden_size=128,
        ):
        
        NN_mean_predictor =  MLP(input_size=inducing_points.size(-1), hidden_size=hidden_size, output_size=1)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))

        variational_strategy = VariationalStrategyNNMeanPredictor(self, inducing_points, 
                                                   variational_distribution, NN_mean_predictor,
                                                   learn_inducing_locations=learn_inducing_locations)
        super(NNSVGP, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if kernel_type == 'se':
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        elif kernel_type == 'matern1/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern3/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5,ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern5/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=ard_num_dims)
         
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode 
        # self.model.eval() 
        self.likelihood.eval()
        dist = self.likelihood(self(X))
        return GPyTorchPosterior(mvn=dist)
