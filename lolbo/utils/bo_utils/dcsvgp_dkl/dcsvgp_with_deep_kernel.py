import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
# from gpytorch.variational import VariationalStrategy
from lolbo.utils.bo_utils.dcsvgp_dkl.variational_strategy_decoupled_feature_extractor import VariationalStrategyDecoupledFeatureExtractors as RegularVariationalStrategyDecoupledFeatureExtractors
from lolbo.utils.bo_utils.dcsvgp_dkl.variational_strategy_decoupled_feature_extractor_shared_u import VariationalStrategyDecoupledFeatureExtractors as SharedInducingVariationalStrategyDecoupledFeatureExtractors


from botorch.posteriors.gpytorch import GPyTorchPosterior
import torch
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class _LinearBlock(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim, swish):
        if swish:
            super().__init__(OrderedDict([
                ("fc", torch.nn.Linear(input_dim, output_dim)),
                ("swish", Swish()),
            ]))
        else:
            super().__init__(OrderedDict([
                ("fc", torch.nn.Linear(input_dim, output_dim)),
                ("norm", torch.nn.BatchNorm1d(output_dim)),
                ("relu", torch.nn.ReLU(True)),
            ]))


class DenseNetwork(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dims, swish=True):
        prev_dims = [input_dim] + list(hidden_dims[:-1])
        layers = OrderedDict([
            (f"hidden{i + 1}", _LinearBlock(prev_dim, current_dim, swish=swish))
            for i, (prev_dim, current_dim) in enumerate(zip(prev_dims, hidden_dims))
        ])
        self.output_dim = hidden_dims[-1]

        super().__init__(layers)


# gp model with deep kernel
class DCSVGP_DKL(ApproximateGP):
    def __init__(
        self, 
        inducing_points, 
        likelihood, 
        hidden_dims=(256, 256),
        shared_inducing_pts=False,
    ):
        feature_extractor_mean = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
        )

        feature_extractor_covar = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
        )   

        inducing_points_mean = feature_extractor_mean(inducing_points)
        inducing_points_covar = feature_extractor_covar(inducing_points)

        variational_distribution = CholeskyVariationalDistribution(inducing_points_covar.size(0))
        if shared_inducing_pts:
            variational_strategy = RegularVariationalStrategyDecoupledFeatureExtractors(
                self,
                inducing_points_mean,
                inducing_points_covar,
                variational_distribution,
                learn_inducing_locations=True
            )
        else:
            variational_strategy = SharedInducingVariationalStrategyDecoupledFeatureExtractors(
                self,
                inducing_points_mean,
                inducing_points_covar,
                variational_distribution,
                learn_inducing_locations=True
            )
        super(DCSVGP_DKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor_mean = feature_extractor_mean
        self.feature_extractor_covar = feature_extractor_covar

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x_mean = self.feature_extractor_mean(x)
        x = self.feature_extractor_covar(x)
        kwargs["x_mean"] = x_mean
        # print("Calling Variational strategy")
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)


if __name__ == "__main__":
    # example to load model and trian on random data 
    N = 100
    train_bsz = 10
    n_epochs = 3
    n_inducing = 10
    dim = 32
    train_x = torch.randn(N, dim)
    train_y = torch.randn(N,1)

    # Initialize model: 
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
    """ NOTE: hidden_dims is a tuple giving the number of nodes 
        in each hidden layer in the neural net 
    """
    model = DCSVGP_DKL(
        inducing_points=train_x[0:n_inducing,:].cuda(), 
        likelihood=likelihood,
        hidden_dims=(16, 16) 
    ).cuda()

    # initialize mll: 
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_x.size(-2))
  
    # train model: 
    model = model.train() 
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr':0.001} ], lr=0.001)
    train_dataset = TensorDataset(train_x.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for e in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda()).sum() 
            loss.backward()
            optimizer.step()
        print(f"epoch: {e}, loss: {loss.item()}")
    
# Expected output: 
# epoch: 0, loss: 16.277769088745117
# epoch: 1, loss: 20.076805114746094
# epoch: 2, loss: 20.745248794555664
# (loss numbers may vary due to randomness)