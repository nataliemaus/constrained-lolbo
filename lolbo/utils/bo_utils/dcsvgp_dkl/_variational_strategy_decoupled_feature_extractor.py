#!/usr/bin/env python3

from abc import ABC, abstractproperty
import torch 
from torch.nn import Module

# DCSVGP_DKL
from gpytorch import settings
from gpytorch.distributions import Delta, MultivariateNormal
from gpytorch.module import Module
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.utils.memoize import cached, clear_cache_hook


class _VariationalStrategyDecoupledFeatureExtractors(Module, ABC):
    """
    Abstract base class for all Variational Strategies.
    """

    def __init__(self, model, inducing_points_mean, inducing_points_covar, variational_distribution, learn_inducing_locations=True):
        super().__init__()
        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        self.register_inducing_points(inducing_points_mean, name="inducing_points_mean")
        self.register_inducing_points(inducing_points_covar, name="inducing_points_covar")

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))
        # self.register_buffer("variational_mean_initialized", torch.tensor(0))
        # self.register_buffer("variational_covar_initialized", torch.tensor(0))

    def register_inducing_points(self, inducing_points, learn_inducing_locations=True, name="inducing_points"):
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        if learn_inducing_locations:
            self.register_parameter(name=name, parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer(name, inducing_points)

    def _clear_cache(self):
        clear_cache_hook(self)

    def _expand_inputs(self, x, inducing_points):
        """
        Pre-processing step in __call__ to make x the same batch_shape as the inducing points
        """
        batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], x.shape[:-2])
        inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
        x = x.expand(*batch_shape, *x.shape[-2:])
        return x, inducing_points

    @abstractproperty
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`p( \mathbf u)`
        """
        raise NotImplementedError

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        return self._variational_distribution()

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        :param torch.Tensor x: Locations :math:`\mathbf X` to get the
            variational posterior of the function values at.
        :param torch.Tensor inducing_points: Locations :math:`\mathbf Z` of the inducing points
        :param torch.Tensor inducing_values: Samples of the inducing function values :math:`\mathbf u`
            (or the mean of the distribution :math:`q(\mathbf u)` if q is a Gaussian.
        :param ~gpytorch.lazy.LazyTensor variational_inducing_covar: If the distribuiton :math:`q(\mathbf u)`
            is Gaussian, then this variable is the covariance matrix of that Gaussian. Otherwise, it will be
            :attr:`None`.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`q( \mathbf f(\mathbf X))`
        """
        raise NotImplementedError

    def kl_divergence(self):
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.

        :rtype: torch.Tensor
        """
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence

    def __call__(self, x, prior=False, **kwargs):
        # print("First")
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x, **kwargs)
        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()
        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            # kwargs = {}
            # if not self.variational_mean_initialized:
            #     kwargs["initialize_mean"] = True
            # if not self.variational_covar_initialized:
            #     kwargs["initialize_covar"] = False
            # kwargs_init = {"initialize_covar": False}
            # try:
            #     a = self.covar_module_mean
            # except:
            #     kwargs_init["initialize_covar"] = True

            prior_dist = self.prior_distribution
            # self._variational_distribution.initialize_variational_distribution(prior_dist, **kwargs_init)
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            # the main delta distribution is initialized using the computed prior mean
            self.variational_params_initialized.fill_(1)
            # self.variational_mean_initialized.fill_(1)
            # self.variational_covar_initialized.fill_(1)

        # Ensure inducing_points and x are the same size
        inducing_points_mean = self.inducing_points_mean
        if inducing_points_mean.shape[:-2] != x.shape[:-2]:
            x, inducing_points_mean = self._expand_inputs(x, inducing_points_mean)
        inducing_points_covar = self.inducing_points_covar
        if inducing_points_covar.shape[:-2] != x.shape[:-2]:
            x, inducing_points_covar = self._expand_inputs(x, inducing_points_covar)


        # Get p(u)/q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f)
        if isinstance(variational_dist_u, MultivariateNormal):
            return super().__call__(
                x,
                inducing_points_mean,
                inducing_points_covar,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                **kwargs,
            )
        elif isinstance(variational_dist_u, Delta):
            return super().__call__(
                x, inducing_points_mean, inducing_points_covar, inducing_values=variational_dist_u.mean, variational_inducing_covar=None, **kwargs
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribuition ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution."
            )
