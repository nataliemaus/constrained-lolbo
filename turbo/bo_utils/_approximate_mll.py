#!/usr/bin/env python3

from abc import ABC, abstractmethod

import torch

from gpytroch.mlls.marginal_log_likelihood import MarginalLogLikelihood

import time

class _ApproximateMarginalLogLikelihood(MarginalLogLikelihood, ABC):
    r"""
    An approximate marginal log likelihood (typically a bound) for approximate GP models.
    We expect that :attr:`model` is a :obj:`gpytorch.models.ApproximateGP`.

    Args:
        :attr:`likelihood` (:obj:`gpytorch.likelihoods.Likelihood`):
            The likelihood for the model
        :attr:`model` (:obj:`gpytorch.models.ApproximateGP`):
            The approximate GP model
        :attr:`num_data` (int):
            The total number of training data points (necessary for SGD)
        :attr:`beta` (float - default 1.):
            A multiplicative factor for the KL divergence term.
            Setting it to 1 (default) recovers true variational inference
            (as derived in `Scalable Variational Gaussian Process Classification`_).
            Setting it to anything less than 1 reduces the regularization effect of the model
            (similarly to what was proposed in `the beta-VAE paper`_).
        :attr:`combine_terms` (bool):
            Whether or not to sum the expected NLL with the KL terms (default True)
    """

    def __init__(self, likelihood, model, num_data, beta=1.0, combine_terms=True, alpha=-1):
        super().__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data
        self.beta = beta
        self.alpha = alpha

    @abstractmethod
    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        raise NotImplementedError

    def forward(self, approximate_dist_f, target, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            :attr:`approximate_dist_f` (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        log_likelihood = self._log_likelihood_term(approximate_dist_f, target, **kwargs).div(num_batch)
        kl_divergence = self.model.variational_strategy.kl_divergence(**kwargs).div(self.num_data / self.beta)
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.alpha > 0:
            penalty = self.alpha*(self.model.covar_module.lengthscale - self.model.variational_strategy.covar_module_mean.lengthscale)**2
        else:
            penalty = 0
            
        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss - penalty
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss, penalty
            else:
                return log_likelihood, kl_divergence, log_prior, penalty
