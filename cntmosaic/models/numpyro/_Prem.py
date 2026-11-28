"""NumPyro model mixin for Prem."""

from typing import Optional

import jax.numpy as jnp
from jax.typing import ArrayLike


class PremNumPyroMixin:
    """Carries the NumPyro ``model()`` for Prem."""

    def model(self, y: Optional[ArrayLike] = None) -> None:
        import numpyro
        from numpyro import distributions as dist
        from numpyro.handlers import plate

        from .distributions import IGMRF2D

        if self.K == 1:
            beta0 = numpyro.sample(
                "beta0", dist.Normal(-9.2, 2.5)
            )  # log(0.0001) approx -9.2
            tau = numpyro.sample("tau", dist.Gamma(2.0, 1.0))
            beta_cd = numpyro.sample(
                "beta_cd",
                IGMRF2D(
                    num_nodes=(self.C, self.D),
                    order=self.order,
                    cond_prec=tau,
                ),
            ).reshape((self.C, self.D))

            log_cint = numpyro.deterministic("log_cint", beta0 + beta_cd)

            if self.random_effects:
                mu_re = numpyro.sample("mu_re", dist.Normal(0.0, 1.0))
                tau_re = numpyro.sample("tau_re", dist.HalfNormal(1.0))
                with plate("random_effects", self.N):
                    sigma_re = numpyro.sample("sigma_re", dist.Normal(mu_re, tau_re))
                log_lambda = log_cint[self.cix, self.dix] + sigma_re[self.iix]
            else:
                log_lambda = log_cint[self.cix, self.dix]

        else:
            with plate("strata", self.K):
                beta0 = numpyro.sample("beta0", dist.Normal(-9.2, 2.5))
                tau = numpyro.sample("tau", dist.Gamma(2.0, 1.0))

            beta_cd = numpyro.sample(
                "beta_cd",
                IGMRF2D(
                    num_nodes=(self.C, self.D),
                    order=self.order,
                    cond_prec=tau,
                )
                .expand([self.K])
                .to_event(1),
            ).reshape((self.K, self.C, self.D))

            log_cint = numpyro.deterministic(
                "log_cint", beta0[:, jnp.newaxis, jnp.newaxis] + beta_cd
            )

            if self.random_effects:
                # Participant-level random effects, as in the unstratified
                # model: one effect per participant, shared by all of their
                # observations (which may span several stratum pairs), with
                # common hyperparameters. (Hyperparameters per stratum pair
                # were indexed per observation, not per participant, and are
                # confounded with beta0 in any case.)
                mu_re = numpyro.sample("mu_re", dist.Normal(0.0, 1.0))
                tau_re = numpyro.sample("tau_re", dist.HalfNormal(1.0))
                with plate("random_effects", self.N):
                    sigma_re = numpyro.sample("sigma_re", dist.Normal(mu_re, tau_re))
                log_lambda = log_cint[self.six, self.cix, self.dix] + sigma_re[self.iix]
            else:
                log_lambda = log_cint[self.six, self.cix, self.dix]

        lambda_param = jnp.exp(log_lambda)
        with plate("data", len(self.y)):
            numpyro.sample("obs", dist.Poisson(lambda_param), obs=y)
