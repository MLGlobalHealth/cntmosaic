from typing import Any, Dict, Optional

import jax.numpy as jnp

from ..dataloader import ContactSurveyLoader
from ._GenMix import GenMix
from .numpyro import AgeMixFCNumPyroMixin
from .numpyro.priors import Hill, PSpline2D


class AgeMixFC(AgeMixFCNumPyroMixin, GenMix):
    """
    Age-only mixing model with fine participant age and coarse contact age resolution.

    AgeMixFC (Age Mixing, Fine-Coarse) estimates social contact matrices at
    single-year age resolution from contact survey data where participant ages are
    at 1-year resolution but contact ages are reported in coarse age groups
    (e.g., 0-4, 5-9, etc.). The model uses smooth priors and an age aggregation
    mechanism to recover fine-age contact rates from the coarse contact age data.

    The model assumes:
    1. Contact rates are smooth functions of participant and contact ages
    2. Rate consistency: forward and reciprocal contact rates are balanced by population
    3. Coarse age groups are aggregations of underlying fine-age contacts
    4. Observation model: Contacts follow Poisson or Negative Binomial distribution

    Mathematical Model
    ------------------
    For each observed contact y_i from participant age a_i to contact age group [b_l, b_u):

        log(rate[a, b]) = β₀ + f(a, b)  for all fine ages b
        log(contact_intensity[a, b]) = log(rate[a, b]) + log(P[b])
        log(μ_i) = logsumexp_{b ∈ [b_l, b_u)} log(cint[a_i, b]) + log(N_i) + log(S_i) + η_i
        y_i ~ Poisson(μ_i) or NegativeBinomial2(μ_i, φ)

    where:
    - β₀: baseline contact rate
    - f(a, b): smooth 2D age-age function (from prior)
    - P[b]: population proportion in age group b
    - N_i: survey sample size for observation i
    - S_i: additional offset (e.g., for different settings)
    - η_i: repeat interview effect (if applicable)
    - φ: overdispersion parameter (negative binomial only)
    - logsumexp: numerically stable log-sum-exp over age group

    The key difference from AgeMixFF is the age aggregation step: instead of using
    a single contact age index bid, we sum over all fine ages within the coarse
    age group using the index_mask_logsumexp function with aid_exp and bid_pad arrays.

    Parameters
    ----------
    dataloader : ContactSurveyLoader
        ContactSurveyLoader object containing processed contact data with columns:
        - y: observed contact counts
        - aid: participant age indices (1-year resolution)
        - aid_exp: expanded participant age indices for aggregation
        - bid_pad: padded contact age indices for coarse age groups
        - cid: coarse contact age group codes (categorical)
        - log_N: log of survey sample sizes
        - log_P: log of population age distribution
        - log_V: log of setting-specific offsets (optional)
        - rid: repeat interview indicators (optional)
    priors : dict, optional
        Dictionary of prior specifications. If None, uses default_priors.
        Must contain:
        - 'rate': Prior2D object for the age-age contact rate function
          (e.g., HSGP2D, Spline2D, PSpline2D, IGMRF2D)
    likelihood : str, default='negbin'
        Observation likelihood:
        - 'negbin': Negative binomial (recommended for overdispersed count data)
        - 'poisson': Poisson (assumes mean = variance)
        - 'quasipoisson': Quasi-Poisson; V(mu) = mu. Dispersion psi fixed at
          1.0 by default; calibrate with set_quasi_dispersion(psi).
        - 'quasinegbin': Quasi-negative-binomial; V(mu) = mu + mu^2/k.
          Concentration k is sampled from a prior; dispersion psi fixed at 1.0
          by default, calibrate with set_quasi_dispersion(psi).

    Attributes
    ----------
    default_priors : dict
        Class-level default prior specifications using PSpline2D with difference-age grid.
    y : jax.Array
        Observed contact counts, shape (n_obs,)
    aid : jax.Array
        Participant age indices (1-year resolution), shape (n_obs,)
    aid_exp : jax.Array
        Expanded participant age indices for aggregation, shape (n_obs, max_int_length)
    bid_pad : jax.Array
        Padded contact age indices for coarse groups, shape (n_obs, max_int_length)
    log_N : jax.Array
        Log of sample sizes, shape (n_obs,)
    log_P : jax.Array
        Log of population age distribution, shape (1, A)
    log_V : jax.Array
        Log of offsets, shape (n_obs,)
    rid : jax.Array, optional
        Repeat interview indicators, shape (n_obs,)
    hill : Hill, optional
        Hill prior for repeat interview effects

    Raises
    ------
    ValueError
        If required data columns (aid_exp, bid_pad, log_N, log_P) are missing.

    Notes
    -----
    The age aggregation mechanism:
    - aid_exp: Repeats participant age indices to match the maximum coarse age group width
    - bid_pad: Contains all fine ages within each coarse group, padded with -1
    - index_mask_logsumexp: Sums over valid ages (non-negative) in log space
    - This allows the model to estimate fine-age contact rates while handling coarse data

    References
    ----------
    Shozen Dan et al., "Estimating fine age structure and time trends in
    human contact patterns from coarse contact data: The Bayesian rate consistency model",
    PLoS Computational Biology. (2023)

    Examples
    --------
    >>> from cntmosaic.dataloader import ContactSurveyLoader, CoordToColumns
    >>> from cntmosaic.models import AgeMixFC
    >>> from cntmosaic.models.numpyro.priors import HSGP2D
    >>> from jax.random import PRNGKey
    >>> import pandas as pd
    >>>
    >>> # Set up dataloader with coarse contact ages
    >>> col_map = CoordToColumns(
    ...     age_part="age_part",
    ...     age_grp_cnt="age_grp_cnt",  # Note: coarse age groups
    ...     age_pop="age",
    ...     P="P"
    ... )
    >>> dataloader = ContactSurveyLoader(df_part, df_cnt, df_age_dist, col_map=col_map)
    >>>
    >>> # Use default priors or specify custom ones
    >>> priors = {
    ...     "rate": HSGP2D(
    ...         grid_type="diff-age",
    ...         prior_type="global",
    ...         M=40,               # basis functions per axis (sets lenscale_min)
    ...         lenscale_scale=0.5  # half-normal scale (sets lenscale_max and C)
    ...     )
    ... }
    >>>
    >>> # Initialize model
    >>> model = AgeMixFC(dataloader, priors, likelihood="negbin")
    >>>
    >>> # Run MCMC inference
    >>> model.run_inference_mcmc(
    ...     PRNGKey(42),
    ...     num_samples=1000,
    ...     num_warmup=1000,
    ...     num_chains=4
    ... )
    >>>
    >>> # Access posterior samples
    >>> samples = model._mcmc_result.get_samples()
    >>> baseline_posterior = samples['baseline']
    >>> log_rate_posterior = samples['log_rate']  # shape (n_samples, A, A)

    See Also
    --------
    AgeMixFF : Age-only mixing model with fine-age resolution for both ages
    GenMixFC : Generalised mixing model with fine participant age and coarse contact age
    ContactSurveyLoader : Data preprocessing utilities
    HSGP2D : Hilbert Space Gaussian Process prior
    """

    # Default priors
    default_priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global", M=15)
    }

    def __init__(
        self,
        dataloader: ContactSurveyLoader,
        priors: Optional[Dict[str, Any]] = None,
        likelihood: str = "negbin",
        backend: Optional[Any] = None,
    ) -> None:
        """
        Initialize AgeMixFC model with fine participant age and coarse contact age.

        Parameters
        ----------
        dataloader : ContactSurveyLoader
            Preprocessed contact data with 1-year participant age and coarse contact age groups.
        priors : Optional[Dict[str, Any]], default=None
            Prior specifications. If None, uses default_priors.
            Must contain 'rate' key with a Prior2D object.
        likelihood : str, default='negbin'
            Observation likelihood ('negbin', 'poisson', 'quasipoisson', or 'quasinegbin').
        backend : InferenceBackend, optional
            Pluggable inference engine (default: NumPyroBackend).
        """
        # Merge user priors with defaults (user priors take precedence)
        effective_priors = self.default_priors.copy()
        if priors is not None:
            effective_priors.update(priors)

        # Initialize parent class with merged priors
        super().__init__(dataloader, effective_priors, likelihood, backend=backend)

        # Convert data to JAX arrays for efficient computation
        self.y = jnp.array(self.data.y)
        self.log_N = jnp.array(self.data.log_N)
        self.log_P = jnp.array(self.data.log_P)

        # Optional offset for different settings (e.g., home, work, school)
        self.log_V = (
            jnp.array(self.data.log_V)
            if self.data.log_V is not None
            else jnp.zeros_like(self.y)
        )

        # Age aggregation indices for coarse-to-fine refinement
        self.aid = jnp.array(self.data.aid, dtype=jnp.int32)
        self.aid_exp = jnp.array(self.data.aid_exp, dtype=jnp.int32)
        self.bid_pad = jnp.array(self.data.bid_pad, dtype=jnp.int32)

        # Optional repeat interview effect
        if self.data.rid is not None:
            self.rid = jnp.array(self.data.rid, dtype=jnp.int32)
            self.hill = Hill(max_value=int(self.data.rid.max()))

