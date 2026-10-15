from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

from typing import Dict, Literal, Optional, Tuple, Union
from numpy.typing import ArrayLike, NDArray

from ..dataloader import ContactSurveyLoader
from ..analysis import ModelSummariser


def _compute_P_sa(dataloader: ContactSurveyLoader) -> Dict[str, np.ndarray]:
    """
    Compute stratified population sizes P^{s}_{a} from a ContactSurveyLoader.

    Combines marginal population sizes P_a from PopulationData with stratum
    proportions Q_{s,a} from StratificationData to produce P^{s}_{a} = Q_{s,a} * P_a.

    Parameters
    ----------
    dataloader : ContactSurveyLoader
        An instantiated ContactSurveyLoader with population and stratification data.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping stratum labels to population size arrays of shape (A,).
    """
    if dataloader.strat_data is None:
        raise ValueError(
            "_compute_P_sa requires a dataloader with strat_data. "
            "The predict pipeline only supports partially-stratified (genmix) models."
        )
    age_col = dataloader.col_map.pop_age or dataloader.col_map.pop_age_grp
    P_a = (
        dataloader.pop_data.groupby(age_col, observed=True)["P"]
        .sum()
        .sort_index()
        .values
    )
    Q_sa, labels = dataloader.strat_data._build_Q_and_labels(
        dataloader.strat_data.strat_var_cols
    )
    return {str(lab): Q_sa[i] * P_a for i, lab in enumerate(labels)}


def z_marginals(
    summariser: ModelSummariser,
    dataloader: ContactSurveyLoader,
) -> Dict[str, NDArray]:
    """
    Compute the expected partially stratified total contact counts z^s_ab.

    The counts are obtained as ``z^s_ab = m^s_ab * P^s_a``, where ``m^s_ab``
    are posterior samples of the partially stratified contact intensities
    and ``P^s_a`` the stratified source population sizes.  They are used to
    derive the feasible mixing bounds and to sample associativity fractions.

    Parameters
    ----------
    summariser : ModelSummariser
        A fitted partially stratified model summariser containing posterior
        samples of the contact intensities ``"cint"``.
    dataloader : ContactSurveyLoader
        An instantiated ContactSurveyLoader. Used to extract stratified source population sizes
        by combining PopulationData (total counts) with StratificationData (proportions).

    Returns
    -------
    Dict[str, NDArray]
        Dictionary mapping source stratum labels to contact count arrays of
        shape ``(S, A, A)``, where S is the number of posterior samples and
        A is the number of age groups.
    """

    # Extract posterior contact intensity samples
    cint_samples = summariser.get_posterior_samples("cint")

    # Extract source labels
    source_labs = [str(key).replace("->All", "") for key in cint_samples.keys()]

    # Extract stratified population sizes: Dict[str, NDArray] with shape (A,)
    P_sa = _compute_P_sa(dataloader)

    def z(lab: str) -> np.ndarray:
        m_s_ab = cint_samples[f"{lab}->All"]
        return m_s_ab * P_sa[lab][np.newaxis, :, np.newaxis]

    zs = {lab: z(lab) for lab in source_labs}

    return zs


def feasible_mixing_bounds(
    z_marginals: Dict[str, np.ndarray],
    return_eta: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Calculate feasible mixing bounds (Frechet-Hoeffding bounds) for every stratum pair.

    For each ``source->target`` pair, the fully stratified contact counts
    ``z^{st}_ab`` must be consistent with the partially stratified marginals
    ``z^s_ab`` (source side) and ``z^t_ba`` (target side, via reciprocity).
    This yields the Frechet-Hoeffding bounds

    .. math::

        \\max(0, z^t_{ba} - \\sum_{k \\neq s} z^k_{ab})
        \\le z^{st}_{ab} \\le
        \\min(z^t_{ba}, z^s_{ab}).

    Parameters
    ----------
    z_marginals : Dict[str, np.ndarray]
        Dictionary mapping stratum labels to partially stratified contact
        count arrays of shape ``(S, A, A)`` (output of :func:`z_marginals`).
    return_eta : bool, optional
        If ``False`` (default), return bounds on the contact counts
        ``z^{st}_ab``.  If ``True``, return bounds on the associativity
        fractions ``eta^{st}_ab = z^{st}_ab / z^s_ab``, clipped to ``[0, 1]``.
        Entries undefined due to division by zero are set to 0 (lower bound)
        and 1 (upper bound).

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        ``(lower, upper)`` where each dictionary maps ``"source->target"``
        stratum pairs to bound arrays of shape ``(S, A, A)``, where S is the
        number of posterior samples and A is the number of age groups.
    """
    labs = list(z_marginals.keys())
    min_dict = {}
    max_dict = {}
    for source_lab in labs:
        for target_lab in labs:

            # Calculate contact flows for source and target groups
            Z_s_ab = z_marginals[source_lab]
            Z_t_ba = z_marginals[target_lab].transpose((0, 2, 1))

            # Sum contact flows from all groups except source group
            Z = 0
            for key, matrix in z_marginals.items():
                if key == source_lab:
                    continue

                Z += matrix

            # Calculate Frechet-Hoeffding bounds
            if return_eta:
                with np.errstate(divide="ignore", invalid="ignore"):
                    eta_min = np.maximum(0, (Z_t_ba - Z) / Z_s_ab)
                    eta_max = np.minimum(1, Z_t_ba / Z_s_ab)

                    # Handle NaNs from division by zero
                    eta_min = np.nan_to_num(eta_min, nan=0.0)
                    eta_max = np.nan_to_num(eta_max, nan=1.0)

                    min_dict[f"{source_lab}->{target_lab}"] = eta_min
                    max_dict[f"{source_lab}->{target_lab}"] = eta_max
            else:
                Z_min = np.maximum(0, Z_t_ba - Z)
                Z_max = np.minimum(Z_t_ba, Z_s_ab)

                min_dict[f"{source_lab}->{target_lab}"] = Z_min
                max_dict[f"{source_lab}->{target_lab}"] = Z_max

    return min_dict, max_dict


def rtruncated_beta(
    a: ArrayLike,
    b: ArrayLike,
    lb: ArrayLike = 0.0,
    ub: ArrayLike = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate random samples from a truncated Beta distribution.

    Samples are drawn via inverse CDF transform: a uniform variate is
    mapped through the Beta CDF restricted to ``[lb, ub]``.

    Parameters
    ----------
    a : array_like
        Alpha (shape) parameter of the Beta distribution.  Must be positive.
    b : array_like
        Beta (shape) parameter of the Beta distribution.  Must be positive.
    lb : array_like, optional
        Lower bound of the truncation interval (default 0.0).
    ub : array_like, optional
        Upper bound of the truncation interval (default 1.0).
    rng : numpy.random.Generator or None, optional
        Random number generator instance.  If ``None`` (default), a new
        generator is created via ``np.random.default_rng()``.

    Returns
    -------
    numpy.ndarray
        Random samples with shape determined by broadcasting
        ``a``, ``b``, ``lb``, and ``ub`` together.
    """
    if rng is None:
        rng = np.random.default_rng()

    a, b, lb, ub = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        np.asarray(lb, dtype=float),
        np.asarray(ub, dtype=float),
    )

    Flb = beta.cdf(lb, a, b)
    Fub = beta.cdf(ub, a, b)
    u = Flb + (Fub - Flb) * rng.uniform(size=a.shape)

    return beta.ppf(u, a, b)


def rtruncated_dirichlet(
    concentration: ArrayLike,
    lb: ArrayLike = 0.0,
    ub: ArrayLike = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate random samples from a truncated Dirichlet distribution.

    Samples are drawn via a sequential conditional stick-breaking procedure.
    Each component is sampled from a truncated Beta distribution conditioned
    on previous components, with bounds adjusted to satisfy the element-wise
    constraints ``lb <= x <= ub`` and the simplex constraint ``sum(x) = 1``.

    Parameters
    ----------
    concentration : array_like
        Concentration (alpha) parameters of the Dirichlet distribution.
        The last axis is the Dirichlet dimension K.  Must be positive.
    lb : array_like, optional
        Element-wise lower bounds for each component (default 0.0).
        Must satisfy ``0 <= lb <= ub <= 1`` and ``sum(lb) <= 1``.
    ub : array_like, optional
        Element-wise upper bounds for each component (default 1.0).
        Must satisfy ``sum(ub) >= 1``.
    rng : numpy.random.Generator or None, optional
        Random number generator instance.  If ``None`` (default), a new
        generator is created via ``np.random.default_rng()``.

    Returns
    -------
    numpy.ndarray
        Random samples on the simplex with shape determined by broadcasting
        ``concentration``, ``lb``, and ``ub`` together.  The last axis has
        size K and sums to 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    concentration, lb, ub = np.broadcast_arrays(
        np.asarray(concentration, dtype=float),
        np.asarray(lb, dtype=float),
        np.asarray(ub, dtype=float),
    )

    K = concentration.shape[-1]  # last axis is the Dirichlet dimension
    if K == 1:
        return np.clip(np.ones_like(concentration), lb, ub)

    x = np.zeros_like(concentration)

    # Sample component K-2 first (marginal truncated Beta)
    lb_km1 = np.maximum(lb[..., K - 2], 1 - np.sum(ub, axis=-1) + ub[..., K - 2])
    ub_km1 = np.minimum(ub[..., K - 2], 1 - np.sum(lb, axis=-1) + lb[..., K - 2])
    x[..., K - 2] = rtruncated_beta(
        concentration[..., K - 2],
        np.sum(concentration, axis=-1) - concentration[..., K - 2],
        lb=lb_km1,
        ub=ub_km1,
        rng=rng,
    )

    # Sample K-3, K-4, ..., 0 via conditional stick-breaking
    for k in range(K - 3, -1, -1):
        R = 1 - np.sum(x[..., k + 1 :], axis=-1)

        ub_rest = np.sum(ub, axis=-1) - np.sum(ub[..., k : K - 1], axis=-1)
        lb_rest = np.sum(lb, axis=-1) - np.sum(lb[..., k : K - 1], axis=-1)

        z_lb = np.maximum(lb[..., k] / R, 1 - ub_rest / R)
        z_ub = np.minimum(ub[..., k] / R, 1 - lb_rest / R)

        beta_b = np.sum(concentration, axis=-1) - np.sum(
            concentration[..., k : K - 1], axis=-1
        )

        x[..., k] = (
            rtruncated_beta(concentration[..., k], beta_b, lb=z_lb, ub=z_ub, rng=rng)
            * R
        )

    x[..., K - 1] = 1 - np.sum(x[..., : K - 1], axis=-1)
    return x


def sample_eta(
    z_marg: Dict[str, np.ndarray],
    eta_lb: Dict[str, np.ndarray],
    eta_ub: Dict[str, np.ndarray],
    dataloader: ContactSurveyLoader,
    alpha: float = 1.0,
    external_eta: Dict[str, np.ndarray] | None = None,
    external_counts: Dict[str, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    """
    Sample an array of associativity fractions.

    For each source stratum, draws associativity fractions
    ``eta[source->target]`` from a truncated Dirichlet distribution whose
    concentration is proportional to the stratified population sizes.  The
    lower triangle (participant age >= contact age) is sampled directly;
    the strictly upper triangle is filled via the reciprocity identity,
    and diagonals of off-diagonal stratum pairs are corrected accordingly.

    Parameters
    ----------
    z_marg : Dict[str, np.ndarray]
        Dictionary mapping stratum labels to marginal contact count arrays
        of shape ``(S, A, A)`` (output of :func:`z_marginals`).
    eta_lb : Dict[str, np.ndarray]
        Lower Frechet bounds for each ``"source->target"`` pair,
        shape ``(S, A, A)``.
    eta_ub : Dict[str, np.ndarray]
        Upper Frechet bounds for each ``"source->target"`` pair,
        shape ``(S, A, A)``.
    dataloader : ContactSurveyLoader
        An instantiated ContactSurveyLoader.  Used to extract stratified population
        sizes via ``_compute_P_sa`` for the Dirichlet concentration.
    alpha : float, optional
        Dirichlet precision — scales the total concentration per age group
        (default 1.0). Higher values pull samples closer to the population-
        proportion prior.
    external_eta : Dict[str, np.ndarray] or None, optional
        External associativity fractions to add to the concentration,
        shape ``(A, A, K)`` per key.  If ``None``, ignored.
    external_counts : Dict[str, np.ndarray] or None, optional
        External contact counts to add to the concentration,
        shape ``(A, A, K)`` per key.  If ``None``, ignored.
    rng : numpy.random.Generator or None, optional
        Random number generator.  If ``None``, a new generator is created.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping ``"source->target"`` labels to associativity
        fraction arrays of shape ``(S, A, A)``.
    """
    strata_names = list(z_marg.keys())
    K = len(strata_names)
    S, A, _ = next(iter(eta_lb.values())).shape

    P_sa = _compute_P_sa(dataloader)
    # Normalize to proportions first, then scale by alpha so that
    # concentration.sum(axis=-1) == alpha for every age group.
    concentration = np.stack([P_sa[t] for t in strata_names], axis=-1)  # (A, K)
    concentration /= concentration.sum(axis=-1, keepdims=True)
    concentration = alpha * concentration

    tril_mask = np.tril(np.ones((A, A), dtype=bool))  # a >= b (sampled)
    striu_mask = ~tril_mask  # a < b  (filled via reciprocity)
    diag_idx = np.arange(A)

    eta_out = {}
    for source in strata_names:
        _lb = np.stack([eta_lb[f"{source}->{t}"] for t in strata_names], axis=-1)
        _ub = np.stack([eta_ub[f"{source}->{t}"] for t in strata_names], axis=-1)

        conc = concentration[np.newaxis, np.newaxis, :, :]
        if external_eta is not None:
            _ext = np.stack(
                [external_eta[f"{source}->{t}"] for t in strata_names], axis=-1
            )  # shape (A, A, K)
            conc = conc + _ext[np.newaxis, :, :, :]  # broadcast to (S, A, A, K)

        if external_counts is not None:
            _ext = np.stack(
                [external_counts[f"{source}->{t}"] for t in strata_names], axis=-1
            )  # shape (A, A, K)
            conc = conc + _ext[np.newaxis, :, :, :]  # broadcast to (S, A, A, K)

        eta_draw = rtruncated_dirichlet(conc, lb=_lb, ub=_ub, rng=rng)

        # Keep only lower triangle (a >= b)
        eta_draw *= tril_mask[np.newaxis, :, :, np.newaxis]

        for j, target in enumerate(strata_names):
            eta_out[f"{source}->{target}"] = eta_draw[..., j]

    # Fill strictly upper triangle via reciprocity
    for ell in strata_names:
        for k in strata_names:
            with np.errstate(divide="ignore", invalid="ignore"):
                fill = (
                    z_marg[k].transpose(0, 2, 1)
                    / z_marg[ell]
                    * eta_out[f"{k}->{ell}"].transpose(0, 2, 1)
                )
                fill = np.nan_to_num(fill, nan=0.0, posinf=0.0, neginf=0.0)
            eta_out[f"{ell}->{k}"] += fill * striu_mask[np.newaxis, :, :]

    # Fix diagonal for off-diagonal category pairs
    for i, ell in enumerate(strata_names):
        for j, k in enumerate(strata_names):
            if i <= j:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                diag_fix = (
                    z_marg[k][:, diag_idx, diag_idx]
                    / z_marg[ell][:, diag_idx, diag_idx]
                    * eta_out[f"{k}->{ell}"][:, diag_idx, diag_idx]
                )
                diag_fix = np.nan_to_num(diag_fix, nan=0.0, posinf=0.0, neginf=0.0)
            eta_out[f"{ell}->{k}"][:, diag_idx, diag_idx] = diag_fix

    return eta_out


def spectral_radius(
    matrices: Dict[str, NDArray],
    dataloader: Optional[ContactSurveyLoader] = None,
    method: Optional[str] = "generalised",
) -> Union[float, NDArray]:
    """Compute the spectral radius of the generalised or age-stratified matrix.

    If matrices contain 2D arrays (A, A), returns a scalar.
    If matrices contain 3D arrays (S, A, A), returns an array of shape (S,).

    Parameters
    ----------
    matrices : dict
        A dictionary mapping "source->target" to a 2D or 3D array
    dataloader : ContactSurveyLoader, optional
        An instantiated ContactSurveyLoader. Required if method="age". Used to extract
        stratified population sizes via StratificationData proportions and
        PopulationData totals.
    method : str, optional
        The method to use: "generalised" or "age". Default is "generalised"

    Returns
    -------
    float or np.ndarray
        The spectral radius of the combined matrix, either a scalar or an array of shape (S,) depending on the input.
    """
    cats = list(dict.fromkeys(k.split("->")[0] for k in matrices))
    first = next(iter(matrices.values()))
    batched = first.ndim == 3

    if method == "generalised":
        # For 2-D blocks: produces (K*A, K*A).
        # For 3-D blocks (S, A, A): np.block concatenates along the last two
        # axes in order, producing (S, K*A, K*A).
        M = np.block([[matrices[f"{s}->{t}"] for t in cats] for s in cats])
    elif method == "age":
        if dataloader is None:
            raise ValueError("dataloader is required for method='age'")
        P_sa = _compute_P_sa(dataloader)
        age_col = dataloader.col_map.pop_age or dataloader.col_map.pop_age_grp
        P_a = dataloader.pop_data.groupby(age_col)["P"].sum().sort_index().values
        M = np.zeros_like(matrices[f"{cats[0]}->{cats[0]}"])
        for source in cats:
            mat_partial = sum(matrices[f"{source}->{t}"] for t in cats)
            P_ka = P_sa[source]
            if batched:
                M += mat_partial * (P_ka / P_a)[None, :, None]
            else:
                M += mat_partial * (P_ka / P_a)[:, None]
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'generalised' or 'age'.")

    if batched:
        eigenvalues = np.linalg.eigvals(M)  # (S, N) complex
        return np.max(np.abs(eigenvalues), axis=1)  # (S,)
    else:
        eigenvalues = np.linalg.eigvals(M)
        return np.max(np.abs(eigenvalues))
