from __future__ import annotations

import warnings
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta

from ..analysis import ModelSummariser
from ..dataloader import ContactSurveyLoader


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
        dataloader.strat_data.strat_var_cols  # type: ignore
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
    source_labs = [str(key).replace("->All", "") for key in cint_samples.keys()]  # type: ignore

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
    mapped through the Beta CDF restricted to ``[lb, ub]``.  Intervals in the
    upper tail (``CDF(lb) > 0.5``) are inverted through the survival function
    instead, for precision.  If the interval holds numerically zero
    probability mass, the bound nearest the Beta mean is returned, which is
    the limit of the truncated distribution.

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

    U = rng.uniform(size=a.shape)
    Flb = beta.cdf(lb, a, b)
    upper = Flb > 0.5  # interval in the upper tail: invert via the survival function
    x = np.empty_like(a)

    lo = ~upper
    Fub = beta.cdf(ub[lo], a[lo], b[lo])
    mass_lo = Fub - Flb[lo]
    x[lo] = beta.ppf(Flb[lo] + mass_lo * U[lo], a[lo], b[lo])

    Slb = beta.sf(lb[upper], a[upper], b[upper])
    Sub = beta.sf(ub[upper], a[upper], b[upper])
    mass_up = Slb - Sub
    x[upper] = beta.isf(Sub + mass_up * U[upper], a[upper], b[upper])

    mass = np.empty_like(a)
    mass[lo], mass[upper] = mass_lo, mass_up
    x = np.where(mass > 0, x, a / (a + b))

    return np.clip(x, lb, ub)


def rtruncated_dirichlet(
    concentration: ArrayLike,
    lb: ArrayLike = 0.0,
    ub: ArrayLike = 1.0,
    rng: np.random.Generator | None = None,
    n_sweeps: int = 20,
) -> np.ndarray:
    """
    Generate random samples from a truncated Dirichlet distribution.

    Draws ``x ~ Dirichlet(concentration)`` restricted to the box
    ``lb <= x <= ub`` on the simplex ``sum(x) = 1``.

    For ``K = 2`` the draw is exact: ``x_0`` follows a truncated Beta on the
    interval implied by both components' bounds.  For ``K >= 3`` a
    pairwise-coordinate Gibbs sampler is used.  Starting from a feasible
    point, each sweep visits pairs ``(i, j)`` in random order, holds
    ``c = x_i + x_j`` fixed and redraws ``x_i / c`` from its exact
    conditional, a Beta(``alpha_i``, ``alpha_j``) truncated so both
    components stay within their bounds.  The chain leaves the truncated
    Dirichlet invariant, so draws are exact in the limit of many sweeps.
    Each element of the batch runs its own chain, so draws are mutually
    independent.

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
    n_sweeps : int, optional
        Number of Gibbs sweeps for ``K >= 3`` (default 20).  Each sweep makes
        K truncated Beta draws per batch element, so the cost grows as
        ``n_sweeps * K``.  Ignored for ``K <= 2``.

    Returns
    -------
    numpy.ndarray
        Random samples on the simplex with shape determined by broadcasting
        ``concentration``, ``lb``, and ``ub`` together.  The last axis has
        size K and sums to 1.

    Raises
    ------
    ValueError
        If the bounds are infeasible: ``lb > ub``, bounds outside
        ``[0, 1]``, ``sum(lb) > 1`` or ``sum(ub) < 1``.
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

    tol = 1e-8
    if (
        np.any(lb > ub + tol)
        or np.any(lb < -tol)
        or np.any(ub > 1 + tol)
        or np.any(lb.sum(axis=-1) > 1 + tol)
        or np.any(ub.sum(axis=-1) < 1 - tol)
    ):
        raise ValueError(
            "Infeasible bounds: require 0 <= lb <= ub <= 1, sum(lb) <= 1 "
            "and sum(ub) >= 1 along the last axis."
        )

    if K == 2:
        x0 = rtruncated_beta(
            concentration[..., 0],
            concentration[..., 1],
            lb=np.maximum(lb[..., 0], 1 - ub[..., 1]),
            ub=np.minimum(ub[..., 0], 1 - lb[..., 1]),
            rng=rng,
        )
        return np.stack([x0, 1 - x0], axis=-1)

    # Feasible starting point: move every component the same fraction w of
    # the way from lb to ub, with w chosen so the components sum to one.
    width = ub - lb
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (1 - lb.sum(axis=-1, keepdims=True)) / width.sum(axis=-1, keepdims=True)
    x = lb + np.nan_to_num(w, nan=0.0) * width

    for _ in range(n_sweeps):
        perm = rng.permutation(K)
        for i, j in zip(perm, np.roll(perm, 1)):
            c = x[..., i] + x[..., j]
            lo = np.maximum(lb[..., i], c - ub[..., j])
            hi = np.minimum(ub[..., i], c - lb[..., j])
            with np.errstate(divide="ignore", invalid="ignore"):
                u_lb = np.clip(lo / c, 0.0, 1.0)
                u_ub = np.clip(hi / c, 0.0, 1.0)
            u = rtruncated_beta(
                concentration[..., i],
                concentration[..., j],
                lb=np.nan_to_num(u_lb, nan=0.0),
                ub=np.nan_to_num(u_ub, nan=0.0),
                rng=rng,
            )
            x_i = np.clip(u * c, lo, hi)
            x_i = np.where(c > 0, x_i, 0.0)
            x[..., i] = x_i
            x[..., j] = c - x_i

    return x


def _slice_step(
    base: np.ndarray,
    sign: np.ndarray,
    weight: np.ndarray,
    rng: np.random.Generator,
    max_shrink: int = 100,
) -> np.ndarray:
    """One shrinkage slice-sampling update of a step ``d`` along a move.

    The move changes m table entries to ``base + sign * d`` and the log
    density is ``sum_j weight_j * log(base_j + sign_j * d)``.

    Parameters
    ----------
    base : np.ndarray
        Current values of the entries, shape ``(..., m)``.
    sign : np.ndarray
        Direction of each entry, +1 or -1, shape ``(m,)``.
    weight : np.ndarray
        Exponent of each entry, broadcastable to ``base``.
    rng : numpy.random.Generator
        Random number generator.
    max_shrink : int, optional
        Maximum number of shrinkage steps (default 100).

    Returns
    -------
    np.ndarray
        Step ``d`` of shape ``base.shape[:-1]``; 0 where the move has no room.
    """
    lo = np.max(np.where(sign > 0, -base, -np.inf), axis=-1)
    hi = np.min(np.where(sign < 0, base, np.inf), axis=-1)

    def log_density(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = weight * np.log(base + sign * d[..., np.newaxis])
        return np.sum(np.where(weight == 0, 0.0, terms), axis=-1)

    todo = hi > lo
    d = np.zeros(lo.shape)
    y = np.where(todo, log_density(d), 0.0) - rng.exponential(size=d.shape)
    left = np.where(todo, lo, 0.0)
    right = np.where(todo, hi, 0.0)
    for _ in range(max_shrink):
        if not todo.any():
            break
        prop = left + (right - left) * rng.uniform(size=d.shape)
        ok = todo & (log_density(prop) > y)
        d = np.where(ok, prop, d)
        todo &= ~ok
        below = todo & (prop < 0)
        left = np.where(below, prop, left)
        right = np.where(todo & ~below, prop, right)
    return d


def sample_eta(
    z_marg: Dict[str, np.ndarray],
    eta_lb: Dict[str, np.ndarray],
    eta_ub: Dict[str, np.ndarray],
    dataloader: ContactSurveyLoader,
    alpha: float = 2.0,
    external_eta: Dict[str, np.ndarray] | None = None,
    external_counts: Dict[str, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
    n_sweeps: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Sample associativity fractions satisfying reciprocity and sum-to-one.

    The fully stratified counts are ``Z^{st}_ab = z^s_ab * eta^{st}_ab``.
    They must satisfy, exactly,

    - reciprocity: ``Z^{st}_ab = Z^{ts}_ba``, and
    - sum-to-one: ``sum_t eta^{st}_ab = 1`` in every cell.

    Both constraints only link a cell ``(a, b)`` with its mirror ``(b, a)``.
    For ``a > b`` the counts ``X[s, t] = Z^{st}_ab`` form a K x K table whose
    row sums are ``z^s_ab`` and whose column sums are ``z^t_ba``; the mirror
    cell holds ``X^T``.  On the diagonal ``X`` is symmetric with row sums
    ``z^s_aa``.  Every non-negative table with these sums satisfies both
    constraints, and its entries lie within the Fréchet bounds.

    The prior is a Dirichlet on each source's fractions at both cells of a
    pair, with concentration ``alpha * Q^t_b`` (the target stratum's
    population share at the contact age) plus any external terms,
    restricted to valid tables.  Tables are sampled by MCMC: starting from
    the independence table, each sweep applies every margin-preserving 2 x 2
    swap move (symmetric moves on the diagonal), drawing each step by slice
    sampling.  Every (posterior sample, age pair) runs its own chain.

    Parameters
    ----------
    z_marg : Dict[str, np.ndarray]
        Dictionary mapping stratum labels to partially stratified contact
        count arrays of shape ``(S, A, A)`` (output of :func:`z_marginals`).
        The totals ``sum_s z^s_ab`` must be reciprocal in ``(a, b)``.
    eta_lb : Dict[str, np.ndarray]
        Lower Fréchet bounds for each ``"source->target"`` pair.  Not used:
        the table constraints imply them.  Kept for backward compatibility.
    eta_ub : Dict[str, np.ndarray]
        Upper Fréchet bounds for each ``"source->target"`` pair.  Not used,
        as for ``eta_lb``.
    dataloader : ContactSurveyLoader
        An instantiated ContactSurveyLoader.  Used to extract stratified
        population sizes via ``_compute_P_sa`` for the Dirichlet concentration.
    alpha : float, optional
        Dirichlet precision (default 2.0).  The concentration of source s on
        target t at cell ``(a, b)`` is ``alpha * Q^t_b``.  Because reciprocity
        ties ``eta^{st}_ab`` to ``eta^{ts}_ba``, the prior is only proper when
        the two concentrations sum to more than 1 for every such pair; a
        ``ValueError`` reports the minimum ``alpha`` otherwise.
    external_eta : Dict[str, np.ndarray] or None, optional
        External associativity fractions to add to the concentration,
        shape ``(A, A)`` per ``"source->target"`` key.  If ``None``, ignored.
    external_counts : Dict[str, np.ndarray] or None, optional
        External contact counts to add to the concentration,
        shape ``(A, A)`` per ``"source->target"`` key.  If ``None``, ignored.
    rng : numpy.random.Generator or None, optional
        Random number generator.  If ``None``, a new generator is created.
    n_sweeps : int, optional
        Number of MCMC sweeps (default 20).  Each sweep makes
        ``(K choose 2)^2`` slice updates per off-diagonal age pair.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping ``"source->target"`` labels to associativity
        fraction arrays of shape ``(S, A, A)``.  Cells where a source has no
        contacts (``z^s_ab = 0``) are set to 0.

    Raises
    ------
    ValueError
        If the prior is improper for the given concentrations.
    """
    if rng is None:
        rng = np.random.default_rng()

    strata_names = list(z_marg.keys())
    K = len(strata_names)
    zm = np.stack([np.asarray(z_marg[s], dtype=float) for s in strata_names], axis=1)
    S, _, A, _ = zm.shape  # (S, K, A, A)

    # Concentration conc[s, t, a, b] of source s on target t at cell (a, b)
    P_sa = _compute_P_sa(dataloader)
    Pt = np.stack([P_sa[t] for t in strata_names], axis=-1)  # (A, K)
    Q = Pt / Pt.sum(axis=-1, keepdims=True)
    conc = np.broadcast_to(alpha * Q.T[np.newaxis, :, np.newaxis, :], (K, K, A, A)).copy()
    for ext in (external_eta, external_counts):
        if ext is not None:
            conc += np.stack(
                [np.stack([ext[f"{s}->{t}"] for t in strata_names]) for s in strata_names]
            )

    ia, ib = np.tril_indices(A, k=-1)  # off-diagonal pairs, a > b
    dg = np.arange(A)

    # Exponents of each table entry in the log density
    e_off = (conc[:, :, ia, ib] - 1).transpose(2, 0, 1) + (
        conc[:, :, ib, ia] - 1
    ).transpose(2, 1, 0)  # (n, K, K): X[s,t] is Z^{st}_ab and Z^{ts}_ba
    e_diag = (conc[:, :, dg, dg] - 1).transpose(2, 0, 1)  # (A, K, K)

    # Properness: entries that vanish together need total exponent > -1
    worst = min(
        e_off.min() if e_off.size else np.inf,
        (e_diag + e_diag.transpose(0, 2, 1))[:, ~np.eye(K, dtype=bool)].min()
        if K > 1
        else np.inf,
    )
    if worst <= -1:
        Q_pair = np.concatenate(
            [
                (Q[ib][:, np.newaxis, :] + Q[ia][:, :, np.newaxis]).ravel(),
                (Q[:, :, np.newaxis] + Q[:, np.newaxis, :])[:, ~np.eye(K, dtype=bool)].ravel(),
            ]
        )
        raise ValueError(
            f"The eta prior is improper for alpha={alpha}: reciprocity ties "
            "eta^{st}_ab to eta^{ts}_ba, so their concentrations must sum to "
            f"more than 1 (smallest sum: {worst + 2:.3f}). Without external "
            f"terms this requires alpha > {1 / Q_pair.min():.3f}."
        )

    # Table margins; rescale column sums so both margins share one total
    r = zm[:, :, ia, ib].transpose(0, 2, 1)  # (S, n, K) row sums z^s_ab
    c = zm[:, :, ib, ia].transpose(0, 2, 1)  # (S, n, K) column sums z^t_ba
    r_tot, c_tot = r.sum(axis=-1), c.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        asym = np.abs(r_tot - c_tot) / np.maximum(r_tot, c_tot)
        if np.nanmax(asym, initial=0.0) > 1e-3:
            warnings.warn(
                "Total contact counts are not reciprocal (max relative asymmetry "
                f"{np.nanmax(asym):.2e}); column sums are rescaled to match.",
                UserWarning,
                stacklevel=2,
            )
        c = c * np.nan_to_num(r_tot / c_tot, nan=1.0)[..., np.newaxis]
        X = np.nan_to_num(r[..., :, np.newaxis] * c[..., np.newaxis, :] / r_tot[..., np.newaxis, np.newaxis])

        rd = zm[:, :, dg, dg].transpose(0, 2, 1)  # (S, A, K)
        rd_tot = rd.sum(axis=-1)
        Xd = np.nan_to_num(rd[..., :, np.newaxis] * rd[..., np.newaxis, :] / rd_tot[..., np.newaxis, np.newaxis])

    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    swap_sign = np.array([1.0, 1.0, -1.0, -1.0])
    diag_sign = np.array([1.0, -1.0, -1.0])
    for _ in range(n_sweeps):
        for s1, s2 in pairs:
            for t1, t2 in pairs:
                # Rows s1, s2 and columns t1, t2: +d on (s1,t1),(s2,t2); -d on the others
                ent = [(s1, t1), (s2, t2), (s1, t2), (s2, t1)]
                base = np.stack([X[..., i, j] for i, j in ent], axis=-1)
                w = np.stack([e_off[:, i, j] for i, j in ent], axis=-1)
                d = _slice_step(base, swap_sign, w, rng)
                for (i, j), sg in zip(ent, swap_sign):
                    X[..., i, j] += sg * d
        for s, t in pairs:
            # Symmetric move: +d on (s,t) and (t,s); -d on (s,s) and (t,t)
            base = np.stack([Xd[..., s, t], Xd[..., s, s], Xd[..., t, t]], axis=-1)
            w = np.stack(
                [e_diag[:, s, t] + e_diag[:, t, s], e_diag[:, s, s], e_diag[:, t, t]],
                axis=-1,
            )
            d = _slice_step(base, diag_sign, w, rng)
            Xd[..., s, t] += d
            Xd[..., t, s] += d
            Xd[..., s, s] -= d
            Xd[..., t, t] -= d

    eta = np.zeros((S, K, K, A, A))
    with np.errstate(divide="ignore", invalid="ignore"):
        # (a, b): eta^{st} = X[s, t] / z^s_ab ; (b, a): eta^{st} = X[t, s] / z^s_ba
        eta[:, :, :, ia, ib] = np.nan_to_num(X / r[..., :, np.newaxis]).transpose(0, 2, 3, 1)
        eta[:, :, :, ib, ia] = np.nan_to_num(
            X.swapaxes(-1, -2) / c[..., :, np.newaxis]
        ).transpose(0, 2, 3, 1)
        eta[:, :, :, dg, dg] = np.nan_to_num(Xd / rd[..., :, np.newaxis]).transpose(0, 2, 3, 1)

    return {
        f"{s}->{t}": eta[:, i, j]
        for i, s in enumerate(strata_names)
        for j, t in enumerate(strata_names)
    }


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
