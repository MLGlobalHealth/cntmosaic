from __future__ import annotations

import warnings

import jax
import numpy as np
import pandas as pd
from scipy.stats import beta

from typing import Dict, List, Literal, Optional, Tuple, Union
from numpy.typing import ArrayLike, NDArray

from ..dataloader import ContactSurveyLoader
from ..analysis import ModelSummariser
from ._polytope_sampler import sample_hit_run
from ._swap_sampler import sample_swap


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


# Relative difference between the row and column totals of a table above
# which ``sample_full_counts`` warns that the margins were reconciled.  Fitted
# GenMix models give totals that agree to about 1e-6 (tutorial 01), so smaller
# differences are reconciled silently.
_TOTALS_RTOL = 1e-4


def _reconcile_totals(
    row_sums: NDArray, col_sums: NDArray
) -> Tuple[NDArray, NDArray, float]:
    """
    Rescale row and column sums to the average of their two totals.

    Parameters
    ----------
    row_sums, col_sums : NDArray
        Margins of shape ``(..., K)``.

    Returns
    -------
    Tuple[NDArray, NDArray, float]
        The rescaled row and column sums, and the largest relative difference
        between the original totals.
    """
    row_total = row_sums.sum(-1, keepdims=True)
    col_total = col_sums.sum(-1, keepdims=True)
    total = (row_total + col_total) / 2
    mismatch = float(np.max(np.abs(row_total - col_total) / total))
    return row_sums * total / row_total, col_sums * total / col_total, mismatch


def _diagonal_constraints(
    row_sums: NDArray, col_sums: NDArray, delta: NDArray, fixed: NDArray
) -> NDArray:
    """
    Strict feasibility conditions of a fixed diagonal as ``h(delta) < 0``.

    For every fixed stratum ``k`` the table needs ``delta_k < r_k``,
    ``delta_k < c_k`` and ``(r_k - delta_k) + (c_k - delta_k) < N - sum(delta)``
    (Proposition 4 of the supplement).  All three are linear in ``delta``.
    Conditions of free strata are set to ``-inf``.

    Returns
    -------
    NDArray
        Array of shape ``(..., 3, K)``.
    """
    N = row_sums.sum(-1, keepdims=True)
    h = np.stack(
        [
            delta - row_sums,
            delta - col_sums,
            row_sums + col_sums - N - 2 * delta + delta.sum(-1, keepdims=True),
        ],
        axis=-2,
    )
    return np.where(fixed, h, -np.inf)


def _shrink_to_feasible(
    row_sums: NDArray, col_sums: NDArray, delta: NDArray, fixed: NDArray
) -> Tuple[NDArray, NDArray]:
    """
    Move infeasible fixed diagonals towards the independence diagonal.

    The diagonal ``delta0_k = r_k c_k / N`` of the independence table is
    strictly feasible, since every cell of that table is positive.  The
    feasible diagonals form a convex set, so along
    ``delta(s) = delta0 + s (delta - delta0)`` the conditions of
    :func:`_diagonal_constraints` hold for all ``s`` below a bound ``s_max``
    that has a closed form.  Infeasible diagonals are replaced by
    ``delta(0.99 s_max)``, which keeps the starting table strictly inside.

    Parameters
    ----------
    row_sums, col_sums, delta : NDArray
        Margins and diagonal values of shape ``(..., K)``, with ``delta = 0``
        for free strata.
    fixed : NDArray
        Boolean array of shape ``(K,)`` marking the fixed strata.

    Returns
    -------
    Tuple[NDArray, NDArray]
        The adjusted diagonal, and a boolean array of shape ``(...)`` marking
        the tables that were adjusted.
    """
    delta0 = fixed * row_sums * col_sums / row_sums.sum(-1, keepdims=True)
    h0 = _diagonal_constraints(row_sums, col_sums, delta0, fixed)
    h1 = _diagonal_constraints(row_sums, col_sums, delta, fixed)

    violated = h1 >= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        s_bound = np.where(violated, h0 / (h0 - h1), np.inf)
    s_max = s_bound.min(axis=(-2, -1))
    adjusted = s_max <= 1
    s = np.where(adjusted, 0.99 * s_max, 1.0)[..., np.newaxis]
    return delta0 + s * (delta - delta0), adjusted


def _assortativity_array(
    assortativity: Dict[str, ArrayLike], strata: List[str], shape: Tuple[int, ...]
) -> Tuple[NDArray, NDArray]:
    """
    Validate assortativity fractions and stack them into one array.

    Parameters
    ----------
    assortativity : Dict[str, ArrayLike]
        Dictionary mapping stratum labels to fractions broadcastable to
        ``shape``, i.e. ``(S, A, A)``.
    strata : List[str]
        Stratum labels, in the order of the tables.
    shape : Tuple[int, ...]
        Shape ``(S, A, A)`` of the partially stratified counts.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Fractions of shape ``(S, K, A, A)``, zero for free strata, and a
        boolean array of shape ``(K,)`` marking the fixed strata.
    """
    unknown = set(assortativity) - set(strata)
    if unknown:
        raise ValueError(
            f"Unknown strata in assortativity: {sorted(unknown)}. "
            f"Known strata: {strata}."
        )
    if len(strata) == 2 and len(assortativity) == 2:
        raise ValueError(
            "With two strata a table has a single free cell, so the "
            "assortativity of at most one stratum can be fixed."
        )

    eta = np.zeros((shape[0], len(strata)) + shape[1:])
    for i, k in enumerate(strata):
        if k not in assortativity:
            continue
        try:
            values = np.asarray(assortativity[k], dtype=float)
            eta[:, i] = np.broadcast_to(values, shape)
        except ValueError:
            raise ValueError(
                f"Assortativity of stratum {k!r} must be a scalar or an array "
                f"broadcastable to {shape} (posterior draws, age, age)."
            ) from None
    if not np.all((eta >= 0) & (eta <= 1)):
        raise ValueError("Assortativity fractions must lie in [0, 1].")
    return eta, np.array([k in assortativity for k in strata])


def _assortative_diagonal(
    eta: NDArray, fixed: NDArray, row_sums: NDArray, col_sums: NDArray
) -> Tuple[NDArray, NDArray, float]:
    """
    Fixed diagonal values ``delta_k = eta_k r_k`` for a batch of tables.

    Parameters
    ----------
    eta, row_sums, col_sums : NDArray
        Assortativity fractions and margins of shape ``(..., K)``.
    fixed : NDArray
        Boolean array of shape ``(K,)`` marking the fixed strata.

    Returns
    -------
    Tuple[NDArray, NDArray, float]
        The diagonal with ``NaN`` for free strata (the format expected by
        :func:`sample_hit_run`), a boolean array marking the
        tables whose diagonal had to be moved to be feasible, and the largest
        resulting change of an assortativity fraction.
    """
    delta, adjusted = _shrink_to_feasible(row_sums, col_sums, eta * row_sums, fixed)
    eta_change = float(np.max(np.abs(eta - delta / row_sums), initial=0.0))
    return np.where(fixed, delta, np.nan), adjusted, eta_change


# Default burn-in of each sampler: sweeps of R * C swap moves, or hit-and-run steps
_DEFAULT_WARMUP = {"swap": 10, "hit_and_run": 1600}


def _sample_tables(
    key: jax.Array,
    row_sums: NDArray,
    col_sums: NDArray,
    sampler: str,
    num_warmup: Optional[int],
    symmetric: bool = False,
    diagonal: Optional[NDArray] = None,
) -> jax.Array:
    """
    Draw one table per pair of margins with the chosen sampler.

    Returns tables of shape ``(..., K, K)`` for margins of shape ``(..., K)``.
    """
    if num_warmup is None:
        num_warmup = _DEFAULT_WARMUP[sampler]
    if sampler == "swap":
        tables = sample_swap(
            key, row_sums, col_sums, num_samples=1, num_warmup_sweeps=num_warmup,
            symmetric=symmetric, diagonal=diagonal,
        )
    else:
        tables = sample_hit_run(
            key, row_sums, col_sums, num_samples=1, num_warmup=num_warmup,
            symmetric=symmetric, diagonal=diagonal,
        )
    return tables[..., 0, :, :]


def sample_full_counts(
    z_marg: Dict[str, NDArray],
    key: jax.Array,
    num_warmup: Optional[int] = None,
    assortativity: Optional[Dict[str, ArrayLike]] = None,
    sampler: Literal["swap", "hit_and_run"] = "swap",
) -> Dict[str, NDArray]:
    """
    Sample fully stratified expected contact counts from their margins.

    For a posterior draw and a pair of age groups ``(a, b)``, the fully
    stratified counts ``X^{kl}_ab`` (contacts from stratum ``k`` at age ``a``
    with stratum ``l`` at age ``b``) form a ``K x K`` table.  Its row sums are
    the partially stratified counts ``z^k_ab``, and by reciprocity its column
    sums are ``z^l_ba``.  Nothing else is known about the table, so it is
    drawn uniformly from all non-negative tables with these margins (the
    transportation polytope), with swap moves (:func:`sample_swap`, the
    default) or hit-and-run (:func:`sample_hit_run`).

    Reciprocity also requires ``X^{kl}_ab = X^{lk}_ba``.  Tables are therefore
    sampled for ``a > b`` only, and the ``(b, a)`` table is the transpose of
    the ``(a, b)`` table.  On the diagonal ``a = b`` the table must itself be
    symmetric and is drawn uniformly from the symmetric tables.

    A table exists only if its row and column totals agree,
    ``sum_k z^k_ab = sum_l z^l_ba``.  Where they differ, both margins are
    rescaled to the average of the two totals and a warning is issued.

    Optionally, the assortativity of some strata is fixed.  The assortativity
    fraction ``eta^{kk}_ab = X^{kk}_ab / z^k_ab`` is the share of the contacts
    made by stratum ``k`` at age ``a`` with age ``b`` that are with stratum
    ``k``.  It may differ between age pairs and is given for the sampled
    tables, i.e. respondent age ``a >= b``.  The diagonal cells
    ``X^{kk}_ab = eta^{kk}_ab z^k_ab`` are then fixed and the remaining cells
    are sampled uniformly.  By reciprocity the fraction in the other direction
    is implied, ``eta^{kk}_ba = eta^{kk}_ab z^k_ab / z^k_ba``, so values given
    for ``a < b`` are ignored.  For some tables
    the fixed diagonal may leave no positive table with the given margins (see
    :func:`diagonal_is_feasible`).  Their diagonal is moved towards that of
    the independence table until it is feasible, and a warning is issued.

    Parameters
    ----------
    z_marg : Dict[str, NDArray]
        Dictionary mapping stratum labels to partially stratified contact
        counts of shape ``(S, A, A)`` (output of :func:`z_marginals`).
    key : jax.Array
        JAX PRNG key.
    num_warmup : int or None, optional
        Burn-in per table: sweeps of ``K * K`` moves for swap moves, steps for
        hit-and-run.  If ``None`` (default), 10 sweeps or 1600 steps; see
        :func:`sample_swap` and :func:`sample_hit_run` for how these were
        chosen.  With two strata the tables are drawn exactly by either
        sampler, and the burn-in is irrelevant.
    assortativity : Dict[str, ArrayLike] or None, optional
        Dictionary mapping stratum labels to assortativity fractions in
        ``[0, 1]``: a scalar (common to all age pairs), an ``(A, A)`` array
        indexed by respondent age and contact age, or an ``(S, A, A)`` array
        per posterior draw.  Only entries with respondent age ``a >= b`` are
        used.  With ``sampler="swap"`` every stratum needs a fraction, since
        the swap sampler fixes the whole diagonal.  With
        ``sampler="hit_and_run"`` strata that are left out are not
        constrained, and with two strata at most one can be given.
    sampler : {"swap", "hit_and_run"}, optional
        Sampler for the tables (default ``"swap"``).

    Returns
    -------
    Dict[str, NDArray]
        Dictionary mapping ``"source->target"`` stratum pairs to fully
        stratified contact counts of shape ``(S, A, A)``.

    Raises
    ------
    ValueError
        If ``sampler`` is not ``"swap"`` or ``"hit_and_run"``, or if the
        assortativity fractions are invalid.
    """
    if sampler not in _DEFAULT_WARMUP:
        raise ValueError(
            f"sampler must be one of {list(_DEFAULT_WARMUP)}, got {sampler!r}."
        )
    strata = list(z_marg)
    Z = np.stack([z_marg[k] for k in strata], axis=1)  # (S, K, A, A)
    S, K, A, _ = Z.shape
    a, b = np.tril_indices(A, k=-1)  # age pairs with a > b
    diag = np.arange(A)
    key_off, key_diag = jax.random.split(key)

    # Off-diagonal age pairs: row sums z^k_ab, column sums z^l_ba
    row_sums, col_sums, mismatch = _reconcile_totals(
        np.moveaxis(Z[:, :, a, b], 1, -1),  # (S, n_pairs, K)
        np.moveaxis(Z[:, :, b, a], 1, -1),
    )
    if mismatch > _TOTALS_RTOL:
        warnings.warn(
            f"Row and column totals of the stratified tables differ by up to "
            f"{mismatch:.2e} (relative); both were rescaled to their average.",
            stacklevel=2,
        )

    # Diagonal age pairs: symmetric tables with row and column sums z^k_aa
    margins_diag = np.moveaxis(Z[:, :, diag, diag], 1, -1)  # (S, A, K)

    # Fixed within-stratum cells X^{kk}_ab = eta^{kk}_ab z^k_ab
    fixed_off = fixed_diag = None
    if assortativity:
        eta, fixed = _assortativity_array(assortativity, strata, (S, A, A))
        if sampler == "swap" and not fixed.all():
            missing = [k for k, f in zip(strata, fixed) if not f]
            raise ValueError(
                "The swap sampler fixes the whole diagonal, so every stratum "
                f"needs an assortativity fraction; missing: {missing}. Give all "
                "strata, or use sampler='hit_and_run' for a partly fixed diagonal."
            )
        fixed_off, adjusted_off, change_off = _assortative_diagonal(
            np.moveaxis(eta[:, :, a, b], 1, -1), fixed, row_sums, col_sums
        )
        fixed_diag, adjusted_diag, change_diag = _assortative_diagonal(
            np.moveaxis(eta[:, :, diag, diag], 1, -1),
            fixed,
            margins_diag,
            margins_diag,
        )
        n_adjusted = int(adjusted_off.sum() + adjusted_diag.sum())
        if n_adjusted:
            warnings.warn(
                f"The assortativity fractions are infeasible for {n_adjusted} of "
                f"{adjusted_off.size + adjusted_diag.size} tables; their diagonal "
                "was moved towards the independence table until feasible "
                "(largest change of a fraction: "
                f"{max(change_off, change_diag):.3f}).",
                stacklevel=2,
            )

    X_off = _sample_tables(
        key_off, row_sums, col_sums, sampler, num_warmup, diagonal=fixed_off
    )  # (S, n_pairs, K, K)
    X_diag = _sample_tables(
        key_diag, margins_diag, margins_diag, sampler, num_warmup,
        symmetric=True, diagonal=fixed_diag,
    )  # (S, A, K, K)

    # Assemble X[s, k, l, a, b]; the (b, a) table is the transpose of (a, b)
    X = np.zeros((S, K, K, A, A))
    X_off = np.moveaxis(np.asarray(X_off), 1, -1)  # (S, K, K, n_pairs)
    X[..., a, b] = X_off
    X[..., b, a] = X_off.swapaxes(1, 2)
    X[..., diag, diag] = np.moveaxis(np.asarray(X_diag), 1, -1)

    return {
        f"{k}->{l}": X[:, i, j]
        for i, k in enumerate(strata)
        for j, l in enumerate(strata)
    }


def counts_to_intensities(
    counts: Dict[str, NDArray],
    P_sa: Dict[str, NDArray],
) -> Dict[str, NDArray]:
    """
    Convert fully stratified contact counts to contact intensities.

    The intensity is ``m^{k->l}_ab = X^{kl}_ab / P^k_a``, the expected number
    of contacts per person in stratum ``k`` at age ``a``, and 0 where
    ``P^k_a = 0``.

    Parameters
    ----------
    counts : Dict[str, NDArray]
        Dictionary mapping ``"source->target"`` stratum pairs to contact
        counts of shape ``(S, A, A)`` (output of :func:`sample_full_counts`).
    P_sa : Dict[str, NDArray]
        Dictionary mapping stratum labels to population sizes of shape ``(A,)``.

    Returns
    -------
    Dict[str, NDArray]
        Dictionary mapping ``"source->target"`` stratum pairs to contact
        intensities of shape ``(S, A, A)``.
    """
    intensities = {}
    for pair, X in counts.items():
        P = P_sa[pair.split("->")[0]][np.newaxis, :, np.newaxis]
        intensities[pair] = np.divide(X, P, out=np.zeros_like(X), where=P > 0)
    return intensities


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


# ----------------------------------------------------------------------
# Deprecated: associativity-fraction (eta) prediction
#
# These functions implemented the earlier prediction of fully stratified
# matrices, which sampled associativity fractions from a truncated Dirichlet
# distribution.  They are no longer used by ``Predictor`` and are kept only
# for backward compatibility.
# ----------------------------------------------------------------------


def feasible_mixing_bounds(
    z_marginals: Dict[str, np.ndarray],
    return_eta: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Calculate feasible mixing bounds (Frechet-Hoeffding bounds) for every stratum pair.

    .. deprecated:: 0.6.0
        No longer used by :class:`Predictor`, which now samples the fully
        stratified tables directly with :func:`sample_full_counts`.

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

    .. deprecated:: 0.6.0
        No longer used by :class:`Predictor`, which now samples the fully
        stratified tables directly with :func:`sample_full_counts`.

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

    .. deprecated:: 0.6.0
        No longer used by :class:`Predictor`, which now samples the fully
        stratified tables directly with :func:`sample_full_counts`.

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

    .. deprecated:: 0.6.0
        No longer used by :class:`Predictor`, which now samples the fully
        stratified tables directly with :func:`sample_full_counts`.

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
