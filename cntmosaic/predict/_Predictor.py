from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import jax
from numpy.typing import ArrayLike, NDArray

from ..analysis import ModelSummariser
from ..dataloader import ContactSurveyLoader
from ._predict import (
    _compute_P_sa,
    counts_to_intensities,
    sample_full_counts,
    spectral_radius as _spectral_radius_fn,
    z_marginals,
)


class Predictor:
    """
    Posterior-predictive engine for fully stratified contact intensity matrices.

    Wraps a fitted ``ModelSummariser`` and its associated ``ContactSurveyLoader``
    to predict fully stratified contact intensity matrices from a partially
    stratified model.  The partially stratified contact counts (z-marginals)
    are computed lazily on first use and cached so repeated calls do not
    recompute them.

    Only genmix models with PARTIAL stratification are supported (i.e. models
    whose ``model_type == "genmix"`` and whose dataloader carries ``strat_data``).

    Parameters
    ----------
    summariser : ModelSummariser
        A fitted partially-stratified model summariser.
    dataloader : ContactSurveyLoader
        The dataloader used to prepare the model data.  Must have
        ``strat_data`` attached.

    Raises
    ------
    TypeError
        If *summariser* is not a ``ModelSummariser`` instance.
    ValueError
        If *summariser* is an agemix model (unsupported).
        If *dataloader* has no ``strat_data``.

    Examples
    --------
    >>> predictor = Predictor(summariser, dataloader)
    >>> cint_dict = predictor.predict_full_matrices(jax.random.key(0))
    >>> cint_dict["Male->Female"].shape   # (S, A, A)
    >>> rho = predictor.spectral_radius(jax.random.key(1))
    """

    def __init__(
        self,
        summariser: ModelSummariser,
        dataloader: ContactSurveyLoader,
    ) -> None:
        if not isinstance(summariser, ModelSummariser):
            raise TypeError(
                f"summariser must be a ModelSummariser instance, "
                f"got {type(summariser).__name__}"
            )
        if summariser.model_type != "genmix":
            raise ValueError(
                "Predictor only supports genmix (partially-stratified) models. "
                f"Got model_type={summariser.model_type!r}."
            )
        if dataloader.strat_data is None:
            raise ValueError(
                "dataloader must have strat_data attached. "
                "The predict pipeline requires stratification proportions."
            )

        self.summariser = summariser
        self.dataloader = dataloader

        self._z_marg: Optional[Dict[str, NDArray]] = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_z_marginals(self) -> None:
        if self._z_marg is None:
            self._z_marg = z_marginals(self.summariser, self.dataloader)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strata(self) -> List[str]:
        """Source stratum labels inferred from the fitted model."""
        self._ensure_z_marginals()
        return list(self._z_marg.keys())

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def predict_full_matrices(
        self,
        key: jax.Array,
        num_warmup: Optional[int] = None,
        assortativity: Optional[Dict[str, ArrayLike]] = None,
        sampler: Literal["swap", "hit_and_run"] = "swap",
    ) -> Dict[str, NDArray]:
        """
        Sample fully stratified contact intensity matrices.

        For every posterior draw and pair of age groups ``(a, b)``, the
        ``K x K`` table of fully stratified contact counts is known only
        through its row sums ``z^k_ab`` and, by reciprocity, its column sums
        ``z^l_ba``.  Each table is drawn uniformly from all non-negative
        tables with these margins, with swap moves by default (see
        :func:`sample_full_counts`), and the counts are divided by the
        stratified population sizes to give contact intensities.

        Optionally, the assortativity of some strata is fixed: the fraction
        ``eta^{kk}_ab`` of the contacts made by respondents of stratum ``k``
        at age ``a`` with contacts at age ``b <= a`` that are with stratum
        ``k``.  The fraction in the other direction (``b > a``) follows by
        reciprocity.  Where the fractions are
        infeasible for a table they are adjusted, with a warning (see
        :func:`sample_full_counts`).

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        num_warmup : int or None, optional
            Burn-in per table: sweeps of ``K * K`` moves for swap moves, steps
            for hit-and-run.  If ``None`` (default), 10 sweeps or 1600 steps.
            With two strata (``K = 2``) the tables are drawn exactly, and the
            burn-in is irrelevant.
        assortativity : Dict[str, ArrayLike] or None, optional
            Dictionary mapping stratum labels to assortativity fractions in
            ``[0, 1]``: a scalar, an ``(A, A)`` array indexed by respondent
            age and contact age, or an ``(S, A, A)`` array per posterior
            draw.  Only entries with respondent age ``a >= b`` are used.
            Strata that are left out are not constrained.  With two strata at
            most one can be given, and it then determines the table.
        sampler : {"swap", "hit_and_run"}, optional
            Sampler for the tables (default ``"swap"``).

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping ``"source->target"`` stratum pairs to predicted
            fully stratified contact intensity matrices of shape ``(S, A, A)``,
            where S is the number of posterior samples and A is the number of
            age groups.
        """
        self._ensure_z_marginals()
        counts = sample_full_counts(
            self._z_marg,
            key,
            num_warmup=num_warmup,
            assortativity=assortativity,
            sampler=sampler,
        )
        return counts_to_intensities(counts, _compute_P_sa(self.dataloader))

    def spectral_radius(
        self,
        key: jax.Array,
        method: str = "generalised",
        num_warmup: Optional[int] = None,
        assortativity: Optional[Dict[str, ArrayLike]] = None,
        sampler: Literal["swap", "hit_and_run"] = "swap",
    ) -> Union[float, NDArray]:
        """
        Compute the spectral radius of the predicted fully stratified matrices.

        Calls :meth:`predict_full_matrices` internally, then passes the result
        to the :func:`spectral_radius` free function.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key passed to :meth:`predict_full_matrices`.
        method : {"generalised", "age"}, optional
            Spectral radius method (default ``"generalised"``).
        num_warmup : int or None, optional
            Burn-in per table, passed to :meth:`predict_full_matrices`.
        assortativity : Dict[str, ArrayLike] or None, optional
            Assortativity fractions passed to :meth:`predict_full_matrices`.
        sampler : {"swap", "hit_and_run"}, optional
            Sampler passed to :meth:`predict_full_matrices` (default
            ``"swap"``).

        Returns
        -------
        float or NDArray
            Scalar for point-estimate matrices; array of shape ``(S,)`` for
            posterior sample matrices.
        """
        matrices = self.predict_full_matrices(
            key, num_warmup=num_warmup, assortativity=assortativity, sampler=sampler
        )
        return _spectral_radius_fn(matrices, self.dataloader, method=method)

    def clear_cache(self) -> None:
        """Clear the cached z-marginals.

        Forces recomputation on the next call to :meth:`predict_full_matrices`
        or :meth:`spectral_radius`.
        """
        self._z_marg = None

    def get_cache_info(self) -> Dict[str, Any]:
        """Return information about the lazy-loaded cache.

        Returns
        -------
        dict
            ``z_marginals_loaded`` — whether z-marginals have been computed.
            ``strata``             — stratum labels, or ``"not loaded"``.
            ``model_type``         — ``"genmix"``.
        """
        loaded = self._z_marg is not None
        return {
            "z_marginals_loaded": loaded,
            "strata": self.strata if loaded else "not loaded",
            "model_type": self.summariser.model_type,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = self._z_marg is not None
        strata_repr = str(self.strata) if loaded else "not loaded"
        return (
            f"Predictor("
            f"model_type={self.summariser.model_type!r}, "
            f"strata={strata_repr}, "
            f"z_marginals_loaded={loaded})"
        )
