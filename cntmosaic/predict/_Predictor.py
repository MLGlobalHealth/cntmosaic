from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..analysis import ModelSummariser
from ..dataloader import ContactSurveyLoader
from ._predict import (
    feasible_mixing_bounds,
    sample_eta,
    spectral_radius as _spectral_radius_fn,
    z_marginals,
)


class Predictor:
    """
    Posterior-predictive engine for fully stratified contact intensity matrices.

    Wraps a fitted ``ModelSummariser`` and its associated ``ContactSurveyLoader``
    to predict fully stratified contact intensity matrices from a partially
    stratified model.  Z-marginals and Fréchet-Hoeffding bounds are computed
    lazily on first use and cached so repeated calls do not recompute them.

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
    >>> cint_dict = predictor.predict_full_matrices(alpha=2.0, rng=rng)
    >>> cint_dict["Male->Female"].shape   # (S, A, A)
    >>> rho = predictor.spectral_radius()

    Keep the sampled associativity fractions for inspection:

    >>> cint_dict = predictor.predict_full_matrices(rng=rng, store_eta=True)
    >>> eta = predictor.get_eta_samples()
    >>> eta["Male->Female"].shape         # (S, A, A)
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
        self._frechet_lb: Optional[Dict[str, NDArray]] = None
        self._frechet_ub: Optional[Dict[str, NDArray]] = None
        self._eta: Optional[Dict[str, NDArray]] = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_bounds(self) -> None:
        if self._z_marg is not None:
            return
        self._z_marg = z_marginals(self.summariser, self.dataloader)
        self._frechet_lb, self._frechet_ub = feasible_mixing_bounds(
            self._z_marg, return_eta=True
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strata(self) -> List[str]:
        """Source stratum labels inferred from the fitted model."""
        self._ensure_bounds()
        return list(self._z_marg.keys())

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def predict_full_matrices(
        self,
        alpha: float = 2.0,
        rng: Optional[np.random.Generator] = None,
        external_eta: Optional[Dict[str, NDArray]] = None,
        external_counts: Optional[Dict[str, NDArray]] = None,
        store_eta: bool = False,
    ) -> Dict[str, NDArray]:
        """
        Sample fully stratified contact intensity matrices.

        The fully stratified intensities are obtained by multiplying the
        partially stratified posterior intensities by sampled associativity
        fractions (eta).  Set ``store_eta=True`` to keep these fractions for
        retrieval via :meth:`get_eta_samples`.

        Parameters
        ----------
        alpha : float, optional
            Dirichlet precision for the associativity fraction prior
            (default 2.0).  Higher values pull samples toward the
            population-proportion prior.  Must be large enough for the
            prior to be proper; see :func:`sample_eta`.
        rng : numpy.random.Generator or None, optional
            Random number generator.  If ``None``, a new generator is created.
        external_eta : Dict[str, NDArray] or None, optional
            External associativity fractions to augment the Dirichlet
            concentration, shape ``(A, A, K)`` per key.
        external_counts : Dict[str, NDArray] or None, optional
            External contact counts to augment the Dirichlet concentration,
            shape ``(A, A, K)`` per key.
        store_eta : bool, optional
            If ``True``, keep the sampled associativity fractions so they can
            be retrieved with :meth:`get_eta_samples` (default ``False``).
            The stored fractions take as much memory as the returned
            matrices: K² arrays of shape ``(S, A, A)``.  Every call replaces
            the stored fractions, or clears them when ``store_eta=False``,
            so they always correspond to the most recent prediction.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping ``"source->target"`` stratum pairs to predicted
            fully stratified contact intensity matrices of shape ``(S, A, A)``,
            where S is the number of posterior samples and A is the number of
            age groups.
        """
        self._ensure_bounds()

        samples_eta = sample_eta(
            self._z_marg,
            self._frechet_lb,
            self._frechet_ub,
            self.dataloader,
            alpha=alpha,
            external_eta=external_eta,
            external_counts=external_counts,
            rng=rng,
        )
        sample_cint = self.summariser.get_posterior_samples("cint")
        strata = self.strata

        cint_dict = {
            f"{s}->{t}": sample_cint[f"{s}->All"] * samples_eta[f"{s}->{t}"]
            for s in strata
            for t in strata
        }
        self._eta = samples_eta if store_eta else None

        return cint_dict

    def spectral_radius(
        self,
        method: str = "generalised",
        alpha: float = 2.0,
        rng: Optional[np.random.Generator] = None,
        store_eta: bool = False,
    ) -> Union[float, NDArray]:
        """
        Compute the spectral radius of the predicted fully stratified matrices.

        Calls :meth:`predict_full_matrices` internally, then passes the result
        to the :func:`spectral_radius` free function.  Because a new
        prediction is drawn, any previously stored associativity fractions
        are replaced (``store_eta=True``) or cleared (``store_eta=False``).

        Parameters
        ----------
        method : {"generalised", "age"}, optional
            Spectral radius method (default ``"generalised"``).
        alpha : float, optional
            Dirichlet precision passed to :meth:`predict_full_matrices`
            (default 2.0).
        rng : numpy.random.Generator or None, optional
            Random number generator passed to :meth:`predict_full_matrices`.
        store_eta : bool, optional
            Passed to :meth:`predict_full_matrices` (default ``False``).

        Returns
        -------
        float or NDArray
            Scalar for point-estimate matrices; array of shape ``(S,)`` for
            posterior sample matrices.
        """
        matrices = self.predict_full_matrices(
            alpha=alpha, rng=rng, store_eta=store_eta
        )
        return _spectral_radius_fn(matrices, self.dataloader, method=method)

    def get_eta_samples(self) -> Dict[str, NDArray]:
        """Return the associativity fractions stored by the last prediction.

        The fractions are only kept when :meth:`predict_full_matrices` (or
        :meth:`spectral_radius`) is called with ``store_eta=True``.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping ``"source->target"`` stratum pairs to
            associativity fraction arrays of shape ``(S, A, A)``, keyed like
            the output of :meth:`predict_full_matrices`.  The dictionary is
            new, but the arrays are shared with the internal store to avoid
            doubling memory; do not modify them in place.

        Raises
        ------
        RuntimeError
            If no associativity fractions are stored, i.e. the most recent
            prediction was made with ``store_eta=False`` or the cache was
            cleared.
        """
        if self._eta is None:
            raise RuntimeError(
                "No eta samples stored. Call "
                "predict_full_matrices(store_eta=True) first."
            )
        return dict(self._eta)

    def clear_cache(self) -> None:
        """Clear cached z-marginals, Fréchet bounds and stored eta samples.

        Forces recomputation on the next call to :meth:`predict_full_matrices`
        or :meth:`spectral_radius`.
        """
        self._z_marg = None
        self._frechet_lb = None
        self._frechet_ub = None
        self._eta = None

    def get_cache_info(self) -> Dict[str, Any]:
        """Return information about the lazy-loaded cache.

        Returns
        -------
        dict
            ``bounds_loaded`` — whether bounds have been computed.
            ``strata``        — stratum labels, or ``"not loaded"``.
            ``model_type``    — ``"genmix"``.
            ``eta_stored``    — whether eta samples are stored.
        """
        loaded = self._z_marg is not None
        return {
            "bounds_loaded": loaded,
            "strata": self.strata if loaded else "not loaded",
            "model_type": self.summariser.model_type,
            "eta_stored": self._eta is not None,
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
            f"bounds_loaded={loaded})"
        )
