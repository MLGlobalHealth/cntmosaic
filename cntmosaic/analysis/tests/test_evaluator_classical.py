"""
End-to-end tests of the SocialMix and Prem evaluators on simulated surveys.

Both summarisers return ``ContactSummary`` objects; the evaluators must read
their ``lower``/``central``/``upper`` fields, and the SocialMix summariser
must depixilate the bootstrap samples to 1-year ages before comparison with
1-year ground truth.
"""

import numpy as np
import pytest

from cntmosaic.analysis import (
    ModelEvaluatorPrem,
    ModelEvaluatorSocialMix,
    ModelSummariserPrem,
    ModelSummariserSocialMix,
)
from cntmosaic.dataloader import ContactData, ParticipantData, PopulationData
from cntmosaic.datasets import load_age_distribution, load_template_patterns
from cntmosaic.models import Prem, SocialMix
from cntmosaic.sim import (
    ContactSampler,
    MatrixSampler,
    ParticipantSampler,
    Population,
    Stratification,
)
from cntmosaic.utils import AgeBins

MAX_AGE = 49
METRIC_COLS = ["rmse", "mae", "mape", "interval_score", "coverage"]


def _survey(stratified: bool, seed: int = 0):
    templates = load_template_patterns("United_States", max_age=MAX_AGE)
    ref = load_age_distribution("United_States", max_age=MAX_AGE)["P"].to_numpy()
    if stratified:
        strat = Stratification("sex", 2, ref, labels=["F", "M"], seed=1)
        pop = Population(strat)
        df_pop = pop.df_P
        truth = MatrixSampler(templates).generate_full(pop, mean_intensity=10.0, seed=seed)
        names = ["sex"]
    else:
        strat = Stratification("general", 1, ref, seed=1)
        pop = Population(strat)
        df_pop = pop.df_P.drop(columns="general")
        truth = MatrixSampler(templates).generate_single(pop, mean_intensity=10.0, seed=seed)
        names = None
    df_part = ParticipantSampler(pop, n_part=400).sample(seed)
    df_cnt = ContactSampler(df_part, truth, random_effects=True).sample(seed)
    part = ParticipantData(df_part, id_col="id", age_col="age", strat_var_cols=names)
    cnt = ContactData(
        df_cnt, id_col="id", age_col="cnt_age",
        strat_var_cols=[f"cnt_{v}" for v in names] if names else None,
    )
    popd = PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols=names)
    return part, cnt, popd, truth


def _check(df, truth):
    assert "all" in set(df["cat"])
    assert set(METRIC_COLS) <= set(df.columns)
    vals = df[METRIC_COLS].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals))
    assert df["coverage"].between(0, 100).all()
    if len(truth) > 1:
        assert set(truth) <= set(df["cat"])


@pytest.mark.parametrize("stratified", [False, True])
def test_socialmix_evaluate_cint(stratified):
    part, cnt, pop, truth = _survey(stratified)
    model = SocialMix(
        part, cnt, AgeBins(0, MAX_AGE, step=5), pop,
        apply_reciprocity=True, adaptive_merge=True,
    )
    model.run_inference_bootstrap(20, random_state=0, progress=False)
    summariser = ModelSummariserSocialMix(model)

    depix = summariser.summarise_cint(alpha=0.05, return_depixilated=True)
    for label, cs in depix.items():
        assert cs.central.shape == truth[label].shape
        assert np.all(cs.lower <= cs.central) and np.all(cs.central <= cs.upper)

    df = ModelEvaluatorSocialMix(summariser, truth, alpha=0.05).evaluate_cint(alpha=0.05)
    _check(df, truth)


@pytest.mark.parametrize("stratified", [False, True])
def test_prem_evaluate_cint_with_random_effects(stratified):
    from jax.random import PRNGKey
    from numpyro.infer.autoguide import AutoNormal

    part, cnt, pop, truth = _survey(stratified)
    model = Prem(part, cnt, AgeBins(min=0, max=MAX_AGE, step=5), random_effects=True)
    model.run_inference_svi(prng_key=PRNGKey(0), guide=AutoNormal(model.model), num_steps=50)
    summariser = ModelSummariserPrem(model, pop, num_samples=50)
    df = ModelEvaluatorPrem(summariser, truth, alpha=0.05).evaluate_cint(alpha=0.05)
    _check(df, truth)
