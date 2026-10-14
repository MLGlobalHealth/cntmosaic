from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import ContactSurveyLoader, StratificationData
from .._GenMixFC import GenMixFC
from ..numpyro.priors import PSpline2D
from .fixtures import partial_coarse_large_sample


def _build_model(fixture, prior_type):
    part_data, cnt_data, pop_data = fixture
    strat_vars = pop_data.get_strat_vars()
    strat_data = StratificationData.from_counts(
        pop_data.data, age_col="age", strat_var_cols=strat_vars, count_col="P"
    )
    dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data, strat_data)
    priors = {"rate": PSpline2D(prior_type="global", M=5)}
    for var in strat_vars:
        priors[var] = PSpline2D(prior_type=prior_type, M=5)
    return GenMixFC(dataloader, priors, "negbin")


class TestNumPyroModel:

    def test_model_callable(self, partial_coarse_large_sample):
        from numpyro.handlers import seed

        model = _build_model(partial_coarse_large_sample, "partial")
        with seed(rng_seed=0):
            model.model(y=model.y)

    def test_print_model_shape(self, partial_coarse_large_sample):
        model = _build_model(partial_coarse_large_sample, "partial")
        model.print_model_shape()


class TestNumPyroInference:

    def test_svi_partial(self, partial_coarse_large_sample):
        model = _build_model(partial_coarse_large_sample, "partial")
        guide = AutoNormal(model.model)
        model.run_inference_svi(PRNGKey(0), guide, num_steps=50, peak_lr=0.01)

        assert model._svi_result is not None
        assert model._svi_result.params is not None

    def test_mcmc_partial(self, partial_coarse_large_sample):
        model = _build_model(partial_coarse_large_sample, "partial")
        model.run_inference_mcmc(PRNGKey(1), num_warmup=10, num_samples=10, num_chains=1)

        assert model._mcmc_result is not None
        samples = model._mcmc_result.get_samples()
        assert "baseline" in samples
        assert samples["baseline"].shape[0] == 10
