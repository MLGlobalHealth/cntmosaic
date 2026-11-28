import numpy as np
import pytest

from ...datasets._base import load_template_patterns
from .._ContactSampler import ContactSampler
from .._MatrixSampler import MatrixSampler
from .._ParticipantSampler import ParticipantSampler
from .._Population import Population
from .._Stratification import Stratification

patterns = load_template_patterns("United_States", max_age=50)
n_ages = patterns["household"].shape[0]

# ============================
# Fixtures for ContactSampler tests
# ============================


@pytest.fixture
def generate_single():
    """Single population case with new API."""
    ref_age_dist = np.random.rand(n_ages) * 1000
    strat = Stratification("group", 1, ref_age_dist, labels=["All"], seed=42)
    pop = Population(strat)

    df_part = ParticipantSampler(pop, n_part=1000).sample(seed=0)
    cint_matrices = MatrixSampler(patterns).generate_single(
        pop, mean_intensity=15.0, seed=0
    )

    return df_part, cint_matrices


@pytest.fixture
def generate_partial():
    """Partial case with two strata."""
    ref_age_dist = np.random.rand(n_ages) * 1000
    region_strat = Stratification(
        "region", 2, ref_age_dist, labels=["Urban", "Rural"], seed=42
    )
    pop = Population(region_strat)

    df_part = ParticipantSampler(pop, n_part=1500).sample(seed=0)
    cint_matrices = MatrixSampler(patterns).generate_partial(
        pop, mean_intensity=15.0, seed=0
    )

    return df_part, cint_matrices


@pytest.fixture
def generate_full():
    """Full case with two strata."""
    ref_age_dist = np.random.rand(n_ages) * 1000
    region_strat = Stratification(
        "region", 2, ref_age_dist, labels=["Urban", "Rural"], seed=42
    )
    pop = Population(region_strat)

    df_part = ParticipantSampler(pop, n_part=1500).sample(seed=0)
    cint_matrices = MatrixSampler(patterns).generate_full(
        pop, mean_intensity=15.0, seed=0
    )

    return df_part, cint_matrices


# ============================
# Tests for ContactSampler
# ============================
def test_single(generate_single):
    """Test single population case."""
    df_part, cint_matrices = generate_single

    cnt_gen = ContactSampler(df_part, cint_matrices)
    df_cnt = cnt_gen.sample(seed=0)

    assert df_cnt.shape[0] > 0, "No contacts generated in single population case"
    assert df_cnt.columns.tolist() == ["id", "cnt_age", "y"]
    assert (
        df_cnt["id"].nunique() <= df_part["id"].nunique()
    ), "More unique IDs in contacts than participants"


def test_partial(generate_partial):
    """Test partial case with stratified participants."""
    df_part, cint_matrices = generate_partial

    cnt_gen = ContactSampler(df_part, cint_matrices)
    df_cnt = cnt_gen.sample(seed=0)

    assert df_cnt.shape[0] > 0, "No contacts generated in partial case"
    assert df_cnt.columns.tolist() == ["id", "cnt_age", "y"]
    assert (
        df_cnt["id"].nunique() <= df_part["id"].nunique()
    ), "More unique IDs in contacts than participants"


def test_full(generate_full):
    """Test full case with all stratum pair interactions."""
    df_part, cint_matrices = generate_full

    cnt_gen = ContactSampler(df_part, cint_matrices)
    df_cnt = cnt_gen.sample(seed=0)

    assert df_cnt.shape[0] > 0, "No contacts generated in full case"
    assert df_cnt.columns.tolist() == ["id", "cnt_age", "cnt_region", "y"]
    assert (
        df_cnt["id"].nunique() <= df_part["id"].nunique()
    ), "More unique IDs in contacts than participants"

    # Check that contacts include both strata
    assert set(df_cnt["cnt_region"].unique()) == {"Urban", "Rural"}


_HASHSEED_SCRIPT = """
import hashlib
import numpy as np
from cntmosaic.datasets import load_template_patterns
from cntmosaic.sim import (
    ContactSampler, MatrixSampler, ParticipantSampler, Population, Stratification,
)

patterns = load_template_patterns("United_States", max_age=30)
ref = np.linspace(500, 1500, patterns["household"].shape[0])
strats = [
    Stratification("sex", 2, ref, labels=["M", "F"], seed=1),
    Stratification("edu", 3, ref, labels=["low", "mid", "high"], seed=2),
]
pop = Population(strats)
df_part = ParticipantSampler(pop, n_part=300).sample(seed=0)
cint = MatrixSampler(patterns).generate_full(pop, mean_intensity=10.0, seed=0)
df_cnt = ContactSampler(df_part, cint, random_effects=True).sample(seed=0)
df_cnt = df_cnt.sort_values(list(df_cnt.columns[:-1])).reset_index(drop=True)
print(hashlib.sha256(df_cnt.to_csv(index=False).encode()).hexdigest())
"""


@pytest.mark.parametrize("hashseeds", [("0", "1"), ("2", "12345")])
def test_full_reproducible_across_hash_seeds(hashseeds):
    """The same seed gives the same contacts whatever PYTHONHASHSEED is.

    The strata were once iterated in set order, so the random stream was
    consumed in an order that depended on string hashing.
    """
    import os
    import subprocess
    import sys

    digests = []
    for hs in hashseeds:
        env = {**os.environ, "PYTHONHASHSEED": hs}
        out = subprocess.run(
            [sys.executable, "-c", _HASHSEED_SCRIPT],
            env=env, capture_output=True, text=True, check=True,
        )
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1]
