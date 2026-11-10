"""
Regression tests for the stratum index order (dev plan D22, Problem 7).

With two or more stratification variables, the point estimate, the stratum
labels and the bootstrap must agree on which array index belongs to which
stratum: composite codes follow ``itertools.product`` order (first variable
most significant), and each dimension is the number of categories (including
unused ones), not ``nunique()``.

Every participant in a stratum reports the same number of contacts, so the
expected contact intensities are known exactly and the bootstrap mean must
equal the point estimate.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from ...dataloader import ContactData, ParticipantData, PopulationData
from ...utils import AgeGroupSpecs
from ..classical._Prem2 import Prem2
from ..classical._SocialMix import SocialMix

SPECS = AgeGroupSpecs(0, 20, 10)  # [0, 10), [10, 20)
AGES = np.repeat(np.arange(20), 3)  # 3 participants per single year
CNT_AGE = 5  # every contact falls in contact age group [0, 10)


def _participants(levels):
    """One block of participants per combination of *levels* values."""
    rows, combos = [], list(itertools.product(*levels.values()))
    pid = 0
    for combo in combos:
        for age in AGES:
            rows.append({"pid": pid, "age": int(age), **dict(zip(levels, combo))})
            pid += 1
    return pd.DataFrame(rows), combos


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _row_sums(summary_dict, label):
    return summary_dict[label].central.sum(axis=1)


class TestParticipantStrata:
    """Partial stratification: participant strata only."""

    @pytest.mark.parametrize(
        "levels",
        [
            {"sex": ["F", "M"], "region": ["A", "B"]},
            {"sex": ["F", "M"], "region": ["A", "B", "C"], "ses": ["lo", "hi"]},
        ],
    )
    def test_point_estimate_matches_labels(self, levels):
        df_part, combos = _participants(levels)
        # Stratum k (in product order) reports k + 1 contacts per participant.
        y_of = {combo: k + 1 for k, combo in enumerate(combos)}
        stratum = [tuple(r[v] for v in levels) for _, r in df_part.iterrows()]
        df_cnt = pd.DataFrame(
            [
                {"pid": pid, "cnt_age": CNT_AGE}
                for pid, s in zip(df_part["pid"], stratum)
                for _ in range(y_of[s])
            ]
        )
        part = _quiet(
            ParticipantData, df_part, id_col="pid", age_col="age",
            strat_var_cols=list(levels),
        )
        cnt = _quiet(ContactData, df_cnt, id_col="pid", age_col="cnt_age")
        sm = _quiet(SocialMix, part, cnt, SPECS, apply_reciprocity=False)

        assert sm.K_part == len(combos)
        cint = sm.cint()
        for combo, y in y_of.items():
            label = "_".join(combo) + "->All"
            np.testing.assert_allclose(_row_sums(cint, label), y)

    def test_bootstrap_mean_matches_point_estimate(self):
        levels = {"sex": ["F", "M"], "region": ["A", "B"]}
        df_part, combos = _participants(levels)
        y_of = {combo: k + 1 for k, combo in enumerate(combos)}
        df_cnt = pd.DataFrame(
            [
                {"pid": r["pid"], "cnt_age": CNT_AGE}
                for _, r in df_part.iterrows()
                for _ in range(y_of[(r["sex"], r["region"])])
            ]
        )
        part = _quiet(
            ParticipantData, df_part, id_col="pid", age_col="age",
            strat_var_cols=list(levels),
        )
        cnt = _quiet(ContactData, df_cnt, id_col="pid", age_col="cnt_age")
        sm = _quiet(SocialMix, part, cnt, SPECS, apply_reciprocity=False)
        boot = _quiet(
            sm.run_inference_bootstrap, n_boot=5, random_state=1, progress=False
        )
        boot_mean = boot.mean(statistic="cint")
        cint = sm.cint()  # after any bootstrap-time merge
        for label in cint:
            np.testing.assert_allclose(boot_mean[label], cint[label].central)


class TestContactStrata:
    """Full and mixed stratification: contact strata and population rows."""

    def test_full_two_vars_and_population_order(self):
        levels = {"sex": ["F", "M"], "region": ["A", "B"]}
        df_part, combos = _participants(levels)
        n_strata = len(combos)
        # Participant stratum p reports (p * n_strata + q + 1) contacts with
        # contact stratum q, so every (p, q) cell is distinct.
        rows = []
        for _, r in df_part.iterrows():
            p = combos.index((r["sex"], r["region"]))
            for q, (c_sex, c_region) in enumerate(combos):
                rows += [
                    {"pid": r["pid"], "cnt_age": CNT_AGE, "sex": c_sex, "region": c_region}
                ] * (p * n_strata + q + 1)
        df_cnt = pd.DataFrame(rows)

        # Population: variables in the opposite order, categories in reverse
        # order, and a distinct size per stratum.
        pop_rows = []
        for q, (c_sex, c_region) in enumerate(combos):
            for age in range(20):
                pop_rows.append(
                    {"age": age, "region": c_region, "sex": c_sex, "P": 100.0 * (q + 1)}
                )
        df_pop = pd.DataFrame(pop_rows)
        df_pop["region"] = pd.Categorical(df_pop["region"], categories=["B", "A"])
        df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["M", "F"])

        part = _quiet(
            ParticipantData, df_part, id_col="pid", age_col="age",
            strat_var_cols=["sex", "region"],
        )
        cnt = _quiet(
            ContactData, df_cnt, id_col="pid", age_col="cnt_age",
            strat_var_cols=["sex", "region"],
        )
        pop = _quiet(
            PopulationData, df_pop, size_col="P", age_col="age",
            strat_var_cols=["region", "sex"],
        )
        sm = _quiet(SocialMix, part, cnt, SPECS, pop, apply_reciprocity=False)

        cint = sm.cint()
        rate = _quiet(sm.rate)
        for p, src in enumerate(combos):
            for q, tgt in enumerate(combos):
                label = f"{'_'.join(src)}->{'_'.join(tgt)}"
                expected = p * n_strata + q + 1
                np.testing.assert_allclose(_row_sums(cint, label), expected)
                # Each contact age group holds 10 single years of population.
                P_target = 10 * 100.0 * (q + 1)
                np.testing.assert_allclose(
                    rate[label].central[:, 0], cint[label].central[:, 0] / P_target
                )

    def test_mixed_with_unused_contact_category(self):
        df_part, _ = _participants({"sex": ["F", "M"]})
        y_of = {"F": {"A": 1, "B": 2}, "M": {"A": 3, "B": 4}}
        rows = []
        for _, r in df_part.iterrows():
            for region, y in y_of[r["sex"]].items():
                rows += [{"pid": r["pid"], "cnt_age": CNT_AGE, "region": region}] * y
        df_cnt = pd.DataFrame(rows)
        df_cnt["region"] = pd.Categorical(df_cnt["region"], categories=["A", "B", "C"])

        part = _quiet(
            ParticipantData, df_part, id_col="pid", age_col="age", strat_var_cols="sex"
        )
        cnt = _quiet(
            ContactData, df_cnt, id_col="pid", age_col="cnt_age", strat_var_cols="region"
        )
        sm = _quiet(SocialMix, part, cnt, SPECS, apply_reciprocity=False)

        assert sm.K_cnt == 3  # unused category C counts as a stratum
        cint = sm.cint()
        for sex, by_region in y_of.items():
            for region, y in by_region.items():
                np.testing.assert_allclose(_row_sums(cint, f"{sex}->{region}"), y)
            np.testing.assert_allclose(_row_sums(cint, f"{sex}->C"), 0.0)


class TestPrem2SharesFix:
    """Prem2 reuses the SocialMix loader, so it gets the same index order."""

    def test_two_participant_strata(self):
        levels = {"sex": ["F", "M"], "region": ["A", "B"]}
        df_part, combos = _participants(levels)
        y_of = {combo: k + 1 for k, combo in enumerate(combos)}
        df_cnt = pd.DataFrame(
            [
                {"pid": r["pid"], "cnt_age": CNT_AGE}
                for _, r in df_part.iterrows()
                for _ in range(y_of[(r["sex"], r["region"])])
            ]
        )
        part = _quiet(
            ParticipantData, df_part, id_col="pid", age_col="age",
            strat_var_cols=list(levels),
        )
        cnt = _quiet(ContactData, df_cnt, id_col="pid", age_col="cnt_age")
        prem2 = _quiet(Prem2, part, cnt, SPECS)

        assert prem2.K_part == len(combos)
        # Y[k] holds the contacts of stratum k; N[k] its participants.
        for k, combo in enumerate(combos):
            np.testing.assert_allclose(
                prem2.Y[k].sum(axis=1) / prem2.N[k], y_of[combo]
            )
