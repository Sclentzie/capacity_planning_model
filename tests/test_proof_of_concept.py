"""Spec Section 9, Session 5: after proof of concept (spec Section 7).

Task 5l. The Section 9 check this file owns:

    - After proof of concept: no QA rate drops before the error-rate gate
      is met, and legally required sign-offs stay at 100% unless the input
      says the requirement changed.

Plus the ones the user asked for explicitly: the gate week matches the
observed series computed by hand, the scenario changes only what its
overrides file names (the shared diff test in test_scenarios.py), and the
baseline pins do not move.

WHY THESE ARE THE RIGHT CHECKS

Both halves of the Section 9 line guard against the same failure, which
is a model being generous with itself.

The FIRST half is a timing bug waiting to happen. The saving and the
evidence for the saving are the same quantity - the QA sample - so an
off-by-one that lowered the rate in the week the gate was met would use
the discount to help earn it. The rate is checked week by week against
the series, and the week the gate is met is pinned at the FULL rate.

The SECOND half is not a bug but a temptation. A legally required
sign-off is the largest of the three return terms on the rows that carry
one, so a scenario looking for a headline would find it there. It cannot
move on evidence about the machine, only on evidence about the law, and
the test holds it at 100% while the input says the requirement is
unchanged - then proves the switch is real by flipping it.

And a third check that is neither: a week that sampled NOTHING must never
count toward the gate. Before go-live there are no automated items and so
no reading at all, and a gate that accepted "no errors found" from a week
that looked at nothing would clear during the weeks the automation did
not exist.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from capacity_model import (
    automation,
    backlog,
    config,
    proof_of_concept as poc,
    scenarios,
    work_volume,
)


@pytest.fixture(scope="module")
def baseline():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def work(baseline, sites):
    return work_volume.build(baseline, sites)["work"]


@pytest.fixture(scope="module")
def scenario(baseline):
    return scenarios.load_scenario(
        scenarios.scenario_dir(baseline) / "after_proof_of_concept.yaml"
    )


@pytest.fixture(scope="module")
def switched_on(scenario, baseline):
    return scenarios.apply_scenario(scenario, baseline)


@pytest.fixture(scope="module")
def on_target(baseline):
    return scenarios.apply_scenario(
        scenarios.load_scenario(
            scenarios.scenario_dir(baseline) / "after_proof_of_concept_true_05.yaml"
        ),
        baseline,
    )


@pytest.fixture(scope="module")
def automated(switched_on, sites, work):
    return automation.build(switched_on, sites, work)


@pytest.fixture(scope="module")
def series(automated, switched_on):
    return poc.observed_series(automated, switched_on)


@pytest.fixture(scope="module")
def report(baseline, sites):
    return scenarios.proof_of_concept_report(baseline, sites)


# ---------------------------------------------------------------------------
# Off unless asked
# ---------------------------------------------------------------------------


def test_the_gate_is_off_everywhere_it_was_not_switched_on(baseline):
    """The baseline, and every scenario that did not ask for it."""
    assert poc.enabled(baseline) is False
    for scenario in scenarios.list_scenarios(baseline):
        loaded = scenarios.load_scenario(scenario)
        merged = scenarios.apply_scenario(loaded, baseline)
        expected = loaded["key"].startswith("after_proof_of_concept")
        assert poc.enabled(merged) is expected, loaded["key"]


def test_the_spec_defaults_are_the_inputs(baseline):
    """Spec 7's "After proof of concept" paragraph, field by field."""
    assert poc.target_error_rate(baseline) == 0.005
    assert poc.weeks_to_hold(baseline) == 12
    assert poc.true_error_rate(baseline) == 0.003
    assert poc.after_gate_qa_multiple(baseline) == 0.5
    # The default the Section 9 check is written about.
    assert poc.legal_requirement_changed(baseline) is False


def test_the_scenario_builds_on_ai_targeted(switched_on, baseline):
    """Siri 11/02: built on AI targeted, not on the full programme."""
    targeted = scenarios.apply_scenario(
        scenarios.load_scenario(scenarios.scenario_dir(baseline) / "ai_targeted.yaml"),
        baseline,
    )
    assert (
        automation.targeted_work_types(switched_on)
        == automation.targeted_work_types(targeted)
        == ["W5", "W8"]
    )
    assert automation.rates(switched_on) == automation.rates(targeted)
    # The gate is the ONLY difference between the two runs.
    assert poc.enabled(switched_on) and not poc.enabled(targeted)


# ---------------------------------------------------------------------------
# Section 9: no QA rate drops before the gate is met
# ---------------------------------------------------------------------------


def test_no_qa_rate_drops_before_the_gate_week(series, switched_on):
    """Spec Section 9, first half, checked week by week.

    The saving and the evidence are the same quantity, so this is a real
    risk and not a formality: the week the gate is MET is pinned at the
    full rate, because that week's reading is part of what cleared it.
    """
    gate = series.loc[series["gate_met"], "week_index"]
    assert len(gate) == 1, "the gate is met once or not at all"
    gate = int(gate.iloc[0])

    before = series[series["week_index"] <= gate]
    after = series[series["week_index"] > gate]
    assert (before["qa_rate_multiple"] == 1.0).all()
    assert (before["qa_rate_lowered"] == False).all()  # noqa: E712
    assert (
        after["qa_rate_multiple"] == poc.after_gate_qa_multiple(switched_on)
    ).all()
    assert (after["qa_rate_lowered"] == True).all()  # noqa: E712


def test_the_hours_carry_the_rate_the_series_reports(
    switched_on, sites, work, series
):
    """And it holds in the HOURS, not just in the series.

    The series could say the right thing while the plan was priced off
    something else, so the QA hours per automated item are divided back
    out week by week and matched against the rate in force.
    """
    built = backlog.required_hours(switched_on, sites, work)
    qa = built["hours"]
    qa = qa[qa["work_kind"] == "qa_sample"]
    minutes = float(
        automation.settings(switched_on)["human_in_the_loop"]["qa_review_minutes"]
    )
    rates = poc.qa_sample_rates(switched_on)
    expected = {
        int(row.week_index): float(row.qa_rate_multiple)
        for row in series.itertuples()
    }

    # The handle time on a qa_sample row is rate x multiple x review
    # minutes, so the multiple falls straight out of it.
    seen = set()
    for row in qa.itertuples():
        base = rates[row.work_type] * minutes
        assert base > 0
        multiple = row.handle_time_minutes / base
        assert multiple == pytest.approx(expected[row.week_index]), (
            row.week_index,
            row.work_type,
        )
        seen.add(multiple)
    # Both rates actually occur, or the test would pass on a run where
    # nothing ever changed.
    assert len(seen) == 2


def test_a_week_that_sampled_nothing_never_counts_toward_the_gate(series):
    """No evidence is not a pass.

    Every week before go-live has no automated items and so no reading.
    A gate that took "no errors found" from a week that reviewed nothing
    would clear during the weeks the machine did not exist.
    """
    empty = series[series["sampled_items"] == 0]
    assert len(empty), "the history weeks sample nothing, by construction"
    assert not empty["at_or_under_target"].any()
    assert (empty["consecutive_weeks"] == 0).all()
    assert not empty["gate_met"].any()
    assert empty["observed_error_rate"].isna().all()


def test_the_exception_hours_do_not_move_with_the_gate(
    baseline, switched_on, sites, work
):
    """Only the QA sample is about trust (spec 5.5).

    An exception is a property of the WORK - the machine could not finish
    that item - so proving the machine accurate changes nothing about it.
    If the gate had been wired to the wrong term this is where it would
    show.
    """
    targeted = scenarios.apply_scenario(
        scenarios.load_scenario(scenarios.scenario_dir(baseline) / "ai_targeted.yaml"),
        baseline,
    )
    gated = backlog.required_hours(switched_on, sites, work)["hours"]
    plain = backlog.required_hours(targeted, sites, work)["hours"]

    for kind, same in (("exception", True), ("human", True), ("qa_sample", False)):
        a = gated[gated["work_kind"] == kind]["required_hours"].sum()
        b = plain[plain["work_kind"] == kind]["required_hours"].sum()
        if same:
            assert float(a) == pytest.approx(float(b)), kind
        else:
            assert float(a) < float(b), kind


# ---------------------------------------------------------------------------
# Section 9: legally required sign-offs stay at 100%
# ---------------------------------------------------------------------------


def test_sign_offs_stay_at_one_hundred_percent(automated, switched_on):
    """Spec Section 9, second half, and spec Section 1's rule.

    A measured error rate is evidence about the machine. It says nothing
    about the statute, so it cannot move a sign-off the law requires.
    """
    assert poc.legal_requirement_changed(switched_on) is False
    rates = poc.sign_off_rate_by_week(automated, switched_on)
    assert set(rates.values()) == {1.0}
    assert (
        automation.settings(switched_on)["human_in_the_loop"]["sign_off_rate"] == 1.0
    )


def test_the_sign_off_minutes_are_unchanged_at_every_rate(switched_on):
    """Priced, not just configured.

    Taken on W7, which is sign-off throughout, so the term is there to
    move. The QA multiple is varied underneath it to prove the two terms
    are independent: lowering the sample must not touch the sign-off.
    """
    full = automation.return_minutes("W7", 15, switched_on, sign_off=True)
    half = automation.return_minutes(
        "W7", 15, switched_on, sign_off=True, qa_multiple=0.5
    )
    assert half["sign_off"] == pytest.approx(full["sign_off"])
    assert half["exception"] == pytest.approx(full["exception"])
    assert half["qa_sample"] == pytest.approx(full["qa_sample"] / 2)


def test_the_legal_switch_is_a_real_switch_and_not_dead_code(
    switched_on, automated
):
    """The other side of the check, so it cannot pass by doing nothing.

    If the law DID change, the rate has to move - otherwise "sign-offs
    stayed at 100%" would be true of a model that simply has no path for
    them to move at all, and the Section 9 check would be vacuous.
    """
    changed = copy.deepcopy(switched_on)
    changed["scenario"]["proof_of_concept"]["legal_requirement_changed"] = True
    changed["scenario"]["proof_of_concept"]["after_gate_sign_off_rate"] = 0.25

    rates = poc.sign_off_rate_by_week(automated, changed)
    series = poc.observed_series(automated, changed)
    gate = int(series.loc[series["gate_met"], "week_index"].iloc[0])
    assert rates[gate] == 1.0  # not in the week it is earned
    assert rates[gate + 1] == 0.25
    assert set(rates.values()) == {1.0, 0.25}

    # And it reaches the priced minutes.
    priced = automation.return_minutes(
        "W7", 15, changed, sign_off=True, sign_off_rate=0.25
    )
    full = automation.return_minutes("W7", 15, changed, sign_off=True)
    assert priced["sign_off"] == pytest.approx(full["sign_off"] / 4)


def test_the_targeted_rows_carry_no_legal_sign_off_at_all(report):
    """The honest headline, and it is not in the spec paragraph.

    W5 and W8 are the two rows that pay after fixed costs, and neither is
    a row a statute speaks to. So on this scenario the sign-off question
    is moot: there is no legally required sign-off in the run to argue
    about, and any sign-off saving would have to come from a wider
    programme that loses money for other reasons.
    """
    assert report["sign_off"]["automated_rows_needing_sign_off"] == []
    assert report["sign_off"]["rate_in_force"] == [1.0]
    assert report["sign_off"]["legal_requirement_changed"] is False
    for key, hours in report["sign_off"]["hours_by_run"].items():
        assert hours == pytest.approx(0.0), key


# ---------------------------------------------------------------------------
# The gate week, by hand
# ---------------------------------------------------------------------------


def test_the_gate_week_matches_the_observed_series_by_hand(series, switched_on):
    """Recomputed from the series' own columns, without the module.

    The gate is the first week with `weeks_to_hold` consecutive readings
    at or under target, and the lower rate starts the week AFTER. Walked
    here in a plain loop so a reader can check the rule against the
    printed series rather than against the code that produced it.
    """
    target = poc.target_error_rate(switched_on)
    hold = poc.weeks_to_hold(switched_on)

    run, by_hand = 0, None
    for row in series.itertuples():
        passes = row.sampled_items > 0 and row.observed_error_rate <= target
        assert passes == row.at_or_under_target, row.week_index
        run = run + 1 if passes else 0
        assert run == row.consecutive_weeks, row.week_index
        if by_hand is None and run >= hold:
            by_hand = int(row.week_index)

    assert by_hand is not None
    assert by_hand == int(series.loc[series["gate_met"], "week_index"].iloc[0])

    # And the figures the summary quotes, pinned (measured 11/02).
    assert by_hand == 179
    assert poc.gate_plan_week(None, switched_on) is None  # no plan, no gate
    cal = switched_on["calendar"]
    assert by_hand - cal["history_weeks"] + 1 == 24


def test_the_observed_rate_is_the_errors_over_the_sample(series):
    """The binomial draw, arithmetically.

    Nothing clever: it exists because an observed rate taken against the
    wrong denominator - the automated items rather than the sample - would
    read far too low and the gate would clear on week one.
    """
    live = series[series["sampled_items"] > 0]
    np.testing.assert_allclose(
        live["observed_error_rate"],
        live["errors_found"] / live["sampled_items"],
    )
    assert (live["errors_found"] <= live["sampled_items"]).all()
    # The sample is a FRACTION of what the machine did, never all of it.
    assert (live["sampled_items"] < live["automated_items"]).all()
    # And the rounded file count is the continuous sample, rounded.
    np.testing.assert_allclose(
        live["sampled_items"],
        (live["sample_at_full_rate"] * live["qa_rate_multiple"]).round(),
    )


def test_the_series_repeats_exactly_on_the_fixed_seed(automated, switched_on):
    """A fixed seed, like every other random line in the model."""
    first = poc.observed_series(automated, switched_on)
    second = poc.observed_series(automated, switched_on)
    pd.testing.assert_frame_equal(first, second)

    moved = copy.deepcopy(switched_on)
    moved["scenario"]["proof_of_concept"]["seed"] += 1
    other = poc.observed_series(automated, moved)
    # A different seed is a different draw, or the seed is not being used.
    assert not other["errors_found"].equals(first["errors_found"])


def test_a_thinner_sample_passes_more_easily_which_is_the_whole_problem():
    """The perverse arithmetic, stated as a test (spec 7, 11/02).

    While one error in n reads above the target - every sample under 200
    items at a 0.5% target - a week passes only on ZERO errors, so

        P(a week passes) = (1 - true rate) ^ n

    which FALLS as the sample grows. Looking at less evidence is the
    cheapest way to clear a gate written on an observed rate, and halving
    the sample after the gate makes the measurement weaker exactly when
    everyone has stopped watching.
    """
    target, truth = 0.005, 0.003
    probabilities = [
        scenarios.week_pass_probability(truth, n, target) for n in (8, 15, 30, 60)
    ]
    assert probabilities == sorted(probabilities, reverse=True)
    for n, got in zip((8, 15, 30, 60), probabilities):
        assert got == pytest.approx((1 - truth) ** n)

    # Past 200 items one error is inside the target, so the curve stops
    # being a pure zero-error race and the relationship breaks - which is
    # the sample size that would make the gate a measurement.
    assert scenarios.week_pass_probability(truth, 400, target) > (1 - truth) ** 400


# ---------------------------------------------------------------------------
# The sensitivity: a machine sitting on its own target
# ---------------------------------------------------------------------------


def test_a_machine_on_its_target_never_clears_the_gate(automated, on_target):
    """The sensitivity run, and the point of running it.

    At a 0.5% true rate against a 0.5% target on a sample of about 30, a
    week passes only if no error turns up at all - about 86% of the time -
    so twelve in a row is a one-in-six shot. The machine is exactly as
    good as it was asked to be and still fails, because the gate measures
    the machine plus the sample size and the operation chose the sample
    size.
    """
    assert poc.true_error_rate(on_target) == 0.005
    assert poc.gate_week(automated, on_target) is None

    series = poc.observed_series(automated, on_target)
    assert not series["gate_met"].any()
    assert (series["qa_rate_multiple"] == 1.0).all()
    live = series[series["sampled_items"] > 0]
    assert int(live["consecutive_weeks"].max()) < poc.weeks_to_hold(on_target)
    # Nothing is freed, because nothing was proved.
    assert poc.hours_freed(series, on_target)["hours"] == 0.0


def test_the_sensitivity_run_is_identical_to_ai_targeted(report):
    """A gate that never clears must leave the plan untouched.

    The strongest possible form of "the model never lowers a rate before
    the gate is met": with the gate unmet for all 52 weeks, every figure
    has to equal the run the scenario is built on, to the cent.
    """
    assert report["gates"]["true_rate_on_target"]["cleared"] is False
    assert report["cost"]["true_rate_on_target"] == pytest.approx(
        report["cost"]["ai_targeted"], abs=1e-6
    )
    gated = report["loop_hours"]["true_rate_on_target"]
    plain = report["loop_hours"]["ai_targeted"]
    assert set(gated) == set(plain)
    for kind in plain:
        assert gated[kind] == pytest.approx(plain[kind]), kind
    assert report["turnaround"]["true_rate_on_target"] == pytest.approx(
        report["turnaround"]["ai_targeted"]
    )


# ---------------------------------------------------------------------------
# What the proof is actually worth
# ---------------------------------------------------------------------------


def test_the_saving_is_only_the_qa_sample_and_it_is_tiny(report, baseline):
    """The finding, pinned (measured 11/02).

    399 QA items not reviewed, 13.3 hours, a few hundred dollars. Pinned
    because the number is the answer to the task's question, and a change
    in it is a change in the finding rather than a rounding difference.
    """
    gate = report["gates"]["proof_of_concept"]
    assert gate["cleared"] is True
    assert gate["gate_plan_week"] == 24
    assert gate["qa_items_not_sampled"] == pytest.approx(399.156, rel=1e-4)
    assert gate["qa_hours_freed"] == pytest.approx(13.3052, rel=1e-4)

    # The QA review is 2 minutes flat (spec 5.5), so the hours follow.
    minutes = baseline["automation"]["human_in_the_loop"]["qa_review_minutes"]
    assert gate["qa_hours_freed"] == pytest.approx(
        gate["qa_items_not_sampled"] * minutes / 60.0
    )

    # And it shows in the plan as a saving of that order, not of a
    # different order: the whole scenario is one term of 5.5.
    saved = -report["cost_vs_ai_targeted"]["proof_of_concept"]
    assert 0 < saved < 5_000
    assert saved == pytest.approx(289.0, abs=20.0)


def test_only_the_qa_term_moves_in_the_plans_own_hours(report):
    """Against AI targeted, kind by kind (spec 5.5).

    The exception hours and the human hours are identical and the QA
    hours fall by the freed amount, so the saving is traceable to one
    term rather than inferred from a net total.
    """
    gated = report["loop_hours"]["proof_of_concept"]
    plain = report["loop_hours"]["ai_targeted"]
    freed = report["gates"]["proof_of_concept"]["qa_hours_freed"]

    assert gated["exception"]["hours"] == pytest.approx(plain["exception"]["hours"])
    assert gated["exception"]["items"] == pytest.approx(plain["exception"]["items"])
    # Exactly, not approximately: both come off the same continuous
    # sample, so the freed figure has to reconcile to the cent with the
    # hours the plan was built on.
    assert gated["qa_sample"]["hours"] == pytest.approx(
        plain["qa_sample"]["hours"] - freed, rel=1e-9
    )
    # The ITEM count is unchanged: the machine did the same work and the
    # sample is a handle time, not a volume. A model that dropped items
    # here would be claiming the automation did less.
    assert gated["qa_sample"]["items"] == pytest.approx(plain["qa_sample"]["items"])


def test_the_gate_changes_nothing_about_service(report):
    """Turnaround and the goal are untouched.

    A review sample is back-office work that never sat on a requester's
    path, so freeing it cannot improve turnaround - and a model showing
    otherwise would have routed the QA hours into the wrong queue.
    """
    for key in ("proof_of_concept", "true_rate_on_target"):
        assert report["turnaround"][key] == pytest.approx(
            report["turnaround"]["ai_targeted"]
        )


def test_the_baseline_pins_do_not_move(report, baseline):
    """The baseline knows nothing about any of this."""
    assert poc.enabled(baseline) is False
    base = report["runs"]["baseline"]
    required = base["prepared"]["required"]
    # No automation in the baseline means no work_kind column at all, and
    # no gate series - the file is byte-identical to the pre-5.4 one.
    assert "work_kind" not in required.columns
    assert base["prepared"]["proof_of_concept"] is None
    assert base["prepared"]["automated"] is not None
    assert len(base["prepared"]["automated"]) == 0
    assert report["loop_hours"]["baseline"] == {}


def test_the_gate_does_not_touch_the_machines_invoice(report):
    """The proof is a labour saving, and only a labour saving.

    The platform fee, the setup and the per-item cost are all the same
    whether or not the gate cleared - proving a machine accurate does not
    make it cheaper to run. So the all-in difference against AI targeted
    has to be exactly the QA labour, which is also the strongest check
    that nothing else moved.
    """
    spend = report["machine_spend"]
    for key in ("proof_of_concept", "true_rate_on_target"):
        assert spend[key]["total"] == pytest.approx(spend["ai_targeted"]["total"])
        assert report["all_in_vs_ai_targeted"][key] == pytest.approx(
            report["cost_vs_ai_targeted"][key]
        )

    # And in context: the invoice dwarfs what the proof is worth. The
    # machine costs about $43,000 a year to run and the gate saves a few
    # hundred, so a proof of concept is not a business case on its own.
    assert spend["ai_targeted"]["total"] == pytest.approx(43280.0, rel=1e-3)
    saved = -report["all_in_vs_ai_targeted"]["proof_of_concept"]
    assert saved < spend["ai_targeted"]["total"] / 100
