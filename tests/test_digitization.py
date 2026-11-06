"""Spec Section 9, Session 5: coordinated digitization (spec Section 7).

Task 5j. The Section 9 check this file owns:

    - Coordinated digitization lowers the non-digital share only by items
      actually converted.

Plus the four the user asked for explicitly: converted items never exceed
hours x the conversion rate, the weeks before the starts week match the
baseline exactly, the scenario changes only what its overrides file
names (covered by the shared diff test in test_scenarios.py), and the
baseline pins do not move.

The check that earns its place most is the first one. "The non-digital
share falls over time" is the kind of claim a model can make for free: a
single multiplier decaying week by week would produce a falling share and
a lovely chart, and it would be fiction. The share here has to be
traceable to charts the project actually had the hours to convert, out of
an archive that actually holds them - which is what these tests pin.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capacity_model import (
    backlog,
    config,
    digitization,
    plan,
    scenarios,
    supply,
    work_volume,
)


@pytest.fixture(scope="module")
def baseline():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def scenario(baseline):
    return scenarios.load_scenario(
        scenarios.scenario_dir(baseline) / "coordinated_digitization.yaml"
    )


@pytest.fixture(scope="module")
def switched_on(scenario, baseline):
    return scenarios.apply_scenario(scenario, baseline)


@pytest.fixture(scope="module")
def walk(switched_on, sites):
    return digitization.converted_by_week(switched_on, sites)


@pytest.fixture(scope="module")
def report(baseline, sites):
    return scenarios.digitization_report(baseline, sites)


# ---------------------------------------------------------------------------
# Off unless asked (and the baseline's share is FIXED)
# ---------------------------------------------------------------------------


def test_the_baseline_holds_the_non_digital_share_fixed(baseline, sites):
    """Stated, not changed - the user asked which way the baseline works.

    The baseline DOES convert records: every onsite retrieval (W11) drags a
    15-minute digitization item (W12) behind it, about 21,000 a year. But
    that conversion never feeds back into record_media, so the share a
    leader would be quoted is 11% in week 1 and 11% in week 52.

    Reactive digitization does the work and throws the benefit away, which
    is precisely the gap the coordinated scenario exists to close. This
    test pins the baseline behaviour so the comparison means something.
    """
    assert digitization.enabled(baseline) is False
    assert digitization.non_digital_share(baseline) == pytest.approx(0.11)

    # There is no week-varying mechanism at all: the share is a scalar
    # read once from record_media and multiplied into every week.
    built = work_volume.build(baseline, sites)
    assert work_volume.media_share_arrays(
        baseline, sites, 208, [s["code"] for s in sites["sites"]]
    ) is None

    # Measured PRE-LAG, on the conservation table. The realized weekly mix
    # does drift a little in any run, because W8 and W9 lag 70/30 while
    # W11 lags 50/50 - so the three do not land in the same proportions
    # they were generated in. The generated shares are the claim being
    # tested here, and they are exactly the record_media table.
    expected = built["conservation"].groupby("work_type")["expected"].sum()
    retrieval = expected[["W8", "W9", "W11"]].sum()
    assert expected["W11"] / retrieval == pytest.approx(0.11, rel=1e-12)
    assert expected["W8"] / retrieval == pytest.approx(0.85, rel=1e-12)
    assert expected["W9"] / retrieval == pytest.approx(0.04, rel=1e-12)

    # And W12 is doing real work all the same - about 79,000 items a plan
    # year of conversion whose benefit is never banked.
    assert expected["W12"] > 50_000
    assert expected["W12"] == pytest.approx(expected["W11"], rel=1e-12)


def test_the_scenario_file_turns_it_on_and_nothing_else(scenario, switched_on):
    """Spec 7: 40 hours, 4 an hour, from week 13, at the RS rate."""
    assert scenario["name"] == "Coordinated digitization"
    assert scenario["starts_week"] == 13
    assert digitization.enabled(switched_on) is True
    assert digitization.items_per_week(switched_on) == pytest.approx(160.0)

    block = digitization.settings(switched_on)
    assert block["hours_per_week"] == 40
    assert block["items_per_hour"] == 4
    assert block["project_role"] == "records_specialist"

    # The archive and the concentration are CALIBRATED, not chosen (11/02);
    # the test that earns them is further down this file.
    assert block["archive_charts"] == 2_000_000
    assert block["concentration_archive_share"] == 0.20
    assert block["concentration_request_share"] == 0.45
    assert block["reactive_conversions_per_week"] == pytest.approx(367.84)
    assert block["history_drift_budget_points"] == 0.5

    # 4 an hour IS W12's 15 minutes, not a number of its own (10/28).
    w12 = next(w for w in switched_on["work_types"] if w["code"] == "W12")
    assert 60.0 / w12["handle_time_minutes"] == pytest.approx(
        block["items_per_hour"]
    )

    # The added hours are paid at the Records Specialist rate.
    hourly = supply.role_settings("records_specialist", switched_on)["hourly_cost"]
    assert digitization.weekly_cost(switched_on) == pytest.approx(40 * hourly)
    assert digitization.weekly_cost(switched_on) == pytest.approx(1200.0)


# ---------------------------------------------------------------------------
# The Section 9 check: the share falls ONLY by items actually converted
# ---------------------------------------------------------------------------


def test_the_non_digital_share_falls_only_by_items_actually_converted(
    walk, switched_on, sites, report
):
    """Spec Section 9. The check this scenario lives or dies by.

    Every week's share has to be reproducible from two numbers: the charts
    the project had the hours to convert by then, and the request
    probability those charts carried. Nothing else is allowed to move it.

    Worked by hand for the whole plan year, site by site, rather than
    spot-checked - the failure mode this guards against is a share that
    decays on a schedule of its own, and a schedule looks right in any
    single week.
    """
    per_week = digitization.items_per_week(switched_on)
    weights = digitization.site_weights(sites)
    archive = digitization.archive_by_site(switched_on, sites)
    start = digitization.from_week(switched_on)
    base_share = digitization.non_digital_share(switched_on)

    hot_archive, hot_requests = digitization.concentration(switched_on)
    rate = digitization.reactive_rate(switched_on)
    network_archive = sum(archive.values())
    lam_hot = hot_requests * rate / (hot_archive * network_archive)
    lam_cold = (1.0 - hot_requests) * rate / ((1.0 - hot_archive) * network_archive)

    running = {code: 0.0 for code in weights}
    hot_done = {code: 0.0 for code in weights}
    cold_done = {code: 0.0 for code in weights}
    for row in walk.itertuples():
        held = archive[row.site]
        hot_tier = hot_archive * held
        cold_tier = held - hot_tier
        # The charts still on paper, after the operation's own reactive
        # conversions have been running since the first history week.
        elapsed = row.week_index + 1
        hot_room = max(
            0.0, hot_tier * np.exp(-lam_hot * elapsed) - hot_done[row.site]
        )
        cold_room = max(
            0.0, cold_tier * np.exp(-lam_cold * elapsed) - cold_done[row.site]
        )
        expected_converted = (
            min(per_week * weights[row.site], hot_room + cold_room)
            if row.week_index >= start
            else 0.0
        )
        assert row.converted_this_week == pytest.approx(expected_converted), (
            row.week_index, row.site
        )
        # Most-likely-first: the hot survivors go before any cold one.
        take_hot = min(expected_converted, hot_room)
        hot_done[row.site] += take_hot
        cold_done[row.site] += expected_converted - take_hot
        running[row.site] = hot_done[row.site] + cold_done[row.site]
        assert row.converted_cumulative == pytest.approx(running[row.site])

        # The requests removed, by hand: each tier's conversions carry
        # that tier's share of the requests, counted against the original
        # pool the flat baseline assumes.
        fraction = running[row.site] / held
        assert row.converted_fraction == pytest.approx(fraction)
        expected_removed = (
            hot_requests * hot_done[row.site] / hot_tier
            + (1.0 - hot_requests) * cold_done[row.site] / cold_tier
        )
        assert row.requests_removed_share == pytest.approx(expected_removed), (
            row.week_index, row.site
        )

        # And while the cap does not bind, that is exactly the two-line
        # curve - the two routes to the same number have to agree.
        if cold_done[row.site] == 0.0:
            assert row.requests_removed_share == pytest.approx(
                float(digitization.requests_removed_share(fraction, switched_on))
            )

    # And the reported network share is the base share times what is left.
    weekly = report["weekly"]
    for row in weekly.itertuples():
        site_rows = walk[walk["week_index"] == row.week_index]
        left = float(
            ((1.0 - site_rows["requests_removed_share"])
             * site_rows["archive_charts"]).sum()
            / site_rows["archive_charts"].sum()
        )
        assert row.non_digital_share == pytest.approx(base_share * left)

    # Before the starts week nothing has been converted, so the share is
    # exactly the baseline's - not approximately.
    before = weekly[weekly["plan_week"] < 13]
    assert len(before) == 12
    assert (before["converted_cumulative"] == 0).all()
    assert (before["non_digital_share"] == base_share).all()

    # After it, the share falls every week and never rises.
    after = weekly[weekly["plan_week"] >= 13]["non_digital_share"].to_numpy()
    assert (np.diff(after) < 0).all()
    assert after[-1] < base_share


def test_converted_items_never_exceed_the_hours_times_the_rate(
    walk, switched_on, sites
):
    """Two caps, and the second one is the one that bites eventually.

    The project cannot convert faster than its hours allow, and it cannot
    convert records a site does not hold. Both have to hold in every week
    of every site, or the scenario is claiming free work.
    """
    per_week = digitization.items_per_week(switched_on)
    weights = digitization.site_weights(sites)
    archive = digitization.archive_by_site(switched_on, sites)

    # Per site and week: never more than that site's slice of the rate.
    for row in walk.itertuples():
        assert row.converted_this_week <= per_week * weights[row.site] + 1e-9, (
            row.week_index, row.site
        )
        assert row.converted_this_week >= 0.0
        # And never more archive than the site has.
        assert row.converted_cumulative <= archive[row.site] + 1e-9
        assert row.converted_fraction <= 1.0 + 1e-12
        assert row.requests_removed_share <= 1.0 + 1e-12

    # Network-wide, each week: never more than hours x rate.
    weekly = walk.groupby("week_index")["converted_this_week"].sum()
    assert (weekly <= per_week + 1e-9).all()

    # The plan year's total is 40 live weeks at the full rate, exactly.
    assert weekly.sum() == pytest.approx(per_week * 40)
    assert weekly.sum() == pytest.approx(6400.0)

    # The archive cap is real, not theoretical: converting the whole
    # archive removes every onsite request and not one more.
    assert digitization.requests_removed_share(1.0, switched_on) == pytest.approx(1.0)
    assert digitization.requests_removed_share(1.5, switched_on) == pytest.approx(1.0)
    assert digitization.requests_removed_share(0.0, switched_on) == pytest.approx(0.0)


def test_the_concentration_curve_is_steep_then_shallow(switched_on):
    """Spec 7, calibrated 11/02: 20% of the archive draws 45% of requests.

    Which is the argument for converting most-likely-first. The hot band
    removes requests 2.25 times faster than its share of the archive; the
    cold tail about three times slower than that. The pair is no longer
    80/20 - it is solved against the flat history, which is what the next
    test pins - but the SHAPE of the curve is unchanged, and this test
    reads the pair from the inputs rather than restating it.
    """
    hot_archive, hot_requests = digitization.concentration(switched_on)
    assert (hot_archive, hot_requests) == (0.20, 0.45)

    hot = digitization.requests_removed_share(hot_archive, switched_on)
    assert hot == pytest.approx(hot_requests)
    assert hot / hot_archive == pytest.approx(2.25)

    # Linear inside each band, with the kink exactly at the hot share.
    assert digitization.requests_removed_share(0.10, switched_on) == pytest.approx(0.225)
    assert digitization.requests_removed_share(0.05, switched_on) == pytest.approx(0.1125)
    assert digitization.requests_removed_share(0.60, switched_on) == pytest.approx(0.725)
    assert digitization.requests_removed_share(1.00, switched_on) == pytest.approx(1.00)

    # The cold tail's slope: the remaining requests over the remaining
    # archive, 0.55 / 0.80.
    cold_slope = (
        digitization.requests_removed_share(0.60, switched_on)
        - digitization.requests_removed_share(0.20, switched_on)
    ) / 0.40
    assert cold_slope == pytest.approx(0.6875)

    # Converting in ARCHIVE order instead would be a straight line, and
    # the first 0.32% of the archive - the plan year's work - would remove
    # 0.32% of requests rather than 0.72%.
    plan_year = 6400 / 2_000_000
    assert digitization.requests_removed_share(
        plan_year, switched_on
    ) == pytest.approx(2.25 * plan_year)
    assert digitization.requests_removed_share(
        plan_year, switched_on
    ) == pytest.approx(0.0072)

    # Concentration is worth at most 1.618x once the history is honoured,
    # because the archive has to grow with D to keep the history flat.
    # That is the finding the recalibration produced, and it is a property
    # of the curve, so it belongs here: the pair chosen is the one that
    # maximises it.
    def leverage(a, b):
        pair = copy.deepcopy(switched_on)
        pair["scenario"]["digitization"]["concentration_archive_share"] = a
        pair["scenario"]["digitization"]["concentration_request_share"] = b
        return (b / a) / digitization.mean_density(pair)

    assert leverage(0.20, 0.45) == pytest.approx(1.6180, abs=1e-4)
    for b in (0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.95):
        assert leverage(0.20, b) <= leverage(0.20, 0.45) + 1e-9
    assert leverage(0.20, 0.20) == pytest.approx(1.0)      # no concentration
    assert leverage(0.20, 0.80) == pytest.approx(1.23077, abs=1e-5)

    # A concentration that claims no concentration is just the line.
    flat = copy.deepcopy(switched_on)
    flat["scenario"]["digitization"]["concentration_request_share"] = 0.20
    for c in (0.1, 0.2, 0.5, 0.9):
        assert digitization.requests_removed_share(c, flat) == pytest.approx(c)


def test_the_calibration_holds_against_the_models_own_history(
    baseline, switched_on, sites, report
):
    """Spec 7, revised 11/02. The test the archive now has to earn.

    The baseline holds the non-digital share fixed at 11% through 156
    weeks of history, and that is only defensible if the share would
    barely have moved anyway - because the baseline is converting charts
    the whole time. Every onsite retrieval (W11) digitizes its chart.

    So the archive and the concentration are not free inputs. Treating a
    chart as sitting on paper until its first request digitizes it, three
    years of reactive conversion has to move the share by less than the
    half-point a flat history can hide. The first defaults failed this by
    eleven times, which is what the recalibration was for.

    Three things are checked, and the first is the one that matters:

      the recorded reactive rate still matches the history the model
        generates - a calibration against a number that has drifted is
        no calibration at all;
      the drift it implies is inside the budget;
      and it is still inside at the PLAN's higher request rate, so the
        answer does not depend on which end of the timeline is read.
    """
    # 1. The recorded rate is the history's own.
    work = work_volume.build(baseline, sites)["work"]
    hist = work[(work["period"] == "history") & (work["work_type"] == "W11")]
    measured = float(hist["work_items"].sum()) / int(
        baseline["calendar"]["history_weeks"]
    )
    recorded = digitization.reactive_rate(switched_on)
    assert measured == pytest.approx(recorded, rel=0.02), (
        f"the archive is calibrated against {recorded:,.2f} onsite retrievals "
        f"a week, but the history now generates {measured:,.2f}"
    )

    # 2. The drift that rate implies is inside the budget.
    budget = digitization.drift_budget_points(switched_on)
    drift = digitization.history_drift_points(switched_on)
    assert 0.0 < drift < budget, (
        f"three years of reactive digitization would move the share "
        f"{drift:.3f} point, against a budget of {budget}"
    )
    assert drift == pytest.approx(0.4276, abs=1e-3)

    # 3. Still inside at the plan year's higher rate.
    plan_rows = work[(work["period"] != "history") & (work["work_type"] == "W11")]
    plan_rate = float(plan_rows["work_items"].sum()) / int(
        baseline["calendar"]["plan_weeks"]
    )
    assert plan_rate > recorded          # volume grows, so this is the harder case
    assert digitization.history_drift_points(switched_on, rate=plan_rate) < budget

    # The old defaults are the counter-example, and they fail loudly.
    old = copy.deepcopy(switched_on)
    old["scenario"]["digitization"]["archive_charts"] = 250_000
    old["scenario"]["digitization"]["concentration_request_share"] = 0.80
    assert digitization.history_drift_points(old) > 10 * budget

    # And the report carries the calibration with it, so the archive is
    # never quoted without the history it was solved against.
    assert report["calibration"]["holds"] is True
    assert report["calibration"]["history_drift_points"] == pytest.approx(drift)
    assert report["calibration"]["mean_density"] == pytest.approx(1.390625)


def test_the_project_cannot_convert_a_chart_already_requested(
    walk, switched_on, sites
):
    """Siri 11/02: most-likely-first AMONG CHARTS NOT YET REQUESTED.

    A chart the operation has already pulled is already digital. The
    project cannot convert it again, and must not be credited for the
    request probability it carried - that probability is already gone.

    The archive the project draws from is therefore the surviving one,
    thinned by the operation's own reactive conversions since the first
    history week. At the calibrated archive the cap never binds inside
    the plan year, which is itself worth pinning: it says the project is
    small relative to the archive, not that the rule is absent. The
    second half of the test makes the rule bite, so a cap that quietly
    stopped working could not pass.
    """
    hot_archive, _hot_requests = digitization.concentration(switched_on)
    archive = digitization.archive_by_site(switched_on, sites)

    for row in walk.itertuples():
        hot_tier = hot_archive * archive[row.site]
        # The survivors are fewer than the tier - the history has been
        # converting all along - and the project never exceeds them.
        assert row.hot_charts_left < hot_tier
        assert row.converted_hot <= row.hot_charts_left + 1e-9
        assert row.converted_cold == pytest.approx(0.0)

    # The history alone has already taken 6% of the hot tier off the
    # shelf before the project converts anything.
    first = walk[walk["week_index"] == walk["week_index"].min()]
    network_hot = hot_archive * sum(archive.values())
    already_gone = 1.0 - float(first["hot_charts_left"].sum()) / network_hot
    assert 0.05 < already_gone < 0.08

    # Now make the rule bite: a small archive, where reactive conversion
    # has emptied most of the hot tier before the project starts. The
    # project is then into the cold tail, and the requests it removes
    # fall WELL short of what the uncapped curve would claim.
    small = copy.deepcopy(switched_on)
    small["scenario"]["digitization"]["archive_charts"] = 60_000
    tight = digitization.converted_by_week(small, sites)
    assert tight["cap_binds"].any(), "the cap has to bite somewhere"
    last = tight[tight["week_index"] == tight["week_index"].max()]
    for row in last.itertuples():
        uncapped = float(
            digitization.requests_removed_share(row.converted_fraction, small)
        )
        assert row.requests_removed_share < uncapped
    # And it still never removes more requests than exist.
    assert (tight["requests_removed_share"] <= 1.0 + 1e-12).all()


def test_a_nonsense_concentration_is_refused(switched_on):
    """An input that cannot describe an archive is an error, not a curve."""
    for bad in (0.0, 1.0, -0.1, 1.5):
        broken = copy.deepcopy(switched_on)
        broken["scenario"]["digitization"]["concentration_archive_share"] = bad
        with pytest.raises(ValueError, match="between 0 and 1"):
            digitization.requests_removed_share(0.1, broken)

    for bad in (0.0, -0.1, 1.5):
        broken = copy.deepcopy(switched_on)
        broken["scenario"]["digitization"]["concentration_request_share"] = bad
        with pytest.raises(ValueError, match="between 0 and 1"):
            digitization.requests_removed_share(0.1, broken)


# ---------------------------------------------------------------------------
# The weeks before the starts week
# ---------------------------------------------------------------------------


def test_the_weeks_before_the_starts_week_match_the_baseline(
    baseline, switched_on, sites
):
    """Spec 7: a scenario's change takes effect from its Starts week.

    Checked on the work volume itself rather than on a summary, because
    this is where the change enters the model - a per-week, per-site
    multiplier on the onsite media shares. If the multiplier leaked into a
    history week, every forecast and every backlog that rolls out of
    history would be quietly wrong.
    """
    base = work_volume.build(baseline, sites)["work"]
    run = work_volume.build(switched_on, sites)["work"]
    assert len(base) == len(run)

    keys = ["week_index", "site", "work_type", "request_type", "media"]
    merged = base.merge(run, on=keys, suffixes=("_base", "_run"))
    assert len(merged) == len(base)

    start = digitization.from_week(switched_on)
    before = merged[merged["week_index"] < start]
    assert len(before) > 0
    np.testing.assert_allclose(
        before["work_items_run"], before["work_items_base"], rtol=0, atol=0
    )

    # And from the starts week onward something really does move - a test
    # that only proved nothing changed would pass on a broken scenario.
    after = merged[merged["week_index"] >= start]
    onsite = after[after["work_type"] == "W11"]
    assert onsite["work_items_run"].sum() < onsite["work_items_base"].sum()


def test_the_retrieval_work_types_still_partition_the_charts(
    baseline, switched_on, sites
):
    """W8, W9 and W11 cover every chart exactly once (Section 9).

    A converted record does not vanish - it moves from a cabinet to the
    client system, so it leaves W11 and arrives at W8. If the scenario
    simply deleted the onsite share, total retrieval volume would fall and
    the model would be claiming the requests stopped coming.
    """
    retrieval = {"W8", "W9", "W11"}

    # PRE-LAG, from the conservation table. W8 and W9 lag 70/30 while W11
    # lags 50/50, so moving share from W11 to W8 moves items between two
    # different lag windows: in-horizon totals then differ by a few items
    # at the edges of the plan year, which is the lag doing its job rather
    # than the partition breaking. The partition is a property of the
    # generated volume, so that is where it is checked.
    def totals(assumptions):
        built = work_volume.build(assumptions, sites)
        c = built["conservation"]
        mine = c[c["work_type"].isin(retrieval)]
        return mine.groupby("work_type")["expected"].sum()

    base = totals(baseline)
    run = totals(switched_on)

    # The total is unchanged: nothing created, nothing destroyed.
    assert run.sum() == pytest.approx(base.sum(), rel=1e-9)
    # W11 falls, W8 rises by the same amount, W9 is untouched.
    assert run["W11"] < base["W11"]
    assert run["W8"] > base["W8"]
    assert run["W9"] == pytest.approx(base["W9"], rel=1e-12)
    assert (base["W11"] - run["W11"]) == pytest.approx(
        run["W8"] - base["W8"], rel=1e-9
    )

    # W12 follows W11 one-for-one, so it falls with it (spec Section 1).
    def w12(assumptions):
        built = work_volume.build(assumptions, sites)
        c = built["conservation"]
        return c[c["work_type"] == "W12"]["expected"].sum()

    assert w12(switched_on) < w12(baseline)
    # And it tracks W11 exactly, before and after.
    assert w12(switched_on) == pytest.approx(run["W11"], rel=1e-12)
    assert w12(baseline) == pytest.approx(base["W11"], rel=1e-12)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_counts_the_added_digital_retrieval_too(report):
    """Spec 7: a conversion saves W11 and W12 - and ADDS a W8.

    Reporting only the two savings would overstate the case by the cost of
    the work that replaces it. A chart in the client system still has to be
    fetched; it just takes 10 minutes instead of 25, 35 or 45.
    """
    w = report["weekly"]
    assert w["w11_hours_saved"].sum() > 0
    assert w["w12_hours_saved"].sum() > 0
    # Negative saving = hours ADDED.
    assert w["w8_hours_saved"].sum() < 0
    assert w["hours_saved_net"].sum() == pytest.approx(
        w["w11_hours_saved"].sum()
        + w["w12_hours_saved"].sum()
        + w["w8_hours_saved"].sum()
    )
    # The net is still a saving, because an onsite retrieval costs more
    # than a digital one even before the digitization behind it.
    assert w["hours_saved_net"].sum() > 0


def test_the_report_charges_every_live_week_and_no_dead_one(report, switched_on):
    """The cost starts at the starts week and never stops.

    Including the weeks where conversion is capped by an exhausted
    archive: the hours are paid whether or not there is anything left to
    convert. Nothing in the plan year hits that cap, but the arithmetic
    has to be right for the projection that runs past it.
    """
    w = report["weekly"]
    weekly_cost = digitization.weekly_cost(switched_on)
    assert report["weekly_added_cost"] == pytest.approx(weekly_cost)

    live = w[w["plan_week"] >= 13]
    dead = w[w["plan_week"] < 13]
    assert len(live) == 40 and len(dead) == 12
    np.testing.assert_allclose(live["added_cost"], weekly_cost, rtol=1e-12)
    assert (dead["added_cost"] == 0.0).all()
    assert w["added_cost"].sum() == pytest.approx(weekly_cost * 40)

    # Cumulative columns are cumulative.
    np.testing.assert_allclose(
        w["cumulative_added_cost"], w["added_cost"].cumsum(), rtol=1e-12
    )
    np.testing.assert_allclose(
        w["cumulative_saving"], w["labour_saving"].cumsum(), rtol=1e-12
    )
    np.testing.assert_allclose(
        w["cumulative_net_cost"],
        w["cumulative_added_cost"] - w["cumulative_saving"],
        rtol=1e-12,
    )


def test_the_break_even_is_a_projection_and_says_so(report):
    """It does not pay back inside the plan year, and the report is explicit.

    40 hours a week against the calibrated 2,000,000-chart archive
    converts 0.32% of it in a year. The saving is real and growing, and it
    is nowhere near $1,200 a week - so the honest answer is a projection
    with its assumption written down, not a week number inside the table.

    The projection holds the CONVERSION RATE constant, which means the
    weekly saving keeps growing: that makes break-even 2C/g rather than a
    division, and the test pins the relationship rather than the number.
    """
    assert report["break_even_plan_week"] is None
    p = report["projection"]
    assert p is not None

    # Still losing money weekly at the end of the plan year, which is why
    # dividing the balance by the last week's net would have returned
    # nothing at all.
    assert report["weekly"]["net_this_week"].iloc[-1] < 0

    cost = p["weekly_cost"]
    slope = p["saving_growth_per_live_week"]
    assert slope > 0
    assert p["live_weeks_to_weekly_break_even"] == pytest.approx(cost / slope)
    assert p["live_weeks_to_cumulative_break_even"] == pytest.approx(
        2 * cost / slope
    )
    # Cumulative break-even is exactly twice the weekly one, which is the
    # signature of a linearly growing saving.
    assert p["live_weeks_to_cumulative_break_even"] == pytest.approx(
        2 * p["live_weeks_to_weekly_break_even"]
    )
    assert p["years_to_cumulative_break_even"] == pytest.approx(
        p["live_weeks_to_cumulative_break_even"] / 52
    )

    # And the projection now steps OUTSIDE its own assumption, which it
    # has to say rather than imply (calibrated 11/02).
    #
    # The straight line holds only while there are hot charts still on
    # paper for the project to convert. At the calibrated archive that
    # runs out at about 1,340 live weeks, and break-even is projected at
    # about 2,036 - so the figure is a FLOOR, not an estimate: past the
    # kink each converted chart is worth less and the slope falls.
    #
    # Before the recalibration this flag read True at 312 weeks against a
    # 143-week break-even, and the projection was safe. The flag flipping
    # is the honest consequence of the archive the history demands.
    assert p["live_weeks_of_hot_charts_on_paper"] == pytest.approx(1340.0)
    assert p["live_weeks_to_cumulative_break_even"] == pytest.approx(
        2036.2, abs=1.0
    )
    assert p["still_inside_the_band_at_break_even"] is False
    assert p["live_weeks_to_cumulative_break_even"] > p[
        "live_weeks_of_hot_charts_on_paper"
    ]
    # Decades, not years - and the report says which.
    assert p["years_to_cumulative_break_even"] > 30


def test_the_break_even_request_probability_answers_the_real_question(report):
    """Siri 11/02: what would have to be true for this to pay?

    The weekly table is the project's own arithmetic. This is the test a
    leader can carry out of the room, and it is the clearest form of the
    answer: converting a chart costs a known amount and saves NOTHING
    unless that chart gets pulled again. So there is a request probability
    at which the project breaks even, and it can be held against what the
    archive actually offers.

    Both sides come from the model's own numbers - the value of an avoided
    paper pull from the hours and rates in the weekly table, the cost per
    chart from the project's hours - so neither can drift away from the
    report above it.
    """
    pc = report["per_chart"]
    assert pc is not None

    # Cost per chart: $1,200 a week for 160 charts.
    assert pc["cost_per_chart_converted"] == pytest.approx(7.50)

    # Value of one avoided paper pull: W11 and W12 saved, W8 added back.
    # Derived from the weekly table, so this ties to it exactly.
    weekly = report["weekly"]
    assert pc["value_of_one_avoided_paper_pull"] == pytest.approx(
        weekly["labour_saving"].sum() / pc["avoided_paper_pulls_in_the_plan_year"]
    )
    assert pc["value_of_one_avoided_paper_pull"] == pytest.approx(16.84, abs=0.05)

    # The break-even probability is the ratio of the two.
    assert pc["break_even_request_probability"] == pytest.approx(
        pc["cost_per_chart_converted"] / pc["value_of_one_avoided_paper_pull"]
    )
    assert 0.40 < pc["break_even_request_probability"] < 0.50

    # Against what a hot chart actually offers, which is about 2% a year.
    assert 0.015 < pc["hot_chart_request_probability_in_a_year"] < 0.030
    assert pc["targeting_multiple_needed"] == pytest.approx(
        pc["break_even_request_probability"]
        / pc["hot_chart_request_probability_in_a_year"]
    )
    assert pc["targeting_multiple_needed"] > 15

    # The starkest single number in the report: 6,400 charts converted
    # bought fewer than 60 avoided paper pulls inside the plan year.
    assert pc["charts_converted_in_the_plan_year"] == pytest.approx(6400.0)
    assert pc["avoided_paper_pulls_in_the_plan_year"] < 100
    hit_rate = (
        pc["avoided_paper_pulls_in_the_plan_year"]
        / pc["charts_converted_in_the_plan_year"]
    )
    assert hit_rate < 0.02
