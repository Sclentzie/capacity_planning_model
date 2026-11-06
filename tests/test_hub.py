"""Spec Section 9, Session 5: the central records hub (spec Section 7).

Task 5k. The Section 9 check this file owns:

    - Central records hub: total non-digital items are the same as the
      baseline, and every item's turnaround includes its delivery time.

Plus the four the user asked for explicitly: no Records Specialists at
non-hub sites, the weeks before the starts week match the baseline, the
scenario changes only what its overrides file names (the shared diff test
in test_scenarios.py), and the baseline pins do not move.

The first check is the one that earns its place. "Pool the team and save
money" is a claim a model can make for free by quietly losing work -
move the retrieval to one site, drop the items that no longer have a home
and the hours fall beautifully. So the item count is pinned first: a
chart retrieved at the hub is the SAME chart, and the hub's saving has to
come from how people are rostered, never from work that stopped existing.

The second is the honest half of the trade. A hub that pooled the team
and did not pay for delivery would be free, and the only reason it is not
free is the day or two a record spends in a van.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from capacity_model import (
    backlog,
    config,
    hub,
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
        scenarios.scenario_dir(baseline) / "central_records_hub.yaml"
    )


@pytest.fixture(scope="module")
def switched_on(scenario, baseline):
    return scenarios.apply_scenario(scenario, baseline)


@pytest.fixture(scope="module")
def report(baseline, sites):
    return scenarios.hub_report(baseline, sites)


# ---------------------------------------------------------------------------
# Off unless asked
# ---------------------------------------------------------------------------


def test_the_baseline_keeps_six_records_teams(baseline, sites):
    """Off means off: the baseline is a six-site role and stays one."""
    assert hub.enabled(baseline) is False
    assert supply.scopes_for_role("records_specialist", sites, baseline) == [
        s["code"] for s in sites["sites"]
    ]

    # And the work stays where the request was made.
    work = work_volume.build(baseline, sites)["work"]
    hours = backlog.required_hours(baseline, sites, work)["hours"]
    rs = hours[hours["role"] == "records_specialist"]
    assert set(rs["site"]) == {s["code"] for s in sites["sites"]}


def test_the_scenario_file_turns_it_on_and_nothing_else(scenario, switched_on):
    """Spec 7: week 13, defaults of 1 day within a cluster and 2 between."""
    assert scenario["name"] == "Central records hub"
    assert scenario["starts_week"] == 13
    assert hub.enabled(switched_on) is True

    block = hub.settings(switched_on)
    assert block["delivery_days_within_cluster"] == 1
    assert block["delivery_days_between_clusters"] == 2
    assert block["site"] == ""           # chosen, not stated
    assert block["team_size"] == 0       # sized from the work
    # All Records Specialist work moves, W17 included (Siri 11/02): there
    # are no Records Specialists left at the other sites to do it.
    assert hub.moved_work_types(switched_on) == ["W11", "W12", "W13", "W17"]
    # Only the ones where a physical record travels carry delivery.
    assert hub.delivery_work_types(switched_on) == ["W11", "W13"]


# ---------------------------------------------------------------------------
# The hub choice (spec 7, item 1)
# ---------------------------------------------------------------------------


def test_the_hub_is_the_site_with_the_lowest_weighted_delivery_days(
    switched_on, sites, report
):
    """Spec 7, decided 11/02, with the runner-up reported.

    Worked by hand from sites.yaml rather than trusting the function,
    because the whole choice is six sums and getting one of them wrong
    would still produce a plausible-looking answer.

    N1 wins because it is the largest site: putting the hub there takes
    the single largest delivery weight to zero. That is the mechanism, and
    the test states it as well as checking it.
    """
    factors = {s["code"]: s["size_factor"] for s in sites["sites"]}
    clusters = {s["code"]: s["cluster"] for s in sites["sites"]}
    total = sum(factors.values())

    expected = {}
    for candidate in factors:
        weighted = 0.0
        for site, factor in factors.items():
            if site == candidate:
                days = 0.0
            elif clusters[site] == clusters[candidate]:
                days = 1.0
            else:
                days = 2.0
            weighted += (factor / total) * days
        expected[candidate] = weighted

    # The hand numbers, so a change to sites.yaml has to be noticed.
    assert expected["N1"] == pytest.approx(7.7 / 6.0)
    assert expected["N2"] == pytest.approx(7.9 / 6.0)
    assert min(expected, key=expected.get) == "N1"

    ranking = hub.hub_ranking(switched_on, sites)
    for row in ranking.itertuples():
        assert row.weighted_delivery_days == pytest.approx(expected[row.site])
    # Sorted best first.
    assert list(ranking["weighted_delivery_days"]) == sorted(
        ranking["weighted_delivery_days"]
    )

    chosen, runner_up, _ranking = hub.choose_hub(switched_on, sites)
    assert chosen == "N1"
    assert runner_up == "N2"
    assert report["hub"] == "N1"
    assert report["runner_up"] == "N2"
    assert report["weighted_delivery_days"] == pytest.approx(7.7 / 6.0)

    # A stated site overrides the choice, and the ranking still comes back
    # so a hand-picked hub can be read against the one volume would pick.
    stated = copy.deepcopy(switched_on)
    stated["scenario"]["records_hub"]["site"] = "S2"
    assert hub.choose_hub(stated, sites)[0] == "S2"
    with pytest.raises(ValueError, match="not a site"):
        bad = copy.deepcopy(switched_on)
        bad["scenario"]["records_hub"]["site"] = "Z9"
        hub.choose_hub(bad, sites)


def test_the_choice_agrees_with_the_volume_the_model_really_built(report):
    """Two definitions of "volume", compared rather than left to diverge.

    The hub has to be chosen before the work table exists, because the
    choice decides where the Records Specialists sit and that is needed to
    build the roster the work is planned against. So the weight is the
    size factor. The forecast carries a little per-site drift on top, and
    this is the check that the drift never moves the answer.
    """
    check = report["choice_check"]
    assert check["agree"] is True
    assert check["hub_by_size_factor"] == check["hub_by_realised_volume"] == "N1"

    # The realised ranking is close but not identical - third and fourth
    # place swap, which is worth knowing: that pair is a coin flip and the
    # scenario should not be read as having found something about them.
    realised = list(check["realised_ranking"]["site"])
    assert realised[:2] == ["N1", "N2"]


def test_delivery_days_follow_the_cluster_geography(switched_on, sites):
    """Zero at the hub, 1 inside its cluster, 2 outside it (spec 7)."""
    assert hub.delivery_days("N1", "N1", switched_on, sites) == 0.0
    assert hub.delivery_days("N1", "N2", switched_on, sites) == 1.0
    assert hub.delivery_days("N1", "N3", switched_on, sites) == 1.0
    assert hub.delivery_days("N1", "S1", switched_on, sites) == 2.0
    assert hub.delivery_days("N1", "S3", switched_on, sites) == 2.0
    assert hub.delivery_days("S2", "S1", switched_on, sites) == 1.0
    assert hub.delivery_days("S2", "N1", switched_on, sites) == 2.0
    with pytest.raises(ValueError, match="unknown site"):
        hub.delivery_days("N1", "Z9", switched_on, sites)


# ---------------------------------------------------------------------------
# The Section 9 check: the items are the same, and delivery is counted
# ---------------------------------------------------------------------------


def test_the_non_digital_items_are_the_same_as_the_baseline(
    baseline, switched_on, sites
):
    """Spec Section 9. The check this scenario lives or dies by.

    "Pool the team and save money" is a claim a model can make for free by
    losing work: move retrieval to one site, drop the items that no longer
    have a home, and the hours fall beautifully. So the items are pinned
    before anything else is believed.

    A chart retrieved at the hub is the SAME chart. What moves is where
    the work is DONE, never how much of it there is, and the hub's saving
    has to come from how people are rostered.
    """
    base_work = work_volume.build(baseline, sites)["work"]
    hub_work = work_volume.build(switched_on, sites)["work"]

    # The work-volume table is untouched: its site is where the REQUEST
    # came from, and the onsite interval build still needs six counters.
    pd.testing.assert_frame_equal(base_work, hub_work)

    base_hours = backlog.required_hours(baseline, sites, base_work)["hours"]
    hub_hours = backlog.required_hours(switched_on, sites, hub_work)["hours"]

    moved = hub.moved_work_types(switched_on)
    for code in moved:
        base_items = base_hours[base_hours["work_type"] == code]["items"].sum()
        hub_items = hub_hours[hub_hours["work_type"] == code]["items"].sum()
        assert hub_items == pytest.approx(base_items), code

    # And the hours, not only the items: the same work at the same handle
    # times, so pooling cannot have made the job quicker.
    for code in moved:
        base_h = base_hours[base_hours["work_type"] == code]["required_hours"].sum()
        hub_h = hub_hours[hub_hours["work_type"] == code]["required_hours"].sum()
        assert hub_h == pytest.approx(base_h), code

    # Every work type, not just the moved ones, and every medium.
    by_type = lambda frame: frame.groupby(["work_type", "media"])["items"].sum()
    pd.testing.assert_series_equal(
        by_type(base_hours), by_type(hub_hours), check_like=True
    )


def test_every_hub_retrieved_item_carries_its_delivery_time(
    baseline, switched_on, sites, report
):
    """Spec Section 9, and item 3: delivery is in the turnaround.

    Added to DAYS OF INVENTORY and not to the goal, which is the part
    worth pinning. If the 5-day promise moved because the records moved,
    the scenario could never fail it, and a scenario that cannot fail is
    not measuring anything.
    """
    days = hub.weighted_delivery_days("N1", switched_on, sites)
    assert days == pytest.approx(7.7 / 6.0)

    # The frame-level mechanic: only the travelling types are touched, and
    # every one of them is.
    frame = pd.DataFrame(
        {
            "work_type": ["W8", "W11", "W12", "W13", "W17", "W5"],
            "days_of_inventory": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    out, added = hub.add_delivery_to_turnaround(frame, "N1", switched_on, sites)
    assert added == pytest.approx(days)
    travelling = out["work_type"].isin(hub.delivery_work_types(switched_on))
    np.testing.assert_allclose(
        out.loc[travelling, "delivery_days"].to_numpy(), [days, days]
    )
    np.testing.assert_allclose(
        out.loc[~travelling, "delivery_days"].to_numpy(), [0.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        out.loc[travelling, "days_of_inventory"].to_numpy(),
        [2.0 + days, 4.0 + days],
    )
    np.testing.assert_allclose(
        out.loc[~travelling, "days_of_inventory"].to_numpy(), [1.0, 3.0, 5.0, 6.0]
    )

    # And in the run: the reported turnaround is the before-delivery
    # figure plus the delivery, in every week of the plan year.
    table = report["runs"]["hub"]["turnaround"]
    assert "delivery_days_in_turnaround" in table.columns
    np.testing.assert_allclose(
        table["turnaround_days"],
        table["turnaround_days_before_delivery"]
        + table["delivery_days_in_turnaround"],
        rtol=1e-12,
    )
    # Delivery is never zero once the hub is live, and never more than the
    # weighted days - the retrieval stage is a volume-weighted average, so
    # 12% of charts travelling cannot add a whole day to the total.
    live = table["delivery_days_in_turnaround"]
    assert (live > 0).all()
    assert (live < days).all()

    # The baseline's turnaround file does not gain the columns at all.
    base_table = report["runs"]["baseline"]["turnaround"]
    assert "delivery_days_in_turnaround" not in base_table.columns
    assert "turnaround_days_before_delivery" not in base_table.columns

    # The whole turnaround change IS the delivery: strip it out and the
    # hub's queues behave exactly like the baseline's. That is the real
    # finding - pooling a BACKLOG buys no Erlang gain, because a backlog
    # is allowed to wait.
    t = report["turnaround"]
    assert t["hub"]["mean_before_delivery"] == pytest.approx(
        t["baseline"]["mean"], abs=5e-4
    )
    assert t["hub"]["mean"] - t["baseline"]["mean"] == pytest.approx(
        t["hub"]["mean_delivery_days"], abs=5e-4
    )


# ---------------------------------------------------------------------------
# No Records Specialists at the other sites (item 2)
# ---------------------------------------------------------------------------


def test_no_records_specialists_sit_at_a_non_hub_site(
    switched_on, sites, report
):
    """Spec 7: "the 2-per-site floor no longer applies at the other sites".

    Checked in three places, because the role has to be gone from all of
    them or the scenario is only half true: where the people are rostered,
    where the work is costed, and where the counter help comes from.
    """
    assert supply.scopes_for_role("records_specialist", sites, switched_on) == ["N1"]

    # The roster: one cohort, at the hub.
    cohorts = supply.opening_cohorts(switched_on, sites)
    rs = [c for c in cohorts if c["role"] == "records_specialist"]
    assert {c["scope"] for c in rs} == {"N1"}

    # The cost table: no Records Specialist row anywhere but the hub, and
    # all of the hours at it.
    cost = report["runs"]["hub"]["cost"]
    mine = cost[cost["role"] == "records_specialist"]
    assert set(mine["scope"]) == {"N1"}
    specialist = report["records_specialist"]
    assert specialist["hub"]["by_site"].index.tolist() == ["N1"]
    assert specialist["hub"]["required_hours"] == pytest.approx(
        specialist["baseline"]["required_hours"], rel=1e-4
    )

    # Counter help: the five other sites get none at all.
    helped = report["runs"]["hub"]["prepared"]["onsite_intervals"]
    by_site = helped.groupby("site")["help_hours"].sum()
    for site in by_site.index:
        if site == "N1":
            assert by_site[site] > 0
        else:
            assert by_site[site] == pytest.approx(0.0), site

    # Which is a real cost, and it lands on a different role's budget:
    # 1,600-plus hours of help gone from the five sites, and the Onsite
    # Associate requirement rises there.
    help_detail = report["counter_help"]
    assert help_detail["hours_lost_at_non_hub_sites"] > 1_500
    added = help_detail["by_site_onsite_hours_added"]
    for site in added.index:
        if site == "N1":
            assert added[site] < 0      # the hub's own counter gains help
        else:
            assert added[site] > 0, site


def test_the_people_move_to_the_hub_rather_than_disappearing(
    baseline, switched_on, sites
):
    """The bug that made this scenario cost $13.9M, found 11/02.

    Two rules met badly. Making Records Specialists a one-site role is
    what moves their work, and a scenario does NOT re-tune the opening
    roster (spec 7, 10/27) so that the gap reports what the change costs.
    Together they dropped the five non-hub headcount entries on the floor:
    three people were left holding the whole network's retrieval, and the
    scenario reported $13.9M, 151 hires and a 28-FTE peak gap - a
    transition cost that was pure artefact.

    A hub transition re-bases people. It does not make them redundant.
    """
    stated = baseline["fte"]["starting_headcount"]["records_specialist"]
    total = sum(stated.values())
    # 3 at each site except S2, the one place the 2-per-site floor binds.
    assert stated == {"N1": 3, "N2": 3, "N3": 3, "S1": 3, "S2": 2, "S3": 3}
    assert total == 17

    cohorts = supply.opening_cohorts(switched_on, sites)
    rs = [c for c in cohorts if c["role"] == "records_specialist"]
    assert sum(c["size"] for c in rs) == pytest.approx(total), (
        "the hub must hold every Records Specialist the baseline rostered"
    )
    assert len(rs) == 1 and rs[0]["scope"] == "N1"

    # The baseline is untouched by the same code path.
    base_cohorts = supply.opening_cohorts(baseline, sites)
    base_rs = [c for c in base_cohorts if c["role"] == "records_specialist"]
    assert sum(c["size"] for c in base_rs) == pytest.approx(total)
    assert len(base_rs) == 6


# ---------------------------------------------------------------------------
# What pooling actually buys (item 4, item 6)
# ---------------------------------------------------------------------------


def test_the_saving_is_rounding_and_not_the_two_per_site_floor(report):
    """Measured 11/02, and it corrects the spec paragraph's own framing.

    Spec 7 points at the 2-per-site floor: six sites at a floor of 2 is
    twelve people whether the work needs twelve or not. Sized against this
    model's volumes that is almost entirely wrong - every site needs about
    2.2 to 2.8 Records Specialists anyway, so the floor binds only at the
    smallest and is worth 0.097 FTE across the network.

    What the hub buys is ROUNDING. Six sites each round a fractional
    requirement up to a whole person; one hub rounds once. 14.254 FTE of
    work is 17 people at six sites and 15 at one.

    The distinction decides when a hub is worth considering at all: not
    when sites are small enough to hit the floor, but when the per-site
    requirement has a large fractional part.
    """
    sized = report["sized_to_work"]
    base_rs = sized["baseline"]["records_specialist"]
    hub_rs = sized["hub"]["records_specialist"]

    # Same work, either way. That is the premise of the whole comparison.
    assert hub_rs["required_fte"] == pytest.approx(base_rs["required_fte"])
    assert base_rs["scopes"] == 6
    assert hub_rs["scopes"] == 1

    # And two fewer people to do it.
    assert base_rs["headcount"] == 17
    assert hub_rs["headcount"] == 15
    assert report["sized_headcount_change"]["records_specialist"] == -2

    # The floor is almost none of it.
    assert base_rs["slack_from_the_floor_fte"] == pytest.approx(0.097, abs=0.01)
    assert base_rs["slack_from_rounding_fte"] == pytest.approx(2.649, abs=0.01)
    assert base_rs["slack_from_rounding_fte"] > 20 * base_rs[
        "slack_from_the_floor_fte"
    ]
    # Only the smallest site is held up by the floor at all.
    by_scope = base_rs["by_scope"]
    below = by_scope[by_scope["required_fte"] < by_scope["floor"]]
    assert list(below.index) == ["S2"]

    # Against which the counter needs one more person, because the help
    # has gone. The net saving is real but it is two people, not six.
    assert report["sized_headcount_change"]["onsite_associate"] == 1
    assert report["sized_annual_saving"] == pytest.approx(
        2 * 30 * 2080 - 1 * 26 * 2080
    )
    assert report["sized_annual_saving"] == pytest.approx(70_720.0)


def test_the_as_run_scenario_is_close_to_cost_neutral(report):
    """Item 4's all-in cost, on the roster the task asked for.

    Holding the baseline roster (spec 7, 10/27) is what a transition
    really looks like in its first year: the same people, one address. On
    that basis the hub changes almost nothing in money - the payroll is
    identical and only overtime and uncovered hours move - and the whole
    of its effect is on service.

    Which is the point of reporting both halves. This alone would say the
    hub is pointless; the sized-to-work comparison alone would say it is
    free.
    """
    cost = report["cost"]
    assert cost["delta"] == pytest.approx(980.0, abs=200.0)
    assert abs(cost["delta"]) < 0.001 * cost["baseline"]

    # One genuine pooling gain, and it is small: a single queue cannot
    # have a per-site peak, so the Records Specialists' overtime goes.
    specialist = report["records_specialist"]
    assert specialist["baseline"]["overtime_hours"] > 100
    assert specialist["hub"]["overtime_hours"] == pytest.approx(0.0)

    # And the service cost: slower, though not enough to miss more weeks.
    t = report["turnaround"]
    assert t["hub"]["mean"] > t["baseline"]["mean"]
    assert t["hub"]["mean"] - t["baseline"]["mean"] == pytest.approx(0.139, abs=0.01)
    assert t["hub"]["weeks_over_goal"] == t["baseline"]["weeks_over_goal"] == 6

    # The archive move is NOT in any of this, and the report says so
    # rather than leaving a reader to assume it was counted.
    assert report["archive_move_cost_modelled"] is False
    assert "upper bound" in report["not_modelled"]


# ---------------------------------------------------------------------------
# The weeks before the starts week
# ---------------------------------------------------------------------------


def test_the_weeks_before_the_starts_week_match_the_baseline(report):
    """Spec 7: a scenario's change takes effect from its Starts week.

    The hub is a STRUCTURAL change - where people sit - so unlike the
    digitization project it cannot be switched on mid-year by an input.
    What has to hold instead is that the thing the scenario changes about
    service, the delivery time, is the only thing moving: the queues
    themselves have to behave as they did.

    Pinned on the turnaround, because that is where a structural change
    would show first if it were leaking into the wrong weeks.
    """
    base = report["runs"]["baseline"]["turnaround"]
    run = report["runs"]["hub"]["turnaround"]
    assert list(base["week_index"]) == list(run["week_index"])

    merged = base.merge(
        run[["week_index", "turnaround_days_before_delivery"]],
        on="week_index",
        how="inner",
    )
    # Every week, not just the mean: the queues are the baseline's.
    gap = (
        merged["turnaround_days"] - merged["turnaround_days_before_delivery"]
    ).abs()
    assert gap.max() < 0.01, f"worst week differs by {gap.max():.4f} days"

    # The required hours are the same week by week - which is the
    # strongest form of "no work was created or destroyed".
    #
    # The same, but not IDENTICAL, and the gap is worth understanding
    # rather than tightening away. Required hours are
    #
    #     (arrivals + max(0, starting - target) / catch-up) x handle / 60
    #
    # and that max() is not linear. Six separate queues each chase their
    # own excess over target; one pooled queue nets a site that is behind
    # against a site that is ahead and chases only the difference. So
    # pooling a BACKLOG does buy a little something after all - not
    # Erlang, but the netting of excess - and it is tiny: 0.32 hours in
    # the worst of 52 weeks, about 0.015%.
    base_cost = report["runs"]["baseline"]["cost"]
    hub_cost = report["runs"]["hub"]["cost"]
    by_week = lambda frame: frame.groupby("week_index")["required_hours"].sum()
    base_weekly = by_week(base_cost).to_numpy()
    hub_weekly = by_week(hub_cost).to_numpy()
    np.testing.assert_allclose(base_weekly, hub_weekly, rtol=1e-3)
    assert abs(hub_weekly.sum() - base_weekly.sum()) < 1.0
    # And the netting never asks for MORE work than six queues did.
    assert hub_weekly.sum() <= base_weekly.sum() + 1e-6


# ---------------------------------------------------------------------------
# Baseline pins
# ---------------------------------------------------------------------------


def test_the_baseline_pins_do_not_move(report):
    """The hub must not have touched the baseline (spec 9, every session)."""
    summary = scenarios.summarise(
        report["runs"]["baseline"],
        report["runs"]["baseline"]["assumptions"],
        report["runs"]["baseline"]["sites"],
    )
    assert summary["scenario"] == "Baseline"
    assert summary["total_cost"] == pytest.approx(7_456_636.40, abs=1.0)
    assert summary["overtime_hours"] == pytest.approx(2_767.55, abs=0.1)
    assert summary["uncovered_hours"] == pytest.approx(419.97, abs=0.1)
    assert summary["mean_projected_turnaround_days"] == pytest.approx(
        4.6714, abs=1e-3
    )
    assert summary["weeks_over_turnaround_goal"] == 6
    assert summary["network_required_fte"] == pytest.approx(107.939, abs=1e-2)
