"""Spec Section 9, Session 4: the gap, overtime and its consequence (6.1-6.4).

build_plan covers the BACK-OFFICE roles - Tier 1, Tier 2, QA and the
Records Specialists. The counter is in test_coverage.py, because it is
checked at a different grain and its consequence is a service level rather
than a backlog.

Checked BY WEEK (spec 6.2, decided 10/26). A back-office queue absorbs
within-week timing - a Wednesday shortfall can wait for Thursday and still
clear inside the turnaround goal - so only a week that comes up short buys
overtime. Live work is the opposite case and is checked by day.

The property worth stating up front, because several tests circle it: a
shortfall does not vanish. Hours nobody worked become work nobody did, and
in a back-office queue that work is still there next Monday. So the tests
check not just that the arithmetic balances but that the consequence
actually lands in the backlog and shows up in turnaround.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from capacity_model import backlog, config, intervals, plan, work_volume


@pytest.fixture(scope="module")
def assumptions():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def work(assumptions, sites):
    return work_volume.build(assumptions, sites)["work"]


@pytest.fixture(scope="module")
def built(assumptions, sites, work):
    return plan.build_plan(assumptions, sites, work)


@pytest.fixture(scope="module")
def fully_staffed(assumptions, sites, work):
    return plan.build_plan(assumptions, sites, work, unconstrained=True)


@pytest.fixture(scope="module")
def turnaround(built, assumptions, sites, work):
    return plan.weekly_turnaround(built["queues"], assumptions, sites, work)


@pytest.fixture(scope="module")
def turnaround_fully_staffed(fully_staffed, assumptions, sites, work):
    return plan.weekly_turnaround(fully_staffed["queues"], assumptions, sites, work)


# ---------------------------------------------------------------------------
# Who is in scope
# ---------------------------------------------------------------------------


def test_the_back_office_roles_are_planned_weekly(built, assumptions):
    """Spec 6.2, 10/26: everyone with a queue, checked by week.

    The Onsite Associates are deliberately absent: a walk-in cannot wait,
    so they are checked by day instead.
    """
    assert set(plan.remote_roles(assumptions)) == {"tier_1", "tier_2", "qa_specialist"}
    assert set(plan.back_office_roles(assumptions)) == {
        "tier_1", "tier_2", "qa_specialist", "records_specialist"
    }
    for frame in (built["gap"], built["cost"], built["queues"]):
        assert set(frame["role"]) == {
            "tier_1", "tier_2", "qa_specialist", "records_specialist"
        }
        remote = frame[frame["role"] != "records_specialist"]
        assert set(remote["scope"]) == {"NETWORK"}
        per_site = frame[frame["role"] == "records_specialist"]
        assert set(per_site["scope"]) == {"N1", "N2", "N3", "S1", "S2", "S3"}


def test_the_plan_runs_over_the_plan_year(built, assumptions):
    cal = assumptions["calendar"]
    weeks = np.sort(built["cost"]["week_index"].unique())
    assert weeks[0] == cal["history_weeks"]
    assert len(weeks) == cal["plan_weeks"] == 52
    assert set(built["gap"]["day_of_week"]) == set(
        assumptions["intraday"]["day_of_week_share"]
    )


# ---------------------------------------------------------------------------
# The daily gap and overtime (spec 6.1, 6.2)
# ---------------------------------------------------------------------------


def test_the_daily_split_adds_back_to_the_week(built, assumptions):
    """Weekly required hours spread by the day-of-week shares (spec 3.4)."""
    shares = assumptions["intraday"]["day_of_week_share"]
    gap = built["gap"]
    weekly = gap.groupby(["week_index", "role", "scope"])["required_hours"].sum()
    stated = built["cost"].set_index(["week_index", "role", "scope"])["required_hours"]
    np.testing.assert_allclose(weekly.sort_index(), stated.sort_index(), rtol=1e-9)

    for _key, rows in gap.groupby(["week_index", "role", "scope"]):
        total = rows["required_hours"].sum()
        for row in rows.itertuples():
            assert row.required_hours == pytest.approx(
                total * shares[row.day_of_week], rel=1e-9
            )


def test_overtime_never_exceeds_its_weekly_cap(built, assumptions):
    """Spec 5.1: 10% of the role's paid hours that week.

    The cap is what keeps "treat overtime as fully productive" honest. Real
    overtime is tired time, and a model that could buy unlimited amounts of
    it would never report a gap at all.
    """
    paid = assumptions["fte"]["paid_hours_per_week"]
    settings = assumptions["supply"]["roles"]
    cost = built["cost"]

    for row in cost.itertuples():
        cap = settings[row.role]["overtime_cap"] * row.paid_headcount * paid
        assert settings[row.role]["overtime_cap"] == 0.10
        assert row.overtime_hours <= cap + 1e-9, f"{row.role} week {row.week_index}"


def test_no_overtime_in_a_week_whose_hours_cover_its_work(built):
    """Spec 6.2, decided 10/26: the check is weekly, so the week decides.

    A week whose available hours cover its required hours buys no overtime
    at all, however badly the days inside it line up. That is the whole
    change: the queue absorbs a Wednesday that ran short because Thursday
    can still clear it inside the turnaround goal.
    """
    cost = built["cost"]
    covered = cost[cost["available_hours"] >= cost["required_hours"] - 1e-9]
    assert len(covered) > 0, "expected some comfortably covered weeks"
    assert covered["overtime_hours"].abs().max() < 1e-9
    assert covered["uncovered_hours"].abs().max() < 1e-9

    # And those weeks are not trivially free of daily mismatch - if they
    # were, the weekly check would be indistinguishable from the daily one.
    gap = built["gap"]
    keys = set(zip(covered["week_index"], covered["role"]))
    inside = gap[[(w, r) in keys for w, r in zip(gap["week_index"], gap["role"])]]
    assert (inside["daily_mismatch_hours"] > 1e-6).any(), (
        "covered weeks should still contain short days"
    )


def test_overtime_is_used_only_when_the_week_is_short(built):
    """Never more than the week's own shortfall, and never without one.

    There is nothing to gain from paying 1.5x for hours with no work
    behind them.
    """
    cost = built["cost"]
    assert (cost["overtime_hours"] >= -1e-12).all()
    assert (cost["overtime_hours"] <= cost["shortfall_hours"] + 1e-9).all()

    short = cost[cost["shortfall_hours"] > 1e-9]
    assert len(short) > 0, "expected some short weeks, or this proves nothing"
    assert short["overtime_hours"].sum() > 0


def test_shortfall_is_what_is_left_after_cluster_coverage(built):
    """Spec 6.1 and 6.2, at the weekly grain, in the 6.2 order.

    Borrowing is step 1, so by the time the word "shortfall" is used for
    the overtime decision it already means "after the cluster helped".
    Measuring it before borrowing would buy overtime for hours a
    neighbouring site had already driven over and worked.

    A surplus week is zero, not negative: treating it as negative would
    let a slack week cancel a short one, and a backlog does not work
    backwards in time.
    """
    cost = built["cost"]
    expected = (cost["required_hours"] - cost["covered_hours"]).clip(lower=0.0)
    np.testing.assert_allclose(cost["shortfall_hours"], expected, rtol=1e-9, atol=1e-12)
    assert (cost["shortfall_hours"] >= 0).all()

    # covered = available, net of lending, plus whatever was borrowed in.
    np.testing.assert_allclose(
        cost["covered_hours"],
        cost["available_hours"] + cost["borrowed_hours"],
        rtol=1e-9,
    )

    surplus = cost[cost["covered_hours"] > cost["required_hours"]]
    assert len(surplus) > 0
    assert surplus["shortfall_hours"].abs().max() < 1e-12

    # And borrowing genuinely moved the answer somewhere, or this would be
    # the same test it was before part 2.
    assert cost["borrowed_hours"].sum() > 0


def test_uncovered_is_the_shortfall_overtime_could_not_reach(built):
    """Spec 6.2: uncovered = shortfall - overtime, never negative."""
    cost = built["cost"]
    np.testing.assert_allclose(
        cost["uncovered_hours"],
        cost["shortfall_hours"] - cost["overtime_hours"],
        rtol=1e-9,
        atol=1e-12,
    )
    assert (cost["uncovered_hours"] >= -1e-9).all()

    # Uncovered hours exist only where the cap ran out.
    uncovered = cost[cost["uncovered_hours"] > 1e-9]
    assert len(uncovered) > 0
    np.testing.assert_allclose(
        uncovered["overtime_hours"], uncovered["overtime_cap_hours"], rtol=1e-9
    )


def test_the_daily_mismatch_is_reported_but_not_acted_on(built):
    """The within-week timing is visible, and absorbed by the queue.

    Monday carries 23% of the volume while Wednesday loses 34.5% of its
    hours to shrinkage, so days inside a balanced week genuinely do not
    line up. Reporting it keeps that fact available without letting it buy
    overtime that the turnaround goal does not need.
    """
    gap = built["gap"]
    np.testing.assert_allclose(
        gap["daily_mismatch_hours"],
        gap["required_hours"] - gap["available_hours"],
        rtol=1e-9,
    )
    assert (gap["daily_mismatch_hours"] > 0).any(), "some days run short"
    assert (gap["daily_mismatch_hours"] < 0).any(), "and some run slack"

    # The week-level columns repeat the weekly decision on every day row.
    for _key, rows in gap.groupby(["week_index", "role", "scope"]):
        assert rows["week_overtime_hours"].nunique() == 1
        assert rows["week_uncovered_hours"].nunique() == 1


def test_overtime_is_shared_out_in_proportion_to_each_day_s_shortfall():
    """Not first-come, which would be an artefact of iteration order.

    A manager authorising overtime on Monday does not yet know Thursday
    will be worse; a weekly plan does, so it spreads the budget rather than
    spending it on whichever day it happened to visit first.
    """
    shortfall = {"monday": 40.0, "tuesday": 0.0, "wednesday": 60.0}

    uncapped = plan.allocate_overtime(shortfall, 1000.0)
    assert uncapped == pytest.approx(shortfall)

    capped = plan.allocate_overtime(shortfall, 50.0)
    assert sum(capped.values()) == pytest.approx(50.0)
    assert capped["monday"] == pytest.approx(20.0)
    assert capped["wednesday"] == pytest.approx(30.0)
    assert capped["tuesday"] == 0.0

    assert plan.allocate_overtime(shortfall, 0.0) == {d: 0.0 for d in shortfall}
    assert plan.allocate_overtime({"monday": 0.0}, 100.0) == {"monday": 0.0}


# ---------------------------------------------------------------------------
# The consequence: the backlog (spec 4.3)
# ---------------------------------------------------------------------------


def test_backlog_work_is_conserved(built):
    """Nothing lost, nothing invented: ending = opening + arrivals - completed.

    The single most important property here. A projection that leaked work
    would make a shortfall look harmless, which is the opposite of what
    this module exists to show.
    """
    queues = built["queues"]
    np.testing.assert_allclose(
        queues["ending_backlog"],
        queues["opening_backlog"] + queues["arrivals"] - queues["completed"],
        rtol=1e-9,
        atol=1e-9,
    )
    assert (queues["ending_backlog"] >= -1e-9).all()
    assert (queues["completed"] >= -1e-9).all()
    assert (
        queues["completed"] <= queues["opening_backlog"] + queues["arrivals"] + 1e-9
    ).all()


def test_each_queue_opens_where_it_closed(built):
    """The backlog rolls, which is what makes a bad week cost later weeks."""
    queues = built["queues"].sort_values("week_index")
    for _key, rows in queues.groupby(
        ["role", "scope", "work_type", "media"], dropna=False
    ):
        opening = rows["opening_backlog"].to_numpy()
        ending = rows["ending_backlog"].to_numpy()
        np.testing.assert_allclose(opening[1:], ending[:-1], rtol=1e-9, atol=1e-9)


def test_a_shortfall_leaves_more_work_in_the_queue(built, fully_staffed):
    """The consequence, measured against the fully staffed counterfactual.

    Not "a short week grows the backlog" - that is not guaranteed, and
    asserting it would be wrong. A queue sitting above target is given
    catch-up hours, so it can still shrink in a week the role as a whole
    came up short.

    Nor is it true queue by queue that less gets completed: a backed-up
    queue is granted catch-up hours, so in a later week it can take a
    LARGER share and complete more than the fully staffed run ever needed
    to. The guarantees live at the level where the work has to balance.
    """
    actual, ideal = built["queues"], fully_staffed["queues"]

    # Same arrivals on both sides, so the comparison is like for like.
    assert actual["arrivals"].sum() == pytest.approx(ideal["arrivals"].sum(), rel=1e-12)

    # Across the year, a shortfall finishes strictly less work...
    assert actual["completed"].sum() < ideal["completed"].sum()

    # ...and every week carries at least as much backlog as it would have.
    weekly = (
        actual.groupby("week_index")["ending_backlog"].sum()
        - ideal.groupby("week_index")["ending_backlog"].sum()
    )
    assert (weekly >= -1e-9).all()
    assert weekly.max() > 1.0, "the shortfall should leave real work behind"

    # The work left behind at the end is exactly the work never completed.
    # Measured on the FINAL week's backlog - a level, not a flow. Summing
    # the backlog across all 52 weeks would add up stock positions and
    # compare to a flow, which is meaningless.
    last = actual["week_index"].max()
    left_behind = (
        actual[actual.week_index == last]["ending_backlog"].sum()
        - ideal[ideal.week_index == last]["ending_backlog"].sum()
    )
    never_done = ideal["completed"].sum() - actual["completed"].sum()
    assert left_behind == pytest.approx(never_done, rel=1e-6)
    assert left_behind > 0

    # And the backlog does visibly grow somewhere, or nothing is accruing.
    grew = actual["ending_backlog"] > actual["opening_backlog"] + 1e-6
    assert grew.any()


def test_queues_are_never_given_more_hours_than_they_asked_for(built):
    """Spec 4.3: required means hold AT target, not empty the queue.

    A healthy back-office queue is not a queue of zero. Without this cap a
    slack week would drive every backlog to nothing and turnaround would
    fall below the goal - which is not what a team with spare hours does.
    """
    queues = built["queues"]
    assert (queues["allocated_hours"] <= queues["required_hours"] + 1e-9).all()


# ---------------------------------------------------------------------------
# Turnaround (spec 4.4)
# ---------------------------------------------------------------------------


def test_with_no_shortfall_turnaround_holds_the_fully_staffed_baseline(
    fully_staffed, turnaround_fully_staffed, assumptions
):
    """Spec 4.4: fully staffed, the first plan week lands on 5.23.

    The first week is the one where every queue opens exactly at target, so
    it is the week the baseline figure describes. Later weeks drift BELOW
    it as demand grows: the 4.3 formula only ever works a backlog down
    toward target, never tops it back up, so a rising arrival rate leaves
    the queue holding fewer days of work than the goal allows.
    """
    queues = fully_staffed["queues"]
    np.testing.assert_allclose(
        queues["allocated_hours"], queues["required_hours"], rtol=1e-9
    )

    first = turnaround_fully_staffed["week_index"].min()
    opening = turnaround_fully_staffed[
        turnaround_fully_staffed.week_index == first
    ].iloc[0]
    assert opening["turnaround_days"] == pytest.approx(5.232, abs=0.005)

    goal = assumptions["backlog"]["overall_turnaround_goal_days"]
    assert opening["turnaround_days"] > goal, "the exception steps push it past 5"


def test_a_shortfall_can_only_make_turnaround_worse(
    turnaround, turnaround_fully_staffed
):
    """Never better, week by week, against the fully staffed counterfactual.

    This is the test that makes the turnaround figure mean something.
    Turnaround moves with holidays and the demand curve as well as with
    staffing - the highest week of the year is a holiday week where
    arrivals collapse and the backlog therefore represents more days of
    work. Comparing like for like isolates what the shortfall itself did.
    """
    merged = turnaround.merge(
        turnaround_fully_staffed, on="week_index", suffixes=("", "_full")
    )
    degradation = merged["turnaround_days"] - merged["turnaround_days_full"]
    assert (degradation >= -1e-9).all(), "a shortfall must never improve turnaround"
    assert degradation.max() > 0.05, "some week should be visibly degraded"


def test_turnaround_is_reported_for_every_stage_and_week(turnaround, assumptions):
    cal = assumptions["calendar"]
    assert len(turnaround) == cal["plan_weeks"]
    for stage in backlog.TURNAROUND_STAGES:
        assert stage in turnaround.columns
        assert (turnaround[stage] > 0).all()
    np.testing.assert_allclose(
        turnaround[list(backlog.TURNAROUND_STAGES)].sum(axis=1),
        turnaround["turnaround_days"],
        rtol=1e-9,
    )


def test_every_queue_on_the_turnaround_path_is_now_projected(built, assumptions):
    """Nothing on the five stages is held at target any more (spec 4.4).

    Part 1 held the Records Specialist steps - onsite retrieval,
    missing-record search, certified copies - at their goal, so the
    turnaround it reported was what the REMOTE shortfall alone did. Now
    every role with a queue is projected, so the figure is the whole one.
    """
    assert "records_specialist" in set(built["queues"]["role"])
    on_path = set(backlog.goal_days(assumptions)) - {"W12", "W21", "W22", "W23"}
    projected = set(built["queues"]["work_type"])
    assert on_path <= projected, sorted(on_path - projected)


# ---------------------------------------------------------------------------
# Cost (spec 6.4)
# ---------------------------------------------------------------------------


def test_regular_cost_pays_everyone_on_the_payroll(built, assumptions):
    """Spec 6.4: headcount x 40 x hourly cost, onboarding included.

    People still in compliance training are paid in full and produce
    nothing, which is precisely why hiring late is expensive. A cost model
    that only paid productive people would hide that.
    """
    paid = assumptions["fte"]["paid_hours_per_week"]
    settings = assumptions["supply"]["roles"]
    cost = built["cost"]

    for row in cost.itertuples():
        expected = row.paid_headcount * paid * settings[row.role]["hourly_cost"]
        assert row.regular_cost == pytest.approx(expected, rel=1e-9)
    assert (cost["regular_cost"] > 0).all()


def test_overtime_costs_time_and_a_half(built, assumptions):
    """Spec 6.4: overtime hours x hourly cost x the role's pay rate."""
    settings = assumptions["supply"]["roles"]
    for row in built["cost"].itertuples():
        rate = settings[row.role]
        assert rate["overtime_pay_multiple"] == 1.5
        expected = row.overtime_hours * rate["hourly_cost"] * rate["overtime_pay_multiple"]
        assert row.overtime_cost == pytest.approx(expected, rel=1e-9)


def test_total_cost_is_regular_plus_overtime(built):
    cost = built["cost"]
    np.testing.assert_allclose(
        cost["total_cost"], cost["regular_cost"] + cost["overtime_cost"], rtol=1e-12
    )
    assert cost["overtime_cost"].sum() > 0
    assert cost["overtime_cost"].sum() < cost["regular_cost"].sum() * 0.10


# ---------------------------------------------------------------------------
# The weekly check (spec 6.2, decided 10/26)
# ---------------------------------------------------------------------------


def test_the_weekly_check_buys_the_same_outcome_for_less_overtime(
    built, assumptions, sites, work
):
    """Why the 10/26 decision is free: the queues cannot tell the difference.

    Hours applied to a queue are capped at required (spec 4.3). So in a week
    that is covered, both checks hand the queue exactly its required hours;
    and in a week short enough to exhaust the cap, both hand it available
    plus the cap. The daily check differed only in the band between - where
    it bought overtime that the cap then threw away.

    Reproduced here by running the daily rule over the same inputs and
    confirming the queue outcomes are identical while the overtime bill is
    not.
    """
    day_shares = intervals.day_of_week_shares(assumptions)
    cost = built["cost"]
    gap = built["gap"].set_index(["week_index", "role", "scope", "day_of_week"])
    settings = assumptions["supply"]["roles"]
    paid = assumptions["fte"]["paid_hours_per_week"]

    daily_overtime = 0.0
    for row in cost.itertuples():
        shortfall = {
            day: max(0.0, float(gap.loc[(row.week_index, row.role, row.scope, day),
                                        "daily_mismatch_hours"]))
            for day in day_shares
        }
        cap = settings[row.role]["overtime_cap"] * row.paid_headcount * paid
        this_week = sum(plan.allocate_overtime(shortfall, cap).values())
        daily_overtime += this_week

        # Week by week: the daily rule never buys less, and the hours that
        # actually reach the queue are identical once capped at required.
        assert this_week >= row.overtime_hours - 1e-9
        # Measured from COVERED hours, so cluster borrowing - which is step 1
        # either way - is held fixed and only the overtime rule differs.
        assert min(row.covered_hours + this_week, row.required_hours) == pytest.approx(
            min(row.covered_hours + row.overtime_hours, row.required_hours),
            rel=1e-9,
            abs=1e-9,
        )

    weekly_overtime = cost["overtime_hours"].sum()
    assert weekly_overtime < daily_overtime, "the weekly check should buy less"
    assert weekly_overtime < daily_overtime * 0.75


def test_the_overtime_cap_still_holds_under_the_weekly_check(built, assumptions):
    """Spec 5.1: 10% of the role's paid hours, whatever the grain."""
    paid = assumptions["fte"]["paid_hours_per_week"]
    settings = assumptions["supply"]["roles"]
    for row in built["cost"].itertuples():
        cap = settings[row.role]["overtime_cap"] * row.paid_headcount * paid
        assert row.overtime_cap_hours == pytest.approx(cap, rel=1e-9)
        assert row.overtime_hours <= cap + 1e-9


def test_the_fully_staffed_baseline_is_unchanged_at_five_point_two_three(
    turnaround_fully_staffed,
):
    """The weekly check must not move the yardstick (spec 4.4).

    Fully staffed there is no shortfall at any grain, so the first plan
    week still opens every queue at target and lands on 5.23.
    """
    first = turnaround_fully_staffed["week_index"].min()
    opening = turnaround_fully_staffed[
        turnaround_fully_staffed.week_index == first
    ].iloc[0]
    assert opening["turnaround_days"] == pytest.approx(5.232, abs=0.005)
