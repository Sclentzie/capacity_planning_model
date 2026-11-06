"""Spec Section 9, Session 4: the hiring plan (spec 6.5) and cost (6.4).

Covers hiring.py, split out of plan.py on 10/26, plus plan.plan_cost. The
Section 9 checks this file owns:

    - Hire-by examples land on weeks 18 (Onsite Associates) and 21 (Tier 2).
    - Class size covers the shortfall after the attrition expected before
      full ramp.
    - Placing a class never makes a later gap worse.
    - The placement loop stops.
    - Floor backfills are one person, outside the minimum class.
    - No trigger fires on a gap that was covered before overtime.

The judgement the whole module turns on (spec 6.5, decided 10/26): the
trigger reads the gap AFTER cluster coverage and BEFORE overtime. Overtime
is there to bridge a dip and to carry a new class through its ramp, so a
trigger measured after overtime would not fire until the cap ran out -
months after the requisition needed signing. That is why Tier 1 triggers
here in a window where overtime covered every single hour.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from capacity_model import config, hiring, plan, supply, work_volume


@pytest.fixture(scope="module")
def assumptions():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def built(assumptions, sites):
    work = work_volume.build(assumptions, sites)["work"]
    return plan.build_all(assumptions, sites, work)


@pytest.fixture(scope="module")
def classes(built):
    return built["hiring"]


# ---------------------------------------------------------------------------
# The gap the trigger reads (spec 6.5, decided 10/26)
# ---------------------------------------------------------------------------


def test_the_gap_is_measured_after_coverage_and_before_overtime(built, assumptions):
    """The 10/26 decision, in the one column the trigger reads.

    For the back office the gap is required minus covered, where covered
    already includes cluster borrowing and does NOT include overtime. If
    overtime were netted off first, a team running on 2,380 hours of
    overtime a year would read as fully staffed.
    """
    before = built["before_hiring"]
    gap = before["gap_fte"].set_index(["week_index", "role", "scope"])
    cost = before["cost"]

    for row in cost.itertuples():
        basis = hiring.fte_basis(row.role, assumptions)
        cell = gap.loc[(row.week_index, row.role, row.scope)]
        assert cell["short_fte"] == pytest.approx(
            row.shortfall_hours / basis, rel=1e-9
        )
        # Overtime is not in it, and for Tier 1 that is a large difference.
        if row.overtime_hours > 1e-9:
            after_overtime = max(
                0.0, row.required_hours - row.covered_hours - row.overtime_hours
            )
            assert cell["short_fte"] > after_overtime / basis - 1e-12

    assert (before["gap_fte"]["short_fte"] >= 0).all()
    assert (before["gap_fte"]["surplus_fte"] >= 0).all()


def test_the_counter_gap_is_the_sum_of_its_own_short_days(built, assumptions):
    """A slack Friday does not repay a short Monday at a live queue.

    So the counter's weekly gap is the sum of the days' own shortfalls
    after borrowing, not the week's net. It is also self-correcting: a site
    whose only problem is a heavy Monday accumulates a fraction of an FTE
    and never triggers hiring, which is right - its problem is the roster
    shape, and another body would sit idle on Friday.
    """
    before = built["before_hiring"]
    basis = hiring.fte_basis("onsite_associate", assumptions)
    daily = before["onsite"]
    gap = before["gap_fte"]
    live = gap[gap["role"] == "onsite_associate"].set_index(["week_index", "scope"])

    summed = daily.groupby(["week_index", "site"])["shortfall_after_borrow_hours"].sum()
    for (week, site), hours in summed.items():
        assert live.loc[(week, site), "short_fte"] == pytest.approx(
            hours / basis, rel=1e-9
        )

    # Net would be a different, smaller number in most weeks, or the
    # distinction above would be decoration.
    net = daily.groupby(["week_index", "site"]).apply(
        lambda g: max(0.0, g["required_hours"].sum() - g["available_hours"].sum()),
        include_groups=False,
    )
    assert (summed > net + 1e-6).any()


def test_a_week_covered_before_overtime_never_raises_a_trigger(built, assumptions):
    """Spec 6.5: there is nothing to hire for where the hours were there.

    Both ways round: no covered week sits inside a trigger window, and a
    gap table with no shortfall at all produces no candidates however much
    overtime the same weeks bought.
    """
    before = built["before_hiring"]
    gap = before["gap_fte"]
    triggers = hiring.find_candidates(gap, before["prepared"], assumptions)
    windows = {
        (c["role"], c["scope"]): range(c["first_short_week"], c["last_short_week"] + 1)
        for c in triggers
        if c["kind"] == "hiring class"
    }
    assert windows, "expected some triggers, or this proves nothing"

    covered = gap[gap["short_fte"] <= 1e-12]
    assert len(covered) > 0
    for row in covered.itertuples():
        window = windows.get((row.role, row.scope))
        assert window is None or row.week_index not in window

    # A fully covered year asks for nobody - even though these same weeks
    # bought 5,597 hours of overtime in the real plan.
    flat = gap.copy()
    flat["short_fte"] = 0.0
    assert [
        c for c in hiring.find_candidates(flat, before["prepared"], assumptions)
        if c["kind"] == "hiring class"
    ] == []


# ---------------------------------------------------------------------------
# The three-week rule (spec 6.5)
# ---------------------------------------------------------------------------


def test_the_three_week_rule_needs_three_consecutive_weeks():
    """One bad week is a scheduling problem; three in a row is a staffing one."""
    weeks = np.arange(10)

    # Two short weeks are not enough.
    assert hiring.runs_over(weeks, np.array([0, 2, 2, 0, 0, 0, 0, 0, 0, 0]), 1.0, 3) == []
    # Three are.
    assert hiring.runs_over(
        weeks, np.array([0, 2, 2, 2, 0, 0, 0, 0, 0, 0]), 1.0, 3
    ) == [(1, 3, 2)]
    # The threshold is strict: exactly 1.0 FTE short is not "more than 1.0".
    assert hiring.runs_over(weeks, np.full(10, 1.0), 1.0, 3) == []
    # Two separate runs are found separately, with each one's own worst.
    found = hiring.runs_over(
        weeks, np.array([2, 2, 3, 0, 0, 4, 2, 2, 0, 0]), 1.0, 3
    )
    assert found == [(0, 2, 3), (5, 7, 4)]
    # A break in the week numbering breaks the run, not just a low value.
    assert hiring.runs_over(np.array([0, 1, 2, 9]), np.full(4, 2.0), 1.0, 3) == [(0, 2, 2)]


def test_every_placed_class_answers_a_real_three_week_window(classes, assumptions):
    """Each row carries the window that justified it (spec 6.5, CSV in 4e.6)."""
    assert len(classes) > 0
    hiring = assumptions["hiring"]
    for row in classes.itertuples():
        if row.kind != "hiring class":
            continue
        assert row.weeks_short >= hiring["trigger_weeks"]
        assert row.worst_short_fte > hiring["trigger_fte"]
        assert row.last_short_week - row.first_short_week + 1 == row.weeks_short
        assert str(row.why)


# ---------------------------------------------------------------------------
# Hire-by date (spec 6.5)
# ---------------------------------------------------------------------------


def test_the_hire_by_worked_examples_still_land_on_weeks_18_and_21(assumptions):
    """Spec 6.5's two worked examples, by hand (Section 9).

    Onsite Associates short from week 30: 30 - 3 ramp - 3 onboarding
    - 6 requisition = week 18. Tier 2 short from week 40: 40 - 4 - 7 - 8
    = week 21. Tier 2 needs nearly five months of warning, which is the
    whole argument for planning ahead rather than reacting.
    """
    assert supply.hire_by_week("onsite_associate", 30, assumptions) == 18
    assert supply.hire_by_week("tier_2", 40, assumptions) == 21

    # And through schedule_class, on real plan week indexes.
    first = assumptions["calendar"]["history_weeks"]
    schedule = hiring.schedule_class("onsite_associate", first + 29, assumptions)
    assert schedule["hire_by_week"] - first + 1 == 18
    assert schedule["flag"] == ""

    schedule = hiring.schedule_class("tier_2", first + 39, assumptions)
    assert schedule["hire_by_week"] - first + 1 == 21


def test_the_schedule_chains_requisition_onboarding_and_ramp(assumptions):
    """Each date follows from the one before, so the arithmetic is checkable.

        requisition -> + lead time -> start
        start       -> + onboarding -> first productive week
    """
    first = assumptions["calendar"]["history_weeks"]
    for role in assumptions["supply"]["roles"]:
        settings = supply.role_settings(role, assumptions)
        schedule = hiring.schedule_class(role, first + 40, assumptions)
        assert schedule["start_week"] == (
            schedule["requisition_week"] + settings["requisition_weeks"]
        )
        assert schedule["first_productive_week"] == (
            schedule["start_week"] + supply.onboarding_weeks(settings)
        )
        # A class that is ON TIME reaches 90% ramp by the short week. The
        # spec's formula lands it one week early, which is deliberate:
        # arriving the week the shortfall starts is arriving late.
        reached = (
            schedule["first_productive_week"]
            + supply.weeks_to_full_speed(settings)
            - 1
        )
        assert reached <= first + 40


def test_a_hire_by_date_in_the_past_is_flagged_and_placed_anyway(assumptions):
    """Spec 6.5, decided 10/26: say it plainly, and place the class.

    Hours cannot be conjured backwards. A plan that silently slid the
    shortfall to match what it could hire would be lying about it, so the
    class goes in at the earliest week the plan can place it, flagged, and
    overtime carries the gap until it ramps.
    """
    first = assumptions["calendar"]["history_weeks"]
    schedule = hiring.schedule_class("tier_2", first + 2, assumptions)
    assert schedule["hire_by_week"] < first
    assert schedule["requisition_week"] == first
    assert schedule["flag"] == "hire-by date passed"
    # Still a real, forward-looking schedule.
    assert schedule["start_week"] > first
    assert schedule["first_productive_week"] > schedule["start_week"]


# ---------------------------------------------------------------------------
# Class size (spec 6.5)
# ---------------------------------------------------------------------------


def test_class_size_covers_the_shortfall_after_expected_attrition(assumptions):
    """Spec Section 9. Hiring the gap exactly means hiring short.

    A class of 4 is not 4 people by the time it reaches full speed. Tier 2
    waits 8 weeks of requisition, 7 of onboarding and 4 of ramp, so some of
    the class has gone before a single productive hour arrives - and the
    shortfall it was hired for is still there.
    """
    for role in assumptions["supply"]["roles"]:
        settings = supply.role_settings(role, assumptions)
        attrition = supply.weekly_attrition(settings["monthly_attrition"])
        for shortfall in (1.2, 2.5, 4.9, 11.0):
            size, survivors, weeks = hiring.class_size(role, shortfall, assumptions)
            # The survivors cover the gap the class was hired for.
            assert size * survivors >= shortfall - 1e-9, f"{role} {shortfall}"
            assert survivors == pytest.approx((1 - attrition) ** weeks)
            assert weeks == supply.onboarding_weeks(settings) + supply.weeks_to_full_speed(
                settings, 1.0
            )
            # Grossed up, not padded: one fewer would not have covered it.
            if size > assumptions["hiring"]["minimum_class"]:
                assert (size - 1) * survivors < shortfall + 1e-9


def test_the_minimum_class_is_two(assumptions):
    """Spec 6.5. A class of one has no cover if that person leaves."""
    assert assumptions["hiring"]["minimum_class"] == 2
    for role in assumptions["supply"]["roles"]:
        size, _survivors, _weeks = hiring.class_size(role, 0.1, assumptions)
        assert size == 2


def test_the_placed_class_sizes_match_the_formula(classes, assumptions):
    """The table and the function cannot disagree."""
    for row in classes.itertuples():
        if row.kind != "hiring class":
            continue
        size, survivors, weeks = hiring.class_size(
            row.role, row.worst_short_fte, assumptions
        )
        assert row.class_size == size
        assert row.gross_up_survivors == pytest.approx(survivors)
        assert row.weeks_to_full_ramp == weeks


# ---------------------------------------------------------------------------
# Floor backfills (spec 6.5, 6.7, decided 10/26)
# ---------------------------------------------------------------------------


def test_a_floor_backfill_is_one_person_outside_the_minimum_class(
    built, assumptions, sites
):
    """Spec 6.5: one person, opened at once, outside the minimum of 2.

    A site with one Records Specialist is a compliance problem, not a
    capacity one - the answer is the seat refilled, not a class of two. And
    it does not wait for the three-week rule, because the floor is already
    breached.

    The baseline reaches no breach at all (the 10/26 rounding rule), so
    this runs on a bench driven under the floor.
    """
    before = built["before_hiring"]
    gap = before["gap_fte"]
    prepared = before["prepared"]

    assert hiring.floor_breach_runs(prepared, assumptions) == []
    baseline = hiring.find_candidates(gap, prepared, assumptions)
    assert all(c["kind"] != "floor backfill" for c in baseline)

    thin = copy.deepcopy(prepared)
    ledger = thin["ledger"]["ledger"].copy()
    mine = (ledger["role"] == "records_specialist") & (ledger["scope"] == "S2")
    ledger.loc[mine, "ending_headcount"] = 1.4
    thin["ledger"] = {**thin["ledger"], "ledger": ledger}

    runs = hiring.floor_breach_runs(thin, assumptions)
    assert len(runs) == 1, "52 breached weeks are one empty seat, not 52 requisitions"
    scope, first, last, short = runs[0]
    assert scope == "S2"
    assert first == assumptions["calendar"]["history_weeks"]
    assert last - first + 1 == assumptions["calendar"]["plan_weeks"]
    assert short == pytest.approx(1.0)

    backfills = [
        c for c in hiring.find_candidates(gap, thin, assumptions)
        if c["kind"] == "floor backfill"
    ]
    assert len(backfills) == 1
    backfill = backfills[0]
    assert backfill["class_size"] == 1
    assert backfill["class_size"] < assumptions["hiring"]["minimum_class"]
    assert backfill["role"] == "records_specialist"
    assert backfill["scope"] == "S2"
    # Opened at once: the shortfall starts in plan week 1, so the hire-by
    # date is already in the past and the flag says so.
    assert backfill["flag"] == "hire-by date passed"
    assert backfill["requisition_week"] == assumptions["calendar"]["history_weeks"]


def test_a_backfill_is_not_opened_twice_for_the_same_empty_seat(
    built, assumptions, sites
):
    """The loop guard: a run already answered is not answered again.

    Without it, a requisition that has not yet ramped would look like a
    fresh breach every week and the plan would hire the same seat 52 times.
    """
    before = built["before_hiring"]
    thin = copy.deepcopy(before["prepared"])
    ledger = thin["ledger"]["ledger"].copy()
    mine = (ledger["role"] == "records_specialist") & (ledger["scope"] == "S2")
    ledger.loc[mine, "ending_headcount"] = 1.4
    thin["ledger"] = {**thin["ledger"], "ledger": ledger}

    first = hiring.find_candidates(before["gap_fte"], thin, assumptions)
    key = next(c["key"] for c in first if c["kind"] == "floor backfill")
    again = hiring.find_candidates(before["gap_fte"], thin, assumptions, placed={key})
    assert all(c["kind"] != "floor backfill" for c in again)


# ---------------------------------------------------------------------------
# The placement loop (spec 6.5, task 4e.4)
# ---------------------------------------------------------------------------


def test_the_loop_stops(built, assumptions):
    """It ends because no candidate is left, not because it ran out of room.

    Rebuilding the ledger after each class is what makes termination
    non-obvious: every class changes the gaps that choose the next one. It
    terminates because a class only ever ADDS hours, so each pass leaves
    strictly fewer trigger windows, and a window already answered is
    recorded and never answered twice.
    """
    assert built["hiring_exhausted"] is False
    assert built["hiring_rounds"] == len(built["hiring"])
    assert built["hiring_rounds"] < assumptions["hiring"]["max_classes"]

    # Nothing is left to trigger on the roster the loop produced.
    leftover = [
        c for c in hiring.find_candidates(
            built["gap_fte"], built["prepared"], assumptions
        )
    ]
    assert leftover == [], [
        (c["kind"], c["role"], c["scope"], c["first_short_week"]) for c in leftover
    ]


def test_placing_a_class_never_makes_a_later_gap_worse(built):
    """Spec Section 9. The monotonicity the loop's termination rests on.

    A class only adds hours, and the one indirect path - more Records
    Specialists means more spare hours at the counter - pushes the same
    way. So no role, site or week may come out worse than it was on the
    opening roster.
    """
    before = built["before_hiring"]["gap_fte"]
    after = built["gap_fte"]
    merged = before.merge(
        after, on=["week_index", "role", "scope"], suffixes=("_before", "_after")
    )
    assert len(merged) == len(before)

    worse = merged[merged["short_fte_after"] > merged["short_fte_before"] + 1e-9]
    assert len(worse) == 0, worse[
        ["week_index", "role", "scope", "short_fte_before", "short_fte_after"]
    ].head(10)

    # And it helped somewhere, or the hiring plan did nothing.
    assert (merged["short_fte_after"] < merged["short_fte_before"] - 1e-9).any()


def test_hiring_never_makes_turnaround_worse(built):
    """The consequence the requester actually experiences (spec 4.4)."""
    before = built["before_hiring"]["turnaround"]
    after = built["turnaround"]
    merged = before.merge(after, on="week_index", suffixes=("_before", "_after"))
    assert (
        merged["turnaround_days_after"] <= merged["turnaround_days_before"] + 1e-9
    ).all()
    assert after["turnaround_days"].mean() < before["turnaround_days"].mean()


def test_the_classes_are_placed_earliest_gap_first(classes):
    """Task 4e.4: earliest first, because an early class changes later gaps."""
    weeks = classes["first_short_week"].to_numpy()
    assert list(weeks) == sorted(weeks)


def test_hiring_replaces_overtime_rather_than_adding_throughput(built):
    """Why the cost goes UP while the overtime bill collapses.

    Hours applied to a queue are capped at required (spec 4.3), so a team
    that was covering its gap with overtime was already getting the work
    done. Hiring does not add throughput there - it buys the same hours at
    1.0x instead of 1.5x, and buys them reliably. The gain shows up in the
    overtime bill and in the uncovered hours, not in the queues.
    """
    before, after = built["before_hiring"], built
    old = before["role_cost"]["overtime_hours"].sum()
    new = after["role_cost"]["overtime_hours"].sum()
    assert new < old * 0.6, (old, new)

    # Uncovered hours fall too - those WERE lost work.
    assert after["cost"]["uncovered_hours"].sum() <= (
        before["cost"]["uncovered_hours"].sum() + 1e-9
    )

    # Regular payroll rises, because the new people are paid from their
    # first week including onboarding (spec 6.4).
    assert (
        after["role_cost"]["regular_cost"].sum()
        > before["role_cost"]["regular_cost"].sum()
    )


# ---------------------------------------------------------------------------
# Surplus flag (spec 6.5)
# ---------------------------------------------------------------------------


def test_the_surplus_flag_recommends_pausing_hiring(built, assumptions):
    """Spec 6.5: over by more than 2 FTE for 6 or more weeks.

    A plan that only ever finds gaps only ever recommends spending. The
    alternative to hiring in a surplus is not doing nothing - it is moving
    the requisition somewhere it is needed.
    """
    hiring = assumptions["hiring"]
    surplus = built["surplus"]
    assert len(surplus) > 0

    gap = built["gap_fte"].set_index(["week_index", "role", "scope"])
    for row in surplus.itertuples():
        assert row.weeks >= hiring["surplus_weeks"]
        assert row.worst_surplus_fte > hiring["surplus_fte"]
        assert row.class_size == 0
        for week in range(row.first_week, row.last_week + 1):
            cell = gap.loc[(week, row.role, row.scope)]
            assert cell["surplus_fte"] > hiring["surplus_fte"]


def test_a_short_surplus_run_is_not_flagged():
    """Five weeks over is a quiet patch; six is a pattern."""
    weeks = np.arange(10)
    assert hiring.runs_over(weeks, np.array([3] * 5 + [0] * 5), 2.0, 6) == []
    assert hiring.runs_over(weeks, np.array([3] * 6 + [0] * 4), 2.0, 6) == [(0, 5, 3)]


# ---------------------------------------------------------------------------
# Cost (spec 6.4)
# ---------------------------------------------------------------------------


def test_regular_cost_pays_everyone_including_people_still_onboarding(
    built, assumptions
):
    """Spec 6.4, and the whole argument for planning ahead.

    People in compliance training are paid in full and produce nothing. A
    cost model that paid only productive people would make a late hire
    look free.
    """
    paid = assumptions["fte"]["paid_hours_per_week"]
    settings = assumptions["supply"]["roles"]
    cost = built["role_cost"]
    ledger = built["prepared"]["ledger"]["ledger"].set_index(
        ["week_index", "role", "scope"]
    )

    for row in cost.itertuples():
        hourly = settings[row.role]["hourly_cost"]
        assert row.regular_cost == pytest.approx(
            row.paid_headcount * paid * hourly, rel=1e-9
        )
        assert row.paid_headcount == pytest.approx(
            ledger.loc[(row.week_index, row.role, row.scope), "starting_headcount"]
        )
        assert row.total_cost == pytest.approx(row.regular_cost + row.overtime_cost)

    # The weeks a class is paid for and produces nothing are real.
    classes = built["hiring"]
    assert len(classes) > 0
    for row in classes.itertuples():
        onboarding = row.first_productive_week - row.start_week
        assert onboarding > 0
        paid_weeks = cost[
            (cost["role"] == row.role)
            & (cost["scope"] == row.scope)
            & (cost["week_index"] >= row.start_week)
            & (cost["week_index"] < row.first_productive_week)
        ]
        assert len(paid_weeks) == onboarding
        assert (paid_weeks["regular_cost"] > 0).all()


def test_every_role_and_week_is_costed(built, assumptions):
    """No role falls between the weekly and the daily check."""
    cost = built["role_cost"]
    assert set(cost["role"]) == set(assumptions["supply"]["roles"])
    assert cost["week_index"].nunique() == assumptions["calendar"]["plan_weeks"]
    assert len(cost) == len(cost.drop_duplicates(["week_index", "role", "scope"]))
    assert (cost["regular_cost"] > 0).all()
    assert (cost["overtime_hours"] >= 0).all()

    # The overtime in the cost table is the overtime the plan actually bought.
    live = built["onsite"]["overtime_hours"].sum()
    queued = built["cost"]["overtime_hours"].sum()
    assert cost["overtime_hours"].sum() == pytest.approx(live + queued, rel=1e-9)
