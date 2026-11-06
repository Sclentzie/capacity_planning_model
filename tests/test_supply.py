"""Spec Section 9, Session 4: the Capacity Ledger (spec 5.1 to 5.3).

    - Ledger reconciles every week: ending HC = starting HC + hires - attrition.
    - With zero attrition and zero hires, headcount stays flat for 52 weeks.
    - A class of 10 Onsite Associates contributes 0, 0, 0, then 5.0, 7.5,
      9.0, 10.0 FTE before attrition.
    - Hire-by examples land on week 18 (Onsite Associates) and week 21 (Tier 2).
    - Breaks are counted exactly once for every role.

The ledger is where "we have 22 people" becomes "we have 648 productive
hours", and almost all of the loss between those two numbers is deliberate.
These tests pin each deduction separately so a change to one cannot be
hidden by a change to another.
"""

import copy
import datetime

import numpy as np
import pandas as pd
import pytest

from capacity_model import config, generate_data, supply


@pytest.fixture(scope="module")
def assumptions():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def built(assumptions, sites):
    return supply.build_ledger(assumptions, sites)


@pytest.fixture(scope="module")
def ledger(built):
    return built["ledger"]


@pytest.fixture(scope="module")
def mondays(built):
    return built["mondays"]


def zero_attrition(assumptions):
    """A copy with nobody leaving, for isolating the other deductions."""
    changed = copy.deepcopy(assumptions)
    for settings in changed["supply"]["roles"].values():
        settings["monthly_attrition"] = 0.0
    return changed


# ---------------------------------------------------------------------------
# Attrition and the weekly recursion
# ---------------------------------------------------------------------------


def test_weekly_attrition_compounds_rather_than_divides():
    """Spec 5.3: weekly = 1 - (1 - monthly) ^ (12 / 52).

    The compounded weekly rate comes out slightly ABOVE the naive
    0.025 x 12 / 52, which is the opposite of what it looks like it should
    do. Each week's leavers come out of a roster the previous week already
    thinned, so a rate applied 52 times to a shrinking base has to be a
    little larger to produce the same annual loss. Dividing the monthly
    rate instead would quietly under-count attrition all year.
    """
    weekly = supply.weekly_attrition(0.025)
    assert weekly == pytest.approx(1 - (1 - 0.025) ** (12 / 52))
    assert weekly == pytest.approx(0.005826, abs=1e-6)

    naive = 0.025 * 12 / 52
    assert weekly > naive, "compounding must sit above the naive division"
    assert (1 - naive) ** 52 > (1 - 0.025) ** 12, "the naive rate loses too few people"

    # A year of weekly attrition reproduces a year of monthly attrition.
    assert (1 - weekly) ** 52 == pytest.approx((1 - 0.025) ** 12, rel=1e-12)
    assert supply.weekly_attrition(0.0) == 0.0


def test_the_ledger_reconciles_every_week(ledger):
    """Spec 9: ending headcount = starting + hires - attrition, every row."""
    rebuilt = (
        ledger["starting_headcount"] + ledger["hires"] - ledger["attrition"]
    )
    np.testing.assert_allclose(ledger["ending_headcount"], rebuilt, rtol=1e-12)
    assert (ledger["attrition"] >= 0).all()
    assert (ledger["ending_headcount"] >= 0).all()


def test_each_week_opens_where_the_last_one_closed(ledger):
    """The roster rolls: no headcount appears or vanishes between weeks."""
    for (_role, _scope), rows in ledger.groupby(["role", "scope"]):
        rows = rows.sort_values("week_index")
        starting = rows["starting_headcount"].to_numpy()
        ending = rows["ending_headcount"].to_numpy()
        np.testing.assert_allclose(starting[1:], ending[:-1], rtol=1e-12)


def test_zero_attrition_and_no_hires_keeps_headcount_flat(assumptions, sites):
    """Spec 9: flat for 52 weeks when nobody leaves and nobody joins.

    The control for everything else: if headcount drifts here, it is the
    machinery and not the attrition rate.
    """
    flat = supply.build_ledger(zero_attrition(assumptions), sites)["ledger"]
    assert flat["week_index"].nunique() == 52
    assert flat["hires"].sum() == 0
    assert flat["attrition"].sum() == pytest.approx(0.0)

    for (_role, _scope), rows in flat.groupby(["role", "scope"]):
        headcount = rows.sort_values("week_index")["ending_headcount"]
        assert headcount.nunique() == 1
        np.testing.assert_allclose(headcount, headcount.iloc[0], rtol=1e-12)


def test_headcount_stays_fractional(ledger):
    """Spec 5.3: fractional inside the maths, rounded only for display.

    Rounding 0.4 of a leaver down to zero every week would quietly rebuild
    the headcount the attrition rate is there to remove.
    """
    ending = ledger["ending_headcount"]
    assert not np.allclose(ending, ending.round()), "headcount has been rounded"
    assert (ledger["attrition"] % 1 != 0).any()


def test_the_opening_roster_matches_the_tuned_headcount(ledger, assumptions, sites):
    """The ledger opens on exactly the roster assumptions.yaml states.

    Headcount is now stated per SCOPE rather than as a network total split
    evenly, because the tuned numbers differ by site: N1 carries five
    Onsite Associates and N3 four, which an even split could not express.
    """
    first = ledger["week_index"].min()
    opening = ledger[ledger.week_index == first]
    stated = assumptions["fte"]["starting_headcount"]

    for role, by_scope in stated.items():
        assert isinstance(by_scope, dict), f"{role} should be stated per scope"
        for scope, count in by_scope.items():
            row = opening[(opening.role == role) & (opening.scope == scope)]
            assert len(row) == 1, f"{role}/{scope}"
            assert row["starting_headcount"].iloc[0] == pytest.approx(count)

    per_site = opening[opening.role == "onsite_associate"]
    assert len(per_site) == len(sites["sites"])
    assert per_site["starting_headcount"].nunique() > 1, "sites should differ"
    assert set(opening[opening.role == "tier_1"]["scope"]) == {"NETWORK"}


# ---------------------------------------------------------------------------
# Cohorts, onboarding and ramp
# ---------------------------------------------------------------------------


def test_a_class_of_ten_onsite_associates_ramps_as_the_spec_says(assumptions, sites):
    """Spec 9: 0, 0, 0, then 5.0, 7.5, 9.0, 10.0 FTE before attrition.

    Three weeks at zero is the onboarding stretch - paid, and productive of
    nothing, because a new hire cannot touch records before training and
    system access are done. Then the ramp. Run with attrition off, which is
    what "before attrition" means.
    """
    flat = zero_attrition(assumptions)
    first = flat["calendar"]["history_weeks"]

    cohorts = supply.opening_cohorts(flat, sites, first)
    cohorts = supply.add_hiring_class(
        cohorts, "onsite_associate", "N1", start_week=first, size=10
    )
    built = supply.build_ledger(flat, sites, cohorts)
    rows = built["ledger"]
    n1 = rows[(rows.role == "onsite_associate") & (rows.scope == "N1")].sort_values(
        "week_index"
    )

    # The site's existing people, fully ramped. Read from the tuned
    # roster rather than assumed, so retuning does not break this.
    opening_fte = flat["fte"]["starting_headcount"]["onsite_associate"]["N1"]
    contributed = n1["ramped_fte"].to_numpy()[:7] - opening_fte
    np.testing.assert_allclose(
        contributed, [0.0, 0.0, 0.0, 5.0, 7.5, 9.0, 10.0], atol=1e-9
    )

    assert n1["hires"].iloc[0] == 10
    assert n1["hires"].iloc[1:].sum() == 0
    assert n1["ending_headcount"].iloc[0] == pytest.approx(opening_fte + 10)


def test_a_new_class_attrits_like_everybody_else(assumptions, sites):
    """Spec 5.3: new hires attrit too, which is why 10 never stays 10.

    The detail that makes a hiring class smaller than it looks a year
    later, and the reason 6.5 grosses a class up for attrition before full
    ramp.
    """
    first = assumptions["calendar"]["history_weeks"]
    cohorts = supply.add_hiring_class(
        supply.opening_cohorts(assumptions, sites, first),
        "tier_2", "NETWORK", start_week=first + 10, size=10,
    )
    rows = supply.build_ledger(assumptions, sites, cohorts)["ledger"]
    tier_2 = rows[rows.role == "tier_2"].sort_values("week_index")

    at_start = tier_2[tier_2.week_index == first + 10]["ending_headcount"].iloc[0]
    a_year_on = tier_2["ending_headcount"].iloc[-1]
    assert a_year_on < at_start, "the class should shrink after it lands"
    assert tier_2["hires"].sum() == 10


def test_adding_a_class_leaves_the_baseline_alone(assumptions, sites):
    """add_hiring_class returns a new list, so scenarios can branch safely."""
    first = assumptions["calendar"]["history_weeks"]
    baseline = supply.opening_cohorts(assumptions, sites, first)
    before = len(baseline)
    changed = supply.add_hiring_class(baseline, "tier_1", "NETWORK", first + 5, 4)
    assert len(baseline) == before
    assert len(changed) == before + 1
    with pytest.raises(ValueError):
        supply.add_hiring_class(baseline, "tier_1", "NETWORK", first + 5, 0)


def test_the_baseline_places_no_hiring_classes(ledger):
    """Task 4a is mechanics only - hiring decisions come in 6.5."""
    assert ledger["hires"].sum() == 0


def test_onboarding_is_paid_and_productive_of_nothing(assumptions):
    """Ramp is 0.0 through compliance and training, then steps up."""
    for role in assumptions["supply"]["roles"]:
        settings = supply.role_settings(role, assumptions)
        onboarding = supply.onboarding_weeks(settings)
        assert onboarding > 0
        for week in range(onboarding):
            assert supply.ramp_fraction(settings, week) == 0.0
        assert supply.ramp_fraction(settings, onboarding) == settings["ramp"][0]
        assert supply.ramp_fraction(settings, onboarding + 50) == settings["ramp"][-1] == 1.0


# ---------------------------------------------------------------------------
# Hire-by (spec 6.5)
# ---------------------------------------------------------------------------


def test_hire_by_matches_the_spec_worked_examples(assumptions):
    """Spec 9: week 18 for Onsite Associates, week 21 for Tier 2.

        Onsite Associates short from week 30: 30 - 3 - 3 - 6 = 18
        Tier 2 short from week 40:           40 - 4 - 7 - 8 = 21

    Tier 2 needs nineteen weeks of warning - nearly five months - because
    everything about it is slower: an 8-week requisition, 7 weeks of
    onboarding, and a ramp that only reaches 90% in its fourth week.
    """
    assert supply.hire_by_week("onsite_associate", 30, assumptions) == 18
    assert supply.hire_by_week("tier_2", 40, assumptions) == 21

    onsite = supply.role_settings("onsite_associate", assumptions)
    assert supply.weeks_to_full_speed(onsite) == 3
    assert supply.onboarding_weeks(onsite) == 3
    assert onsite["requisition_weeks"] == 6

    tier_2 = supply.role_settings("tier_2", assumptions)
    assert supply.weeks_to_full_speed(tier_2) == 4, "40/60/80/100 first clears 90 at week 4"
    assert supply.onboarding_weeks(tier_2) == 7
    assert tier_2["requisition_weeks"] == 8


def test_a_hire_by_date_already_passed_is_reported_not_hidden(assumptions):
    """Spec 6.5: say so plainly rather than clamping to week zero.

    "You needed to start three weeks before the plan began" is the useful
    answer; a zero would look like an action that is still available.
    """
    assert supply.hire_by_week("tier_2", 5, assumptions) == -14
    assert supply.hire_by_week("onsite_associate", 12, assumptions) == 0


def test_every_role_has_a_hire_by_lead_time(assumptions):
    """Lead times differ enough that one number for all roles would mislead."""
    leads = {
        role: supply.hire_by_week(role, 100, assumptions)
        for role in assumptions["supply"]["roles"]
    }
    assert leads["onsite_associate"] == 88     # 12 weeks of lead time
    assert leads["tier_2"] == 81               # 19 weeks, the slowest
    assert min(leads.values()) < max(leads.values())


# ---------------------------------------------------------------------------
# Shrinkage (spec 5.2)
# ---------------------------------------------------------------------------


def test_each_shrinkage_line_averages_its_annual_percentage(built, assumptions):
    """Spec 9: the patterns redistribute time, they never create it.

    A line can be concentrated on Wednesdays or in August and still cost
    exactly its stated percentage over the year. If that stopped holding,
    the timing detail would be quietly changing the total.
    """
    rates = {line["code"]: line["rate"] for line in assumptions["shrinkage"]["lines"]}
    for role, scopes in built["shrinkage_lines"].items():
        for scope, lines in scopes.items():
            for code, values in lines.items():
                assert values.mean() == pytest.approx(
                    rates[code], rel=1e-9
                ), f"{role}/{scope}/{code}"


def test_role_totals_match_the_spec_lines(built, assumptions):
    """Onsite Associates 21.75%, back office 28% (spec 5.2).

    Onsite sits lower by exactly the 6.25% paid-break line, which 4.2
    builds into the interval schedule instead.
    """
    totals = {
        (role, scope): sum(values.mean() for values in lines.values())
        for role, scopes in built["shrinkage_lines"].items()
        for scope, lines in scopes.items()
    }
    for (role, scope), total in totals.items():
        expected = 0.2175 if role == "onsite_associate" else 0.28
        assert total == pytest.approx(expected, rel=1e-9), f"{role}/{scope}"

    totals = {role: value for (role, _scope), value in totals.items()}
    assert totals["onsite_associate"] == pytest.approx(0.2175, rel=1e-9)
    for role in ("records_specialist", "tier_1", "tier_2", "qa_specialist"):
        assert totals[role] == pytest.approx(0.28, rel=1e-9), role
    assert totals["onsite_associate"] == pytest.approx(totals["tier_1"] - 0.0625, rel=1e-9)


def test_vacation_is_heavier_in_summer_and_at_year_end(built, mondays, assumptions):
    """Decided 10/26, and it has to show up in the weeks it names."""
    vacation = built["shrinkage_lines"]["tier_1"]["NETWORK"]["vacation"].mean(axis=1)
    months = np.array([m.month for m in mondays])
    days = np.array([m.day for m in mondays])

    summer = (months >= 6) & (months <= 8)
    year_end = (months == 12) & (days >= 18)
    ordinary = ~(summer | year_end)

    assert vacation[summer].mean() > vacation[ordinary].mean() * 1.5
    assert vacation[year_end].mean() > vacation[ordinary].mean() * 1.5
    assert vacation.mean() == pytest.approx(0.10, rel=1e-9)


def test_sick_time_is_higher_in_the_first_quarter_and_varies_week_to_week(
    built, mondays
):
    """Decided 10/26: about a quarter higher January to March, plus noise.

    Sick is the one line nobody schedules, so it carries week-to-week
    variation on top of the season - and it is the line tagged unplanned,
    because it cannot be moved when coverage is short.
    """
    sick = built["shrinkage_lines"]["tier_1"]["NETWORK"]["sick"].mean(axis=1)
    months = np.array([m.month for m in mondays])
    winter = months <= 3

    assert sick[winter].mean() > sick[~winter].mean()
    assert sick[winter].mean() / sick[~winter].mean() == pytest.approx(1.25, rel=0.25)
    assert sick.std() > 0, "sick time should not be identical every week"
    assert sick.mean() == pytest.approx(0.04, rel=1e-9)


def test_the_same_seed_gives_the_same_sick_weeks(assumptions, sites):
    """Fixed seed (spec 5.2), so a plan can be rebuilt and compared."""
    for role, scope in (("tier_1", "NETWORK"), ("onsite_associate", "N1")):
        first = supply.build_ledger(assumptions, sites)["shrinkage_lines"][role][scope]
        again = supply.build_ledger(assumptions, sites)["shrinkage_lines"][role][scope]
        np.testing.assert_array_equal(first["sick"], again["sick"])


def test_training_lands_tuesday_to_thursday_and_meetings_on_wednesday(built):
    """Decided 10/26. The day pattern is the point, not the weekly average.

    A team losing 1% to meetings every day is a different operation from
    one losing 5% every Wednesday, and it is the Wednesday that breaks
    coverage. Monday and Friday carry neither training nor meetings.
    """
    lines = built["shrinkage_lines"]["tier_1"]["NETWORK"]
    days = list(supply.WEEKDAYS)

    for code in ("digital_training", "offsite_training"):
        by_day = lines[code].mean(axis=0)
        assert by_day[days.index("monday")] == pytest.approx(0.0)
        assert by_day[days.index("friday")] == pytest.approx(0.0)
        for day in ("tuesday", "wednesday", "thursday"):
            assert by_day[days.index(day)] > 0

    meetings = lines["meetings"].mean(axis=0)
    assert meetings[days.index("wednesday")] == pytest.approx(0.01 * 5, rel=1e-9)
    for day in ("monday", "tuesday", "thursday", "friday"):
        assert meetings[days.index(day)] == pytest.approx(0.0)

    even = lines["one_on_ones"].mean(axis=0)
    np.testing.assert_allclose(even, even[0], rtol=1e-12)


def test_wednesday_is_the_thinnest_day_of_the_week(built):
    """The consequence of the 10/26 placement, worth knowing before coverage.

    Training and meetings both land midweek, so Wednesday loses about half
    again as much time as Monday - which is where a coverage plan will
    first feel it.
    """
    daily = built["daily"]
    tier_1 = daily[daily.role == "tier_1"].groupby("day_of_week")["shrinkage"].mean()
    assert tier_1.idxmax() == "wednesday"
    assert tier_1["wednesday"] > tier_1["monday"]
    assert tier_1["monday"] == pytest.approx(tier_1["friday"], rel=1e-9)


def test_daily_placement_adds_back_to_the_weekly_total(built, assumptions):
    """Spec 9: each line falls on its own days and still sums to its week.

    The whole justification for modelling days at all: it must change WHEN
    time is lost without changing HOW MUCH.
    """
    ledger, daily = built["ledger"], built["daily"]

    rebuilt = daily.groupby(["week_index", "role", "scope"])["productive_hours"].sum()
    stated = ledger.set_index(["week_index", "role", "scope"])["productive_hours"]
    np.testing.assert_allclose(rebuilt.sort_index(), stated.sort_index(), rtol=1e-9)

    mean_shrinkage = daily.groupby(["week_index", "role", "scope"])["shrinkage"].mean()
    weekly = ledger.set_index(["week_index", "role", "scope"])["shrinkage"]
    np.testing.assert_allclose(mean_shrinkage.sort_index(), weekly.sort_index(), rtol=1e-9)


def test_every_line_is_tagged_planned_or_unplanned(assumptions):
    """Spec 5.2: the tag decides what a coverage plan may move.

    Sick is the only unplanned line, and it is the one a short week cannot
    reschedule - which is exactly why it is worth separating.
    """
    lines = {line["code"]: line for line in assumptions["shrinkage"]["lines"]}
    assert lines["sick"]["planned"] is False
    for code, line in lines.items():
        assert isinstance(line["planned"], bool), code
        if code != "sick":
            assert line["planned"] is True, code


def test_shrinkage_can_never_take_the_whole_day(assumptions, sites):
    """The productive-hours formula divides nothing by zero, but it would
    silently return no hours - so an impossible total is refused instead."""
    broken = copy.deepcopy(assumptions)
    for line in broken["shrinkage"]["lines"]:
        if line["code"] == "vacation":
            line["rate"] = 0.95
    with pytest.raises(ValueError, match="cannot be away"):
        supply.build_ledger(broken, sites)


# ---------------------------------------------------------------------------
# Productive hours (spec 5.3)
# ---------------------------------------------------------------------------


def test_productive_hours_follow_the_spec_formula(built, assumptions):
    """Spec 5.3: ramped FTE x paid hours x (1 - that day's shrinkage)."""
    paid = assumptions["fte"]["paid_hours_per_week"]
    per_day = paid / assumptions["calendar"]["business_days_per_week"]
    assert per_day == 8.0

    daily = built["daily"]
    expected = daily["ramped_fte"] * per_day * (1 - daily["shrinkage"])
    np.testing.assert_allclose(daily["productive_hours"], expected, rtol=1e-12)
    assert (daily["productive_hours"] >= 0).all()


def test_productive_hours_sit_well_below_paid_hours(built, assumptions):
    """The gap the whole module exists to make visible.

    Twenty-two Tier 1 Specialists are 880 paid hours a week and about 650
    productive ones. A plan built on the first number would be a quarter
    short before a single person took a day off.
    """
    ledger = built["ledger"]
    first = ledger["week_index"].min()
    week = ledger[ledger.week_index == first]
    paid = assumptions["fte"]["paid_hours_per_week"]

    for row in week.itertuples():
        paid_hours = row.ramped_fte * paid
        if paid_hours > 0:
            assert row.productive_hours < paid_hours
            assert row.productive_hours > paid_hours * 0.6


def test_the_ledger_covers_the_whole_plan_year(built, assumptions):
    cal = assumptions["calendar"]
    ledger = built["ledger"]
    weeks = np.sort(ledger["week_index"].unique())
    assert weeks[0] == cal["history_weeks"]
    assert len(weeks) == cal["plan_weeks"] == 52
    assert set(ledger["role"]) == set(assumptions["supply"]["roles"])


# ---------------------------------------------------------------------------
# Breaks counted exactly once (spec Section 9) - finished here
# ---------------------------------------------------------------------------


def test_paid_breaks_are_counted_exactly_once_for_every_role(assumptions):
    """Spec 9: by interval for Onsite Associates, in shrinkage for the rest.

    The check spans three mechanisms that each looked right on their own,
    which is how a break gets paid for twice:

      4.2  Onsite Associates' breaks are built into the interval schedule
      5.2  back-office roles carry a 6.25% paid-break shrinkage line
      4.2  their email spillover hours carry the same 6.25% allowance

    Every role must appear in exactly one of the first two, and the
    allowance in the third must be the same rate rather than a second,
    independently drifting number.
    """
    by_interval = set(assumptions["breaks_and_lunch"]["built_into_intervals_for"])
    in_shrinkage = {
        role
        for line in assumptions["shrinkage"]["lines"]
        if line["code"] == "paid_breaks"
        for role in line.get("roles", assumptions["supply"]["roles"])
    }
    everyone = set(assumptions["supply"]["roles"])

    assert by_interval | in_shrinkage == everyone, "some role has no breaks at all"
    assert not (by_interval & in_shrinkage), "some role is paid breaks twice"
    assert by_interval == {"onsite_associate"}

    break_line = next(
        line for line in assumptions["shrinkage"]["lines"] if line["code"] == "paid_breaks"
    )
    allowance = assumptions["onsite_team"]["support_work"]["spillover_break_allowance"]
    assert allowance == break_line["rate"] == 0.0625


def test_onsite_shrinkage_is_the_back_office_total_less_the_break_line(built):
    """The arithmetic that proves the two mechanisms do not overlap.

    Onsite Associates come out at exactly the back-office total minus
    6.25%. If their interval build and their shrinkage ever both carried
    breaks, this difference would close.
    """
    totals = {
        role: sum(values.mean() for values in next(iter(scopes.values())).values())
        for role, scopes in built["shrinkage_lines"].items()
    }
    assert "paid_breaks" not in built["shrinkage_lines"]["onsite_associate"]["N1"]
    assert "paid_breaks" in built["shrinkage_lines"]["tier_1"]["NETWORK"]
    assert totals["tier_1"] - totals["onsite_associate"] == pytest.approx(0.0625, rel=1e-9)


# ---------------------------------------------------------------------------
# Records Specialists help the counter (spec 4.5, decided 10/26)
# ---------------------------------------------------------------------------

HELP_WEEKS = [156, 160]


@pytest.fixture(scope="module")
def required_hours(assumptions, sites):
    from capacity_model import backlog as backlog_module

    return backlog_module.required_hours(assumptions, sites)["hours"]


@pytest.fixture(scope="module")
def onsite(assumptions, sites):
    from capacity_model import intervals as intervals_module
    from capacity_model import work_volume

    work = work_volume.build(assumptions, sites)["work"]
    sample = work[work["week_index"].isin(HELP_WEEKS)]
    return intervals_module.onsite_intervals(assumptions, sites, sample)["intervals"]


@pytest.fixture(scope="module")
def spare(built, required_hours, assumptions):
    return supply.records_specialist_spare_hours(
        built["daily"], required_hours, assumptions
    )


@pytest.fixture(scope="module")
def helped(onsite, spare, assumptions):
    return supply.apply_records_specialist_help(onsite, spare, assumptions)


def test_spare_hours_are_productive_time_left_after_their_own_work(
    spare, built, required_hours, assumptions
):
    """Spec 4.5: onsite retrieval first, then digitization, then the counter.

    Their own work comes off before anything is offered, and the remainder
    is floored at zero per DAY rather than per week - a Monday where the
    chart list runs long cannot lend negative help to make up for a quiet
    Friday.
    """
    settings = assumptions["onsite_team"]["records_specialist_help"]
    assert settings["own_work_types"] == ["W11", "W12", "W13", "W17"]

    assert (spare["spare_hours"] >= 0).all()
    np.testing.assert_allclose(
        spare["spare_hours"],
        (spare["productive_hours"] - spare["own_work_hours"]).clip(lower=0.0),
        rtol=1e-12,
    )

    # Daily flooring means the weekly spare is at least the weekly net.
    weekly = spare.groupby(["week_index", "site"])[
        ["productive_hours", "own_work_hours", "spare_hours"]
    ].sum()
    net = weekly["productive_hours"] - weekly["own_work_hours"]
    assert (weekly["spare_hours"] >= net - 1e-9).all()


def test_help_never_exceeds_half_the_spare_hours(spare, helped, assumptions):
    """Spec 4.5, decided 10/26: half, not all.

    They are not reliably free when the counter needs them, so only half of
    the spare is promised - and what the counter can actually use is capped
    again by what it needed in the first place.
    """
    share = assumptions["onsite_team"]["records_specialist_help"]["share_of_spare"]
    assert share == 0.5

    np.testing.assert_allclose(
        spare["help_hours_offered"], spare["spare_hours"] * share, rtol=1e-12
    )
    assert (spare["help_hours_offered"] <= spare["spare_hours"] / 2 + 1e-12).all()

    # Applied help can only be less than offered, never more.
    assert (helped["help_staff"] <= helped["help_staff_offered"] + 1e-12).all()
    used = helped.groupby(["week_index", "site", "day_of_week"])["help_hours"].sum()
    offered = spare.set_index(["week_index", "site", "day_of_week"])["help_hours_offered"]
    common = used.index.intersection(offered.index)
    assert (used.loc[common] <= offered.loc[common] + 1e-9).all()


def test_no_spare_hours_means_no_help(onsite, spare, assumptions):
    """A site whose Records Specialists are fully booked lends nothing.

    Checked by zeroing the spare rather than by hoping no such day exists,
    so the zero case is exercised on every row.
    """
    nothing = spare.copy()
    nothing["spare_hours"] = 0.0
    nothing["help_hours_offered"] = 0.0

    unchanged = supply.apply_records_specialist_help(onsite, nothing, assumptions)
    assert unchanged["help_staff"].sum() == pytest.approx(0.0)
    np.testing.assert_allclose(
        unchanged["scheduled_staff"], onsite["scheduled_staff"], rtol=1e-12
    )
    np.testing.assert_allclose(
        unchanged["scheduled_hours"], onsite["scheduled_hours"], rtol=1e-12
    )


def test_a_records_specialist_with_no_productive_hours_offers_nothing(
    built, required_hours, assumptions
):
    """The other end of the same case, from the ledger side."""
    empty = built["daily"].copy()
    empty.loc[empty["role"] == "records_specialist", "productive_hours"] = 0.0
    spare = supply.records_specialist_spare_hours(empty, required_hours, assumptions)
    assert spare["spare_hours"].sum() == pytest.approx(0.0)
    assert spare["help_hours_offered"].sum() == pytest.approx(0.0)


def test_help_never_crosses_a_site(onsite, spare, helped, assumptions):
    """Spec 4.5: they are standing in one building.

    Measured by removing one site's spare entirely and checking that no
    other site's schedule moves by so much as a rounding error. A pooled
    implementation would quietly spread the loss across all six.
    """
    assert assumptions["onsite_team"]["records_specialist_help"]["cross_site"] is False

    without_n1 = spare.copy()
    mask = without_n1["site"] == "N1"
    without_n1.loc[mask, "help_hours_offered"] = 0.0
    changed = supply.apply_records_specialist_help(onsite, without_n1, assumptions)

    for site in sorted(helped["site"].unique()):
        before = helped[helped.site == site].sort_values(["week_index", "day_of_week", "interval"])
        after = changed[changed.site == site].sort_values(["week_index", "day_of_week", "interval"])
        if site == "N1":
            assert after["help_staff"].sum() == pytest.approx(0.0)
            assert after["scheduled_staff"].sum() > before["scheduled_staff"].sum()
        else:
            np.testing.assert_allclose(
                after["scheduled_staff"].to_numpy(),
                before["scheduled_staff"].to_numpy(),
                rtol=1e-12,
                err_msg=f"{site} moved when N1 lost its spare hours",
            )


def test_total_coverage_in_every_interval_is_unchanged(onsite, helped):
    """The property the whole change rests on: WHO, not HOW MUCH.

    Erlang C still says the same number of people must be on the floor in
    every single interval; some of them now wear a different job title. If
    this ever stopped holding, the model would be quietly lowering service
    rather than reallocating people.
    """
    covered = helped["scheduled_staff"] + helped["help_staff"]
    np.testing.assert_allclose(
        covered, helped["scheduled_staff_before_help"], rtol=1e-9
    )
    np.testing.assert_allclose(
        helped["scheduled_staff_before_help"].to_numpy(),
        onsite["scheduled_staff"].to_numpy(),
        rtol=1e-12,
    )

    # Required coverage is untouched - only the scheduled split moved.
    np.testing.assert_array_equal(
        helped["required_staff"].to_numpy(), onsite["required_staff"].to_numpy()
    )


def test_onsite_staffing_never_goes_below_zero(helped):
    """Offering more help than an interval needs buys no credit."""
    assert (helped["scheduled_staff"] >= -1e-12).all()
    assert (helped["scheduled_hours"] >= -1e-12).all()
    assert (helped["help_staff"] >= -1e-12).all()

    capped = helped[helped["help_staff_offered"] > helped["scheduled_staff_before_help"]]
    if len(capped):
        np.testing.assert_allclose(
            capped["scheduled_staff"], 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            capped["help_staff"], capped["scheduled_staff_before_help"], rtol=1e-9
        )


def test_help_is_spread_evenly_across_the_day(helped, assumptions):
    """Spread, not promised for a particular half-hour (spec 4.5).

    A chart pull takes as long as it takes, so the help cannot be aimed at
    the 11:00 peak. Within a site-day the offered help is identical in
    every interval.
    """
    per_day = assumptions["intraday"]["site_hours"]["intervals_per_day"]
    for (_week, _site, _day), rows in helped.groupby(
        ["week_index", "site", "day_of_week"]
    ):
        assert len(rows) == per_day
        offered = rows["help_staff_offered"].to_numpy()
        np.testing.assert_allclose(offered, offered[0], rtol=1e-12)


def test_help_converts_hours_to_bodies_correctly(helped, spare, assumptions):
    """help staff per interval = help hours / (intervals x interval hours).

    Eighteen half-hour intervals are nine hours of floor time, so one spare
    hour buys a ninth of a person all day rather than one person for an
    hour - which is what "spread evenly" means in staffing terms.
    """
    per_day = assumptions["intraday"]["site_hours"]["intervals_per_day"]
    interval_hours = assumptions["live_queues"]["interval_seconds"] / 3600.0
    assert per_day * interval_hours == 9.0

    offered = spare.set_index(["week_index", "site", "day_of_week"])["help_hours_offered"]
    sample = helped.drop_duplicates(["week_index", "site", "day_of_week"])
    for row in sample.itertuples():
        hours = offered[(row.week_index, row.site, row.day_of_week)]
        assert row.help_staff_offered == pytest.approx(hours / 9.0, rel=1e-9)


def test_the_help_can_be_switched_off(onsite, spare, assumptions):
    """A scenario needs to be able to compare with and without (spec 7)."""
    off = copy.deepcopy(assumptions)
    off["onsite_team"]["records_specialist_help"]["enabled"] = False
    unchanged = supply.apply_records_specialist_help(onsite, spare, off)
    assert unchanged["help_staff"].sum() == 0.0
    np.testing.assert_allclose(
        unchanged["scheduled_staff"], onsite["scheduled_staff"], rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Tuning the opening roster (spec 5.1, decided 10/26)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tuned(assumptions, sites):
    return supply.tune_starting_headcount(assumptions, sites)


@pytest.fixture(scope="module")
def gaps(assumptions, sites):
    return supply.gap_by_quarter(assumptions, sites)


def test_retuning_the_tuned_values_changes_nothing(tuned, assumptions):
    """The rule has to be at its own fixed point (spec 5.1).

    If rerunning it moved the numbers, the roster in assumptions.yaml
    would not be the answer the rule gives - it would be one step along a
    sequence, and nobody could say which step was right.

    Not circular, because the required hours the rule divides do not depend
    on headcount for the back office. They do for the counter, which is why
    the back office is tuned first and the ledger rebuilt before the
    counter is sized against the help it will actually get.
    """
    stated = assumptions["fte"]["starting_headcount"]
    for row in tuned.itertuples():
        assert stated[row.role][row.scope] == row.headcount, f"{row.role}/{row.scope}"

    # And every scope in the file is one the rule produced.
    produced = {(row.role, row.scope) for row in tuned.itertuples()}
    written = {(role, scope) for role, by in stated.items() for scope in by}
    assert produced == written


def test_first_quarter_coverage_clears_one_hundred_percent_everywhere(gaps):
    """Spec 5.1: the plan opens BALANCED, so Q1 is covered for every team.

    Every role at every site, not just the network averages - a network
    surplus that hides a short site is exactly the failure the per-site
    tuning exists to prevent.
    """
    first_quarter = gaps[gaps.quarter == 1]
    by_scope = first_quarter.groupby(["role", "scope"]).agg(
        coverage=("coverage", "mean"), gap_fte=("gap_fte", "mean")
    )
    assert (by_scope["coverage"] >= 1.0).all(), by_scope[by_scope["coverage"] < 1.0]
    assert (by_scope["gap_fte"] >= 0).all()


def test_first_quarter_is_not_over_staffed_beyond_rounding(gaps):
    """Under 1 FTE over, because the only slack allowed is the round-up.

    Rounding up a fractional requirement can only ever add less than one
    person. Anything above that would mean the rule had padded the roster,
    and the Capacity Plan's gaps would be measuring the padding rather than
    the operation.
    """
    first_quarter = gaps[gaps.quarter == 1]
    by_scope = first_quarter.groupby(["role", "scope"])["gap_fte"].mean()
    assert (by_scope < 1.0).all(), by_scope[by_scope >= 1.0]


def test_the_rounding_slack_is_what_the_tuner_said_it_was(tuned):
    """The slack is the round-up and nothing else."""
    np.testing.assert_allclose(
        tuned["rounding_slack_fte"], tuned["headcount"] - tuned["required_fte"], rtol=1e-12
    )
    unfloored = tuned[tuned["headcount"] > tuned["floor"]]
    assert (unfloored["rounding_slack_fte"] >= 0).all()
    assert (unfloored["rounding_slack_fte"] < 1.0).all()


def test_the_records_specialist_floor_holds(tuned, assumptions):
    """Spec Section 1: never below 2 a site, so records cover survives.

    S2 is the site where it binds - its chart work only needs 1.9 people -
    and the floor is the reason a quiet site still has somebody who can
    walk to the archive room.
    """
    records = tuned[tuned.role == "records_specialist"]
    assert (records["headcount"] >= 2).all()
    assert len(records) == 6

    bound = records[records["required_fte"] < 2]
    assert len(bound) >= 1, "expected the floor to bind somewhere"
    for row in bound.itertuples():
        assert row.headcount == 2, f"{row.scope} should sit exactly on the floor"

    stated = assumptions["fte"]["starting_headcount"]["records_specialist"]
    assert min(stated.values()) == 2


def test_per_person_hours_come_from_the_ledger_not_a_flat_rate(assumptions, sites):
    """Spec 5.1: seasonal shrinkage AND attrition, both from the ledger.

    The two pull opposite ways over the first quarter and neither is
    negligible:

      seasonality  January to March has the lightest vacation and the
                   heaviest sick time; on balance a person delivers MORE
                   than the annual average
      attrition    but a roster of N is not N people for 13 weeks, which
                   takes more back than the season gave

    A flat 13 x 40 x (1 - 28%) would get both wrong at once.
    """
    weeks, _mondays = supply.plan_window(assumptions)
    window = weeks[:13]
    flat = 13 * assumptions["fte"]["paid_hours_per_week"] * (
        1 - assumptions["fte"]["paid_shrinkage"]["tier_1"]
    )

    from_ledger = supply.productive_hours_per_person("tier_1", window, assumptions, sites)
    assert from_ledger != pytest.approx(flat), "the flat rate must not be the answer"
    assert from_ledger < flat, "attrition should outweigh the lighter Q1 vacation"
    assert from_ledger == pytest.approx(367.07, abs=0.5)

    # With attrition switched off, the season alone leaves it ABOVE flat.
    no_leavers = supply.productive_hours_per_person(
        "tier_1", window, zero_attrition(assumptions), sites
    )
    assert no_leavers > flat


def test_seasonal_factors_are_normalised_over_the_year_not_the_window(assumptions):
    """The bug this nearly shipped with, pinned.

    Normalising a 13-week slice on its own forces that slice to average
    the annual rate, which silently deletes the seasonality: Q1 would come
    out looking exactly average however heavy its sick season was.
    """
    weeks, _mondays = supply.plan_window(assumptions)
    year = supply.plan_week_factors(assumptions)
    quarter = supply.plan_week_factors(assumptions, weeks[:13])

    for name, values in year.items():
        assert values.mean() == pytest.approx(1.0, abs=1e-12), name

    assert quarter["winter_sickness"].mean() > 1.2, "Q1 sickness must sit above the year"
    assert quarter["summer_and_year_end"].mean() < 0.9, "Q1 vacation must sit below it"

    with pytest.raises(ValueError, match="outside the plan window"):
        supply.plan_week_factors(assumptions, [0])


def test_the_tuned_roster_is_bigger_than_the_placeholder_it_replaced(assumptions):
    """A sanity check on direction, not a target.

    The placeholders were 12 / 12 / 22 / 12 / 6 = 64 people against
    required hours that Session 3 showed at roughly twice that. The tuned
    roster has to be materially larger or the tuning did nothing.
    """
    stated = assumptions["fte"]["starting_headcount"]
    totals = {role: sum(by.values()) for role, by in stated.items()}
    placeholders = {
        "onsite_associate": 12, "records_specialist": 12,
        "tier_1": 22, "tier_2": 12, "qa_specialist": 6,
    }
    assert sum(totals.values()) > sum(placeholders.values())
    for role, placeholder in placeholders.items():
        assert totals[role] >= placeholder, role


def test_the_counter_is_sized_after_the_help_it_receives(tuned, assumptions, sites):
    """Ordering matters: Records Specialists first, then the counter.

    Onsite Associate demand is the scheduled hours LEFT after the help, so
    sizing the counter before the back office would hire people to do work
    the Records Specialists were about to absorb.
    """
    counter = tuned[tuned.role == "onsite_associate"]
    assert len(counter) == 6
    assert (counter["required_hours"] > 0).all()

    # Without the help the counter would need more hours, so more people.
    no_help = copy.deepcopy(assumptions)
    no_help["onsite_team"]["records_specialist_help"]["enabled"] = False
    unhelped = supply.tune_starting_headcount(no_help, sites)
    unhelped_counter = unhelped[unhelped.role == "onsite_associate"]

    assert unhelped_counter["required_hours"].sum() > counter["required_hours"].sum()
    assert unhelped_counter["headcount"].sum() >= counter["headcount"].sum()


# ---------------------------------------------------------------------------
# The gap with no hiring (spec 6.1)
# ---------------------------------------------------------------------------


def test_the_gap_erodes_across_the_year_with_no_hiring(gaps, assumptions):
    """Opening balanced and then hiring nobody is a slow slide.

    Attrition thins the roster week by week while demand grows 6% a year,
    so a plan that opens covered does not stay covered. This is the whole
    reason 6.5 computes a hire-by date rather than waiting for the gap to
    appear.
    """
    network = gaps.groupby("quarter")["gap_fte"].sum()
    assert network[1] > 0, "Q1 should open covered"
    assert network[3] < network[1], "the gap should be worse by Q3 than Q1"

    weekly = gaps.groupby("week_index")["gap_fte"].sum()
    first_week = weekly.index.min()
    assert weekly.loc[first_week] > weekly.min()


def test_gap_fte_follows_the_spec_formula(gaps, assumptions, sites):
    """Spec 6.1: gap FTE = (available - required) / (40 x (1 - shrinkage))."""
    built = supply.build_ledger(assumptions, sites)
    ledger = built["ledger"].set_index(["week_index", "role", "scope"])
    paid = assumptions["fte"]["paid_hours_per_week"]

    sample = gaps.sample(200, random_state=0)
    for row in sample.itertuples():
        shrinkage = ledger.loc[(row.week_index, row.role, row.scope), "shrinkage"]
        expected = (row.available_hours - row.required_hours) / (paid * (1 - shrinkage))
        assert row.gap_fte == pytest.approx(expected, rel=1e-9)


def test_the_gap_covers_every_role_scope_and_plan_week(gaps, assumptions):
    cal = assumptions["calendar"]
    assert gaps["week_index"].nunique() == cal["plan_weeks"] == 52
    assert set(gaps["quarter"]) == {1, 2, 3, 4}
    assert set(gaps["role"]) == set(assumptions["supply"]["roles"])
    assert (gaps["required_hours"] > 0).all()
    assert (gaps["available_hours"] > 0).all()


# ---------------------------------------------------------------------------
# Sick days drawn separately for each site (spec 5.2, decided 10/26)
# ---------------------------------------------------------------------------


def test_each_site_draws_its_own_sick_days(built, sites):
    """Sites differ day to day, which is what makes coverage possible.

    Sick time is the one line nobody schedules. When every site loses the
    same share of the same Tuesday, every site is short on the same
    Tuesday - and cluster coverage has nothing to work with, because a
    site can only cover a neighbour on a day it is better off than them.
    """
    codes = [site["code"] for site in sites["sites"]]
    series = {
        code: built["shrinkage_lines"]["onsite_associate"][code]["sick"]
        for code in codes
    }

    # Every pair of sites differs, and differs day to day rather than only
    # by a weekly level shift.
    for i, left in enumerate(codes):
        for right in codes[i + 1:]:
            assert not np.allclose(series[left], series[right]), f"{left} vs {right}"
            difference = series[left] - series[right]
            assert difference.std() > 0
            # Within a single week the two sites still diverge, so the draw
            # is genuinely daily and not one number scaled across the week.
            assert np.abs(difference[0] - difference[0].mean()).max() > 1e-6


def test_every_site_sick_line_still_averages_its_annual_rate(built, sites, assumptions):
    """The draw redistributes sick time; it never creates or destroys it.

    Each site's series is normalised over the whole plan year, so N1 having
    a bad February does not make N1 a sicker site than N2 over the year.
    """
    rate = next(
        line["rate"] for line in assumptions["shrinkage"]["lines"] if line["code"] == "sick"
    )
    for site in sites["sites"]:
        series = built["shrinkage_lines"]["onsite_associate"][site["code"]]["sick"]
        assert series.mean() == pytest.approx(rate, rel=1e-9), site["code"]


def test_the_planned_lines_are_still_shared_across_sites(built, sites):
    """Only the unplanned line is drawn per site.

    Vacation, training and meetings are scheduled centrally, so they land
    the same way everywhere. If a planned line started wandering between
    sites, the per-site draw would have leaked past the line it belongs to.
    """
    codes = [site["code"] for site in sites["sites"]]
    for code in ("vacation", "digital_training", "meetings", "one_on_ones"):
        reference = built["shrinkage_lines"]["onsite_associate"][codes[0]][code]
        for site in codes[1:]:
            np.testing.assert_allclose(
                built["shrinkage_lines"]["onsite_associate"][site][code],
                reference,
                rtol=1e-12,
                err_msg=f"{code} at {site}",
            )


def test_the_network_pool_keeps_the_shared_sick_series(assumptions, sites, mondays):
    """A network pool is not a site, so it is not drawn per site.

    One remote team means one sick pattern, and it also means the remote
    roles' hours are untouched by the 10/26 per-site decision - which is
    what lets part 1's plan figures be compared with today's.
    """
    shared = supply.daily_shrinkage("tier_1", mondays, assumptions)
    pooled = supply.daily_shrinkage("tier_1", mondays, assumptions, scope="NETWORK")
    np.testing.assert_array_equal(shared, pooled)

    # And a site scope genuinely departs from it.
    at_site = supply.daily_shrinkage("records_specialist", mondays, assumptions, scope="N1")
    assert not np.allclose(at_site, supply.daily_shrinkage("records_specialist", mondays, assumptions))


def test_a_site_draw_does_not_depend_on_python_string_hashing(assumptions, sites):
    """Seeded by position in the sorted site list, not hash() (CLAUDE.md).

    Python randomises string hashing per process, so a hash-seeded draw
    would differ between runs and quietly break the fixed-seed promise.
    """
    index = supply.site_draw_index(assumptions, sites)
    assert index == {"N1": 0, "N2": 1, "N3": 2, "S1": 3, "S2": 4, "S3": 5}
