"""Spec Section 9, Session 4: onsite coverage and the coverage events log.

Covers plan.build_onsite (spec 6.2, 6.3) and coverage.py (spec 6.7), which
was split out of plan.py on 10/26. The Section 9 checks this file owns:

    - Coverage never borrows more than a lender's surplus, never crosses
      clusters, and every borrow creates a coverage event.
    - Travel time is subtracted from the covering site's hours on every
      cluster borrow, and travel cost = miles x mileage rate.
    - Every coverage event has all model-filled fields; planned or
      unplanned always matches its cause.
    - No site drops below its Records Specialist floor in the baseline; any
      idle specialist hours flow to walk-ins and phones, never to remote
      work.

Plus the two rules the 10/26 defaults added: a half-day minimum on every
borrow, and a round trip charged per covering person-day.

Why the counter is tested separately from the back office: it is checked by
DAY, not by week (spec 6.2, 10/26). A walk-in cannot wait until Thursday,
so the day is the unit of decision - and the consequence of a short day is
a service level, not a backlog.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from capacity_model import config, coverage, erlang, plan, work_volume


@pytest.fixture(scope="module")
def assumptions():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def prepared(assumptions, sites):
    work = work_volume.build(assumptions, sites)["work"]
    return plan.prepare(assumptions, sites, work)


@pytest.fixture(scope="module")
def back_office(assumptions, sites, prepared):
    return plan.build_plan(assumptions, sites, prepared=prepared)


@pytest.fixture(scope="module")
def onsite(assumptions, sites, prepared):
    return plan.build_onsite(assumptions, sites, prepared=prepared)


@pytest.fixture(scope="module")
def events(back_office, onsite, prepared, assumptions, sites):
    return coverage.coverage_events(back_office, onsite, prepared, assumptions, sites)


@pytest.fixture(scope="module")
def borrows(back_office, onsite):
    return pd.concat(
        [back_office["borrows"], onsite["borrows"]], ignore_index=True
    )


# ---------------------------------------------------------------------------
# The travel table (sites.yaml, spec 6.7)
# ---------------------------------------------------------------------------


def test_the_travel_table_covers_every_pair_inside_a_cluster(sites):
    """Spec 6.2 never borrows across a cluster, so the table need not either.

    Three sites in each of two clusters is three pairs each, six in all.
    A pair that crossed a cluster would be a quiet invitation to break the
    rule, so travel_pairs refuses to load one.
    """
    members, _of_site = plan.cluster_lookup(sites)
    pairs = plan.travel_pairs(sites)

    expected = {
        (a, b)
        for group in members.values()
        for a in group
        for b in group
        if a != b
    }
    assert set(pairs) == expected
    assert len(pairs) == 12  # six pairs, both directions

    # Symmetric: the drive is the same length in either direction.
    for (a, b), leg in pairs.items():
        assert pairs[(b, a)] == leg

    # Within the illustrative range the 10/26 defaults set.
    for leg in pairs.values():
        assert 20 <= leg["minutes"] <= 35
        assert 12 <= leg["miles"] <= 24


def test_a_cross_cluster_travel_pair_is_refused(sites):
    """The rule lives in the loader, not only in the solver."""
    broken = {
        "clusters": sites["clusters"],
        "sites": sites["sites"],
        "travel": {
            **sites["travel"],
            "pairs": [{"sites": ["N1", "S1"], "minutes": 25, "miles": 16}],
        },
    }
    with pytest.raises(ValueError, match="crosses clusters"):
        plan.travel_pairs(broken)


# ---------------------------------------------------------------------------
# The borrow solver (spec 6.2 step 1, 6.7)
# ---------------------------------------------------------------------------


def test_a_borrow_never_costs_the_lender_more_than_its_surplus():
    """The constraint the whole solver exists to respect.

        hours borrowed + travel hours <= the lender's surplus

    If it did not hold, helping a neighbour would push the LENDER into a
    shortfall and the model would have invented hours out of nothing.
    """
    for shortfall in (4.0, 6.0, 9.5, 20.0, 40.0):
        for surplus in (4.0, 5.0, 8.0, 13.5, 30.0):
            for minutes in (22, 31, 34):
                hours, days, travel = plan.borrow_offer(
                    shortfall, surplus, minutes, 6.26, 4.0
                )
                assert hours + travel <= surplus + 1e-9
                assert hours <= shortfall + 1e-9
                if hours > 0:
                    assert days >= 1
                    assert travel == pytest.approx(days * 2 * minutes / 60)
                    # Enough trips to deliver the hours, and no more.
                    assert hours <= days * 6.26 + 1e-9
                    assert hours > (days - 1) * 6.26 - 1e-9


def test_no_borrow_under_half_a_day():
    """Decided 10/26: nobody drives 30 minutes each way to cover one hour.

    Both ways round: a shortfall under the minimum buys nothing, and a
    lender whose surplus cannot fund four hours plus the drive lends
    nothing either.
    """
    minimum = 4.0
    assert plan.borrow_offer(3.9, 100.0, 22, 6.26, minimum) == (0.0, 0, 0.0)
    # Four hours wanted, but the lender has only 4.5 and the round trip is
    # 0.73 - leaving 3.77, under the minimum, so no trip is made.
    assert plan.borrow_offer(4.0, 4.5, 22, 6.26, minimum) == (0.0, 0, 0.0)
    # One more hour of surplus and it goes ahead.
    hours, days, _travel = plan.borrow_offer(4.0, 5.5, 22, 6.26, minimum)
    assert hours == pytest.approx(4.0)
    assert days == 1


def test_borrowing_never_crosses_a_cluster(assumptions, sites):
    """Spec 6.2: never across clusters, whatever the shortfall.

    A whole cluster short while the other sits on a mountain of surplus is
    the case that would expose a leak, so that is the case tested.
    """
    shortfalls = {"N1": 40.0, "N2": 40.0, "N3": 40.0, "S1": 0.0, "S2": 0.0, "S3": 0.0}
    surpluses = {"N1": 0.0, "N2": 0.0, "N3": 0.0, "S1": 200.0, "S2": 200.0, "S3": 200.0}
    assert (
        plan.allocate_borrowing(
            shortfalls, surpluses, "records_specialist", assumptions, sites
        )
        == []
    )


def test_the_nearest_lender_goes_first(assumptions, sites):
    """Travel is the only thing borrowing costs, so nearest is cheapest.

    N1 is 22 minutes from N2 and 31 from N3. Both can cover, so N2 should
    be asked first - which is also the plan a supervisor would make.
    """
    borrows = plan.allocate_borrowing(
        {"N1": 30.0}, {"N2": 20.0, "N3": 20.0}, "records_specialist", assumptions, sites
    )
    assert [b["covering_site"] for b in borrows] == ["N2", "N3"]


def test_a_lender_is_never_asked_for_more_than_it_has(assumptions, sites):
    """Across several borrows from the same site in one pass."""
    surpluses = {"N2": 12.0, "N3": 12.0}
    borrows = plan.allocate_borrowing(
        {"N1": 100.0}, surpluses, "records_specialist", assumptions, sites
    )
    assert borrows
    spent = {}
    for borrow in borrows:
        spent.setdefault(borrow["covering_site"], 0.0)
        spent[borrow["covering_site"]] += borrow["hours"] + borrow["travel_hours"]
    for site, total in spent.items():
        assert total <= surpluses[site] + 1e-9


def test_travel_cost_is_miles_times_the_rate(borrows, sites):
    """Spec Section 9, and spec 6.7's editable mileage rate."""
    rate = sites["travel"]["mileage_rate"]
    assert rate == 0.70
    assert len(borrows) > 0, "no borrows to check, so this would prove nothing"
    np.testing.assert_allclose(
        borrows["travel_cost"], borrows["travel_miles"] * rate, rtol=1e-12
    )


def test_travel_is_a_round_trip_for_every_covering_person_day(borrows, sites):
    """Decided 10/26: each covering person-day carries the drive both ways.

    The YAML holds a one-way leg, so a borrow of two person-days between
    sites 22 minutes apart costs 2 x 2 x 22 minutes of the lender's time
    and 2 x 2 x 14 miles of somebody's expense claim.
    """
    pairs = plan.travel_pairs(sites)
    for borrow in borrows.itertuples():
        leg = pairs[(borrow.short_site, borrow.covering_site)]
        assert borrow.travel_hours == pytest.approx(
            borrow.person_days * 2 * leg["minutes"] / 60
        )
        assert borrow.travel_miles == pytest.approx(
            borrow.person_days * 2 * leg["miles"]
        )


# ---------------------------------------------------------------------------
# Borrowing inside the plan (spec 6.2, Section 9)
# ---------------------------------------------------------------------------


def test_the_plan_borrows_and_the_lender_pays_for_the_drive(back_office):
    """Travel comes off the COVERING site's hours (spec 6.7 F19).

    The lender loses the hours worked AND the hours driven; the borrower
    gains only the hours worked. The difference is the real cost of
    coverage, and leaving it out would make borrowing look free.
    """
    cost = back_office["cost"].set_index(["week_index", "role", "scope"])
    borrows = back_office["borrows"]
    assert len(borrows) > 0

    for (week, role), group in borrows.groupby(["week_index", "role"]):
        for site, taken in group.groupby("short_site")["hours"]:
            assert cost.loc[(week, role, site), "borrowed_hours"] == pytest.approx(
                taken.sum(), rel=1e-9
            )
        for site, lent in group.groupby("covering_site")[["hours", "travel_hours"]]:
            assert cost.loc[(week, role, site), "lent_hours"] == pytest.approx(
                lent["hours"].sum() + lent["travel_hours"].sum(), rel=1e-9
            )


def test_lending_never_makes_the_lender_short(back_office, onsite):
    """The property that makes step 1 of 6.2 safe to run first.

    Only surplus is ever offered, and travel is taken out of that same
    surplus, so a site that lends still finishes with at least its own
    required hours. If this failed, the plan would be robbing one site to
    pay another and calling it coverage.
    """
    cost = back_office["cost"]
    lenders = cost[cost["lent_hours"] > 1e-9]
    assert len(lenders) > 0
    assert (lenders["covered_hours"] >= lenders["required_hours"] - 1e-9).all()

    daily = onsite["daily"]
    live_lenders = daily[daily["lent_hours"] > 1e-9]
    assert (
        live_lenders["available_hours"] >= live_lenders["required_hours"] - 1e-9
    ).all()


def test_every_borrow_creates_a_coverage_event(events, borrows):
    """Spec Section 9, and spec 6.2 step 2: each borrow is written down."""
    logged = events[events["event_type"] == "E1"]
    assert len(logged) == len(borrows)

    expected = {
        (b.week_index, b.day_of_week, b.role, b.short_site, b.covering_site, round(b.hours, 9))
        for b in borrows.itertuples()
    }
    actual = {
        (r.week_index, r.day_of_week, r.role, r.short_site, r.covering_site,
         round(r.hours_covered, 9))
        for r in logged.itertuples()
    }
    assert actual == expected


# ---------------------------------------------------------------------------
# The counter, checked by day (spec 6.2, 6.3)
# ---------------------------------------------------------------------------


def test_the_counter_is_checked_by_day_and_the_back_office_by_week(
    onsite, back_office, assumptions
):
    """Spec 6.2, decided 10/26: the two grains, and who is on which."""
    assert assumptions["coverage_events"]["daily_check_roles"] == ["onsite_associate"]
    assert set(onsite["daily"]["role"]) == {"onsite_associate"}
    assert set(onsite["daily"]["day_of_week"]) == set(plan.WEEKDAYS)
    assert "onsite_associate" not in set(back_office["cost"]["role"])


def test_the_counter_requirement_is_the_schedule_after_help_plus_spillover(
    onsite, prepared, assumptions
):
    """Spec 5.1 and 4.5, decided 10/26.

    The Onsite Associates owe the hours the schedule still asks of THEM -
    after the Records Specialists have taken their share of the floor -
    plus any email that would not fit into idle time. Required coverage in
    each interval is untouched by the help; what changes is who pays.
    """
    live = (
        prepared["onsite_intervals"]
        .groupby(["week_index", "site", "day_of_week"])["scheduled_hours"]
        .sum()
    )
    email = prepared["onsite_hours"].set_index(
        ["week_index", "site", "day_of_week"]
    )["email_paid_hours"]

    for row in onsite["daily"].head(200).itertuples():
        key = (row.week_index, row.site, row.day_of_week)
        assert row.required_hours == pytest.approx(
            float(live.loc[key]) + float(email.loc[key]), rel=1e-9
        )

    # The help is real, or the sentence above would be decoration.
    assert prepared["onsite_intervals"]["help_hours"].sum() > 0


def test_overtime_at_the_counter_stays_inside_the_weekly_cap(onsite, assumptions):
    """Spec 5.1: 10% of the role's paid hours - a WEEKLY budget, spent daily.

    The decision is the day's, because a walk-in cannot wait. The budget is
    the week's, because that is how overtime is authorised. So the days of a
    week have to be solved together.
    """
    paid = assumptions["fte"]["paid_hours_per_week"]
    settings = assumptions["supply"]["roles"]["onsite_associate"]
    daily = onsite["daily"]
    assert settings["overtime_cap"] == 0.10

    for (week, site), group in daily.groupby(["week_index", "site"]):
        cap = settings["overtime_cap"] * group["paid_headcount"].iloc[0] * paid
        assert group["overtime_cap_hours"].nunique() == 1
        assert group["overtime_cap_hours"].iloc[0] == pytest.approx(cap, rel=1e-9)
        assert group["overtime_hours"].sum() <= cap + 1e-9

    # Never more overtime than the day had work for.
    assert (
        daily["overtime_hours"] <= daily["shortfall_after_borrow_hours"] + 1e-9
    ).all()
    assert (daily["overtime_hours"] >= -1e-12).all()


def test_the_coverage_order_adds_up_at_the_counter(onsite):
    """Spec 6.2: borrow, then overtime, then accept the gap."""
    daily = onsite["daily"]
    np.testing.assert_allclose(
        daily["covered_hours"],
        daily["available_hours"] + daily["borrowed_hours"] + daily["overtime_hours"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        daily["uncovered_hours"],
        (daily["required_hours"] - daily["covered_hours"]).clip(lower=0.0),
        rtol=1e-9,
        atol=1e-12,
    )
    assert (daily["uncovered_hours"] >= -1e-12).all()
    assert daily["uncovered_hours"].sum() > 0, "some days genuinely go uncovered"


# ---------------------------------------------------------------------------
# Service level at the staff actually available (spec 6.3)
# ---------------------------------------------------------------------------


def test_a_fully_covered_day_reproduces_the_designed_service_level(
    onsite, prepared, assumptions
):
    """Spec 6.3 recomputes the outcome; it must not move the yardstick.

    At full coverage the recomputation has to hand back the service levels
    the schedule was built to deliver - 80% of calls in 20 seconds and 90%
    of walk-ins in 10 minutes. Otherwise every miss the plan reports would
    be partly an artefact of the recomputation itself.
    """
    bundle, phones, walk_ins = plan._interval_bundle(
        prepared["onsite_intervals"], assumptions
    )
    daily = onsite["daily"]
    covered = daily[daily["covered_hours"] >= daily["required_hours"] - 1e-9]
    assert len(covered) > 0

    for row in covered.head(60).itertuples():
        result = plan.service_levels(
            bundle[(row.week_index, row.site, row.day_of_week)], 1.0, phones, walk_ins
        )
        assert result["phones"] >= phones["service_level"] - 1e-9
        assert result["walk_ins"] >= walk_ins["service_level"] - 1e-9
        assert row.service_level_phones_after == pytest.approx(result["phones"], rel=1e-9)


def test_service_level_falls_when_the_staff_do(onsite, prepared, assumptions):
    """Erlang C at fewer people, which is the consequence 6.3 asks for.

    Monotone in coverage: a day covered to 70% can never report a better
    service level than the same day covered to 90%. Erlang C guarantees it
    interval by interval, and the arrival-weighted average inherits it.
    """
    bundle, phones, walk_ins = plan._interval_bundle(
        prepared["onsite_intervals"], assumptions
    )
    key = sorted(bundle)[200]
    levels = [
        plan.service_levels(bundle[key], ratio, phones, walk_ins)["phones"]
        for ratio in (0.5, 0.7, 0.8, 0.9, 1.0)
    ]
    assert levels == sorted(levels)
    assert levels[0] < levels[-1], "the day has to be staffing-sensitive"

    # And a short day in the plan reports worse than its own fully covered self.
    daily = onsite["daily"]
    short = daily[daily["uncovered_hours"] > 1.0]
    assert len(short) > 0
    for row in short.head(40).itertuples():
        full = plan.service_levels(
            bundle[(row.week_index, row.site, row.day_of_week)], 1.0, phones, walk_ins
        )
        assert row.service_level_phones_after <= full["phones"] + 1e-9


def test_a_fractional_person_is_blended_not_floored(prepared, assumptions):
    """Half a person cannot take a call, but half an interval can have one.

    Erlang C only answers for whole people. Flooring 2.98 to 2 would turn a
    rounding remainder into a service-level collapse, so a fractional
    staffing is read as the interval running at 2 for part of the half-hour
    and 3 for the rest - and the service level is the blend.
    """
    offered = np.array([8.0])
    handle = np.array([300.0])
    low = erlang.service_level(2, 8.0, 20, 300.0)
    high = erlang.service_level(3, 8.0, 20, 300.0)

    for weight in (0.0, 0.25, 0.5, 0.75):
        blended = plan._blended_service_level(
            np.array([2.0 + weight]), offered, handle, 20
        )[0]
        assert blended == pytest.approx((1 - weight) * low + weight * high)

    # At a whole number it IS Erlang C, with nothing interpolated.
    assert plan._blended_service_level(np.array([3.0]), offered, handle, 20)[0] == (
        pytest.approx(high)
    )


# ---------------------------------------------------------------------------
# The Records Specialist floor (spec Section 1, 6.7 E4)
# ---------------------------------------------------------------------------


def test_no_floor_breach_while_the_rounded_headcount_holds_at_two(
    events, prepared, assumptions
):
    """Spec 6.7, decided 10/26: a fraction of a person is not a person.

    Headcount is continuous here, so a site that opens at exactly the floor
    of 2 reads 1.993 the week attrition starts. Logging that as a breach
    produced 52 events a year describing nobody leaving. The breach now
    counts only when the ROUNDED headcount falls below 2, and in the
    baseline plan year no site gets there - S2 opens at 2 and ends at 1.67,
    which still rounds to 2.
    """
    floor = assumptions["supply"].get("records_specialist_floor_per_site", 2)
    assert floor == 2
    assert assumptions["coverage_events"]["floor_breach"]["round_headcount"] is True

    ledger = prepared["ledger"]["ledger"]
    specialists = ledger[ledger["role"] == "records_specialist"]

    # Fractional dips are real and are deliberately NOT breaches.
    assert (specialists["ending_headcount"] < floor).any()
    assert (specialists["ending_headcount"].round() >= floor).all()
    assert len(events[events["event_type"] == "E4"]) == 0


def thin_the_records_bench(prepared, scope="S2", headcount=1.4):
    """A copy of `prepared` with one site's Records bench below the floor.

    The baseline never gets there - which is the point of the 10/26
    rounding rule - so the E4 path is proved on a roster driven under it
    rather than left untested.
    """
    thin = copy.deepcopy(prepared)
    ledger = thin["ledger"]["ledger"].copy()
    mine = (ledger["role"] == "records_specialist") & (ledger["scope"] == scope)
    ledger.loc[mine, "ending_headcount"] = headcount
    thin["ledger"] = {**thin["ledger"], "ledger": ledger}
    return thin


def test_a_whole_person_gone_logs_a_breach_and_records_the_backfill(
    back_office, onsite, prepared, assumptions, sites
):
    """E4, and F24 carrying the backfill (spec 6.7, decided 10/26).

    The requisition is recorded as the follow-up rather than as a field of
    its own, because it IS the follow-up - and a field only one event type
    ever fills would have been a column of blanks on 1,300 rows.
    """
    events = coverage.coverage_events(
        back_office, onsite, thin_the_records_bench(prepared), assumptions, sites
    )
    breaches = events[events["event_type"] == "E4"]
    assert len(breaches) == assumptions["calendar"]["plan_weeks"]
    assert set(breaches["short_site"]) == {"S2"}
    assert set(breaches["role"]) == {"records_specialist"}
    assert (breaches["cause"] == "vacancy").all()
    assert (breaches["planned_or_unplanned"] == "unplanned").all()

    # F24 is filled by the MODEL here, not drawn from the sample pool.
    stated = assumptions["coverage_events"]["floor_breach"]["follow_up"]
    assert stated == "backfill requisition opened"
    assert (breaches["follow_up"] == stated).all()

    # And no other event type can draw that value by accident, because it
    # is deliberately absent from the sample pool.
    pool = assumptions["coverage_events"]["sample_text"]["follow_up"]
    assert stated not in pool
    others = events[events["event_type"] != "E4"]
    assert set(others["follow_up"]) <= set(pool)

    # 1.4 rounds to 1, so the hours short are one person-week of productive
    # time, not the fractional 0.6.
    per_week = plan.productive_hours_per_person_day("records_specialist", assumptions) * 5
    assert breaches["hours_short_before_cover"].iloc[0] == pytest.approx(per_week)


def test_idle_specialist_hours_go_to_the_counter_and_not_to_remote_work(
    back_office, prepared
):
    """Spec Section 9: never to remote work.

    The hours the Records Specialists spend at the counter are taken off
    their own available hours, or the same hour would be spent twice. Only
    SPARE hours are ever offered, so this can never push their own queue
    into a shortfall - which is the property that makes 4.5 safe.
    """
    cost = back_office["cost"]
    specialists = cost[cost["role"] == "records_specialist"]
    helped = specialists[specialists["help_to_counter_hours"] > 1e-9]
    assert len(helped) > 0

    offered = prepared["onsite_intervals"].groupby(["week_index", "site"])["help_hours"].sum()
    for row in helped.itertuples():
        assert row.help_to_counter_hours == pytest.approx(
            float(offered.loc[(row.week_index, row.scope)]), rel=1e-9
        )

    # The help never came out of the time their own queue needed.
    assert (helped["uncovered_hours"] <= 1e-9).all()

    # And no remote role ever receives it.
    remote = cost[cost["role"] != "records_specialist"]
    assert remote["help_to_counter_hours"].abs().max() < 1e-12


# ---------------------------------------------------------------------------
# The event log (spec 6.7)
# ---------------------------------------------------------------------------


def test_every_event_has_all_of_its_model_fields(events):
    """Spec Section 9. Blank only where spec 6.7 says the field does not apply.

    6.7 already scopes two fields by event type - F7 is "E1 only", F21 is
    "E2 only" - so a blank there is correct rather than missing. The same
    reasoning covers the two consequence fields: a live queue has a service
    level and no inventory, and a back-office queue has it the other way
    round. field_applies holds that rule in one place so this test and the
    builder cannot drift apart.
    """
    assert len(events) > 0
    for field, columns, _types, _grain in coverage.MODEL_FIELDS:
        for column in columns:
            assert column in events.columns, f"{field} missing column {column}"
            expected = [
                coverage.field_applies(field, event_type, grain)
                for event_type, grain in zip(events["event_type"], events["grain"])
            ]
            filled = events[column].notna().to_numpy()
            np.testing.assert_array_equal(
                filled, np.array(expected), err_msg=f"{field} / {column}"
            )

    # The person-entered fields carry sample values on every row (6.7).
    for field, columns in coverage.PERSON_FIELDS:
        for column in columns:
            assert events[column].notna().all(), f"{field} / {column}"
            assert (events[column].astype(str).str.len() > 0).all()


def test_planned_or_unplanned_always_matches_the_cause(events, assumptions):
    """Spec Section 9. F14 is derived from F13, never typed beside it.

    Two fields that mean the same thing drift apart the moment a person
    edits one of them, so the model never offers the chance: the tag is a
    lookup on the cause.
    """
    causes = assumptions["coverage_events"]["causes"]
    assert set(events["cause"]) <= set(causes)
    for row in events.itertuples():
        assert row.planned_or_unplanned == causes[row.cause]
    assert set(events["planned_or_unplanned"]) <= {"planned", "unplanned"}

    # The notes a person left agree with the tag, so the sample data is not
    # self-contradicting the moment anyone reads it.
    samples = assumptions["coverage_events"]["sample_text"]["notes"]
    for row in events.itertuples():
        assert row.cause_notes in samples[row.planned_or_unplanned]


def test_the_baseline_log_carries_the_three_types_it_can(events):
    """Spec 6.7: E1 to E3 occur in the baseline. E4 and E5 cannot.

    E5 is an automation outage and there is no automation until the Session
    5 scenarios. E4 needs a whole Records Specialist gone, and 52 weeks of
    attrition on a roster of 2 does not reach that (decided 10/26). Both
    types exist in the spec and no row claims either, which is the honest
    way to leave them - the shape of the file does not change with the
    data, so the columns are there either way.
    """
    assert set(events["event_type"]) == {"E1", "E2", "E3"}
    assert not {"E4", "E5"} & set(events["event_type"])
    for _field, columns, _types, _grain in coverage.MODEL_FIELDS:
        for column in columns:
            assert column in events.columns


def test_the_action_taken_matches_the_event_type(events, assumptions):
    """F22, the model's suggestion, editable by a person (spec 6.7)."""
    actions = assumptions["coverage_events"]["actions"]
    for row in events.itertuples():
        assert row.action_taken == actions[row.event_type]


def test_remote_roles_log_their_events_against_the_network(events, assumptions):
    """Spec 6.7: a network pool has no site, so it files under "network".

    And no E1: there is nobody to borrow from when you already are every
    site, so step 1 of the 6.2 order is absent rather than empty.
    """
    remote = set(plan.remote_roles(assumptions))
    rows = events[events["role"].isin(remote)]
    assert len(rows) > 0
    assert set(rows["short_site"]) == {"network"}
    assert set(rows["cluster"]) == {"network"}
    assert set(rows["event_type"]) <= {"E2", "E3"}


def test_the_hours_on_an_event_reconcile(events):
    """F10, F11 and F12: short before cover, covered, still uncovered."""
    assert (events["hours_short_before_cover"] >= -1e-9).all()
    assert (events["hours_covered"] >= -1e-9).all()
    assert (events["hours_uncovered"] >= -1e-9).all()
    assert (
        events["hours_covered"] <= events["hours_short_before_cover"] + 1e-6
    ).all()

    # E3 is the type that exists precisely because something was left over.
    gaps = events[events["event_type"] == "E3"]
    assert len(gaps) > 0
    assert (gaps["hours_uncovered"] > 0).all()

    # E1 and E2 both covered something, or they would not have been logged.
    for code in ("E1", "E2"):
        rows = events[events["event_type"] == code]
        assert len(rows) > 0
        assert (rows["hours_covered"] > 0).all()


def test_event_ids_are_unique_and_deterministic(
    events, back_office, onsite, prepared, assumptions, sites
):
    """The log has to mean the same thing on a second run (CLAUDE.md seed rule)."""
    assert events["event_id"].is_unique
    assert len(events) == events["event_id"].nunique()

    again = coverage.coverage_events(back_office, onsite, prepared, assumptions, sites)
    pd.testing.assert_frame_equal(events, again)


def test_the_cause_is_one_of_the_three_the_spec_names(events, assumptions):
    """F13: a volume spike, a vacancy, or a shrinkage line.

    Only lines spec 6.7 lists are eligible, which is why paid breaks and
    corporate needs never appear: they are real 5.2 lines, but "you were
    short because of paid breaks" is not a finding anybody can act on.
    """
    lines = {line["code"] for line in assumptions["shrinkage"]["lines"]}
    listed = set(assumptions["coverage_events"]["causes"])
    assert set(events["cause"]) <= listed
    assert not {"paid_breaks", "corporate_needs"} & set(events["cause"])
    assert {"paid_breaks", "corporate_needs"} <= lines, "still real 5.2 lines"

    # All three kinds of explanation actually occur, or the field would be
    # a constant dressed up as an analysis.
    assert {"volume_spike", "vacancy"} <= set(events["cause"])
    assert set(events["cause"]) & lines


def test_the_cause_names_the_line_most_above_its_own_normal_level(
    events, prepared, assumptions
):
    """Spec 6.7, decided 10/26 - and why the rule had to change.

    Scored as the line's share that day against its own annual average, so
    a 1% line landing entirely on Wednesdays beats a 10% line spread
    evenly. The old rule compared raw sizes and therefore answered
    "vacation" every single time: at 10% it is the largest line in every
    week of the year, and the field carried no information at all.
    """
    lines = prepared["ledger"]["shrinkage_lines"]
    weeks = list(prepared["ledger"]["weeks"])
    position = {week: i for i, week in enumerate(weeks)}
    listed = set(assumptions["coverage_events"]["causes"])

    by_line = events[~events["cause"].isin(["volume_spike", "vacancy"])]
    assert len(by_line) > 0

    for row in by_line.head(400).itertuples():
        scope = "NETWORK" if row.short_site == "network" else row.short_site
        series = lines[row.role][scope]
        index = position[row.week_index]
        best = None
        for code, values in sorted(series.items()):
            if code not in listed:
                continue
            normal = float(values.mean())
            if row.day_of_week == plan.ALL_WEEK:
                share = float(values[index].mean())
            else:
                share = float(values[index, plan.WEEKDAYS.index(row.day_of_week)])
            candidate = (round(share / normal, 9), round(normal, 9), code)
            best = candidate if best is None else max(best, candidate)
        assert row.cause == best[2], (
            f"{row.event_id} {row.role}/{scope} week {row.week_index} "
            f"{row.day_of_week}: said {row.cause}, expected {best[2]}"
        )

    # The new rule has to beat the old one at its own job: naming more than
    # one thing. Vacation alone no longer explains the year.
    assert events["cause"].nunique() >= 5
    assert (events["cause"] == "vacation").mean() < 0.5


def test_a_meeting_heavy_wednesday_and_winter_sickness_both_surface(events):
    """The two cases spec 6.7 names as the reason for the 10/26 change.

    Meetings are 1% of the year and all of it lands on Wednesday, so a
    Wednesday runs five times its own normal level - the highest score any
    line reaches. Sick time is drawn per site and peaks in winter, so it
    wins on days when nothing scheduled is unusual.
    """
    wednesdays = events[events["day_of_week"] == "wednesday"]
    assert len(wednesdays) > 0
    scheduled = wednesdays[~wednesdays["cause"].isin(["volume_spike", "vacancy"])]
    assert (scheduled["cause"] == "meetings").all()

    assert "sick" in set(events["cause"])
    sick = events[events["cause"] == "sick"]
    # Never on a Wednesday, where meetings always score higher.
    assert "wednesday" not in set(sick["day_of_week"])


# ---------------------------------------------------------------------------
# The remote roles are untouched by the 10/26 changes
# ---------------------------------------------------------------------------


def test_the_remote_role_results_from_part_one_are_unchanged(back_office):
    """A regression pin on the figures task 4d part 1 reported.

    The per-site sick draw must not reach the network pools, and the two
    coverage-log changes must not reach the plan arithmetic at all. These
    numbers were produced before any of the three 10/26 decisions, so if
    one of them leaked into supply, this is where it shows.
    """
    cost = back_office["cost"]
    remote = cost[cost["role"] != "records_specialist"]

    totals = remote.groupby("role")["overtime_hours"].sum()
    assert totals["tier_1"] == pytest.approx(2380.8, abs=0.1)
    assert totals["tier_2"] == pytest.approx(825.8, abs=0.1)
    assert totals["qa_specialist"] == pytest.approx(91.6, abs=0.1)
    assert remote["overtime_hours"].sum() == pytest.approx(3298.2, abs=0.1)
    assert remote["uncovered_hours"].sum() == pytest.approx(13.7, abs=0.1)

    # And the Records Specialists DID move, because they are per site and
    # now carry their own sick draw - otherwise this test would be proving
    # that nothing happened anywhere.
    specialists = cost[cost["role"] == "records_specialist"]
    assert specialists["overtime_hours"].sum() != pytest.approx(124.5, abs=0.1)


def test_the_fully_staffed_turnaround_baseline_still_holds(
    back_office, prepared, assumptions, sites
):
    """5.23 days in the first plan week (spec 4.4), as it has been since 10/22."""
    turnaround = plan.weekly_turnaround(
        back_office["queues"], assumptions, sites, prepared=prepared
    )
    first = turnaround["week_index"].min()
    opening = turnaround[turnaround["week_index"] == first].iloc[0]
    assert opening["turnaround_days"] == pytest.approx(5.232, abs=0.005)
