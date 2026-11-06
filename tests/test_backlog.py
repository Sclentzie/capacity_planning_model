"""Spec Section 9, Session 3: required hours by role (spec 4.3, 4.4).

    - No request is assigned to a tier without that skill.
    - Media shares sum to 100% at every site, and onsite retrieval hours
      match non-digital items x retrieval time by media.

This is the step where items become hours and acquire an owner, so the two
things that can go wrong are routing work to somebody who cannot do it, and
losing or duplicating it on the way. Both are checked against the
work-volume file rather than against this module's own totals.

No automation, deflection or backlog catch-up yet - these are straight
workload hours, and a test pins that too, so the day someone adds a
catch-up term it cannot slip in silently.
"""

import numpy as np
import pandas as pd
import pytest

from capacity_model import backlog, config, work_volume

FIRST_PLAN_WEEK = 156


@pytest.fixture(scope="module")
def assumptions():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def skills():
    return config.load_skills()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def work(assumptions, sites):
    return work_volume.build(assumptions, sites)["work"]


@pytest.fixture(scope="module")
def result(assumptions, sites, work):
    return backlog.required_hours(assumptions, sites, work)


@pytest.fixture(scope="module")
def hours(result):
    return result["hours"]


# ---------------------------------------------------------------------------
# Routing by skill (spec 4.4)
# ---------------------------------------------------------------------------


def test_no_work_reaches_a_role_without_the_skill(work, assumptions, skills):
    """Spec 9: no request is assigned to a tier without that skill.

    Only the "by request type" rows route by skill - a named owner in the
    Section 1 table owns its row outright, which is why intake is Tier 1
    even for a Tier 2 request. For the rows that DO route by skill, the
    tier must be the one skills.yaml says holds it.

    Checked on every routed item rather than on the summary, because the
    summary has already collapsed request types away.
    """
    by_code = {w["code"]: w for w in assumptions["work_types"]}
    tiers = backlog.request_type_tiers(skills)
    routed = work[~work["work_type"].isin(backlog.onsite_work_types(assumptions))]

    by_request_type = [
        code for code, wt in by_code.items() if wt["owner"] == backlog.BY_REQUEST_TYPE
    ]
    assert set(by_request_type) == {"W5", "W8", "W10", "W24"}, (
        "spec Section 1 marks these rows by request type; W24 was added 10/22"
    )

    for code, request_type in set(zip(routed["work_type"], routed["request_type"])):
        role = backlog.role_for(by_code[code], request_type, tiers)
        if code in by_request_type:
            assert role == tiers[request_type], f"{code}/{request_type} went to {role}"
        else:
            assert role == by_code[code]["owner"]
        assert role in {r["code"] for r in skills["roles"]}


def test_the_skill_matrix_agrees_with_the_request_type_table(assumptions, skills):
    """skills.yaml and assumptions.yaml must not disagree about a tier.

    The tier is written in both files - once as the routing matrix, once as
    an attribute of the request type - and a silent disagreement would send
    work to a tier that cannot do it while every other test still passed.
    """
    matrix = backlog.request_type_tiers(skills)
    stated = {rt["code"]: rt["skill_tier"] for rt in assumptions["request_types"]}
    assert matrix == stated


def test_named_owners_match_the_section_one_table(assumptions):
    """The owner column, pinned so a routing change has to be deliberate."""
    owners = {w["code"]: w["owner"] for w in assumptions["work_types"]}
    assert owners["W14"] == owners["W22"] == "qa_specialist"
    for code in ("W11", "W12", "W13", "W17"):
        assert owners[code] == "records_specialist", code
    for code in ("W1", "W2", "W3", "W4", "W6", "W9", "W16", "W18"):
        assert owners[code] == "tier_1", code
    for code in ("W7", "W15", "W23"):
        assert owners[code] == "tier_2", code
    for code in ("W19", "W20", "W21"):
        assert owners[code] == "onsite_associate", code


def test_onsite_associates_get_no_hours_here(hours, result, assumptions):
    """W19 to W21 belong to intervals.py; routing them here would pay twice.

    The live queues are staffed by Erlang C and the email work is absorbed
    into their idle time, so these three must be visibly set aside rather
    than quietly missing.
    """
    assert "onsite_associate" not in set(hours["role"])
    assert backlog.onsite_work_types(assumptions) == {"W19", "W20", "W21"}
    assert set(result["routed_elsewhere"]["work_type"]) == {"W19", "W20", "W21"}


# ---------------------------------------------------------------------------
# Hours arithmetic
# ---------------------------------------------------------------------------


def test_hours_are_items_times_handle_time_on_every_row(hours):
    """Spec 4.4: hours = items x handle time / 60. No other term, anywhere."""
    np.testing.assert_allclose(
        hours["required_hours"], hours["items"] * hours["handle_time_minutes"] / 60, rtol=1e-12
    )
    assert (hours["required_hours"] > 0).all()
    assert (hours["handle_time_minutes"] > 0).all()


def test_every_work_item_is_routed_exactly_once(work, result, assumptions):
    """Nothing lost, nothing counted twice, measured against work_volume.

    The routed items plus the three onsite types must add back to every
    item the work-volume build produced. This is the check that would catch
    a work type silently dropped by a routing rule, which is the easiest
    mistake to make here and the hardest to see.
    """
    routed = result["hours"]["items"].sum()
    elsewhere = result["routed_elsewhere"]["work_items"].sum()
    assert routed + elsewhere == pytest.approx(work["work_items"].sum(), rel=1e-9)

    # And per work type, so a compensating pair of errors cannot hide.
    expected = work.groupby("work_type")["work_items"].sum()
    got = result["hours"].groupby("work_type")["items"].sum()
    for code, items in expected.items():
        if code in backlog.onsite_work_types(assumptions):
            assert code not in got.index
        else:
            assert got[code] == pytest.approx(items, rel=1e-9), code


def test_no_automation_deflection_or_catch_up_has_crept_in(hours):
    """Session 3 task 3 is straight workload (spec 4.3 comes later)."""
    forbidden = {
        "automated_items", "deflection", "exception_hours", "backlog",
        "catch_up_hours", "days_of_inventory",
    }
    assert not forbidden & set(hours.columns)


# ---------------------------------------------------------------------------
# The three hand checks for the first plan week
# ---------------------------------------------------------------------------


def test_w5_splits_between_the_tiers_at_different_handle_times(hours, work, skills):
    """W5: 5 minutes Tier 1, 12 minutes Tier 2 (spec Section 1).

    The clearest case of handle time following the skill. Both tiers see
    the same number of items here - R1 to R3 are 50% of the mix and R4 to
    R11 the other 50% - so Tier 2 costs exactly 2.4x the hours for the same
    workload. Validating a subpoena is simply a longer job.

    Rebuilt from the work-volume file so the expected figure is not this
    module's own output.
    """
    tiers = backlog.request_type_tiers(skills)
    source = work[(work.work_type == "W5") & (work.week_index == FIRST_PLAN_WEEK)].copy()
    source["role"] = source["request_type"].map(tiers)
    items = source.groupby("role")["work_items"].sum()

    week = hours[(hours.work_type == "W5") & (hours.week_index == FIRST_PLAN_WEEK)]
    got = week.groupby("role")[["items", "required_hours"]].sum()

    assert set(got.index) == {"tier_1", "tier_2"}
    for role, minutes in (("tier_1", 5), ("tier_2", 12)):
        assert got.loc[role, "items"] == pytest.approx(items[role], rel=1e-9)
        assert got.loc[role, "required_hours"] == pytest.approx(
            items[role] * minutes / 60, rel=1e-9
        )

    assert got.loc["tier_1", "items"] == pytest.approx(got.loc["tier_2", "items"], rel=1e-3)
    assert got.loc["tier_2", "required_hours"] / got.loc["tier_1", "required_hours"] == (
        pytest.approx(12 / 5, rel=1e-3)
    )


def test_w11_onsite_retrieval_is_priced_by_media(hours, work, assumptions):
    """Spec 9: onsite retrieval hours match non-digital items x time by media.

    25 / 35 / 45 minutes for paper / microfiche / archive room, read from
    the record-media table since 10/22. Paper is 3.5x the volume of either
    other medium but only 55% of the hours, because the old formats are
    what actually cost the time - which is the case for the digitization
    programme in spec 7.
    """
    all_times = backlog.retrieval_times(assumptions)
    onsite = next(w for w in assumptions["work_types"] if w["code"] == "W11")["media"]
    times = {medium: all_times[medium] for medium in onsite}
    assert times == {"paper": 25, "microfiche": 35, "archive_room": 45}

    source = work[(work.work_type == "W11") & (work.week_index == FIRST_PLAN_WEEK)]
    items = source.groupby("media")["work_items"].sum()

    week = hours[(hours.work_type == "W11") & (hours.week_index == FIRST_PLAN_WEEK)]
    got = week.groupby("media")[["items", "required_hours"]].sum()

    assert set(got.index) == {"paper", "microfiche", "archive_room"}
    assert set(week["role"]) == {"records_specialist"}
    for medium, minutes in times.items():
        assert got.loc[medium, "items"] == pytest.approx(items[medium], rel=1e-9)
        assert got.loc[medium, "required_hours"] == pytest.approx(
            items[medium] * minutes / 60, rel=1e-9
        )

    assert got.loc["paper", "items"] > 3 * got.loc["microfiche", "items"]
    assert got.loc["paper", "required_hours"] < 0.6 * got["required_hours"].sum()


def test_w14_quality_review_hand_check(hours, work):
    """W14 at 6 minutes on the charts it actually reviews (revised 10/22).

    Review is no longer one per chart: since 10/22 it is full coverage for
    R5, R7 and R8 plus redaction charts, and a 10% sample of the rest. The
    hand check therefore compares against the REVIEWED items from the
    work-volume file, and separately against the self-check (W24), which is
    the row that still covers every chart.

    Spec 9 also asks that QA work never lands on the tier that did the
    work, checked here by the role being qa_specialist and nothing else.
    """
    reviewed = work[(work.work_type == "W14") & (work.week_index == FIRST_PLAN_WEEK)][
        "work_items"
    ].sum()
    charts = work[(work.work_type == "W24") & (work.week_index == FIRST_PLAN_WEEK)][
        "work_items"
    ].sum()

    week = hours[(hours.work_type == "W14") & (hours.week_index == FIRST_PLAN_WEEK)]
    assert set(week["role"]) == {"qa_specialist"}
    assert week["handle_time_minutes"].unique().tolist() == [6.0]
    assert week["items"].sum() == pytest.approx(reviewed, rel=1e-9)
    assert week["required_hours"].sum() == pytest.approx(reviewed * 6 / 60, rel=1e-9)

    # Coverage sits where the rule puts it: well under every chart, well
    # over the 10% sample alone, because R5 bulk audits dominate the charts
    # and are reviewed in full.
    assert 0.5 < reviewed / charts < 0.8

    self_check = hours[
        (hours.work_type == "W24") & (hours.week_index == FIRST_PLAN_WEEK)
    ]
    assert self_check["items"].sum() == pytest.approx(charts, rel=1e-9)
    assert self_check["handle_time_minutes"].unique().tolist() == [1.0]
    assert set(self_check["role"]) == {"tier_1", "tier_2"}, "owner is by request type"


def test_rework_costs_half_of_the_review_it_reworks(hours, assumptions):
    """W22: half the original time (spec Section 1), derived not typed.

    Reading it off W14 means the two cannot drift apart if the review time
    is ever retuned.
    """
    by_code = {w["code"]: w for w in assumptions["work_types"]}
    link = by_code["W22"]["handle_time_fraction_of"]
    assert link == {"work_type": "W14", "fraction": 0.5}

    derived = backlog.handle_time_minutes(
        by_code["W22"], "qa_specialist", None, by_code, backlog.retrieval_times(assumptions)
    )
    assert derived == pytest.approx(3.0)

    week = hours[(hours.work_type == "W22") & (hours.week_index == FIRST_PLAN_WEEK)]
    assert week["handle_time_minutes"].unique().tolist() == [3.0]
    assert set(week["role"]) == {"qa_specialist"}


# ---------------------------------------------------------------------------
# FTE (spec 5.1)
# ---------------------------------------------------------------------------


def test_remote_roles_pool_across_the_network(hours, assumptions):
    """Spec 4.4: remote roles are one queue, not six site teams.

    Site is kept as a reporting field so the plan can show where demand
    arises, but the FTE division happens after the network rollup. Six
    sites each needing a third of a person is two people in one queue, not
    six people.
    """
    week = hours[hours.week_index == FIRST_PLAN_WEEK]
    assert week["site"].nunique() == 6

    summary = backlog.fte_by_role(week, assumptions)
    per_site = (
        week.groupby(["role", "site"])["required_hours"].sum().groupby("role").apply(
            lambda s: (s / 28.8).apply(np.ceil).sum()
        )
    )
    for row in summary.itertuples():
        if row.role == "records_specialist":
            continue  # genuinely site-based, not a pool
        assert row.fte < per_site[row.role], f"{row.role} should gain from pooling"


def test_fte_uses_the_spec_shrinkage_for_each_role(hours, assumptions):
    """Spec 5.1: back-office roles at 40 x (1 - 28%) = 28.8 productive hours. Corrected 10/26 from 26%, an early placeholder: the 5.2 lines add to 28%, and the 21.75% onsite figure is that same 28% less the 6.25% break line."""
    week = hours[hours.week_index == FIRST_PLAN_WEEK]
    summary = backlog.fte_by_role(week, assumptions).set_index("role")

    for role in ("tier_1", "tier_2", "qa_specialist", "records_specialist"):
        assert summary.loc[role, "productive_hours_per_fte"] == pytest.approx(28.8)
        assert summary.loc[role, "fte"] == pytest.approx(
            summary.loc[role, "required_hours"] / 28.8, rel=1e-12
        )

    # Onsite Associates sit lower, because 4.2 already paid their breaks.
    onsite = assumptions["fte"]["paid_shrinkage"]["onsite_associate"]
    assert 40 * (1 - onsite) == pytest.approx(31.30)


def test_starting_headcount_is_stated_per_scope(assumptions):
    """Tuned 10/26 (spec 5.1), and now stated per site or network pool.

    The placeholder totals it replaced - 12 / 12 / 22 / 12 / 6 - are kept
    in a comment in assumptions.yaml for the record.
    """
    headcount = assumptions["fte"]["starting_headcount"]
    assert set(headcount) == {
        "onsite_associate", "records_specialist", "tier_1", "tier_2", "qa_specialist"
    }
    for role, by_scope in headcount.items():
        assert isinstance(by_scope, dict), role
        assert all(isinstance(v, int) and v > 0 for v in by_scope.values()), role

    for role in ("onsite_associate", "records_specialist"):
        assert set(headcount[role]) == {"N1", "N2", "N3", "S1", "S2", "S3"}
    for role in ("tier_1", "tier_2", "qa_specialist"):
        assert set(headcount[role]) == {"NETWORK"}


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_the_table_covers_the_whole_horizon(hours, assumptions):
    cal = assumptions["calendar"]
    weeks = np.sort(hours["week_index"].unique())
    assert weeks[0] == 0
    assert weeks[-1] == cal["history_weeks"] + cal["plan_weeks"] - 1
    assert set(hours["period"]) == {"history", "plan"}
    assert set(hours["role"]) == {
        "tier_1", "tier_2", "qa_specialist", "records_specialist"
    }


# ---------------------------------------------------------------------------
# Retrieval times come from one table only (spec Section 1, revised 10/22)
# ---------------------------------------------------------------------------


def test_retrieval_times_come_only_from_the_record_media_table(assumptions):
    """Spec 9: W8, W9 and W11 take their times from record_media only.

    The two tables briefly both carried 12 minutes for digital retrieval.
    That is the kind of duplication that stays correct right up until
    somebody retunes one of them, so the work types now name a medium and
    carry no time of their own - and this refuses to let one creep back.
    """
    by_code = {w["code"]: w for w in assumptions["work_types"]}
    times = backlog.retrieval_times(assumptions)
    assert times == {
        "digital": 10, "email_archive": 20,
        "paper": 25, "microfiche": 35, "archive_room": 45,
    }

    for code in ("W8", "W9", "W11"):
        work_type = by_code[code]
        assert "media" in work_type, code
        assert "handle_time_minutes" not in work_type, (
            f"{code} must not carry its own retrieval time"
        )
        assert "handle_time_minutes_by_media" not in work_type, code

    with pytest.raises(ValueError, match="record_media only"):
        backlog.handle_time_minutes(
            {**by_code["W8"], "handle_time_minutes": 12},
            "tier_1", "digital", by_code, times,
        )


def test_digital_retrieval_is_ten_minutes_and_drives_the_hours(hours, assumptions):
    """Revised 10/22 from 12 minutes: a site converting to a workforce system.

    Spec Section 1 notes each minute is worth about 1.6 FTE across the
    network, which is why this value earns a row of its own rather than
    being a constant inside a work type.
    """
    week = hours[(hours.work_type == "W8") & (hours.week_index == FIRST_PLAN_WEEK)]
    assert week["handle_time_minutes"].unique().tolist() == [10.0]
    assert set(week["media"]) == {"digital"}

    productive = assumptions["fte"]["paid_hours_per_week"] * (
        1 - assumptions["fte"]["paid_shrinkage"]["tier_1"]
    )
    per_minute_fte = week["items"].sum() / 60 / productive
    assert per_minute_fte == pytest.approx(1.6, abs=0.15)


# ---------------------------------------------------------------------------
# The backlog (inventory) model, weekly (spec 4.3, 4.4)
# ---------------------------------------------------------------------------

# Siri's hand roll, docs/validation/backlog_hand_roll.xlsx, 10/22.
# One queue, 4 weeks: handle time 6 min, goal 1 business day, catch-up 4
# weeks, opening backlog 200. These are her Excel values, transcribed; the
# code was written to meet them, and
# test_hand_roll_constants_match_the_workbook proves they came from her file
# rather than from this module.
HAND_ROLL_INPUTS = {
    "handle_time_minutes": 6,
    "goal_business_days": 1,
    "catch_up_weeks": 4,
    "starting_backlog": 200,
}
HAND_ROLL_ARRIVALS = [1000, 1200, 900, 1100]
HAND_ROLL_AVAILABLE_HOURS = [100, 100, 110, 100]

HAND_ROLL_ENDING_BACKLOG = [200, 400, 200, 300]
HAND_ROLL_DAYS_OF_INVENTORY = [1.00, 1.67, 1.11, 1.36]
HAND_ROLL_REQUIRED_HOURS = [100, 120, 95.5, 110]
HAND_ROLL_GOAL_MET = [True, False, False, False]

HAND_ROLL_WORKBOOK = "docs/validation/backlog_hand_roll.xlsx"


@pytest.fixture(scope="module")
def hand_roll():
    return backlog.roll_forward(
        arrivals=HAND_ROLL_ARRIVALS,
        available_hours=HAND_ROLL_AVAILABLE_HOURS,
        **HAND_ROLL_INPUTS,
    )


def test_hand_roll_matches_siris_four_week_table(hand_roll):
    """Spec 9: a 4-week table built by hand, matched by the code week for week.

    The whole inventory model in one table. Week 2 is the interesting row:
    arrivals jump to 1,200 but only 1,000 items of capacity are available,
    so 200 items fall through and the backlog doubles. Week 3 then asks for
    95.5 hours rather than 90 - the extra 5.5 is a quarter of the 220-item
    excess, which is the catch-up period refusing to demand it all at once.
    """
    np.testing.assert_allclose(
        hand_roll["ending_backlog"], HAND_ROLL_ENDING_BACKLOG, rtol=1e-12
    )
    np.testing.assert_allclose(
        hand_roll["days_of_inventory"].round(2), HAND_ROLL_DAYS_OF_INVENTORY, rtol=1e-12
    )
    np.testing.assert_allclose(
        hand_roll["required_hours"], HAND_ROLL_REQUIRED_HOURS, rtol=1e-12
    )
    assert hand_roll["goal_met"].tolist() == HAND_ROLL_GOAL_MET
    assert hand_roll["goal_met"].sum() == 1, "the goal is met in week 1 only"


def test_the_backlog_rolls_forward_between_weeks(hand_roll):
    """Each week opens with what the last one left behind.

    This is what makes it an inventory model rather than four unconnected
    weeks, and it is the mechanism by which one bad week costs hours in the
    next three.
    """
    endings = hand_roll["ending_backlog"].tolist()
    openings = hand_roll["starting_backlog"].tolist()
    assert openings[0] == HAND_ROLL_INPUTS["starting_backlog"]
    assert openings[1:] == endings[:-1]


def test_hand_roll_constants_match_the_workbook():
    """The constants above must be Siri's numbers, not ones this code made.

    A reference case copied out of the thing it checks proves nothing, so
    this reads the workbook and confirms the hard-coded values came from it.
    """
    openpyxl = pytest.importorskip("openpyxl")
    path = config.resolve_path(HAND_ROLL_WORKBOOK)
    if not path.exists():
        pytest.skip(f"{HAND_ROLL_WORKBOOK} not present")

    sheet = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = [row for row in sheet.iter_rows(values_only=True) if any(row)]

    inputs = {row[0]: row[1] for row in rows if isinstance(row[0], str)}
    assert inputs["Handle time"] == HAND_ROLL_INPUTS["handle_time_minutes"]
    assert inputs["Goal"] == HAND_ROLL_INPUTS["goal_business_days"]
    assert inputs["Catch-up period"] == HAND_ROLL_INPUTS["catch_up_weeks"]
    assert inputs["Starting backlog"] == HAND_ROLL_INPUTS["starting_backlog"]

    weeks = [row for row in rows if isinstance(row[0], int)]
    assert len(weeks) == 4
    for index, row in enumerate(weeks):
        (_week, arrivals, available, _start, _target, required,
         _capacity, _completed, ending, days, status) = row[:11]
        assert arrivals == HAND_ROLL_ARRIVALS[index]
        assert available == HAND_ROLL_AVAILABLE_HOURS[index]
        assert required == pytest.approx(HAND_ROLL_REQUIRED_HOURS[index])
        assert ending == pytest.approx(HAND_ROLL_ENDING_BACKLOG[index])
        assert round(days, 2) == HAND_ROLL_DAYS_OF_INVENTORY[index]
        assert (status == "Met") == HAND_ROLL_GOAL_MET[index]


# ---------------------------------------------------------------------------
# The pieces of the model
# ---------------------------------------------------------------------------


def test_target_backlog_is_littles_law(assumptions):
    """Spec 4.3: target backlog = daily arrivals x the goal in days.

    A healthy queue is not empty. Zero backlog would mean the team had
    nothing to pick up the moment it finished an item, which costs more
    than it saves.
    """
    business_days = assumptions["calendar"]["business_days_per_week"]
    assert business_days == 5
    assert backlog.target_backlog(1000, 1, business_days) == pytest.approx(200)
    assert backlog.target_backlog(1000, 2, business_days) == pytest.approx(400)
    assert backlog.target_backlog(0, 5, business_days) == pytest.approx(0)


def test_only_the_excess_over_target_is_chased(assumptions):
    """Spec 4.3: a queue on target asks for nothing beyond its arrivals.

    And one below target asks for no less - the catch-up term is a max at
    zero, so a quiet spell does not let the plan under-staff the next week.
    """
    business_days = assumptions["calendar"]["business_days_per_week"]
    on_target = backlog.required_hours_for_week(1000, 200, 6, 1, 4, business_days)
    assert on_target == pytest.approx(1000 * 6 / 60)

    below = backlog.required_hours_for_week(1000, 50, 6, 1, 4, business_days)
    assert below == pytest.approx(on_target), "being ahead does not reduce the ask"

    above = backlog.required_hours_for_week(1000, 400, 6, 1, 4, business_days)
    assert above == pytest.approx((1000 + 200 / 4) * 6 / 60)
    assert above > on_target


def test_zero_arrivals_and_zero_backlog_need_zero_hours(assumptions):
    """Nothing to do means nothing to staff, with no division by zero.

    Days of inventory has no meaning without arrivals, so an empty queue
    reports zero rather than a nan that would poison every total above it.
    """
    business_days = assumptions["calendar"]["business_days_per_week"]
    hours = backlog.required_hours_for_week(0, 0, 6, 1, 4, business_days)
    assert hours == 0.0

    outcome = backlog.project_week(0, 0, 0, 6, 1, business_days)
    assert float(outcome["completed"]) == 0.0
    assert float(outcome["ending_backlog"]) == 0.0
    assert float(outcome["days_of_inventory"]) == 0.0
    assert bool(outcome["goal_met"])

    rolled = backlog.roll_forward([0, 0], [0, 0], 6, 1, 4, 0, business_days)
    assert (rolled["required_hours"] == 0).all()
    assert (rolled["ending_backlog"] == 0).all()
    assert rolled["goal_met"].all()


def test_a_queue_on_target_given_exactly_its_hours_stays_on_target(assumptions):
    """The steady state: required hours hold the line, they do not catch up.

    A queue opening at target and given exactly its required hours completes
    precisely its arrivals and ends the week back at target. That is the
    property the whole plan rests on - "required" means "enough to stay
    healthy", not "enough to reach zero".
    """
    business_days = assumptions["calendar"]["business_days_per_week"]
    arrivals, handle, goal = 1000, 6, 1
    target = backlog.target_backlog(arrivals, goal, business_days)
    required = backlog.required_hours_for_week(
        arrivals, target, handle, goal, 4, business_days
    )

    rolled = backlog.roll_forward(
        [arrivals] * 6, [required] * 6, handle, goal, 4, target, business_days
    )
    np.testing.assert_allclose(rolled["ending_backlog"], target, rtol=1e-12)
    np.testing.assert_allclose(rolled["days_of_inventory"], goal, rtol=1e-12)
    assert rolled["goal_met"].all()
    np.testing.assert_allclose(rolled["required_hours"], required, rtol=1e-12)


def test_the_backlog_can_never_go_negative_or_bank_phantom_output(assumptions):
    """Completed is capped by the work available, not just by capacity.

    Given far more hours than there is work, a queue clears to zero and
    stops - it cannot run negative and carry credit into next week.
    """
    business_days = assumptions["calendar"]["business_days_per_week"]
    rolled = backlog.roll_forward(
        [100, 100, 100], [1000, 1000, 1000], 6, 1, 4, 50, business_days
    )
    assert (rolled["ending_backlog"] >= 0).all()
    assert (rolled["completed"] <= rolled["starting_backlog"] + rolled["arrivals"] + 1e-12).all()
    assert (rolled["completed"] <= rolled["capacity_items"] + 1e-12).all()
    assert rolled["ending_backlog"].iloc[-1] == pytest.approx(0.0)


def test_no_hours_means_the_backlog_grows_by_the_arrivals(assumptions):
    """The other extreme, to pin the direction of the model."""
    business_days = assumptions["calendar"]["business_days_per_week"]
    rolled = backlog.roll_forward([500, 500], [0, 0], 6, 1, 4, 0, business_days)
    assert rolled["ending_backlog"].tolist() == [500, 1000]
    assert not rolled["goal_met"].any()


# ---------------------------------------------------------------------------
# Queues, scope and the baseline (spec 4.3, 4.4)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def queues(assumptions, sites, hours):
    return backlog.build_queues(assumptions, sites, hours)


def test_remote_roles_share_one_queue_and_records_specialists_do_not(
    queues, assumptions
):
    """Spec 4.4: remote roles are one network queue; onsite work is per site.

    A Tier 1 Specialist works the pipeline wherever the request came from,
    so six site queues would be a fiction that costs real people. A Records
    Specialist walks to a filing cabinet in their own building, so their
    queue cannot be pooled.
    """
    scopes = queues.groupby("role")["scope"].unique()
    network = assumptions["backlog"]["network_scope"]

    for role in ("tier_1", "tier_2", "qa_specialist"):
        assert list(scopes[role]) == [network], role
    assert set(scopes["records_specialist"]) == {"N1", "N2", "N3", "S1", "S2", "S3"}
    assert assumptions["backlog"]["per_site_roles"] == ["records_specialist"]


def test_every_queue_has_a_goal(queues, assumptions):
    """A queue with no goal has no target, so the model would have no plan."""
    assert queues["goal_business_days"].notna().all()
    assert (queues["goal_business_days"] > 0).all()
    goals = backlog.goal_days(assumptions)
    assert set(queues["work_type"]) <= set(goals)


def test_the_baseline_opens_every_queue_at_target(queues, assumptions):
    """Spec 4.3: the baseline starts at target, so it starts healthy.

    Beginning at zero would flatter the plan for a few weeks and then
    understate it; beginning anywhere else would be an invented number.
    Starting at target means the turnaround the model reports is the one
    the goals describe, not an artefact of where it happened to begin.
    """
    assert assumptions["backlog"]["start_at_target"] is True
    np.testing.assert_allclose(
        queues["starting_backlog"], queues["target_backlog"], rtol=1e-12
    )
    live = queues[queues["arrivals"] > 0]
    np.testing.assert_allclose(
        live["days_of_inventory"], live["goal_business_days"], rtol=1e-9
    )


def test_at_target_required_hours_are_just_the_workload(queues):
    """No catch-up term fires when nothing is above target.

    required_hours and the straight workload must agree exactly here, which
    ties this module back to the hours table it was built from.
    """
    np.testing.assert_allclose(
        queues["required_hours"], queues["workload_hours"], rtol=1e-9
    )



def test_the_turnaround_path_excludes_onsite_and_follow_up(assumptions):
    """Only the steps a request passes through count toward turnaround.

    Walk-ins and calls are contacts about a request, not steps in it;
    rework and amendment tracking happen after it has gone out; and
    digitization carries its own 10-day goal under spec 4.6.
    """
    in_path = {
        w["code"] for w in assumptions["work_types"] if w.get("in_turnaround")
    }
    for code in ("W19", "W20", "W21", "W22", "W23", "W12"):
        assert code not in in_path, code
    assert len(in_path) == 18

    by_code = {w["code"]: w for w in assumptions["work_types"]}
    assert by_code["W12"]["goal_business_days"] == 10, "spec 4.6"


# ---------------------------------------------------------------------------
# Turnaround: five stages added, steps within a stage averaged (spec 4.4)
# ---------------------------------------------------------------------------

# Stage goals from spec 4.4, approved 10/22. The five add to the 5-day
# overall goal before any exception step lifts them.
STAGE_GOALS = {
    "intake": 0.5,
    "validation": 1.0,
    "retrieval": 2.0,
    "review": 1.0,
    "release": 0.5,
}


@pytest.fixture(scope="module")
def plan_turnaround(queues, assumptions):
    plan_week = queues[queues.week_index == FIRST_PLAN_WEEK]
    return backlog.turnaround(plan_week, assumptions)


def test_the_approved_stage_goals_add_to_the_overall_goal(assumptions):
    """Spec 4.4: the five stage goals add to the 5-business-day goal."""
    assert sum(STAGE_GOALS.values()) == 5.0
    assert assumptions["backlog"]["overall_turnaround_goal_days"] == 5

    by_code = {w["code"]: w for w in assumptions["work_types"]}
    expected = {
        "W1": 0.5, "W2": 0.5, "W3": 0.5, "W4": 0.5,
        "W5": 1, "W6": 1, "W7": 2,
        "W8": 2, "W9": 2, "W10": 2, "W11": 3, "W13": 3,
        "W14": 1, "W15": 1, "W24": 1,
        "W16": 0.5, "W17": 1, "W18": 0.5,
        "W12": 10, "W21": 1, "W22": 1, "W23": 10,
    }
    for code, goal in expected.items():
        assert by_code[code]["goal_business_days"] == goal, code


def test_off_path_work_types_are_excluded_from_turnaround(
    plan_turnaround, queues, assumptions
):
    """Spec 4.4: digitization, rework, amendments and onsite email are off path.

    Each has a goal of its own - they are tracked, not ignored - but a
    request is released before any of them happens, so counting them would
    add delay the requester never waits through.
    """
    by_stage, by_step, _total = plan_turnaround
    off_path = {"W12", "W19", "W20", "W21", "W22", "W23"}
    assert not off_path & set(by_step["work_type"])

    by_code = {w["code"]: w for w in assumptions["work_types"]}
    for code in ("W12", "W21", "W22", "W23"):
        assert by_code[code]["in_turnaround"] is False, code
        assert by_code[code]["goal_business_days"] > 0, f"{code} is tracked, not ignored"

    # W12 is in the retrieval STAGE but must not reach the retrieval figure.
    assert by_code["W12"]["stage"] == "retrieval"
    assert "W12" in set(queues["work_type"]), "digitization still has a queue"
    assert set(by_stage["stage"]) == set(STAGE_GOALS)


def test_each_stage_is_a_volume_weighted_average_of_its_steps(plan_turnaround):
    """Spec 4.4: average within a stage, never add.

    Steps inside a stage are alternatives or run side by side - a chart is
    retrieved digitally OR from a filing cabinet, not both - so adding them
    would invent delay no single request experiences. Rebuilt here from the
    step detail so the weighting is checked, not assumed.
    """
    by_stage, by_step, total = plan_turnaround

    for row in by_stage.itertuples():
        steps = by_step[by_step.stage == row.stage]
        weighted = np.average(
            steps["days_of_inventory"], weights=steps["arrivals"]
        )
        assert row.days_of_inventory == pytest.approx(weighted, rel=1e-12)

        # An average, so it must sit inside the range of its steps - and
        # well below their sum wherever a stage has more than one.
        assert steps["days_of_inventory"].min() - 1e-12 <= row.days_of_inventory
        assert row.days_of_inventory <= steps["days_of_inventory"].max() + 1e-12
        if len(steps) > 1:
            assert row.days_of_inventory < steps["days_of_inventory"].sum()

    assert total == pytest.approx(by_stage["days_of_inventory"].sum(), rel=1e-12)


def test_a_rare_slow_step_barely_moves_its_stage(plan_turnaround):
    """The point of weighting by volume, measured on retrieval.

    Onsite retrieval takes 3 days against 2 for the digital path, but it
    touches only about a tenth of charts. Weighted, it lifts the stage to
    about 2.1; an unweighted mean of the five retrieval steps would say 2.4
    and the sum would say 12.
    """
    _by_stage, by_step, _total = plan_turnaround
    retrieval = by_step[by_step.stage == "retrieval"]

    slow = retrieval[retrieval.goal_business_days == 3]
    assert set(slow["work_type"]) == {"W11", "W13"}
    share = slow["arrivals"].sum() / retrieval["arrivals"].sum()
    assert 0.08 < share < 0.14, f"the slow steps are {share:.1%} of retrieval volume"

    weighted = np.average(
        retrieval["days_of_inventory"], weights=retrieval["arrivals"]
    )
    assert weighted == pytest.approx(2.1, abs=0.05), "spec 4.4 says about 2.1"
    assert weighted < retrieval["days_of_inventory"].mean(), "weighting must beat a flat mean"
    assert weighted < retrieval["days_of_inventory"].sum() / 2


def test_the_fully_staffed_baseline_lands_just_past_the_goal(
    plan_turnaround, assumptions
):
    """Spec 4.4: old-format records push turnaround just past the 5-day goal.

    The baseline opens every queue at target, so each step sits at its own
    goal and the total is decided entirely by the three exception steps.
    It decomposes exactly:

        retrieval  +0.098  W11 and W13 at 3 days on ~10% of charts
        validation +0.101  W7 at 2 days on ~10% of validation volume
        release    +0.033  W17 at 1 day on ~7% of release volume
        -------------------------------------------------------------
        total      +0.232  giving 5.23 days against a 5-day goal

    Spec 4.4 estimates "about 5.1", which counts the retrieval exception
    only. The validation exception is actually the larger of the two, so
    the real figure is 5.23. Flagged to Siri; the model follows the rule
    4.4 states rather than the figure it estimates.
    """
    by_stage, _by_step, total = plan_turnaround
    goal = assumptions["backlog"]["overall_turnaround_goal_days"]

    assert total > goal, "the exception steps must push it past the goal"
    assert total == pytest.approx(5.23, abs=0.05)

    lifts = {}
    for row in by_stage.itertuples():
        lifts[row.stage] = row.days_of_inventory - STAGE_GOALS[row.stage]

    assert lifts["intake"] == pytest.approx(0.0, abs=1e-9)
    assert lifts["review"] == pytest.approx(0.0, abs=1e-9)
    assert lifts["retrieval"] == pytest.approx(0.098, abs=0.02)
    assert lifts["validation"] == pytest.approx(0.101, abs=0.02)
    assert lifts["release"] == pytest.approx(0.033, abs=0.02)
    assert sum(lifts.values()) == pytest.approx(total - goal, abs=1e-9)


def test_stages_without_an_exception_sit_exactly_on_their_goal(plan_turnaround):
    """Intake and review have no exception step, so they land on the number.

    Which is the control: it shows the 0.23 overshoot comes from the three
    exceptions and not from the baseline construction drifting.
    """
    by_stage, by_step, _total = plan_turnaround
    for stage in ("intake", "review"):
        steps = by_step[by_step.stage == stage]
        assert steps["goal_business_days"].nunique() == 1, f"{stage} has no exception"
        row = by_stage[by_stage.stage == stage].iloc[0]
        assert row["days_of_inventory"] == pytest.approx(STAGE_GOALS[stage], abs=1e-9)
