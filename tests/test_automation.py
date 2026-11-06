"""Spec Section 9, Session 4: automation and human-in-the-loop (5.4, 5.5, 6.6).

Task 4f. The Section 9 checks this file owns:

    - Work conservation: for every eligible work type, human items +
      automated items = total items each week; automation never exceeds its
      capacity cap x uptime x ramp.
    - Human-in-the-loop hours match the hand formula, and are never also
      counted inside the handle-time uplift.
    - Every automated item on a human sign-off row (W7, R8 releases,
      notarization) gets a person's final decision; sign-off hours equal
      those items x sign-off time.
    - Every QA sample, quality review, and rework item lands on the QA
      team, never on the tier that did the work.
    - Next Dollar worked example reproduces $1.94 (human, easy item) and
      $0.87 (automated), revised 10/28 for easy-item valuation
      (automated, before fixed costs; $1.15 until 10/27, when QA hours
      moved to the QA team's rate).

Plus the four points Siri settled on 10/27: W17's notarization is the R8
path, the $500 platform fee is one shared fee split by automated items,
Next Dollar ranks by savings per dollar AFTER fixed costs, and any weekly
automation-against-by-hand figure carries the human-in-the-loop labour and
the platform share so both sides cover the same items and the same work.
A fifth, settled later the same day: QA review hours are priced at the QA
team's rate everywhere, including 6.6's per-item formula, so the per-item
and weekly figures reconcile.

The baseline keeps every rate at 0% (spec 7.1; turning them up is a
Session 5 scenario), so almost everything here runs against a test config
with the 7.1 defaults switched on. The last section does the opposite and
proves the baseline is untouched.

The idea worth holding on to while reading: AUTOMATED WORK COMES BACK. An
8% target share does not free 8% of the hours, because exceptions return to
the owning tier, the QA sample goes to the QA team, and a legally required
sign-off sends every single item to a person. A plan that assumed otherwise
would promise savings it cannot deliver, which is exactly what spec 5.5
exists to prevent.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from capacity_model import automation, backlog, config, plan, supply, work_volume

# Spec 7.1's supply-side defaults, which give the 8% target share named in
# 5.4. Hard-coded here with the spec reference, the way the Erlang hand
# check is: a test that asserts the spec's own numbers has to carry them.
SPEC_7_1_DEFAULTS = {
    "simple_task_share": 0.40,
    "ai_handling": 0.10,
    "automation_displacement": 0.10,
    "realization": 1.00,
}


@pytest.fixture(scope="module")
def baseline():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def switched_on(baseline):
    """A test config with spec 7.1's supply-side rates turned up."""
    assumptions = copy.deepcopy(baseline)
    assumptions["automation"]["rates"].update(SPEC_7_1_DEFAULTS)
    return assumptions


@pytest.fixture(scope="module")
def work(baseline, sites):
    return work_volume.build(baseline, sites)["work"]


@pytest.fixture(scope="module")
def built(switched_on, sites, work):
    return backlog.required_hours(switched_on, sites, work)


# ---------------------------------------------------------------------------
# The target share (spec 5.4, 7.1)
# ---------------------------------------------------------------------------


def test_the_target_share_is_eight_percent_at_the_spec_defaults(switched_on):
    """Spec 5.4's input table names 8%, and 7.1 says where it comes from.

        simple task share x (AI handling + automation displacement)
        x realization = 40% x 20% x 100% = 8%
    """
    assert automation.target_share(switched_on) == pytest.approx(0.08)
    assert automation.enabled(switched_on) is True


def test_rates_over_one_hundred_percent_are_refused(switched_on):
    """Spec 7.1: refused, not clipped.

    A scenario asking for 60% AI plus 60% automation has a typo in it.
    Silently capping at 100% would hide the typo and the plan would then be
    argued from a number nobody chose.
    """
    broken = copy.deepcopy(switched_on)
    broken["automation"]["rates"].update(ai_handling=0.60, automation_displacement=0.60)
    with pytest.raises(ValueError, match="over 100%"):
        automation.target_share(broken)


def test_applying_to_all_volume_drops_the_simple_share(switched_on):
    """Spec 7.1: each rate can apply to all volume instead of the simple share."""
    everywhere = copy.deepcopy(switched_on)
    everywhere["automation"]["rates"]["apply_to_all_volume"] = True
    assert automation.target_share(everywhere) == pytest.approx(0.20)


def test_the_physical_and_self_check_rows_stay_at_zero(switched_on):
    """Spec Section 1 pins W11 and W24 to 0% whatever the network rates say.

    W11 is onsite retrieval - somebody physically walks to the archive room
    - and W24 is the fulfiller's own one-minute self-check. Both are listed
    as eligible in principle and set to zero in practice, and an override
    of zero is a real value rather than a missing one.
    """
    assert automation.row_target_share("W11", switched_on) == 0.0
    assert automation.row_target_share("W24", switched_on) == 0.0
    assert automation.row_target_share("W1", switched_on) == pytest.approx(0.08)


def test_the_demand_side_work_types_are_not_eligible(switched_on):
    """W19, W20 and W21 are deflection, not automation (spec Section 1, 7.1).

    A self-service kiosk and call deflection mean the contact never
    arrives, so there is nothing for a supply-side machine to do. Counting
    them here as well would take the same saving twice.
    """
    eligible = set(automation.eligible_codes(switched_on))
    assert not {"W19", "W20", "W21"} & eligible
    assert len(eligible) == 21


# ---------------------------------------------------------------------------
# The ramp and the cap (spec 5.4)
# ---------------------------------------------------------------------------


def test_the_go_live_ramp_is_twenty_five_fifty_seventy_five_one_hundred(switched_on):
    """Spec 5.4, and nothing before go-live."""
    go_live = automation.go_live_week_index(switched_on)
    first_plan_week = switched_on["calendar"]["history_weeks"]
    assert go_live == first_plan_week + switched_on["automation"]["go_live_week"] - 1

    expected = [0.0, 0.0, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00]
    actual = [
        automation.ramp_fraction(week - go_live, switched_on)
        for week in range(go_live - 2, go_live + 6)
    ]
    assert actual == expected

    # Uptime multiplies the ramp, so nothing ever reaches 100% of the cap.
    assert automation.available_fraction(go_live + 5, switched_on) == pytest.approx(0.98)
    assert automation.available_fraction(go_live - 1, switched_on) == 0.0


def test_automation_never_exceeds_its_cap_times_uptime_times_ramp(built):
    """Spec Section 9, and spec 5.4's formula as a minimum of two things.

    The minimum is the whole mechanism: the machine takes the smaller of
    what it was aimed at and what it can physically get through, and the
    remainder stays with people. A single percentage would have quietly
    assumed infinite capacity and perfect uptime.
    """
    table = built["automation"]
    assert len(table) > 0
    assert (table["automated_items"] <= table["capacity_items"] + 1e-9).all()
    assert (
        table["automated_items"]
        <= table["eligible_items"] * table["target_share"] + 1e-9
    ).all()
    assert (table["automated_items"] <= table["eligible_items"] + 1e-9).all()
    assert (table["automated_items"] >= -1e-12).all()

    # The formula IS the minimum of the two, not something near it.
    aimed = table["eligible_items"] * table["target_share"]
    np.testing.assert_allclose(
        table["automated_items"],
        np.minimum(aimed, table["capacity_items"]),
        rtol=1e-9,
        atol=1e-12,
    )


def test_a_binding_cap_leaves_the_rest_with_people(switched_on, sites, work):
    """Spec 5.4: an outage shows up as human workload, never as lost work.

    Proved with a cap tight enough to bind, because the default cap is
    sized to the target share and so only uptime and the ramp hold it back.
    """
    capped = copy.deepcopy(switched_on)
    capped["automation"]["capacity_cap_items_per_week"] = 50

    table = automation.build(capped, sites, work)
    live = table[table["ramp"] > 0]
    assert len(live) > 0
    assert (live["automated_items"] <= 50 * 0.98 + 1e-9).all()
    assert (live["capacity_items"] <= 50 + 1e-9).all()

    # Binding somewhere, or the cap proved nothing.
    aimed = live["eligible_items"] * live["target_share"]
    assert (live["automated_items"] < aimed - 1e-6).any()

    # And every item it could not take is still with a person.
    np.testing.assert_allclose(
        live["human_items"],
        live["eligible_items"] - live["automated_items"],
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Work conservation (spec Section 9)
# ---------------------------------------------------------------------------


def test_human_items_plus_automated_items_equal_the_total_every_week(
    built, work, switched_on
):
    """Spec Section 9, for every eligible work type and every week.

    The one check that proves the split is a split and not a discount. If
    it ever failed, hours would be disappearing from the plan without
    anybody doing the work.
    """
    eligible = set(automation.eligible_codes(switched_on))
    total = (
        work[work["work_type"].isin(eligible)]
        .groupby(["week_index", "work_type"])["work_items"]
        .sum()
    )
    human = built["human"]
    human = (
        human[human["work_type"].isin(eligible)]
        .groupby(["week_index", "work_type"])["items"]
        .sum()
    )
    machine = built["automation"].set_index(["week_index", "work_type"])[
        "automated_items"
    ]

    joined = (
        total.to_frame("total")
        .join(human.to_frame("human"))
        .join(machine.to_frame("machine"))
        .fillna(0.0)
    )
    assert len(joined) > 4000, "expected every week and work type"
    np.testing.assert_allclose(
        joined["human"] + joined["machine"], joined["total"], rtol=1e-9, atol=1e-9
    )

    # Automation genuinely took some of it, or conservation is trivial.
    # Only from go-live onward: the 156 history weeks and the 12 plan weeks
    # before week 13 have no machine in them at all, which is itself the
    # ramp working.
    assert joined["machine"].sum() > 0
    go_live = automation.go_live_week_index(switched_on)
    weeks = joined.index.get_level_values("week_index")
    before = joined[weeks < go_live]
    after = joined[weeks >= go_live]
    assert before["machine"].abs().max() == 0.0
    # Every eligible row except the two pinned to 0% (W11, W24).
    assert (after["machine"] > 0).mean() == pytest.approx(19 / 21, abs=0.01)


def test_the_ineligible_work_types_are_untouched(built, work, switched_on):
    """Only eligible rows split. W19 to W21 and the pinned rows keep every item."""
    eligible = set(automation.eligible_codes(switched_on))
    zero_share = {
        code
        for code in eligible
        if automation.row_target_share(code, switched_on) == 0.0
    }
    untouched = (set(work["work_type"]) - eligible) | zero_share
    assert {"W11", "W24"} <= untouched

    human = built["human"]
    for code in sorted(untouched):
        total = work[work["work_type"] == code]["work_items"].sum()
        kept = human[human["work_type"] == code]["items"].sum()
        if kept == 0:
            continue  # routed elsewhere: W19, W20, W21
        assert kept == pytest.approx(total, rel=1e-9), code


# ---------------------------------------------------------------------------
# Human-in-the-loop (spec 5.5)
# ---------------------------------------------------------------------------


def test_human_in_the_loop_hours_match_the_hand_formula(switched_on):
    """Spec Section 9, against 5.5's formula worked by hand.

        hours = automated x (exception rate x exception time
                             + QA sample rate x review time
                             + sign-off rate x sign-off time) / 60

    Both multiples are against the EASY-item time since 10/28, which for a
    6-minute W1 average is 3 minutes.

    W1 intake, 6-minute average, 3-minute easy item, 1,000 automated:
        exceptions  10% x 1.5 x 3 min = 0.45 min an item ->  7.5 h
        QA sample    5% x 2 min       = 0.10 min an item ->  1.6667 h
        sign-off    none

    The QA term does NOT halve: two minutes is two minutes whether the
    item was simple or not. Only the two multiples move.
    """
    easy = automation.easy_item_minutes(6, switched_on)
    assert easy == pytest.approx(3.0)

    hours = automation.human_in_the_loop_hours("W1", 1000, 6, switched_on)
    assert hours["exception"] == pytest.approx(1000 * 0.10 * 1.5 * easy / 60)
    assert hours["exception"] == pytest.approx(7.5)
    assert hours["qa_sample"] == pytest.approx(1000 * 0.05 * 2 / 60)
    assert hours["qa_sample"] == pytest.approx(1.6666667)
    assert hours["sign_off"] == 0.0

    # W7, a sign-off row: 15-minute average, 7.5-minute easy item, 100% QA
    # and 100% sign-off at half the EASY time.
    easy_7 = automation.easy_item_minutes(15, switched_on)
    assert easy_7 == pytest.approx(7.5)
    hours = automation.human_in_the_loop_hours("W7", 1000, 15, switched_on)
    assert hours["exception"] == 0.0
    assert hours["qa_sample"] == pytest.approx(1000 * 1.00 * 2 / 60)
    assert hours["sign_off"] == pytest.approx(1000 * 1.0 * 0.5 * easy_7 / 60)
    returned = sum(hours.values())
    # 3.75 + 2 = 5.75 minutes back on an easy item worth 7.5 - so the
    # machine frees about 23% of the AVERAGE item's 15 minutes, where the
    # old arithmetic said 37%. Still not the 100% a target share implies.
    assert returned * 60 / 1000 == pytest.approx(5.75)
    assert returned / (1000 * 15 / 60) == pytest.approx(5.75 / 15, rel=1e-9)


def test_the_hours_in_the_table_match_the_formula_row_by_row(built, switched_on):
    """The table and the function cannot disagree."""
    loop = built["human_in_the_loop"]
    assert len(loop) > 0
    for row in loop.head(400).itertuples():
        per_item = row.required_hours / row.items * 60
        assert per_item == pytest.approx(row.handle_time_minutes, rel=1e-9)
        assert row.work_kind in automation.RETURN_KINDS


def test_the_uplift_is_not_also_charged_on_the_loop_work(switched_on, built):
    """Spec 5.5: the uplift and human-in-the-loop are counted once each.

    They are different things. The uplift (7.1) says the work LEFT for
    people is harder; human-in-the-loop says automated work comes BACK to
    people. Charging both for the same hour would double-count the cost of
    automating.

    So the human rows carry the mix uplift (the items a machine took were
    the easy ones) and the loop rows are priced off the BASE handle time at
    their own 5.5 multiples - an exception is 1.5x the normal job, not 1.5x
    an already-uplifted one. The extra uplift stays at its 7.1 default of
    zero here, so the only thing moving the human rows is the mix.
    """
    assert switched_on["automation"]["extra_uplift"] == 0.0

    human = built["human"]
    by_code = {w["code"]: w for w in switched_on["work_types"]}
    base = by_code["W1"]["handle_time_minutes"]
    simple = human[(human["work_type"] == "W1") & (human["role"] == "tier_1")]
    assert len(simple) > 0

    # Every human handle time is the base times that week's mix uplift, and
    # the ramp means the share removed - so the uplift - differs by week.
    easy = switched_on["automation"]["easy_handle_time_relative"]
    shares = dict(
        zip(
            zip(built["automation"]["week_index"], built["automation"]["work_type"]),
            built["automation"]["automated_fraction"],
        )
    )
    for row in simple.itertuples():
        share = shares.get((row.week_index, row.work_type), 0.0)
        expected = base * (1 + automation.mix_uplift(share, easy))
        assert row.handle_time_minutes == pytest.approx(expected, rel=1e-12)
    assert len(set(simple["handle_time_minutes"].round(9))) > 1, "the ramp must show"

    # The loop rows are off the BASE, with no mix uplift anywhere in them.
    # W1: 10% exceptions x 1.5 x 3 min (the EASY item) = 0.45 a head, and
    # 5% QA x 2 flat minutes = 0.10. Before 10/28 the exception line read
    # 0.90, off 1.5 x the 6-minute average.
    loop = built["human_in_the_loop"]
    w1_loop = loop[loop["work_type"] == "W1"]
    assert set(w1_loop["handle_time_minutes"].round(6)) == {0.45, 0.1}


def test_exceptions_go_to_the_tier_that_owns_the_work(built, switched_on):
    """Spec 5.5. An item the machine could not finish is the hard end of the
    distribution, so it goes back to whoever would have done it anyway - at
    one and a half times the handle time, because it is the hard end.
    """
    loop = built["human_in_the_loop"]
    human = built["human"]
    owners = set(zip(human["work_type"], human["media"], human["role"]))

    exceptions = loop[loop["work_kind"] == "exception"]
    assert len(exceptions) > 0
    for row in exceptions.itertuples():
        assert (row.work_type, row.media, row.role) in owners

    # Never the QA team, unless the QA team owns the work type (W14, W22).
    qa_owned = {
        code
        for code, _media, role in owners
        if role == automation.QA_ROLE
    }
    stray = exceptions[
        (exceptions["role"] == automation.QA_ROLE)
        & (~exceptions["work_type"].isin(qa_owned))
    ]
    assert len(stray) == 0


def test_every_qa_sample_lands_on_the_qa_team(built):
    """Spec Section 9: every QA item on the QA team, never on whoever did it.

    For automated items that means never back on the machine's own owning
    tier either - a team checking its own machine's work is not a check.
    """
    loop = built["human_in_the_loop"]
    samples = loop[loop["work_kind"] == "qa_sample"]
    assert len(samples) > 0
    assert set(samples["role"]) == {automation.QA_ROLE}
    assert samples["required_hours"].sum() > 0


def test_quality_review_and_rework_still_belong_to_the_qa_team(built, switched_on):
    """The same Section 9 check, for the human side of W14 and W22.

    Automation must not have quietly rerouted the work types the QA team
    owned in the first place.
    """
    human = built["human"]
    for code in ("W14", "W22"):
        rows = human[human["work_type"] == code]
        assert len(rows) > 0
        assert set(rows["role"]) == {automation.QA_ROLE}, code


# ---------------------------------------------------------------------------
# Human sign-off (spec Section 1, 5.5)
# ---------------------------------------------------------------------------


def test_every_sign_off_item_gets_a_person_s_decision(built, switched_on):
    """Spec Section 9, for W7, R8 releases and notarization.

    Where a legal requirement says a person must decide, automation may do
    the prep but every item goes to a person. So the sign-off item count
    equals the automated item count for those rows - 100%, never a sample.
    """
    loop = built["human_in_the_loop"]
    sign_off = loop[loop["work_kind"] == "sign_off"]
    assert len(sign_off) > 0
    assert set(sign_off["work_type"]) == {"W7", "W16", "W17"}

    loop_settings = switched_on["automation"]["human_in_the_loop"]
    assert loop_settings["sign_off_rate"] == 1.0

    # Sign-off hours = those items x sign-off time (half the handle time).
    by_code = {w["code"]: w for w in switched_on["work_types"]}
    for row in sign_off.itertuples():
        expected = row.items * row.handle_time_minutes / 60
        assert row.required_hours == pytest.approx(expected, rel=1e-9)
    assert sign_off["handle_time_minutes"].min() > 0

    # W7 signs off on everything; W16 and W17 only on the subpoena path.
    assert automation.needs_sign_off("W7", "R1", switched_on) is True
    assert automation.needs_sign_off("W7", "R8", switched_on) is True
    assert automation.needs_sign_off("W16", "R8", switched_on) is True
    assert automation.needs_sign_off("W16", "R1", switched_on) is False
    assert automation.needs_sign_off("W17", "R8", switched_on) is True
    assert automation.needs_sign_off("W17", "R7", switched_on) is False
    assert automation.needs_sign_off("W1", "R8", switched_on) is False


def test_sign_off_hours_go_to_the_owning_role(built):
    """Spec 5.5: "the owning role" decides, not the QA team.

    W7 is Tier 2's legal review, W16 is Tier 1's release, W17 is the
    Records Specialists' certified copies. A QA reviewer has no authority
    to release records under a court order.
    """
    loop = built["human_in_the_loop"]
    sign_off = loop[loop["work_kind"] == "sign_off"]
    expected = {"W7": "tier_2", "W16": "tier_1", "W17": "records_specialist"}
    for code, role in expected.items():
        rows = sign_off[sign_off["work_type"] == code]
        assert len(rows) > 0, code
        assert set(rows["role"]) == {role}, code


def test_a_sign_off_row_returns_most_of_its_own_handle_time(switched_on):
    """Which is the planning lesson, and the reason 5.5 exists.

    W7 is a 15-minute review, and the items a machine takes are the easy
    ones - 7.5 minutes. Automating one returns 2 minutes of QA plus 3.75 of
    sign-off, so 5.75 of the 7.5 comes straight back: the machine frees
    under a quarter of an easy item, not all of it.

    Revised 10/28. The old arithmetic read 9.5 minutes back out of 15,
    which was a bigger-looking number built on the wrong item - it priced
    the sign-off as though a person were re-reading an AVERAGE review.
    Either way the lesson holds and gets sharper: a business case built on
    the target share alone is out by a factor of four here.
    """
    minutes = automation.return_minutes("W7", 15, switched_on)
    easy = automation.easy_item_minutes(15, switched_on)
    assert sum(minutes.values()) == pytest.approx(5.75)
    # Most of the EASY item, which is the item the machine actually took.
    assert sum(minutes.values()) / easy > 0.75


# ---------------------------------------------------------------------------
# Next Dollar (spec 6.6)
# ---------------------------------------------------------------------------


def test_the_next_dollar_worked_example_reproduces_1_94_and_0_87(baseline):
    """Spec Section 9 and 6.6's worked example, intake, before fixed costs.

    Revised 10/28 for easy-item valuation. A machine does not take a random
    sample of the work, it takes the SIMPLE end - so both sides of this
    comparison are priced on a simple item, not on the average one.

    Human: $28 / (1 - 28%) = $38.89 a productive hour. The easy item is 50%
    of the 6-minute average, so 3 minutes, and costs $1.94.

    Automated (10/27: each term at the rate of whoever does the work):
        $0.50 an item
        + 10% x 4.5 min at Tier 1's $38.89      = $0.29
        + 5%  x 2   min at the QA team's $44.44 = $0.07
        = $0.87

    The 4.5 minutes is 1.5 x the EASY item's 3 minutes. Earlier versions
    used 1.5 x the 6-minute average and valued the by-hand side at the full
    average too: $3.89 against $1.16. That was wrong in BOTH directions at
    once - it charged a 9-minute exception on an item it simultaneously
    claimed was worth 6 minutes of saving, which cannot both be true of the
    same item.

    The rates here are the row's own (10% exceptions, 5% QA from Section
    1), so this holds whether or not the network rates are switched on -
    which is why it runs on the baseline config.
    """
    assert plan.productive_hour_cost("tier_1", baseline) == pytest.approx(38.888889)
    assert plan.productive_hour_cost("qa_specialist", baseline) == pytest.approx(44.444444)

    easy = automation.easy_item_minutes(6, baseline)
    assert easy == pytest.approx(3.0)
    assert plan.human_cost_per_item("tier_1", easy, baseline) == pytest.approx(
        1.94, abs=0.005
    )

    machine = plan.automated_cost_per_item("W1", "tier_1", 6, baseline)
    assert machine["item_cost"] == pytest.approx(0.50)
    assert machine["exception_cost"] == pytest.approx(0.29, abs=0.005)
    assert machine["qa_cost"] == pytest.approx(0.07, abs=0.005)
    assert machine["total"] == pytest.approx(0.87, abs=0.005)
    assert machine["sign_off_cost"] == 0.0

    # Exactly, so a later change to either rate shows up here first.
    assert machine["exception_cost"] == pytest.approx(
        0.10 * 1.5 * 3 / 60 * 38.888889
    )
    assert machine["qa_cost"] == pytest.approx(0.05 * 2 / 60 * 44.444444)
    assert machine["total"] == pytest.approx(0.865741, abs=1e-6)

    # The by-hand side is HALF what it was, and the automated side is lower
    # too - so the saving per item falls by more than half. That is the
    # whole consequence of the decision, in one line.
    assert plan.human_cost_per_item("tier_1", 6, baseline) == pytest.approx(
        3.89, abs=0.005
    ), "the general helper still prices an average item"
    before = 3.888889 - 1.157407
    after = 1.944444 - 0.865741
    assert after < before / 2


def test_the_lead_times_match_the_spec(baseline):
    """Spec 6.6: about 14 weeks to a productive Tier 1 hire.

    6 weeks of requisition + 5 of compliance and training + 3 to 90% ramp.
    Measured to the same 90% threshold 6.5 uses for the hire-by date, so
    the two outputs cannot quote different numbers. Automation's ramp is
    four weeks and has no requisition queue, which is the comparison 6.6 is
    really making.
    """
    assert plan.human_lead_weeks("tier_1", baseline) == 14
    assert plan.automation_lead_weeks(baseline) == 4

    for role in baseline["supply"]["roles"]:
        settings = supply.role_settings(role, baseline)
        assert plan.human_lead_weeks(role, baseline) == (
            settings["requisition_weeks"]
            + supply.onboarding_weeks(settings)
            + supply.weeks_to_full_speed(settings, 0.90)
        )


def test_the_next_dollar_table_ranks_by_savings_per_dollar_after_fixed_costs(
    switched_on, sites, built
):
    """Spec 6.6, decided 10/27: ranked on the AFTER-fixed figure.

    Before fixed costs is the per-item price on its own; after is what an
    item really costs. A row doing twelve items a week carries a crushing
    share of the $10,000 setup and cannot achieve the per-item price its
    rate implies, so ranking on the before-fixed figure would recommend it.
    """
    table = plan.next_dollar(switched_on, sites, required=built)
    assert len(table) > 0

    ranked = table["savings_per_dollar"].to_numpy()
    rankable = ranked[~np.isnan(ranked)]
    assert list(rankable) == sorted(rankable, reverse=True)
    # The unrankable rows sort last, never first.
    assert np.isnan(ranked[len(rankable):]).all()

    live = table[table["automated"]]
    np.testing.assert_allclose(
        live["automated_cost_per_item_after_fixed"],
        live["automated_cost_per_item"] + live["fixed_cost_per_item"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        live["savings_per_item_after_fixed"],
        live["human_cost_per_item"] - live["automated_cost_per_item_after_fixed"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        live["savings_per_dollar"],
        live["savings_per_item_after_fixed"]
        / live["automated_cost_per_item_after_fixed"],
        rtol=1e-9,
    )
    # Before fixed costs is still reported, and is still 6.6's three terms.
    np.testing.assert_allclose(
        table["automated_cost_per_item"],
        table["automated_item_cost"]
        + table["automated_exception_cost"]
        + table["automated_qa_cost"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        table["savings_per_item_before_fixed"],
        table["human_cost_per_item"] - table["automated_cost_per_item"],
        rtol=1e-9,
    )

    # Only eligible rows, and never a demand-side one.
    eligible = set(automation.eligible_codes(switched_on))
    assert set(table["work_type"]) <= eligible
    assert not {"W19", "W20", "W21"} & set(table["work_type"])

    # Lead time is on every row, because 6.6 says the comparison needs it.
    assert (table["human_lead_weeks"] > table["automation_lead_weeks"]).all()


def test_the_sign_off_cost_is_reported_but_not_folded_into_the_total(
    switched_on, sites, built
):
    """6.6's formula has no sign-off term, so neither does the total.

    For W7, an R8 release or a notarisation it is the largest of the three
    human terms, so leaving it invisible would be worse than leaving it out
    of a formula Siri did not write. It gets its own column.
    """
    table = plan.next_dollar(switched_on, sites, required=built)
    sign_off_rows = table[table["work_type"].isin(["W7", "W16", "W17"])]
    assert len(sign_off_rows) > 0
    assert (sign_off_rows["sign_off_cost_per_item"] > 0).all()

    other = table[~table["work_type"].isin(["W7", "W16", "W17"])]
    assert (other["sign_off_cost_per_item"] == 0).all()

    # The total is the three terms 6.6 names, with sign-off outside it.
    for row in sign_off_rows.itertuples():
        assert row.automated_cost_per_item == pytest.approx(
            row.automated_item_cost
            + row.automated_exception_cost
            + row.automated_qa_cost,
            rel=1e-9,
        )


# ---------------------------------------------------------------------------
# The baseline is untouched (the point of the whole task)
# ---------------------------------------------------------------------------


def test_automation_is_off_in_the_baseline(baseline):
    """Spec 7.1: every supply-side rate is 0% until a Session 5 scenario."""
    rates = baseline["automation"]["rates"]
    assert rates["simple_task_share"] == 0.0
    assert rates["ai_handling"] == 0.0
    assert rates["automation_displacement"] == 0.0
    assert automation.target_share(baseline) == 0.0
    assert automation.enabled(baseline) is False
    assert baseline["automation"]["extra_uplift"] == 0.0
    # No automation means no share removed, so no mix uplift either.
    assert automation.supply_handle_time_multiple("W1", 10**6, baseline) == 1.0


def test_the_baseline_required_hours_table_is_byte_for_byte_what_it_was(
    baseline, sites, work
):
    """No automation column, no extra rows, no changed numbers.

    The baseline has no automation, so it has no business carrying an
    automation column. This is the test that lets "a reorganization only"
    be said with a straight face.
    """
    result = backlog.required_hours(baseline, sites, work)
    table = result["hours"]
    assert list(table.columns) == [
        "week_index", "week_start", "period", "site", "role", "stage",
        "work_type", "media", "handle_time_minutes", "items", "required_hours",
    ]
    assert "work_kind" not in table.columns
    assert result["human_in_the_loop"].empty
    assert result["automation"].empty
    assert result["hours"] is result["human"]

    # Every item is still a person's, and nothing has been split off.
    handled_elsewhere = backlog.onsite_work_types(baseline)
    expected = work[~work["work_type"].isin(handled_elsewhere)]["work_items"].sum()
    assert table["items"].sum() == pytest.approx(expected, rel=1e-12)


def test_the_automation_table_and_file_are_empty_when_it_is_off(baseline, sites, work):
    """Nothing is written, so the baseline's output directory is unchanged."""
    table = automation.build(baseline, sites, work)
    assert table.empty
    assert automation.write_outputs(table, baseline) is None


def test_switching_automation_on_moves_the_numbers(built, baseline, sites, work):
    """The mirror of the test above: the machinery is not inert.

    If every assertion in this file could pass with the code doing nothing,
    none of them would be worth running.
    """
    off = backlog.required_hours(baseline, sites, work)
    on = built
    assert on["human"]["required_hours"].sum() < off["hours"]["required_hours"].sum()
    assert on["human_in_the_loop"]["required_hours"].sum() > 0
    assert on["automation"]["automated_items"].sum() > 0

    # And the net saving is real but much smaller than the hours the
    # machine took off people, because most of the difference comes back
    # as exceptions, QA and sign-off (spec 5.5).
    gross = off["hours"]["required_hours"].sum() - on["human"]["required_hours"].sum()
    returned = on["human_in_the_loop"]["required_hours"].sum()
    net = off["hours"]["required_hours"].sum() - on["hours"]["required_hours"].sum()
    assert net == pytest.approx(gross - returned, rel=1e-9)
    assert 0 < net < gross
    assert returned / gross > 0.1, "a meaningful share comes back"


# ---------------------------------------------------------------------------
# The shared platform fee (spec 5.4, decided 10/27)
# ---------------------------------------------------------------------------


def test_the_platform_fee_shares_sum_to_exactly_five_hundred_every_live_week(built):
    """Spec 5.4, decided 10/27: ONE fee for the whole network.

    There is one platform and one invoice. Charging $500 per work type
    would multiply a single bill by however many rows happened to be
    switched on - twenty-one of them here, so $10,500 a week for a $500
    platform.
    """
    table = built["automation"]
    assert len(table) > 0
    fee = 500.0

    live_weeks = sorted(table[table["automated_items"] > 0]["week_index"].unique())
    assert len(live_weeks) == 40, "40 plan weeks from go-live in week 13"

    by_week = table.groupby("week_index")["platform_cost"].sum()
    for week in live_weeks:
        assert by_week[week] == pytest.approx(fee, abs=1e-9), week

    # Split by each row's share of that week's automated items, not evenly:
    # the same $500 on a row doing 12 items and one doing 2,000 would make
    # the small row look uneconomic for a reason that is not about it.
    for week, group in table[table["week_index"].isin(live_weeks)].groupby("week_index"):
        items = group["automated_items"].sum()
        np.testing.assert_allclose(
            group["platform_cost"], fee * group["automated_items"] / items, rtol=1e-9
        )
    assert table[table["week_index"].isin(live_weeks)]["platform_cost"].nunique() > 1


def test_a_work_type_with_no_automated_items_carries_no_platform_share(built):
    """Spec 5.4, decided 10/27. You do not pay for a machine you did not run.

    Two cases, and both occur: a week before go-live, where nothing is
    live at all, and W11 and W24, which spec Section 1 pins to 0% in every
    week.
    """
    table = built["automation"]
    idle = table[table["automated_items"] <= 1e-12]
    assert len(idle) > 0
    assert idle["platform_cost"].abs().max() == 0.0

    # Before go-live, the whole network carries nothing.
    go_live = automation.go_live_week_index(config.load_assumptions())
    before = table[table["week_index"] < go_live]
    assert len(before) > 0
    assert before["platform_cost"].abs().max() == 0.0

    # And the pinned rows carry nothing even in a live week.
    pinned = table[table["work_type"].isin(["W11", "W24"])]
    assert len(pinned) > 0
    assert pinned["automated_items"].abs().max() == 0.0
    assert pinned["platform_cost"].abs().max() == 0.0


def test_the_fee_is_not_charged_per_work_type(built):
    """The mistake the 10/27 decision exists to prevent, stated as a number."""
    table = built["automation"]
    live = table[table["automated_items"] > 0]
    weeks = live["week_index"].nunique()
    assert live["platform_cost"].sum() == pytest.approx(500.0 * weeks, abs=1e-6)
    # Per work type it would have been twenty-one times that.
    assert live["work_type"].nunique() == 19  # 21 eligible less the two pinned
    assert live["platform_cost"].sum() < 500.0 * weeks * 2


# ---------------------------------------------------------------------------
# Fixed costs and the after-fixed ranking (spec 6.6, decided 10/27)
# ---------------------------------------------------------------------------


def test_the_after_fixed_ranking_reorders_a_hand_built_two_work_type_case():
    """Hand arithmetic, checkable on paper, where fixed costs flip the order.

    Two work types in one week, sharing one $500 platform fee:

        A: 2,000 automated items   B: 20 automated items
        setup   $10,000 / 52     = $192.3077 each
        platform $500 x 2000/2020 = $495.0495   x 20/2020 = $4.9505
        fixed per item            = $0.343679              = $9.862910

    With A at $4.00 by hand against $1.15 automated, and B at $10.00
    against $1.50:

        before fixed  A (4.00-1.15)/1.15 = 2.478   B (10-1.50)/1.50 = 5.667
        after  fixed  A (4.00-1.4937)/1.4937 = 1.678
                      B (10-11.3629)/11.3629 = -0.120

    So before fixed costs B looks more than twice as good as A, and after
    them B loses money. Automating twenty items a week cannot carry a
    $10,000 setup, and the ranking has to say so.
    """
    setup = 10000.0 / 52
    table = pd.DataFrame(
        [
            {"week_index": 200, "work_type": "A", "automated_items": 2000.0,
             "setup_cost": setup, "platform_cost": 500.0 * 2000 / 2020},
            {"week_index": 200, "work_type": "B", "automated_items": 20.0,
             "setup_cost": setup, "platform_cost": 500.0 * 20 / 2020},
        ]
    )
    fixed = plan.fixed_cost_per_item(table, week_index=200)
    assert fixed["A"] == pytest.approx(0.343679, abs=1e-6)
    assert fixed["B"] == pytest.approx(9.862910, abs=1e-6)

    before = {
        "A": plan._savings_per_dollar(4.00, 1.15),
        "B": plan._savings_per_dollar(10.00, 1.50),
    }
    after = {
        "A": plan._savings_per_dollar(4.00, 1.15 + fixed["A"]),
        "B": plan._savings_per_dollar(10.00, 1.50 + fixed["B"]),
    }
    assert before["A"] == pytest.approx(2.478, abs=0.001)
    assert before["B"] == pytest.approx(5.667, abs=0.001)
    assert after["A"] == pytest.approx(1.678, abs=0.001)
    assert after["B"] == pytest.approx(-0.120, abs=0.001)

    # The order genuinely flips, which is the whole point of the decision.
    assert sorted(before, key=before.get, reverse=True) == ["B", "A"]
    assert sorted(after, key=after.get, reverse=True) == ["A", "B"]
    assert after["B"] < 0, "automating B destroys money once setup is counted"


def test_the_fixed_cost_basis_is_the_plan_year_unless_a_week_is_named(
    switched_on, built
):
    """6.6 prices the ranking on the plan's actual automated volume.

    A single week would make the ranking swing with that week's ramp; the
    plan-year figure is the volume-weighted average of the weekly ones,
    which is what a leader is deciding about.
    """
    table = built["automation"]
    year = plan.fixed_cost_per_item(table, assumptions=switched_on)
    week = plan.fixed_cost_per_item(table, week_index=171, assumptions=switched_on)
    assert set(year) == set(week)
    assert year["W1"] != pytest.approx(week["W1"])

    # The plan-year figure IS the volume-weighted average of the weeks.
    plan_weeks, _mondays = supply.plan_window(switched_on)
    live = table[table["week_index"].isin(plan_weeks)]
    w1 = live[live["work_type"] == "W1"]
    spend = w1["setup_cost"].sum() + w1["platform_cost"].sum()
    assert year["W1"] == pytest.approx(spend / w1["automated_items"].sum())

    # A work type the machine never touched has no entry at all.
    assert "W11" not in year
    assert "W24" not in year


def test_a_row_with_no_automated_volume_is_reported_but_not_ranked(
    switched_on, sites, built
):
    """W11 and W24 are pinned to 0% by spec Section 1, so there is nothing to buy.

    Calling their fixed cost zero would have made them the best investment
    in the network - $0.50 an item against $31.25 by hand, a 61x return -
    on the strength of a machine the spec forbids running. Not measurable
    is not the same as free.
    """
    table = plan.next_dollar(switched_on, sites, required=built)
    pinned = table[table["work_type"].isin(["W11", "W24"])]
    assert len(pinned) > 0
    assert (~pinned["automated"]).all()
    assert pinned["fixed_cost_per_item"].isna().all()
    assert pinned["automated_cost_per_item_after_fixed"].isna().all()
    assert pinned["savings_per_dollar"].isna().all()

    # They are at the bottom, and something real is at the top.
    assert table["automated"].iloc[0]
    assert not table["automated"].iloc[-1]
    assert table["work_type"].iloc[0] == "W8"


# ---------------------------------------------------------------------------
# The weekly comparison (spec 6.6, decided 10/27)
# ---------------------------------------------------------------------------


def test_the_weekly_comparison_covers_the_same_items_both_sides(
    switched_on, sites, built
):
    """Decided 10/27, and the reason the figure is worth printing.

    Both sides price the same number of items. If the by-hand side were
    computed on the whole week's volume while the automation side priced
    only what the machine took, automation would win every comparison by
    arithmetic rather than by merit.
    """
    week = 171
    result = plan.weekly_cost_comparison("W1", week, switched_on, sites, built)
    assert result is not None
    assert result["attributed_items"] == pytest.approx(
        result["automated_items"], rel=1e-9
    )
    # W1 is single-role, so the by-hand side is one role at one handle
    # time - and the time it uses is the ORIGINAL 6.00 minutes, halved to
    # the easy item's 3.00 (10/28). Two corrections landed here together
    # and the test pins both:
    #
    #   the easy ratio, because the items the machine took are the simple
    #     ones and 6.6 values them at what THEY would have cost;
    #   the uplift divided back out, because the human rows read 6.26
    #     after 7.1's mix uplift and half of 6.26 is not an easy item -
    #     that let the uplift feed back into the valuation of the very
    #     items whose removal caused it.
    human = built["human"]
    mine = human[(human["work_type"] == "W1") & (human["week_index"] == week)]
    assert set(mine["role"]) == {"tier_1"}
    uplifted = mine["handle_time_minutes"].unique()
    assert len(uplifted) == 1
    assert uplifted[0] > 6, "the mix uplift must be in the row"

    # The share ACTUALLY removed that week, not the nominal 8%: uptime is
    # 98%, so the real share is a little under target and the uplift that
    # has to be divided out is the one that was applied.
    auto = built["automation"]
    mine_auto = auto[(auto["work_type"] == "W1") & (auto["week_index"] == week)]
    assert len(mine_auto) == 1
    removed = float(mine_auto["automated_fraction"].iloc[0])
    assert 0 < removed < automation.target_share(switched_on)

    easy = automation.easy_item_minutes(
        uplifted[0], switched_on, code="W1", week_index=week,
        removed_share=removed, uplifted=True,
    )
    assert easy == pytest.approx(3.0), "half of the original 6.00"
    assert result["by_hand_cost_per_item"] == pytest.approx(
        plan.human_cost_per_item("tier_1", easy, switched_on), rel=1e-9
    )
    assert result["by_hand_cost_per_item"] == pytest.approx(1.9444, abs=0.0005)

    # Strictly below both of the figures it replaced: the average item
    # ($3.89) and the naive half-of-the-uplifted-time ($2.03).
    assert result["by_hand_cost_per_item"] < plan.human_cost_per_item(
        "tier_1", uplifted[0], switched_on
    )
    naive = automation.easy_item_minutes(uplifted[0], switched_on)
    assert easy < naive
    assert result["by_hand_cost_per_item"] < plan.human_cost_per_item(
        "tier_1", naive, switched_on
    )


def test_the_weekly_comparison_includes_the_loop_labour_and_the_platform_share(
    switched_on, sites, built
):
    """Decided 10/27: the automation side carries the work that comes back.

    Per-item fee, setup share, platform share, and the exception and QA
    hours at a human's productive-hour cost. Leaving the loop labour off
    would be comparing a machine that never needs a second look against
    people who do; leaving the platform share off would hand a leader a
    number that gets worse the moment finance sees it.
    """
    week = 171
    result = plan.weekly_cost_comparison("W1", week, switched_on, sites, built)

    assert result["item_cost"] > 0
    assert result["setup_cost"] == pytest.approx(10000.0 / 52)
    assert result["platform_cost"] > 0
    assert result["exception_cost"] > 0
    assert result["qa_cost"] > 0

    assert result["automation_total"] == pytest.approx(
        result["item_cost"]
        + result["setup_cost"]
        + result["platform_cost"]
        + result["exception_cost"]
        + result["qa_cost"],
        rel=1e-9,
    )

    # The loop labour is priced at the role that is actually paid for it,
    # so the QA hours go at the QA team's rate (spec 5.5), not Tier 1's.
    qa_hour = plan.productive_hour_cost("qa_specialist", switched_on)
    assert result["qa_cost"] == pytest.approx(result["qa_hours"] * qa_hour, rel=1e-9)
    tier_1_hour = plan.productive_hour_cost("tier_1", switched_on)
    assert result["exception_cost"] == pytest.approx(
        result["exception_hours"] * tier_1_hour, rel=1e-9
    )

    # And W1 now LOSES, which is the consequence of the 10/28 easy-item
    # decision and the reason the targeted list had to be re-picked.
    #
    # Nothing about the machine changed. What changed is the honest value
    # of the items it takes: a 3-minute intake item is worth $1.94 by hand,
    # and the machine's fee, exception labour, QA sample and share of the
    # setup come to more than that. Automation was only ever winning this
    # row on a saving measured against an item it was not doing.
    assert result["saving"] < 0
    # Not by much, and not because of the per-item fee: the fixed costs
    # are the whole story on this row.
    assert result["item_cost"] < result["setup_cost"] + result["platform_cost"]
    # The loop labour is still a sixth of the machine's weekly cost - down
    # from a quarter, because the exception time halved with the easy-item
    # decision (10/28). Smaller, and still far too big to leave out.
    loop_share = (result["exception_cost"] + result["qa_cost"]) / result[
        "automation_total"
    ]
    assert loop_share == pytest.approx(0.158, abs=0.005)
    assert loop_share > 0.1


def test_the_weekly_comparison_reports_sign_off_outside_the_total(
    switched_on, sites, built
):
    """6.6's formula names three terms, so the total carries three.

    For W7 the sign-off is the largest human term of the lot, so it is
    reported on its own line rather than buried in or out of sight.
    """
    week = 171
    result = plan.weekly_cost_comparison("W7", week, switched_on, sites, built)
    assert result["sign_off_hours"] > 0
    assert result["sign_off_cost"] > 0
    assert result["automation_total"] == pytest.approx(
        result["item_cost"]
        + result["setup_cost"]
        + result["platform_cost"]
        + result["exception_cost"]
        + result["qa_cost"],
        rel=1e-9,
    )
    # W7 has no exceptions at all - automation preps, a person decides -
    # so its returned work is QA and sign-off.
    assert result["exception_cost"] == 0.0

    # And with the sign-off counted, automating a legal review barely pays.
    with_sign_off = result["automation_total"] + result["sign_off_cost"]
    assert with_sign_off > result["automation_total"]
    assert with_sign_off / result["by_hand_total"] > 0.5


def test_the_per_item_and_weekly_qa_figures_agree(switched_on, sites, built):
    """Decided 10/27: one rate for QA hours, everywhere.

    The weekly comparison prices each hour at the rate of whoever is paid
    for it, and since 10/27 so does 6.6's per-item formula. They therefore
    have to reconcile to the cent - and before the decision they did not,
    because the formula used the owning role's rate for the QA term and
    the weekly figure used the QA team's. On W1 that was $0.0648 against
    $0.0741, under a cent an item, which is exactly the sort of gap that
    goes unnoticed until somebody reconciles two reports.
    """
    week = 171
    weekly = plan.weekly_cost_comparison("W1", week, switched_on, sites, built)
    per_item = plan.automated_cost_per_item("W1", "tier_1", 6, switched_on)

    # Both QA figures come from the QA team's productive hour.
    assert weekly["qa_cost"] / weekly["automated_items"] == pytest.approx(
        per_item["qa_cost"], rel=1e-9
    )
    assert per_item["qa_cost"] == pytest.approx(
        0.05 * 2 / 60 * plan.productive_hour_cost("qa_specialist", switched_on)
    )

    # And so does the exception term, from Tier 1's - and BOTH sides now
    # measure it against the easy item (10/28). If one side were moved to
    # the easy time and the other left on the average, this is the
    # assertion that would catch it: the two differ by exactly 2x.
    assert weekly["exception_cost"] / weekly["automated_items"] == pytest.approx(
        per_item["exception_cost"], rel=1e-9
    )
    easy = automation.easy_item_minutes(6, switched_on)
    assert per_item["exception_cost"] == pytest.approx(
        0.10 * 1.5 * easy / 60 * plan.productive_hour_cost("tier_1", switched_on)
    )
    on_the_average = plan.automated_cost_per_item(
        "W1", "tier_1", 6 / automation.settings(switched_on)["easy_handle_time_relative"],
        switched_on,
    )
    assert on_the_average["exception_cost"] == pytest.approx(
        2 * per_item["exception_cost"], rel=1e-9
    )

    # Which means the whole variable side reconciles: the weekly total less
    # its fixed costs, per item, IS the before-fixed per-item figure.
    variable = (
        weekly["automation_total"] - weekly["setup_cost"] - weekly["platform_cost"]
    )
    assert variable / weekly["automated_items"] == pytest.approx(
        per_item["total"], rel=1e-9
    )


def test_the_built_loop_hours_match_the_hand_formula_with_easy_times(
    switched_on, sites, built
):
    """Spec 5.5 and Section 9, revised 10/28, against the built table.

    Not the function against itself: the TABLE the plan actually bills,
    against 5.5's formula worked with easy-item times.

    Checked three ways, because each catches a different slip:

      - every row's hours are its own minutes x its own items;
      - an exception and a sign-off on the SAME work type stand in the
        ratio 1.5 : 0.5, which holds only if both are measured against the
        same easy item - move one to the average and the ratio breaks;
      - W1 and W7 exactly, by hand.
    """
    loop = built["human_in_the_loop"]
    assert len(loop) > 0
    hil = automation.settings(switched_on)["human_in_the_loop"]

    for row in loop.head(800).itertuples():
        assert row.required_hours == pytest.approx(
            row.items * row.handle_time_minutes / 60, rel=1e-9
        )

    # The ratio test. W7 is 100% sign-off with no exceptions, so use W16
    # and W17, which carry both kinds on the same work type.
    both = 0
    for code, grp in loop.groupby("work_type"):
        kinds = set(grp["work_kind"])
        if not {"exception", "sign_off"} <= kinds:
            continue
        exc = grp[grp["work_kind"] == "exception"]["handle_time_minutes"].max()
        sign = grp[grp["work_kind"] == "sign_off"]["handle_time_minutes"].max()
        rate = automation.work_type_settings(code, switched_on)["exception_rate"]
        expected = (
            hil["exception_handle_time_multiple"] * rate
        ) / (hil["sign_off_handle_time_multiple"] * hil["sign_off_rate"])
        assert exc / sign == pytest.approx(expected, rel=1e-9), code
        both += 1
    assert both > 0, "no work type carries both kinds"

    # W1 exactly: 10% x 1.5 x 3 min (the easy half of a 6-minute average)
    # = 0.45 minutes an item, where the old arithmetic gave 0.90.
    w1 = loop[(loop["work_type"] == "W1") & (loop["work_kind"] == "exception")]
    assert len(w1) > 0
    assert set(w1["handle_time_minutes"].round(9)) == {0.45}
    assert automation.easy_item_minutes(6, switched_on) == pytest.approx(3.0)

    # W7 exactly: 100% sign-off x 0.5 x 7.5 min = 3.75 an item.
    w7 = loop[(loop["work_type"] == "W7") & (loop["work_kind"] == "sign_off")]
    assert len(w7) > 0
    assert set(w7["handle_time_minutes"].round(9)) == {3.75}


def test_there_is_no_weekly_comparison_before_go_live(switched_on, sites, built):
    """Nothing automated means nothing to compare, not a zero."""
    first_plan_week = switched_on["calendar"]["history_weeks"]
    assert plan.weekly_cost_comparison("W1", first_plan_week, switched_on, sites, built) is None
    assert plan.weekly_cost_comparison("W11", 171, switched_on, sites, built) is None


# ---------------------------------------------------------------------------
# W17 notarization (spec Section 1, decided 10/27)
# ---------------------------------------------------------------------------


def test_w17_notarization_is_the_r8_path(switched_on, built):
    """Decided 10/27, and now in spec Section 1's W17 row.

    A notarised certified copy is what a court order gets. The same work
    type's routine copies for R7 and R4 automate like anything else.
    """
    assert automation.needs_sign_off("W17", "R8", switched_on) is True
    assert automation.needs_sign_off("W17", "R7", switched_on) is False
    assert automation.needs_sign_off("W17", "R4", switched_on) is False

    row = switched_on["automation"]["work_types"]["W17"]
    assert row["sign_off"] == "request_types"
    assert row["sign_off_request_types"] == ["R8"]

    # And the hours land on the Records Specialists, who hold the seal.
    loop = built["human_in_the_loop"]
    w17 = loop[(loop["work_type"] == "W17") & (loop["work_kind"] == "sign_off")]
    assert len(w17) > 0
    assert set(w17["role"]) == {"records_specialist"}
