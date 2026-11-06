"""Spec Section 9, Session 5: the scenario engine (spec Section 7).

Task 5a. The Section 9 checks this file owns:

    - Diff test: each scenario changes only the inputs its overrides file
      names.
    - Direction checks: more volume or attrition never lowers required
      staff or cost.
    - A looser phone goal never needs more staff than a stricter one, and
      the walk-in goal still holds at every one of them.
    - A wider lunch window never raises the staff needed to cover lunch in
      the half-hours both windows share.
    - The combined away share never exceeds the 25% cap in any interval of
      any scenario, and the cap never changes a window's total relief.
    - Reshaping Onsite Associate days holds weekly paid hours exactly, puts
      part-time hours only on Monday and Tuesday at 75/25, and never
      touches required hours.

Plus the two rules Siri settled on 10/27 and the one the user asked for
explicitly: every scenario starts from the baseline roster, a scenario's
changes take effect from its Starts week with the weeks before it matching
the baseline, and an unknown override key is REFUSED.

That last one earns its place more than any other test here. A scenario
file is the one place in this model where a typo produces no error and no
effect: `atrition_multiplier: 2.0` would merge in happily, change nothing,
and hand back a run that looks exactly like the baseline. Nobody catches
that by reading the output, because the output is perfectly reasonable - it
is just the answer to a question nobody asked.
"""

import collections
import copy

import numpy as np
import pandas as pd
import pytest

from capacity_model import (
    automation,
    backlog,
    config,
    erlang,
    hiring,
    intervals,
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
def files(baseline):
    return scenarios.list_scenarios(baseline)


@pytest.fixture(scope="module")
def loaded(files):
    return [scenarios.load_scenario(path) for path in files]


@pytest.fixture(scope="module")
def result(files, baseline, sites):
    return scenarios.run(files, baseline, sites)


@pytest.fixture(scope="module")
def comparison(result):
    return result["comparison"].set_index("scenario")


def late(key, value, starts_week):
    """A scenario built in the test, to exercise a later Starts week.

    Both scenarios on disk start in plan week 1, so neither has any week
    before its own start. The mechanism still has to work, because spec 7
    has scenarios starting in weeks 13, 20 and 27.
    """
    return {
        "key": "late",
        "path": None,
        "name": f"Late {key}",
        "starts_week": starts_week,
        "what_it_changes": "built in the test",
        "overrides": {"scenario": {key: value}},
    }


# ---------------------------------------------------------------------------
# The roster of files (spec 7)
# ---------------------------------------------------------------------------


def test_the_roster_is_the_baseline_plus_the_scenarios_built_so_far(loaded):
    """Spec 7's table, as far as tasks 5a and 5b build it.

    The baseline is a file too, even though it overrides nothing, so the
    runner reads every scenario the same way and the comparison table's
    reference column is not a special case.
    """
    assert [s["key"] for s in loaded] == [
        "baseline",
        "after_proof_of_concept",
        "after_proof_of_concept_true_05",
        "ai_and_automation",
        "ai_and_automation_50",
        "ai_targeted",
        "attrition_doubles",
        "central_phones",
        "central_records_hub",
        "combined",
        "combined_ai_100",
        "coordinated_digitization",
        "monday_ideal_match",
        "monday_part_time",
        "phone_goal_80_30",
        "phone_goal_80_60",
        "site_shortfall_n1",
        "site_shortfall_n1_borrow_2h",
        "volume_plus_15",
        "wider_lunch_break_moved",
        "wider_lunch_window",
    ]
    assert loaded[0]["name"] == "Baseline"
    assert loaded[0]["overrides"] == {}

    by_name = {s["name"]: s for s in loaded}
    assert by_name["Volume +15%"]["what_it_changes"] == "Requests x 1.15, all sites"
    assert by_name["Attrition doubles"]["what_it_changes"] == (
        "Monthly attrition x 2, all roles"
    )
    # Spec 7's table: all start in week 1 except the site shortfall,
    # which starts in week 20.
    starts = {s["name"]: s["starts_week"] for s in loaded}
    assert starts.pop("Site shortfall N1") == 20
    assert starts.pop("Site shortfall N1, 2h borrow") == 20
    assert starts.pop("Centralized phones") == 13
    assert starts.pop("AI and automation") == 13
    assert starts.pop("AI and automation, 50%") == 13
    assert starts.pop("AI targeted") == 13
    assert starts.pop("Coordinated digitization") == 13
    assert starts.pop("Central records hub") == 13
    # Spec 7's table dates "After proof of concept" at week 27. Nothing in
    # the run depends on it - the automation goes live in week 13 with AI
    # targeted and the gate is an OUTPUT - so it is the table's label for
    # when the effect is expected, carried here so the file and the spec
    # cannot drift apart.
    assert starts.pop("After proof of concept") == 27
    assert starts.pop("After PoC, true rate 0.5%") == 27
    # Both Combined runs start in week 1, because that is when the VOLUME
    # starts; their machine goes live on the automation block's own week 13.
    assert set(starts.values()) == {1}


def test_a_scenario_file_is_data_and_nothing_else(loaded):
    """Only two top-level keys, and the overrides are plain values."""
    for scenario in loaded:
        assert set(scenario) == {
            "key", "path", "name", "starts_week", "what_it_changes",
            "overrides", "site_overrides",
        }
        assert isinstance(scenario["overrides"], dict)
        assert isinstance(scenario["site_overrides"], dict)


# ---------------------------------------------------------------------------
# The diff test (spec Section 9)
# ---------------------------------------------------------------------------


def test_each_scenario_changes_only_the_inputs_its_file_names(loaded, baseline, sites):
    """Spec Section 9's diff test, both directions.

    Only the inputs the file names change, and the ones it names DO change
    - the second half matters as much as the first, because a merge that
    reached the right keys and left them at their baseline values would
    pass the first half while doing nothing at all.
    """
    for scenario in loaded:
        merged = scenarios.apply_scenario(scenario, baseline)
        changed = set(scenarios.changed_keys(baseline, merged))
        declared = set(scenarios.declared_keys(scenario))

        # Nothing beyond what the file declares. A declared key whose value
        # already matches the baseline is allowed to be absent from
        # `changed` - both scenarios set starts_week to 1, which it already
        # is - so this is containment, not equality.
        assert changed <= declared, sorted(changed - declared)

        # And every mechanism value the file names really did land, at
        # whatever depth it sits. Walked as leaves, because an override
        # block like scenario.onsite_schedule.shape names one field of a
        # block the baseline fills out with four.
        def leaves(node, path=()):
            for key, value in node.items():
                where = path + (str(key),)
                if isinstance(value, dict):
                    yield from leaves(value, where)
                elif scenarios.is_coded_list(value):
                    for entry in value:
                        row = dict(entry)
                        yield from leaves(row, where + (str(row.pop("code")),))
                else:
                    yield where, value

        def resolve(node, path):
            for step in path:
                if scenarios.is_coded_list(node):
                    node = {entry["code"]: entry for entry in node}[step]
                else:
                    node = node[step]
            return node

        for where, value in leaves(scenario["overrides"]):
            assert resolve(merged, where) == value, ".".join(where)

        # And the same for sites.yaml, which a scenario may also reach.
        merged_sites = scenarios.apply_site_overrides(scenario, sites)
        site_changed = set(scenarios.changed_keys(sites, merged_sites))
        site_declared = set(scenarios.declared_site_keys(scenario))
        assert site_changed <= site_declared, sorted(site_changed - site_declared)
        for where, value in leaves(scenario.get("site_overrides") or {}):
            assert resolve(merged_sites, where) == value, ".".join(where)


def test_the_baseline_file_changes_nothing_at_all(baseline):
    """The strongest form of the diff test: an empty overrides block."""
    scenario = scenarios.load_scenario(
        scenarios.scenario_dir(baseline) / "baseline.yaml"
    )
    merged = scenarios.apply_scenario(scenario, baseline)
    assert scenarios.changed_keys(baseline, merged) == []
    assert merged == baseline


def test_merging_never_mutates_the_baseline(loaded, baseline):
    """A scenario run must not leave a mark on the next one.

    Deep-copied on the way in, so a list or a nested dict handed to one
    scenario cannot be shared with another. Without this, running three
    scenarios in a row would quietly accumulate their changes.
    """
    before = copy.deepcopy(baseline)
    for scenario in loaded:
        merged = scenarios.apply_scenario(scenario, baseline)
        merged["scenario"]["demand_multiplier"] = 99.0
        merged["supply"]["roles"]["tier_1"]["hourly_cost"] = 999
    assert baseline == before


# ---------------------------------------------------------------------------
# A typo must fail loudly (the point of the loader)
# ---------------------------------------------------------------------------


def test_an_unknown_override_key_is_refused(baseline):
    """The failure mode that would otherwise be invisible.

    `atrition_multiplier` is not an input. Merged silently, it would change
    nothing and the run would come back identical to the baseline - a
    perfectly plausible-looking answer to a question nobody asked.
    """
    with pytest.raises(KeyError, match="not a baseline input"):
        scenarios.merge_overrides(baseline, {"scenario": {"atrition_multiplier": 2.0}})

    with pytest.raises(KeyError, match="not a baseline input"):
        scenarios.merge_overrides(baseline, {"nonsense": 1})

    # Nested, several levels down, where a typo is easiest to make.
    with pytest.raises(KeyError, match="supply.roles.tier_1.monthly_atrition"):
        scenarios.merge_overrides(
            baseline, {"supply": {"roles": {"tier_1": {"monthly_atrition": 0.04}}}}
        )
    with pytest.raises(KeyError, match="supply.roles.teir_1"):
        scenarios.merge_overrides(
            baseline, {"supply": {"roles": {"teir_1": {"monthly_attrition": 0.04}}}}
        )


def test_a_block_where_the_baseline_has_a_value_is_refused(baseline):
    """Shape matters too, not just spelling."""
    with pytest.raises(TypeError, match="is a block, but the baseline input"):
        scenarios.merge_overrides(
            baseline, {"scenario": {"demand_multiplier": {"all_sites": 1.15}}}
        )


def test_a_real_key_is_accepted_at_any_depth(baseline):
    """The mirror: the loader is strict, not merely obstructive."""
    merged = scenarios.merge_overrides(
        baseline, {"supply": {"roles": {"tier_1": {"monthly_attrition": 0.04}}}}
    )
    assert merged["supply"]["roles"]["tier_1"]["monthly_attrition"] == 0.04
    assert merged["supply"]["roles"]["tier_2"] == baseline["supply"]["roles"]["tier_2"]
    # A list is replaced whole, because there is no sensible way to merge one.
    merged = scenarios.merge_overrides(
        baseline, {"automation": {"ramp": [0.5, 1.0]}}
    )
    assert merged["automation"]["ramp"] == [0.5, 1.0]


def test_a_malformed_scenario_file_is_refused(tmp_path):
    """A missing identity block, or a stray top-level key."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("scenario:\n  name: 'x'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs"):
        scenarios.load_scenario(bad)

    bad.write_text(
        "scenario:\n  name: 'x'\n  starts_week: 1\n  what_it_changes: 'y'\n"
        "overides: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected top-level keys"):
        scenarios.load_scenario(bad)


# ---------------------------------------------------------------------------
# The baseline roster is held (spec 7, decided 10/27)
# ---------------------------------------------------------------------------


def test_every_scenario_starts_from_the_baseline_roster(loaded, baseline):
    """Spec 7, decided 10/27: tuned once, on the baseline, and held.

    A scenario that re-tuned its own roster would open balanced and report
    no gap at all. It would be answering "what if we had already hired for
    this", which is not the question - the question is what the change
    costs the operation it has.
    """
    opening = baseline["fte"]["starting_headcount"]
    for scenario in loaded:
        merged = scenarios.apply_scenario(scenario, baseline)
        assert merged["fte"]["starting_headcount"] == opening, scenario["key"]
        assert "fte" not in scenario["overrides"]


def test_the_scenarios_open_on_the_same_headcount(result):
    """And it holds through the run, not just in the config.

    Amended 11/02 for the central records hub, which is the first
    scenario to change the SHAPE of the roster rather than its size. It
    re-bases every Records Specialist to the hub - the same seventeen
    people at one address instead of six - so its per-scope map cannot
    match the baseline's and should not.

    What still has to hold, and is what this test was really for, is that
    no scenario quietly re-tunes its way out of a gap. So the per-role
    TOTAL is pinned for every scenario including the hub, and the
    per-scope map is pinned for every scenario and every role that has
    not deliberately moved.
    """
    rosters = {}
    for key, built in result["runs"].items():
        ledger = built["prepared"]["ledger"]["ledger"]
        first = ledger["week_index"].min()
        opening = ledger[ledger["week_index"] == first]
        rosters[key] = (
            opening.set_index(["role", "scope"])["starting_headcount"].round(9).to_dict()
        )
    reference = rosters["baseline"]

    def totals(roster):
        out = {}
        for (role, _scope), heads in roster.items():
            out[role] = round(out.get(role, 0.0) + heads, 9)
        return out

    for key, roster in rosters.items():
        # Nobody is hired, and nobody is lost, in any scenario.
        assert totals(roster) == totals(reference), key
        if key == "central_records_hub":
            # The hub: Records Specialists all at the hub, everyone else
            # exactly where the baseline put them.
            moved = {k: v for k, v in roster.items() if k[0] == "records_specialist"}
            assert set(moved) == {("records_specialist", "N1")}
            assert moved[("records_specialist", "N1")] == pytest.approx(17.0)
            rest = {k: v for k, v in roster.items() if k[0] != "records_specialist"}
            base_rest = {
                k: v for k, v in reference.items() if k[0] != "records_specialist"
            }
            assert rest == base_rest
            continue
        assert roster == reference, key


# ---------------------------------------------------------------------------
# The Starts week (spec 7, decided 10/27)
# ---------------------------------------------------------------------------


def test_the_starts_week_converts_from_a_plan_week(baseline):
    """The YAML says "week 20" the way the spec table does."""
    first = baseline["calendar"]["history_weeks"]
    assert config.scenario_starts_week(baseline) == first
    merged = scenarios.apply_scenario(late("demand_multiplier", 1.15, 20), baseline)
    assert config.scenario_starts_week(merged) == first + 19


def test_a_demand_scenario_leaves_the_weeks_before_its_start_untouched(
    baseline, sites
):
    """Spec 7, decided 10/27. Bit-identical, not merely close.

    "Volume rises 15% in week 27" is a different question from "volume was
    always 15% higher", and the 156 weeks of history are never touched
    either way - the forecast is fitted to what actually happened, and
    inflating the history would move the trend and the seasonal index.
    """
    first = baseline["calendar"]["history_weeks"]
    switch = first + 26  # plan week 27
    merged = scenarios.apply_scenario(late("demand_multiplier", 1.15, 27), baseline)

    base = work_volume.build(baseline, sites)["work"]
    after = work_volume.build(merged, sites)["work"]
    assert len(base) == len(after)

    before = base["week_index"] < switch
    np.testing.assert_array_equal(
        base.loc[before, "work_items"].to_numpy(),
        after.loc[before, "work_items"].to_numpy(),
    )
    # The history is in there, untouched, and so are plan weeks 1 to 26.
    assert (base["week_index"] < first).sum() > 0
    assert ((base["week_index"] >= first) & before).sum() > 0

    # And from the starts week the requests really are 15% higher. Work
    # ITEMS come out a shade under, because the lag carries work from week
    # 26 into week 27 and that work was never multiplied - which is the
    # lag doing its job across the boundary, not a leak.
    intake = lambda frame: frame[
        (frame["work_type"] == "W1") & (frame["week_index"] >= switch)
    ]["work_items"].sum()
    assert intake(after) / intake(base) == pytest.approx(1.15, rel=1e-6)
    late_items = after[after["week_index"] >= switch]["work_items"].sum()
    base_items = base[base["week_index"] >= switch]["work_items"].sum()
    assert 1.14 < late_items / base_items < 1.15


def test_an_attrition_scenario_leaves_the_weeks_before_its_start_untouched(
    baseline, sites
):
    """The same rule on the supply side, and bit-identical for a reason.

    The rate is constant either side of the change, so the survival stays
    a closed form - one power for the weeks before and one for the weeks
    after. A product walked week by week would disagree with the baseline
    in the last bits of a float and this test would need a tolerance.
    """
    first = baseline["calendar"]["history_weeks"]
    switch = first + 26
    merged = scenarios.apply_scenario(late("attrition_multiplier", 2.0, 27), baseline)

    base = supply.build_ledger(baseline, sites)["ledger"]
    after = supply.build_ledger(merged, sites)["ledger"]
    joined = base.merge(after, on=["week_index", "role", "scope"], suffixes=("_b", "_s"))

    early = joined[joined["week_index"] < switch]
    assert len(early) > 0
    for column in ("ending_headcount", "starting_headcount", "productive_hours"):
        np.testing.assert_array_equal(
            early[column + "_b"].to_numpy(), early[column + "_s"].to_numpy()
        )

    # And afterwards the roster drains faster, which is the whole scenario.
    final = joined[joined["week_index"] == joined["week_index"].max()]
    assert final["ending_headcount_s"].sum() < final["ending_headcount_b"].sum()


def test_a_week_one_scenario_keeps_the_exact_closed_form(baseline):
    """Which is what keeps the two task 5a scenarios cheap and exact."""
    first = baseline["calendar"]["history_weeks"]
    weeks = range(first, first + baseline["calendar"]["plan_weeks"])
    merged = scenarios.apply_scenario(late("attrition_multiplier", 2.0, 1), baseline)
    uniform, by_week = supply.attrition_profile("tier_1", weeks, merged)
    assert by_week is None
    assert uniform == supply.weekly_attrition(0.04)

    # Mid-plan, it has to vary.
    merged = scenarios.apply_scenario(late("attrition_multiplier", 2.0, 27), baseline)
    uniform, by_week = supply.attrition_profile("tier_1", weeks, merged)
    assert uniform is None
    assert by_week[25] == supply.weekly_attrition(0.02)
    assert by_week[26] == supply.weekly_attrition(0.04)


def test_the_survival_fraction_splits_at_the_switch_by_hand():
    """Hand arithmetic on the piecewise closed form.

    A cohort opening in week 155 with the change landing in week 182: by
    week 200 it has served 26 weeks at the old rate (156 to 181) and 19 at
    the new one (182 to 200), which is 45 weeks in all - the whole age.
    """
    switch = {"week": 182, "before": 0.01, "after": 0.02}
    assert supply.survival_fraction(155, 155, switch) == 1.0
    assert supply.survival_fraction(155, 160, switch) == pytest.approx(0.99**5)
    assert supply.survival_fraction(155, 181, switch) == pytest.approx(0.99**26)
    assert supply.survival_fraction(155, 200, switch) == pytest.approx(
        0.99**26 * 0.98**19
    )
    # A cohort that starts after the change only ever sees the new rate.
    assert supply.survival_fraction(190, 200, switch) == pytest.approx(0.98**10)
    # And the week counts always add back to the age.
    for at_week in range(156, 208):
        before = max(0, min(at_week, 181) - 155)
        after = max(0, at_week - max(155, 181))
        assert before + after == at_week - 155


# ---------------------------------------------------------------------------
# Direction checks (spec Section 9)
# ---------------------------------------------------------------------------


def test_more_volume_never_lowers_required_staff_or_cost(comparison):
    """Spec Section 9's direction check, on the demand side.

    Obvious enough to feel unnecessary, and exactly the kind of thing that
    breaks quietly: a multiplier applied in the wrong place, or to the
    history instead of the plan, can easily come out LOWER somewhere.
    """
    base = comparison.loc["Baseline"]
    more = comparison.loc["Volume +15%"]

    assert more["plan_year_requests"] > base["plan_year_requests"]
    assert more["network_required_fte"] > base["network_required_fte"]
    assert more["onsite_required_fte"] > base["onsite_required_fte"]
    assert more["total_cost"] > base["total_cost"]
    assert more["overtime_hours"] > base["overtime_hours"]
    assert more["uncovered_hours"] > base["uncovered_hours"]
    assert more["peak_gap_fte"] > base["peak_gap_fte"]
    assert more["weeks_short"] > base["weeks_short"]

    # Requests rise by the multiplier, give or take the lag at the seam.
    ratio = more["plan_year_requests"] / base["plan_year_requests"]
    assert ratio == pytest.approx(1.15, rel=1e-3)

    # Service gets worse, not better. This is the one a leader needs: the
    # cost per request actually FALLS, because the held roster absorbs the
    # extra work with overtime rather than headcount - and the price is
    # paid in uncovered hours and turnaround, not in dollars.
    assert more["phone_service_level"] < base["phone_service_level"]
    assert more["mean_projected_turnaround_days"] > base[
        "mean_projected_turnaround_days"
    ]


def test_more_attrition_never_lowers_required_staff_or_cost(comparison):
    """The same check on the supply side.

    Attrition does not change the WORK, so required FTE barely moves - it
    shifts only through the Records Specialists, whose spare hours help the
    counter and so change what the counter needs. What moves is everything
    about covering it.
    """
    base = comparison.loc["Baseline"]
    worse = comparison.loc["Attrition doubles"]

    assert worse["plan_year_requests"] == pytest.approx(base["plan_year_requests"])
    assert worse["network_required_fte"] >= base["network_required_fte"]
    assert worse["total_cost"] > base["total_cost"]
    assert worse["overtime_hours"] > base["overtime_hours"]
    assert worse["uncovered_hours"] > base["uncovered_hours"]
    assert worse["peak_gap_fte"] > base["peak_gap_fte"]
    assert worse["hired_people"] > base["hired_people"]


def test_a_doubled_attrition_rate_really_is_doubled_monthly(baseline):
    """Not doubled weekly, which is a different and gentler scenario.

    2.5% a month is 0.5826% a week. Doubling the monthly rate to 5% gives
    1.1767% a week, not twice 0.5826% (1.1651%) - and a model that doubled
    the weekly figure would answer an easier question than the one asked.
    """
    merged = scenarios.apply_scenario(late("attrition_multiplier", 2.0, 1), baseline)
    for role in baseline["supply"]["roles"]:
        monthly = baseline["supply"]["roles"][role]["monthly_attrition"]
        assert supply.scenario_monthly_attrition(role, merged) == pytest.approx(
            monthly * 2
        )

    weekly_base = supply.weekly_attrition(0.025)
    weekly_double = supply.weekly_attrition(0.05)
    assert weekly_base == pytest.approx(0.00582554, abs=1e-8)
    assert weekly_double == pytest.approx(0.01176713, abs=1e-8)
    assert weekly_double > 2 * weekly_base
    assert weekly_double / weekly_base == pytest.approx(2.0199, abs=1e-4)


def test_a_harsher_scenario_sizes_its_hiring_classes_for_its_own_attrition(
    result, baseline
):
    """Spec 6.5's gross-up has to read the scenario's rate, not the baseline's.

    A class of four is not four people by the time it reaches full speed,
    and at double the attrition it is fewer still. Sizing against the
    baseline rate would hire short exactly where hiring short hurts most.
    """
    merged = result["runs"]["attrition_doubles"]["assumptions"]
    plain = hiring.class_size("tier_1", 3.0, baseline)
    harsh = hiring.class_size("tier_1", 3.0, merged)
    assert harsh[1] < plain[1], "fewer survive to full ramp"
    assert harsh[0] >= plain[0]


# ---------------------------------------------------------------------------
# The comparison table (spec 7, the Capacity Scenarios output)
# ---------------------------------------------------------------------------


def test_the_comparison_lines_up_every_scenario_with_the_baseline_first(result):
    """Spec 7: "lines them up side by side"."""
    table = result["comparison"]
    assert len(table) == 21
    assert table["scenario"].iloc[0] == "Baseline"
    assert set(table["scenario"]) == {
        "Baseline", "Attrition doubles", "Volume +15%",
        "Phone goal 80/30", "Phone goal 80/60", "Wider lunch window",
        "Wider lunch, break moved", "Monday ideal match", "Monday part-time",
        "Site shortfall N1", "Site shortfall N1, 2h borrow",
        "Centralized phones", "Coordinated digitization",
        "Central records hub",
        "AI and automation", "AI and automation, 50%", "AI targeted",
        "After proof of concept", "After PoC, true rate 0.5%",
        "Combined", "Combined, AI 100%",
    }

    expected = {
        "scenario", "starts_week", "what_it_changes", "plan_year_requests",
        "peak_gap_fte", "weeks_short", "overtime_hours",
        "uncovered_hours", "total_cost", "labour_cost", "machine_cost",
        "cost_per_request", "hiring_classes",
        "hired_people", "phone_service_level", "walk_in_service_level",
        "monday_phone_service_level", "friday_phone_service_level",
        "onsite_required_fte", "back_office_required_fte",
        "network_required_fte",
        "opening_week_turnaround_days", "mean_projected_turnaround_days",
        "worst_week_turnaround_days", "weeks_over_turnaround_goal",
    }
    # The planner's view of turnaround (4.4, decided 10/28). Only the runs
    # with a machine have one, so these three are the only columns in the
    # table allowed to be empty - and the scenarios that leave them empty
    # are exactly the ones with no automation.
    planner = {
        "mean_human_queue_turnaround_days",
        "worst_week_human_queue_turnaround_days",
        "weeks_over_goal_human_queue",
    }
    assert set(table.columns) == expected | planner

    numeric = expected - {"scenario", "what_it_changes"}
    assert table[sorted(numeric)].notna().all().all()

    with_machine = {
        "AI and automation", "AI and automation, 50%", "AI targeted",
        # Every run that switches a machine on gets the planner's view,
        # whichever other lever it also pulls (spec 4.4, 10/28).
        "After proof of concept", "After PoC, true rate 0.5%",
        "Combined", "Combined, AI 100%",
    }
    by_scenario = table.set_index("scenario")
    assert by_scenario.loc[sorted(with_machine), sorted(planner)].notna().all().all()
    rest = sorted(set(table["scenario"]) - with_machine)
    assert by_scenario.loc[rest, sorted(planner)].isna().all().all()


def test_the_comparison_arithmetic_ties_to_the_runs(result, baseline):
    """Every figure traceable to the run it came from."""
    table = result["comparison"].set_index("scenario")
    for key, built in result["runs"].items():
        row = table.loc[built["scenario"]["name"]]
        assert row["overtime_hours"] == pytest.approx(
            built["role_cost"]["overtime_hours"].sum()
        )
        # Three cost columns since 11/03 (spec 6.4), and each ties to a
        # different place: labour to the plan's cost table, machine to
        # the automation plan's own invoice, all-in to their sum. Pinned
        # separately because the whole point of splitting them was that
        # one number could not honestly be both.
        assert row["labour_cost"] == pytest.approx(
            built["role_cost"]["total_cost"].sum()
        )
        assert row["machine_cost"] == pytest.approx(
            scenarios.machine_spend(built)["total"]
        )
        assert row["total_cost"] == pytest.approx(
            row["labour_cost"] + row["machine_cost"]
        )
        # No machine, no invoice - which is why the baseline pins and
        # every people-only scenario did not move when this changed.
        if not automation.enabled(built["assumptions"]):
            assert row["machine_cost"] == 0.0
            assert row["total_cost"] == pytest.approx(row["labour_cost"])
        else:
            assert row["machine_cost"] > 0.0
        assert row["uncovered_hours"] == pytest.approx(
            built["cost"]["uncovered_hours"].sum()
            + built["onsite"]["uncovered_hours"].sum()
        )
        assert row["hiring_classes"] == len(built["hiring"])
        assert row["mean_projected_turnaround_days"] == pytest.approx(
            built["turnaround"]["turnaround_days"].mean()
        )
        assert row["worst_week_turnaround_days"] == pytest.approx(
            built["turnaround"]["turnaround_days"].max()
        )
        assert row["cost_per_request"] == pytest.approx(
            row["total_cost"] / row["plan_year_requests"]
        )
        # Peak gap is read BEFORE overtime (spec 6.5, 10/26): the number
        # that says how far short the roster was, not how hard it was worked.
        assert row["peak_gap_fte"] == pytest.approx(
            built["gap_fte"]["short_fte"].max()
        )


def test_the_baseline_run_reproduces_the_regression_pins(result):
    """The scenario engine must not have moved the baseline (Session 4's pins).

    Same four numbers test_model_regression.py holds, reached through the
    scenario runner instead of plan.main - so the merge, the roster rule and
    the two neutral multipliers are all proved inert on the baseline.
    """
    built = result["runs"]["baseline"]
    assert len(built["hiring"]) == 3
    assert built["role_cost"]["overtime_hours"].sum() == pytest.approx(2767.6, abs=0.05)
    uncovered = (
        built["cost"]["uncovered_hours"].sum()
        + built["onsite"]["uncovered_hours"].sum()
    )
    assert uncovered == pytest.approx(420.0, abs=0.05)
    assert built["role_cost"]["total_cost"].sum() == pytest.approx(7_456_636, abs=1.0)


def test_the_output_is_labelled_synthetic(baseline):
    """CLAUDE.md: generated data is labelled synthetic in its file name."""
    assert "SYNTHETIC" in baseline["paths"]["scenario_comparison_csv"]


# ---------------------------------------------------------------------------
# The two turnaround figures measure different things (asked 10/27)
# ---------------------------------------------------------------------------


def test_the_opening_week_turnaround_is_the_goals_and_not_an_outcome(result):
    """Why the table carries two turnaround columns, not one.

    Plan week 1 opens every queue AT its target backlog (spec 4.3,
    start_at_target), and days of inventory at target IS the goal by
    definition. So week 1's total is the weighted sum of the five stage
    goals and tells you nothing about the staffing - it is the 5.23 days
    Session 3 recorded, and it is a yardstick.
    """
    built = result["runs"]["baseline"]
    queues = built["queues"]
    first = queues[queues["week_index"] == queues["week_index"].min()]

    assert np.allclose(first["opening_backlog"], first["target_backlog"])
    assert np.allclose(first["days_of_inventory"], first["goal_business_days"])

    table = result["comparison"].set_index("scenario")
    assert table.loc["Baseline", "opening_week_turnaround_days"] == pytest.approx(
        5.232, abs=0.005
    )

    # Week 2 onward carries what week 1 actually left, so it is no longer
    # pinned to the goals.
    second = queues[queues["week_index"] == queues["week_index"].min() + 1]
    assert not np.allclose(second["opening_backlog"], second["target_backlog"])


def test_the_projected_mean_moves_with_the_arrival_stream(result):
    """And why the mean comes in BELOW the goal without the plan excelling.

    Days of inventory divides a backlog carried from last week by THIS
    week's arrival rate, so a week whose volume jumps reads lower on the
    same pile of work and a quiet week reads higher. Over the plan year
    that correlation is strongly negative, which is most of the gap
    between 5.23 and 4.67 - not staffing.
    """
    built = result["runs"]["baseline"]
    arrivals = built["queues"].groupby("week_index")["arrivals"].sum()
    turnaround = built["turnaround"].set_index("week_index")["turnaround_days"]
    joined = pd.DataFrame({"arrivals": arrivals, "days": turnaround}).dropna()
    change = joined["arrivals"].pct_change().dropna()
    days = joined["days"].loc[change.index]

    assert change.corr(days) < -0.4
    assert days[change > 0].mean() < days[change <= 0].mean()

    table = result["comparison"].set_index("scenario")
    row = table.loc["Baseline"]
    assert row["mean_projected_turnaround_days"] < row["opening_week_turnaround_days"]
    assert row["mean_projected_turnaround_days"] == pytest.approx(4.671, abs=0.005)

    # The numbers a leader should read are the worst week and the count
    # over the goal, and both are in the table.
    assert row["worst_week_turnaround_days"] > row["opening_week_turnaround_days"]
    assert row["weeks_over_turnaround_goal"] == 6


# ---------------------------------------------------------------------------
# Phone service goal (spec 7)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phone_report(baseline, sites):
    return scenarios.phone_goal_report(baseline, sites)


def test_a_looser_phone_goal_never_needs_more_staff(phone_report):
    """Spec Section 9's direction check, on the service goal.

    Erlang C's service level rises with staff and falls with a shorter
    target, so a longer target can never need more people. If it ever did,
    the pooled solve would be finding a worse answer for an easier problem.
    """
    assert list(phone_report["target_seconds"]) == [20, 30, 60]
    fte = phone_report["onsite_required_fte"].to_numpy()
    hours = phone_report["onsite_required_hours"].to_numpy()
    peak = phone_report["mean_daily_peak_staff"].to_numpy()

    assert list(fte) == sorted(fte, reverse=True)
    assert list(hours) == sorted(hours, reverse=True)
    assert list(peak) == sorted(peak, reverse=True)
    assert fte[0] > fte[-1], "a 60-second target must be cheaper than 20"


def test_a_looser_phone_goal_never_needs_more_staff_in_any_interval(baseline, sites):
    """The same check interval by interval, which is where it is decided.

    The aggregate could hide a single interval going the wrong way, and a
    single interval is what a requester experiences.
    """
    work = work_volume.build(baseline, sites)["work"]
    tables = {}
    for seconds in (20, 60):
        variant = scenarios.phone_goal_variant(seconds, baseline)
        tables[seconds] = scenarios.onsite_shape(variant, sites, work)["intervals"]

    strict, loose = tables[20], tables[60]
    assert len(strict) == len(loose)
    assert (loose["required_staff"].to_numpy() <= strict["required_staff"].to_numpy()).all()
    assert (loose["scheduled_staff"].to_numpy() <= strict["scheduled_staff"].to_numpy() + 1e-9).all()
    # And it is a real difference somewhere, not a tie everywhere.
    assert (loose["required_staff"].to_numpy() < strict["required_staff"].to_numpy()).any()


def test_the_walk_in_goal_still_holds_at_every_phone_goal(baseline, sites):
    """Spec 4.1: the blended team is solved at the phone target, then topped
    up until the counter's 90% within 10 minutes holds as well.

    Which matters most at the LOOSEST phone goal: as the phone target
    relaxes, the phone solve asks for fewer people and the walk-in goal
    becomes the binding constraint. If the top-up were ever skipped, this
    is where it would show.
    """
    work = work_volume.build(baseline, sites)["work"]
    goals = {g["code"]: g for g in baseline["live_queues"]["goals"]}
    walk_ins = goals["walk_ins"]
    assert walk_ins["service_level"] == 0.90
    assert walk_ins["target_seconds"] == 600

    for seconds in scenarios.PHONE_GOAL_SECONDS:
        variant = scenarios.phone_goal_variant(seconds, baseline)
        table = scenarios.onsite_shape(variant, sites, work)["intervals"]
        busy = table[table["traffic_pooled"] > 0]
        assert len(busy) > 0
        achieved = np.array(
            [
                erlang.service_level(
                    int(staff), traffic, walk_ins["target_seconds"], handle
                )
                for staff, traffic, handle in zip(
                    busy["required_staff"],
                    busy["traffic_pooled"],
                    busy["pooled_handle_seconds"],
                )
            ]
        )
        assert (achieved >= walk_ins["service_level"] - 1e-9).all(), seconds


def test_the_phone_goal_barely_moves_occupancy_and_the_cap_never_binds(phone_report, baseline):
    """Session 3's finding, which is why the saving is so small.

    Blended occupancy sits near 32% at every goal and nowhere near the 85%
    cap. The counter is staffed to be AVAILABLE, not to be busy, so
    letting the phone ring longer does not change how many people have to
    be there to answer it.
    """
    cap = baseline["live_queues"]["occupancy_cap"]
    assert cap == 0.85
    occupancy = phone_report["blended_occupancy"].to_numpy()
    assert (occupancy < 0.40).all()
    assert (occupancy > 0.30).all()
    # Looser goal, slightly busier people - the right direction, tiny size.
    assert list(occupancy) == sorted(occupancy)
    assert occupancy[-1] - occupancy[0] < 0.02


def test_what_ten_seconds_of_hold_time_actually_buys(phone_report):
    """The answer a leader needs, and it is "about a quarter of an FTE".

    Across the whole six-site network, for ten more seconds of permitted
    wait. Tripling the target from 20 to 60 seconds buys about one FTE in
    total - which is the occupancy finding stated as money, and the reason
    this scenario is worth running before anybody argues about hold time.
    """
    marginal = phone_report["fte_saved_per_10_seconds"].to_numpy()
    assert np.isnan(marginal[0]), "the baseline has no step before it"
    assert 0.2 < marginal[1] < 0.35
    assert 0.2 < marginal[2] < 0.35

    total = phone_report["fte_saved_vs_baseline"].to_numpy()
    assert total[0] == 0.0
    assert 0.2 < total[1] < 0.4
    assert 0.9 < total[2] < 1.3
    # Nearly linear over this range, so the second ten seconds is worth
    # about the same as the first - not a diminishing return worth hunting.
    assert marginal[2] / marginal[1] > 0.8


# ---------------------------------------------------------------------------
# Wider lunch window (spec 7)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lunch_report(baseline, sites):
    return scenarios.lunch_window_report(baseline, sites)


def test_a_wider_lunch_window_never_raises_the_staff_needed_to_cover_lunch(
    lunch_report,
):
    """Compared like for like, on the half-hours both windows share.

    "The lunch window" is six half-hours in one run and seven in the
    other, so comparing that would compare different things and call the
    difference an effect. On the six they share - 11:00 to 13:30 - the
    wider window sends fewer people away at once and needs fewer on the
    floor, which is the claim the scenario is making.
    """
    narrow, wide = lunch_report.iloc[0], lunch_report.iloc[1]
    assert list(lunch_report["run"]) == [
        "Baseline", "Wider lunch", "Wider lunch, break moved"
    ]
    assert narrow["lunch_window"] == "11:00 to 14:00"
    assert wide["lunch_window"] == "11:00 to 14:30"
    assert narrow["window_half_hours"] == 6
    assert wide["window_half_hours"] == 7

    assert wide["shared_mean_away_share"] < narrow["shared_mean_away_share"]
    assert wide["shared_mean_scheduled_staff"] < narrow["shared_mean_scheduled_staff"]
    assert wide["shared_peak_scheduled_staff"] <= narrow["shared_peak_scheduled_staff"]

    # The same relief, spread thinner: 30 minutes over 7 half-hours rather
    # than 6, so 14.3% away instead of 16.7%.
    assert narrow["nominal_away_share"] == pytest.approx(1 / 6)
    assert wide["nominal_away_share"] == pytest.approx(1 / 7)
    assert narrow["shared_mean_away_share"] == pytest.approx(1 / 6, abs=0.001)
    assert wide["shared_mean_away_share"] == pytest.approx(1 / 7, abs=0.002)


def test_the_wider_window_moves_the_daily_peak_out_of_the_midday_hole(lunch_report):
    """Spec 7's actual claim: "thinning the midday coverage hole".

    Session 3 found the lunch window driving the day's staffing peak on
    about half of all site-days. Under the wider window it drives it on
    one day in eight. That is the scenario working.
    """
    narrow, wide = lunch_report.iloc[0], lunch_report.iloc[1]
    assert narrow["day_peak_in_lunch_window_share"] == pytest.approx(0.50, abs=0.02)
    assert wide["day_peak_in_lunch_window_share"] < 0.20


def test_widening_lunch_alone_is_self_cancelling(lunch_report, baseline):
    """The finding, kept as a run (spec 7, decided 10/27).

    14:00 belongs to the AFTERNOON BREAK window as well, so under the
    wider lunch window it carries both reliefs at once and runs right up
    against the cap. That one overlapping half-hour costs about what the
    six thinned ones save, so network onsite FTE moves the wrong way - by
    a hair, but the wrong way.
    """
    narrow, wide = lunch_report.iloc[0], lunch_report.iloc[1]
    cap = baseline["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"]
    assert cap == 0.25

    # The overlap, and what it costs.
    assert wide["overlap_away_share"] > narrow["overlap_away_share"] * 2
    assert wide["overlap_scheduled_staff"] > narrow["overlap_scheduled_staff"]
    # Right up against the combined cap, and not past it (spec 3.4, 10/27).
    assert wide["overlap_away_share"] == pytest.approx(cap, abs=1e-9)
    assert wide["max_away_share_any_interval"] <= cap + 1e-9

    # So the network figures move the wrong way, by a little.
    assert wide["onsite_required_fte"] >= narrow["onsite_required_fte"]
    assert wide["break_cover_hours"] >= narrow["break_cover_hours"]
    assert wide["onsite_required_fte"] / narrow["onsite_required_fte"] < 1.001


def test_moving_the_break_turns_the_idea_into_a_saving(lunch_report, baseline):
    """Spec 7, added 10/27: the run that answers whether the idea pays.

    With the afternoon break at 14:30 the two windows stop overlapping, so
    lunch genuinely spreads over seven half-hours and nothing stacks. Every
    figure then moves the right way - and all of them only a little, which
    is the honest headline: it is free, so it is worth doing, and it is not
    a solution to anything.
    """
    narrow, wide, moved = (lunch_report.iloc[i] for i in range(3))
    assert moved["afternoon_break_window"] == "14:30 to 16:00"

    # Better than the baseline, and better than widening lunch alone.
    assert moved["onsite_required_fte"] < narrow["onsite_required_fte"]
    assert moved["onsite_required_fte"] < wide["onsite_required_fte"]
    assert moved["break_cover_hours"] < narrow["break_cover_hours"]
    assert moved["mean_daily_peak_staff"] < narrow["mean_daily_peak_staff"]
    assert moved["lunch_window_peak_staff"] < narrow["lunch_window_peak_staff"]

    # No stacking left: 14:00 carries lunch only.
    assert moved["overlap_away_share"] < wide["overlap_away_share"]
    assert moved["max_away_share_any_interval"] < wide["max_away_share_any_interval"]

    # And small. A fortieth of an FTE across six sites, for a year.
    saving = narrow["onsite_required_fte"] - moved["onsite_required_fte"]
    assert 0 < saving < 0.1
    assert 0 < narrow["break_cover_hours"] - moved["break_cover_hours"] < 200


def test_the_lunch_window_scenario_leaves_the_other_reliefs_alone(baseline):
    """The merge changed one field of one relief and nothing else.

    Restating the whole reliefs list to move one window is how a scenario
    file acquires a copy-paste error that nothing can detect, which is why
    coded lists merge by code.
    """
    merged = scenarios.wider_lunch_variant("14:30", baseline)
    assert scenarios.changed_keys(baseline, merged) == [
        "breaks_and_lunch.reliefs.lunch.window.to"
    ]
    by_code = {r["code"]: r for r in merged["breaks_and_lunch"]["reliefs"]}
    assert by_code["lunch"]["window"] == {"from": "11:00", "to": "14:30"}
    assert by_code["lunch"]["minutes"] == 30
    assert by_code["lunch"]["paid"] is False
    assert by_code["morning_break"]["window"] == {"from": "09:00", "to": "11:00"}
    assert by_code["afternoon_break"]["window"] == {"from": "14:00", "to": "16:00"}


# ---------------------------------------------------------------------------
# Coded lists merge by code (the engine change task 5b needed)
# ---------------------------------------------------------------------------


def test_a_coded_table_is_merged_row_by_row(baseline):
    """The live-queue goals, the work types, the shrinkage lines.

    Order means nothing in these and the code is the key, so a scenario
    names one row of one of them. A plain list has no key to merge on and
    is replaced whole.
    """
    merged = scenarios.merge_overrides(
        baseline,
        {"live_queues": {"goals": [{"code": "phones", "target_seconds": 45}]}},
    )
    goals = {g["code"]: g for g in merged["live_queues"]["goals"]}
    assert goals["phones"]["target_seconds"] == 45
    assert goals["phones"]["service_level"] == 0.80, "untouched"
    assert goals["walk_ins"] == {
        g["code"]: g for g in baseline["live_queues"]["goals"]
    }["walk_ins"]

    # A plain list is still replaced whole.
    assert scenarios.is_coded_list(baseline["automation"]["ramp"]) is False
    merged = scenarios.merge_overrides(baseline, {"automation": {"ramp": [1.0]}})
    assert merged["automation"]["ramp"] == [1.0]


def test_an_unknown_code_in_a_coded_table_is_refused(baseline):
    """The same strictness one level deeper, where a typo is just as quiet."""
    with pytest.raises(KeyError, match="live_queues.goals.phoness"):
        scenarios.merge_overrides(
            baseline,
            {"live_queues": {"goals": [{"code": "phoness", "target_seconds": 45}]}},
        )
    with pytest.raises(KeyError, match="work_types.W99"):
        scenarios.merge_overrides(
            baseline, {"work_types": [{"code": "W99", "handle_time_minutes": 5}]}
        )
    # And an unknown FIELD inside a row that does exist.
    with pytest.raises(KeyError, match="live_queues.goals.phones.target_secs"):
        scenarios.merge_overrides(
            baseline, {"live_queues": {"goals": [{"code": "phones", "target_secs": 45}]}}
        )


# ---------------------------------------------------------------------------
# The combined away cap (spec 3.4, decided 10/27)
# ---------------------------------------------------------------------------


def test_the_combined_away_share_never_exceeds_the_cap_in_any_scenario(loaded, baseline):
    """Spec 3.4, decided 10/27: the cap limits EVERYONE away, across reliefs.

    A cap counted per relief would allow 25% at lunch plus 25% on a break
    in the same half-hour - half the team gone, and exactly the rota the
    cap exists to forbid. It only shows up when two windows overlap, which
    no baseline window does and the wider-lunch scenario does.
    """
    cap = baseline["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"]
    assert cap == 0.25

    for scenario in loaded:
        merged = scenarios.apply_scenario(scenario, baseline)
        total = intervals.away_share(merged)
        assert total.max() <= cap + 1e-9, scenario["key"]
        assert (total >= 0).all()

    # And the one that would have breached it does reach the cap exactly,
    # so the test above is not passing by never being tested.
    wide = scenarios.wider_lunch_variant("14:30", baseline)
    assert intervals.away_share(wide).max() == pytest.approx(cap, abs=1e-9)


def test_the_cap_never_changes_a_window_s_total_relief(loaded, baseline):
    """The same people take the same breaks, whatever the cap does.

    A relief of M minutes inside a window of W minutes puts M/W of the team
    away across that window. The cap moves relief BETWEEN half-hours; it
    may never remove any, because the break still has to be taken.
    """
    for scenario in loaded:
        merged = scenarios.apply_scenario(scenario, baseline)
        shares = intervals.relief_shares(merged)
        windows = intervals.relief_windows(merged)
        step = merged["intraday"]["site_hours"]["interval_minutes"]
        for relief in merged["breaks_and_lunch"]["reliefs"]:
            code = relief["code"]
            inside = windows[code]
            nominal = relief["minutes"] / (step * int(inside.sum()))
            assert shares[code][inside].mean() == pytest.approx(
                nominal, rel=1e-12
            ), f"{scenario['key']}/{code}"
            # Nothing placed outside its own window.
            assert shares[code][~inside].max() == 0.0


def test_a_per_interval_headroom_is_water_filled_by_hand():
    """The mechanism, on numbers checkable on paper.

    Three intervals with headroom 0.25, 0.05 and 0.25 - the middle one
    already half-spent by an earlier relief - and 0.40 of relief to place
    with equal weights. Even sharing would want 0.1333 each, which the
    middle one cannot take, so it pegs at 0.05 and the other two split the
    remaining 0.35.
    """
    shares = intervals._allocate_relief(
        np.ones(3), 0.40, np.array([0.25, 0.05, 0.25])
    )
    assert shares.sum() == pytest.approx(0.40)
    assert shares[1] == pytest.approx(0.05)
    assert shares[0] == pytest.approx(0.175)
    assert shares[2] == pytest.approx(0.175)

    # A scalar cap still works, and still binds the same way.
    shares = intervals._allocate_relief(np.ones(3), 0.40, 0.15)
    assert shares.sum() == pytest.approx(0.40)
    assert shares.max() <= 0.15 + 1e-12

    # More relief than headroom is refused, not silently clipped: a break
    # that cannot be placed is a rota that cannot be worked.
    with pytest.raises(ValueError, match="of headroom between them"):
        intervals._allocate_relief(np.ones(3), 0.80, np.array([0.25, 0.05, 0.25]))


def test_reliefs_are_placed_in_the_order_the_day_runs(baseline):
    """Where two windows overlap, the earlier one has first call.

    A modelling choice, and the order a rota is actually built - the
    morning break is settled before anyone argues about lunch. Under the
    wider lunch window, 14:00 belongs to both lunch and the afternoon
    break, and lunch gets it.
    """
    wide = scenarios.wider_lunch_variant("14:30", baseline)
    moved = scenarios.wider_lunch_variant("14:30", baseline, "14:30")
    labels = intervals.interval_labels(wide)
    at_two = labels.index("14:00")
    cap = wide["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"]
    overlapping = intervals.relief_shares(wide)
    apart = intervals.relief_shares(moved)

    # Lunch is placed first, so its share at 14:00 is whatever inverse load
    # asked for - identical whether or not the break overlaps it. That is
    # what "first call" means.
    assert overlapping["lunch"][at_two] == pytest.approx(apart["lunch"][at_two])
    np.testing.assert_allclose(overlapping["lunch"], apart["lunch"], rtol=1e-12)

    # The break takes what is left, and the combined share lands exactly on
    # the cap rather than through it.
    headroom = cap - overlapping["lunch"][at_two]
    assert overlapping["afternoon_break"][at_two] == pytest.approx(headroom)
    assert intervals.away_share(wide)[at_two] == pytest.approx(cap, abs=1e-9)
    assert apart["afternoon_break"][at_two] == pytest.approx(0.0, abs=1e-12)

    # And the break still takes its full relief, in the half-hours it has.
    windows = intervals.relief_windows(wide)
    inside = windows["afternoon_break"]
    assert overlapping["afternoon_break"][inside].sum() == pytest.approx(
        0.125 * int(inside.sum())
    )


def test_the_baseline_windows_do_not_overlap_so_the_cap_change_is_inert(baseline):
    """Which is why the baseline is byte-identical under the new rule.

    The three windows touch at their boundaries and share no half-hour, so
    a shared headroom is the same thing as three separate ones - and the
    baseline never reaches the cap at all.
    """
    windows = intervals.relief_windows(baseline)
    codes = sorted(windows)
    for i, left in enumerate(codes):
        for right in codes[i + 1:]:
            overlap = windows[left] & windows[right]
            assert not overlap.any(), f"{left} and {right} overlap"

    cap = baseline["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"]
    assert intervals.away_share(baseline).max() < cap


def test_an_impossible_rota_is_refused_with_the_relief_named(baseline):
    """A cap so tight the breaks cannot be placed must fail loudly.

    Silently dropping relief would make the model cheaper than reality by
    pretending nobody took a break.
    """
    tight = copy.deepcopy(baseline)
    tight["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"] = 0.05
    with pytest.raises(ValueError, match="of headroom between them") as problem:
        intervals.relief_shares(tight)
    # The message names which relief could not be placed - the morning
    # break, because day order places it first and it fails first.
    assert "morning_break" in str(problem.value)
    assert "widen the window" in str(problem.value)


# ---------------------------------------------------------------------------
# Monday scheduling (spec 7, decided 10/27)
# ---------------------------------------------------------------------------

MONDAY_RUNS = ("monday_ideal_match", "monday_part_time")


def test_weekly_paid_hours_are_identical_to_the_baseline_in_both_runs(baseline):
    """Spec 7: "holding total paid hours constant".

    The whole claim rests on this. Reshaping that quietly added hours would
    be hiring with extra steps, and the saving it reported would be the
    hours, not the schedule.
    """
    paid = baseline["fte"]["paid_hours_per_week"]
    for key in MONDAY_RUNS:
        merged = scenarios.apply_scenario(
            scenarios.load_scenario(scenarios.scenario_dir(baseline) / f"{key}.yaml"),
            baseline,
        )
        for role in baseline["supply"]["roles"]:
            hours = supply.paid_hours_by_day(role, merged)
            assert sum(hours.values()) == pytest.approx(paid, rel=1e-12), f"{key}/{role}"
            weights = supply.day_weights(role, merged)
            assert weights.sum() == pytest.approx(
                baseline["calendar"]["business_days_per_week"], rel=1e-12
            )
            assert (weights > 0).all()


def test_only_the_onsite_associates_are_reshaped(baseline):
    """Spec 7 names the Onsite Associates, and the YAML names them too.

    A reshaped Records Specialist would change how much help the counter
    gets (4.5) and so quietly move the demand side of the comparison.
    """
    for key in MONDAY_RUNS:
        merged = scenarios.apply_scenario(
            scenarios.load_scenario(scenarios.scenario_dir(baseline) / f"{key}.yaml"),
            baseline,
        )
        assert merged["scenario"]["onsite_schedule"]["roles"] == ["onsite_associate"]
        for role in baseline["supply"]["roles"]:
            weights = supply.day_weights(role, merged)
            if role == "onsite_associate":
                assert not np.allclose(weights, 1.0)
            else:
                np.testing.assert_array_equal(weights, np.ones(5))


def test_the_ideal_run_matches_the_day_of_week_pattern_exactly(baseline):
    """Spec 7's upper bound: Monday 23%, Friday 17%, on the same 40 hours.

    Exactly, not approximately - the weights ARE the pattern scaled to
    average 1.0, which is what makes this a bound rather than a proposal.
    """
    merged = scenarios.apply_scenario(
        scenarios.load_scenario(
            scenarios.scenario_dir(baseline) / "monday_ideal_match.yaml"
        ),
        baseline,
    )
    shares = intervals.day_of_week_shares(baseline)
    weights = supply.day_weights("onsite_associate", merged)
    hours = supply.paid_hours_by_day("onsite_associate", merged)
    paid = baseline["fte"]["paid_hours_per_week"]

    for day, weight in zip(supply.WEEKDAYS, weights):
        assert weight == pytest.approx(shares[day] * 5, rel=1e-12)
        assert hours[day] == pytest.approx(shares[day] * paid, rel=1e-12)

    assert hours["monday"] == pytest.approx(9.2)
    assert hours["friday"] == pytest.approx(6.8)
    # And it is an upper bound, so it must be at least as Monday-heavy as
    # the pattern it is matching.
    assert hours["monday"] > hours["tuesday"] > hours["wednesday"]
    assert hours["wednesday"] > hours["thursday"] > hours["friday"]


def test_part_time_hours_land_only_on_monday_and_tuesday_at_75_25(baseline):
    """Spec 7, decided 10/27, and the 75/25 is DERIVED.

    Monday sits 3 points above an even 20% of the week and Tuesday 1, so
    the part-time hours follow 3:1. Derived from the day-of-week pattern
    rather than typed beside it, so the two cannot disagree if the pattern
    is ever revised.

    The full-time reduction comes off ALL FIVE days, because that is what
    part-timing a post costs - you cannot take the cut only from Friday.
    """
    merged = scenarios.apply_scenario(
        scenarios.load_scenario(
            scenarios.scenario_dir(baseline) / "monday_part_time.yaml"
        ),
        baseline,
    )
    settings = merged["scenario"]["onsite_schedule"]
    assert settings["shape"] == "part_time"
    # Decided 10/27 after the sweep: 2%, the only part of the curve that
    # pays for itself. The 10% the scenario opened with is still in the
    # sweep, as its last and worst step.
    share = settings["part_time_share"]
    assert share == 0.02
    assert settings["part_time_days"] == ["monday", "tuesday"]

    weights = supply.day_weights("onsite_associate", merged)
    share_of_week = weights / 5
    flat = 0.20

    # Full-timers keep the rest, evenly, on every day.
    full_time = (1 - share) / 5
    assert full_time == pytest.approx(0.196)
    for day in ("wednesday", "thursday", "friday"):
        index = supply.WEEKDAYS.index(day)
        assert share_of_week[index] == pytest.approx(full_time), day
        assert share_of_week[index] < flat, "every day gives up its share"

    # Part-time hours split 75/25 on top.
    monday = share_of_week[supply.WEEKDAYS.index("monday")] - full_time
    tuesday = share_of_week[supply.WEEKDAYS.index("tuesday")] - full_time
    assert monday == pytest.approx(share * 0.75)
    assert tuesday == pytest.approx(share * 0.25)
    assert monday / (monday + tuesday) == pytest.approx(0.75)

    # Nothing lands anywhere else.
    for day in ("wednesday", "thursday", "friday"):
        assert share_of_week[supply.WEEKDAYS.index(day)] - full_time == pytest.approx(0.0)

    # At 2% Monday lands UNDER its 23% of demand rather than over it,
    # which is the whole difference from the 10% the sweep rejected.
    assert share_of_week[0] == pytest.approx(0.211)
    assert share_of_week[0] < intervals.day_of_week_shares(baseline)["monday"]


def test_the_split_is_derived_from_the_pattern_not_hard_coded(baseline):
    """Revise the day-of-week pattern and the split follows it."""
    changed = copy.deepcopy(baseline)
    changed["intraday"]["day_of_week_share"] = {
        "monday": 0.24,
        "tuesday": 0.22,
        "wednesday": 0.20,
        "thursday": 0.18,
        "friday": 0.16,
    }
    changed["scenario"]["onsite_schedule"].update(
        shape="part_time", part_time_share=0.10
    )
    weights = supply.day_weights("onsite_associate", changed) / 5
    full_time = 0.18
    monday = weights[0] - full_time
    tuesday = weights[1] - full_time
    # 4 points above even against 2, so 2:1 rather than 3:1.
    assert monday / (monday + tuesday) == pytest.approx(2 / 3)
    assert weights.sum() == pytest.approx(1.0, rel=1e-12)


def test_reshaping_the_roster_never_changes_required_hours(baseline, sites):
    """The demand side must not move, or the comparison means nothing.

    Required hours come from the work volume and the interval solve, and
    neither knows the roster exists. If reshaping reached them, a run
    could report a smaller gap by shrinking the work rather than by
    covering it better.
    """
    work = work_volume.build(baseline, sites)["work"]
    base_required = backlog.required_hours(baseline, sites, work)["hours"]

    for key in MONDAY_RUNS:
        merged = scenarios.apply_scenario(
            scenarios.load_scenario(scenarios.scenario_dir(baseline) / f"{key}.yaml"),
            baseline,
        )
        reshaped_work = work_volume.build(merged, sites)["work"]
        np.testing.assert_array_equal(
            work["work_items"].to_numpy(), reshaped_work["work_items"].to_numpy()
        )
        required = backlog.required_hours(merged, sites, reshaped_work)["hours"]
        np.testing.assert_array_equal(
            base_required["required_hours"].to_numpy(),
            required["required_hours"].to_numpy(),
        )
        # The counter's own requirement is unchanged too: Erlang C staffs
        # to the arrivals, and the arrivals did not move.
        plan_weeks, _mondays = supply.plan_window(merged)
        shaped = intervals.onsite_intervals(
            merged, sites, reshaped_work[reshaped_work["week_index"].isin(plan_weeks)]
        )["intervals"]
        flat = intervals.onsite_intervals(
            baseline, sites, work[work["week_index"].isin(plan_weeks)]
        )["intervals"]
        np.testing.assert_array_equal(
            shaped["required_staff"].to_numpy(), flat["required_staff"].to_numpy()
        )


def test_the_reshaping_shows_up_only_in_the_ledger_s_daily_hours(baseline, sites):
    """And it shows up there EXACTLY as the weights, nothing more.

    Headcount, ramp and shrinkage are untouched, so each day's productive
    hours are the baseline's times that day's weight. An exact ratio is
    the strongest statement available: it means the reshaping did one
    thing and nothing else.
    """
    base = supply.build_ledger(baseline, sites)["daily"]
    for key in MONDAY_RUNS:
        merged = scenarios.apply_scenario(
            scenarios.load_scenario(scenarios.scenario_dir(baseline) / f"{key}.yaml"),
            baseline,
        )
        shaped = supply.build_ledger(merged, sites)["daily"]
        joined = base.merge(
            shaped,
            on=["week_index", "role", "scope", "day_of_week"],
            suffixes=("_b", "_s"),
        )
        assert len(joined) == len(base)
        np.testing.assert_allclose(joined["ramped_fte_b"], joined["ramped_fte_s"], rtol=0, atol=0)
        np.testing.assert_allclose(joined["shrinkage_b"], joined["shrinkage_s"], rtol=0, atol=0)

        weights = dict(
            zip(supply.WEEKDAYS, supply.day_weights("onsite_associate", merged))
        )
        counter = joined[joined["role"] == "onsite_associate"]
        expected = counter["productive_hours_b"] * counter["day_of_week"].map(weights)
        np.testing.assert_allclose(
            counter["productive_hours_s"], expected, rtol=1e-12
        )

        # And every other role is untouched to the last bit.
        others = joined[joined["role"] != "onsite_associate"]
        np.testing.assert_allclose(
            others["productive_hours_b"], others["productive_hours_s"], rtol=0, atol=0
        )

        # Weekly paid hours per site unchanged: the weights sum to 5.
        for (_week, scope), group in counter.groupby(["week_index", "scope"]):
            shaped_week = (
                group["productive_hours_s"] / (1 - group["shrinkage_s"])
            ).sum()
            base_week = (group["productive_hours_b"] / (1 - group["shrinkage_b"])).sum()
            assert shaped_week == pytest.approx(base_week, rel=1e-12)


def test_an_unknown_schedule_shape_is_refused(baseline):
    """A typo in the shape must not quietly fall back to flat."""
    broken = copy.deepcopy(baseline)
    broken["scenario"]["onsite_schedule"]["shape"] = "monday_heavy"
    with pytest.raises(ValueError, match="unknown onsite schedule shape"):
        supply.day_weights("onsite_associate", broken)

    broken = copy.deepcopy(baseline)
    broken["scenario"]["onsite_schedule"].update(
        shape="part_time", part_time_share=1.5
    )
    with pytest.raises(ValueError, match="part_time_share must be in"):
        supply.day_weights("onsite_associate", broken)

    broken = copy.deepcopy(baseline)
    broken["scenario"]["onsite_schedule"].update(
        shape="part_time", part_time_share=0.1, part_time_days=["satrday"]
    )
    with pytest.raises(ValueError, match="unknown days"):
        supply.day_weights("onsite_associate", broken)

    # Part-time days that are all at or below an even share have nothing
    # to aim at, which is a scenario that does not mean anything.
    broken = copy.deepcopy(baseline)
    broken["scenario"]["onsite_schedule"].update(
        shape="part_time", part_time_share=0.1, part_time_days=["friday"]
    )
    with pytest.raises(ValueError, match="more than an even share"):
        supply.day_weights("onsite_associate", broken)


def test_both_monday_runs_pay_and_the_ideal_match_bounds_them(comparison):
    """Both runs now pay, which they did not at the 10% the scenario opened on.

    A flat roster meets a 23% Monday and a 17% Friday with the same hours,
    so Monday is where the counter misses and Friday is where it idles.
    Lifting Monday fixes that - until the lift overshoots, and every point
    of the share has to come off all five days to pay for it.

    At the 2% settled on 10/27 the part-time run saves money and covers
    more hours. At 10% it did neither, which the sweep still records as
    its last step.
    """
    base = comparison.loc["Baseline"]
    ideal = comparison.loc["Monday ideal match"]
    part_time = comparison.loc["Monday part-time"]

    # The baseline's Monday is its weak day.
    assert base["monday_phone_service_level"] < base["friday_phone_service_level"]

    for run in (ideal, part_time):
        # Monday improves, and nothing is bought with money: paid hours
        # hold, so the requirement is untouched - and the onsite figure is
        # the one that would move if it were not.
        assert run["monday_phone_service_level"] > base["monday_phone_service_level"]
        assert run["onsite_required_fte"] == pytest.approx(
            base["onsite_required_fte"], rel=1e-9
        )
        assert run["network_required_fte"] == pytest.approx(
            base["network_required_fte"], rel=1e-9
        )
        assert run["plan_year_requests"] == pytest.approx(base["plan_year_requests"])
        assert run["hired_people"] == base["hired_people"]

        # And both now net a saving on both measures.
        assert run["uncovered_hours"] < base["uncovered_hours"]
        assert run["overtime_hours"] < base["overtime_hours"]
        assert run["total_cost"] < base["total_cost"]

    # The ideal match is an upper bound, so nothing may beat it.
    assert ideal["uncovered_hours"] <= part_time["uncovered_hours"] + 1e-9
    assert ideal["total_cost"] <= part_time["total_cost"] + 1e-6
    # And it is a bound nobody reaches: the realistic run captures a
    # fraction of it, which is the sweep's headline.
    saved = base["total_cost"] - part_time["total_cost"]
    bound = base["total_cost"] - ideal["total_cost"]
    assert 0 < saved < bound * 0.5


# ---------------------------------------------------------------------------
# The part-time share sweep (spec 7, decided 10/27)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sweep(baseline, sites):
    return scenarios.part_time_sweep(baseline, sites)


def test_the_sweep_runs_zero_to_ten_points_in_one_point_steps(sweep):
    """Spec 7, decided 10/27."""
    assert list(sweep["part_time_share"]) == [round(0.01 * i, 2) for i in range(11)]
    assert len(sweep) == 11


def test_the_zero_point_step_reproduces_the_baseline_exactly(sweep, baseline, sites):
    """Not approximately. The weights come out as ones and the multiply is
    a no-op, so the first row of the sweep is a check on the whole
    mechanism - if reshaping leaked anywhere it would show here first.
    """
    merged = scenarios.apply_scenario(scenarios.part_time_variant(0.0, baseline), baseline)
    weights = supply.day_weights("onsite_associate", merged)
    assert (weights == 1.0).all()

    # Bit-identical daily hours, not merely close.
    base_daily = supply.build_ledger(baseline, sites)["daily"]
    zero_daily = supply.build_ledger(merged, sites)["daily"]
    np.testing.assert_array_equal(
        base_daily["productive_hours"].to_numpy(),
        zero_daily["productive_hours"].to_numpy(),
    )

    # And the whole plan agrees: the pinned baseline figures, reached
    # through the sweep.
    first = sweep.iloc[0]
    assert first["part_time_share"] == 0.0
    assert first["monday_share_of_paid_hours"] == pytest.approx(0.20)
    assert first["wednesday_share_of_paid_hours"] == pytest.approx(0.20)
    assert first["uncovered_hours"] == pytest.approx(420.0, abs=0.05)
    assert first["overtime_hours"] == pytest.approx(2767.6, abs=0.05)
    assert first["total_cost"] == pytest.approx(7_456_636, abs=1.0)
    assert first["cost_change_vs_baseline"] == 0.0
    assert first["uncovered_change_vs_baseline"] == 0.0


def test_weekly_paid_hours_are_unchanged_at_every_step(baseline):
    """The invariant the whole sweep rests on: only WHEN, never how many."""
    paid = baseline["fte"]["paid_hours_per_week"]
    days = baseline["calendar"]["business_days_per_week"]
    for share in scenarios.PART_TIME_SWEEP:
        merged = scenarios.apply_scenario(
            scenarios.part_time_variant(share, baseline), baseline
        )
        weights = supply.day_weights("onsite_associate", merged)
        assert weights.sum() == pytest.approx(days, rel=1e-12), share
        hours = supply.paid_hours_by_day("onsite_associate", merged)
        assert sum(hours.values()) == pytest.approx(paid, rel=1e-12), share
        assert all(value > 0 for value in hours.values())


def test_the_shares_follow_the_arithmetic(sweep):
    """Monday is 20% + 0.55 x share; every other day loses 20% x share.

    Stated here because it is what makes the curve predictable: the
    instrument has two destination days and takes from five, so it cannot
    match the pattern however it is tuned.
    """
    for row in sweep.itertuples():
        assert row.monday_share_of_paid_hours == pytest.approx(
            0.20 + 0.55 * row.part_time_share
        )
        assert row.wednesday_share_of_paid_hours == pytest.approx(
            0.20 - 0.20 * row.part_time_share
        )

    # Monday passes its 23% of demand partway through the sweep, and
    # Wednesday is already under its 20% by then.
    crossing = sweep[sweep["monday_share_of_paid_hours"] >= 0.23]
    assert len(crossing) > 0
    first_over = crossing.iloc[0]
    assert first_over["part_time_share"] == pytest.approx(0.06)
    assert first_over["wednesday_share_of_paid_hours"] < 0.20


def test_the_sweep_has_an_interior_best_step(sweep):
    """The curve turns, which is the finding the sweep exists to produce.

    If cost fell all the way to 10% the answer would be "more", and if it
    rose from the first point the answer would be "none". It does neither:
    there is a share that beats both ends.
    """
    best_cost = sweep.loc[sweep["total_cost"].idxmin()]
    best_uncovered = sweep.loc[sweep["uncovered_hours"].idxmin()]

    assert 0.0 < best_cost["part_time_share"] < 0.10
    assert best_cost["total_cost"] < sweep.iloc[0]["total_cost"]
    assert best_cost["total_cost"] < sweep.iloc[-1]["total_cost"]
    assert best_uncovered["uncovered_hours"] < sweep.iloc[0]["uncovered_hours"]

    # The 10-point end is worse than doing nothing, which is what the
    # task 5c run reported.
    assert sweep.iloc[-1]["cost_change_vs_baseline"] > 0
    assert sweep.iloc[-1]["uncovered_change_vs_baseline"] > 0


def test_no_step_beats_the_ideal_match(sweep, comparison):
    """The ideal match is an upper bound, so the sweep must stay under it."""
    ideal = comparison.loc["Monday ideal match"]
    assert sweep["uncovered_hours"].min() >= ideal["uncovered_hours"] - 1e-6
    assert sweep["total_cost"].min() >= ideal["total_cost"] - 1e-6


# ---------------------------------------------------------------------------
# Site shortfall (spec 7, 6.2)
# ---------------------------------------------------------------------------

SHORTFALL_SITE = "N1"
SHORTFALL_SHARE = 0.20


@pytest.fixture(scope="module")
def shortfall(baseline, sites):
    """The site-shortfall run beside the baseline, on a shared demand side."""
    work = work_volume.build(baseline, sites)["work"]
    demand = plan.demand_side(baseline, sites, work)
    scenario = scenarios.load_scenario(
        scenarios.scenario_dir(baseline) / "site_shortfall_n1.yaml"
    )
    merged = scenarios.apply_scenario(scenario, baseline)
    return {
        "assumptions": merged,
        "weeks": supply.shortfall_weeks(merged),
        "base": plan.build_onsite(
            baseline, sites, prepared=plan.prepare(baseline, sites, demand=demand)
        ),
        "short": plan.build_onsite(
            merged, sites, prepared=plan.prepare(merged, sites, demand=demand)
        ),
        "base_ledger": supply.build_ledger(baseline, sites)["daily"],
        "short_ledger": supply.build_ledger(merged, sites)["daily"],
    }


def test_the_window_is_four_weeks_from_plan_week_twenty(shortfall, baseline):
    """Spec 7's "Site shortfall" row: week 20, four weeks."""
    first = baseline["calendar"]["history_weeks"]
    assert shortfall["weeks"] == [first + 19, first + 20, first + 21, first + 22]
    assert [week - first + 1 for week in shortfall["weeks"]] == [20, 21, 22, 23]


def test_exactly_twenty_percent_of_that_site_s_hours_are_lost_and_only_there(
    shortfall,
):
    """And nowhere else: not another site, not another role, not another week.

    Checked as an exact ratio, because the shortfall is a single multiply
    and anything other than 0.80 or 1.00 would mean it had leaked.
    """
    joined = shortfall["base_ledger"].merge(
        shortfall["short_ledger"],
        on=["week_index", "role", "scope", "day_of_week"],
        suffixes=("_b", "_s"),
    )
    assert len(joined) == len(shortfall["base_ledger"])

    inside = joined["week_index"].isin(shortfall["weeks"])
    hit = (
        inside
        & (joined["scope"] == SHORTFALL_SITE)
        & (joined["role"] == "onsite_associate")
    )
    assert hit.sum() == len(shortfall["weeks"]) * 5, "four weeks of five days"

    np.testing.assert_allclose(
        joined.loc[hit, "productive_hours_s"],
        joined.loc[hit, "productive_hours_b"] * (1 - SHORTFALL_SHARE),
        rtol=1e-12,
    )
    # Everything else is bit-identical: other weeks, other sites, the
    # Records Specialists at the same site, the remote pools.
    np.testing.assert_array_equal(
        joined.loc[~hit, "productive_hours_b"].to_numpy(),
        joined.loc[~hit, "productive_hours_s"].to_numpy(),
    )

    # And headcount is untouched - they are on leave, not gone (spec 7).
    np.testing.assert_array_equal(
        joined["ramped_fte_b"].to_numpy(), joined["ramped_fte_s"].to_numpy()
    )


def test_the_weeks_outside_the_window_match_the_baseline(shortfall):
    """Bit for bit, across every site, not just the short one.

    The counter is checked by day and its overtime cap is weekly, so
    nothing should escape the four weeks. If anything did - a borrow
    spilling into week 24, an overtime budget shared across weeks - this
    is where it would show.
    """
    joined = shortfall["base"]["daily"].merge(
        shortfall["short"]["daily"],
        on=["week_index", "site", "day_of_week"],
        suffixes=("_b", "_s"),
    )
    outside = joined[~joined["week_index"].isin(shortfall["weeks"])]
    assert len(outside) > 0
    for column in (
        "required_hours", "available_hours", "borrowed_hours", "lent_hours",
        "overtime_hours", "uncovered_hours", "service_level_phones_after",
    ):
        np.testing.assert_array_equal(
            outside[column + "_b"].to_numpy(),
            outside[column + "_s"].to_numpy(),
            err_msg=column,
        )

    # And no borrow is dated outside the window.
    borrows = shortfall["short"]["borrows"]
    assert len(borrows) > 0
    assert set(borrows["week_index"]) <= set(shortfall["weeks"])


def test_every_borrowed_hour_comes_from_a_same_cluster_site_with_spare_hours(
    shortfall, sites
):
    """Spec 6.2, and the two things a borrow must be able to prove.

    The lender is in the same cluster - never across one - and it had the
    hours to give on that very day, before anybody drove anywhere.
    """
    borrows = shortfall["short"]["borrows"]
    assert len(borrows) > 0
    _members, of_site = plan.cluster_lookup(sites)
    daily = shortfall["short"]["daily"].set_index(
        ["week_index", "site", "day_of_week"]
    )

    for borrow in borrows.itertuples():
        assert of_site[borrow.short_site] == of_site[borrow.covering_site]
        assert borrow.cluster == of_site[borrow.short_site]
        assert borrow.short_site != borrow.covering_site

        # The lender's surplus BEFORE it lent: the daily frame is already
        # net of lending, so the hours it gave go back on.
        lender = daily.loc[
            (borrow.week_index, borrow.covering_site, borrow.day_of_week)
        ]
        spare_before = (
            float(lender["available_hours"])
            + float(lender["lent_hours"])
            - float(lender["required_hours"])
        )
        assert spare_before >= borrow.hours + borrow.travel_hours - 1e-9

        # And the borrow is a real half-day or more, with the drive paid.
        assert borrow.hours >= sites["travel"]["minimum_borrow_hours"] - 1e-9
        assert borrow.travel_hours > 0
        assert borrow.travel_cost == pytest.approx(
            borrow.travel_miles * sites["travel"]["mileage_rate"]
        )


def test_lending_never_leaves_the_lender_uncovered(shortfall):
    """Spec 6.2's first step must not rob one site to pay another.

    Only surplus is ever offered and travel comes out of that same
    surplus, so a lender finishes with at least its own required hours -
    no overtime bought and nothing uncovered because of lending.
    """
    daily = shortfall["short"]["daily"]
    lenders = daily[daily["lent_hours"] > 1e-9]
    assert len(lenders) > 0

    for row in lenders.itertuples():
        assert row.available_hours >= row.required_hours - 1e-9
        assert row.uncovered_hours == pytest.approx(0.0, abs=1e-9)

    # And the same day at the same site was not worked harder than it
    # would have been, so lending did not buy overtime for the lender.
    base = shortfall["base"]["daily"].set_index(
        ["week_index", "site", "day_of_week"]
    )
    for row in lenders.itertuples():
        before = base.loc[(row.week_index, row.site, row.day_of_week)]
        assert row.overtime_hours <= float(before["overtime_hours"]) + 1e-9


def test_the_shortfall_is_covered_in_the_six_two_order(shortfall):
    """Borrow, then overtime, then accept the gap - and it adds up."""
    daily = shortfall["short"]["daily"]
    inside = daily[daily["week_index"].isin(shortfall["weeks"])]
    np.testing.assert_allclose(
        inside["covered_hours"],
        inside["available_hours"] + inside["borrowed_hours"] + inside["overtime_hours"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        inside["uncovered_hours"],
        (inside["required_hours"] - inside["covered_hours"]).clip(lower=0.0),
        rtol=1e-9,
        atol=1e-12,
    )

    base = shortfall["base"]["daily"]
    base_in = base[base["week_index"].isin(shortfall["weeks"])]
    site = inside[inside["site"] == SHORTFALL_SITE]
    base_site = base_in[base_in["site"] == SHORTFALL_SITE]

    assert site["available_hours"].sum() < base_site["available_hours"].sum()
    assert site["overtime_hours"].sum() > base_site["overtime_hours"].sum()
    assert site["uncovered_hours"].sum() > base_site["uncovered_hours"].sum()
    # Required hours are untouched: the work still arrives (spec 6.2).
    np.testing.assert_allclose(
        site["required_hours"].to_numpy(),
        base_site["required_hours"].to_numpy(),
        rtol=1e-12,
    )


def test_cluster_coverage_barely_helps_because_the_sites_are_short_together(
    shortfall,
):
    """The finding, and the same one the baseline's zero borrows showed.

    All six sites share one day-of-week pattern, so the days N1 needs help
    are the days its neighbours are tight too, and the day they are flush
    (Friday) is the day N1 does not need it. Even the best case the
    network can offer borrows a single half-day.
    """
    borrows = shortfall["short"]["borrows"]
    daily = shortfall["short"]["daily"]
    inside = daily[daily["week_index"].isin(shortfall["weeks"])]
    site = inside[inside["site"] == SHORTFALL_SITE]

    borrowed = borrows["hours"].sum()
    gap = (site["required_hours"] - site["available_hours"]).clip(lower=0).sum()
    assert borrowed > 0, "something must borrow, or this proves nothing"
    assert borrowed / gap < 0.10, "and it covers under a tenth of the hole"

    # Overtime does the work instead.
    assert site["overtime_hours"].sum() > borrowed * 10

    # N1 is short on the early days and slack on Friday, which is exactly
    # when its neighbours have hours to spare.
    short_by_day = {}
    for day, group in site.groupby("day_of_week"):
        short_by_day[day] = (
            group["required_hours"] - group["available_hours"]
        ).mean()
    assert short_by_day["friday"] < short_by_day["monday"]
    assert short_by_day["monday"] > 4


# ---------------------------------------------------------------------------
# A scenario can reach sites.yaml (spec 7, decided 10/27)
# ---------------------------------------------------------------------------


def test_a_scenario_can_override_sites_yaml_and_typos_still_fail(baseline, sites):
    """The borrow minimum lives in sites.yaml, so a scenario must reach it.

    Same strict merge as assumptions.yaml, so a typo there fails the same
    way rather than silently leaving the default in place - which on this
    particular input would produce a sensitivity run that was not
    sensitive to anything.
    """
    scenario = scenarios.load_scenario(
        scenarios.scenario_dir(baseline) / "site_shortfall_n1_borrow_2h.yaml"
    )
    assert scenarios.declared_site_keys(scenario) == ["travel.minimum_borrow_hours"]

    merged = scenarios.apply_site_overrides(scenario, sites)
    assert merged["travel"]["minimum_borrow_hours"] == 2
    assert sites["travel"]["minimum_borrow_hours"] == 4, "baseline untouched"
    # Everything else in the file survives.
    assert merged["travel"]["mileage_rate"] == sites["travel"]["mileage_rate"]
    assert len(merged["travel"]["pairs"]) == len(sites["travel"]["pairs"])
    assert merged["sites"] == sites["sites"]

    with pytest.raises(KeyError, match="travel.minimum_borow_hours"):
        scenarios.merge_overrides(sites, {"travel": {"minimum_borow_hours": 2}})

    # A scenario with no site overrides leaves the file alone.
    plain = scenarios.load_scenario(scenarios.scenario_dir(baseline) / "baseline.yaml")
    assert scenarios.declared_site_keys(plain) == []
    assert scenarios.apply_site_overrides(plain, sites) == sites


def test_a_lower_borrow_minimum_can_only_increase_borrowing(baseline, sites):
    """A direction check on the sensitivity run.

    Relaxing the minimum cannot remove a borrow that already qualified, so
    the 2-hour run must borrow at least as much as the 4-hour one - and
    strictly more, or the sensitivity would be telling us nothing.
    """
    work = work_volume.build(baseline, sites)["work"]
    demand = plan.demand_side(baseline, sites, work)
    runs = {}
    for key in ("site_shortfall_n1", "site_shortfall_n1_borrow_2h"):
        scenario = scenarios.load_scenario(
            scenarios.scenario_dir(baseline) / f"{key}.yaml"
        )
        merged = scenarios.apply_scenario(scenario, baseline)
        merged_sites = scenarios.apply_site_overrides(scenario, sites)
        runs[key] = plan.build_onsite(
            merged,
            merged_sites,
            prepared=plan.prepare(merged, merged_sites, demand=demand),
        )

    strict, loose = runs["site_shortfall_n1"], runs["site_shortfall_n1_borrow_2h"]
    assert len(loose["borrows"]) > len(strict["borrows"])
    assert loose["borrows"]["hours"].sum() > strict["borrows"]["hours"].sum()
    # Every borrow still clears its own minimum.
    assert loose["borrows"]["hours"].min() >= 2 - 1e-9
    assert strict["borrows"]["hours"].min() >= 4 - 1e-9

    # More borrowing, so no more uncovered work anywhere.
    joined = strict["daily"].merge(
        loose["daily"], on=["week_index", "site", "day_of_week"], suffixes=("_s", "_l")
    )
    assert (joined["uncovered_hours_l"] <= joined["uncovered_hours_s"] + 1e-9).all()
    assert joined["uncovered_hours_l"].sum() < joined["uncovered_hours_s"].sum()


# ---------------------------------------------------------------------------
# Centralized phones (spec 7, Section 9)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def central(baseline, sites):
    work = work_volume.build(baseline, sites)["work"]
    scenario = scenarios.load_scenario(
        scenarios.scenario_dir(baseline) / "central_phones.yaml"
    )
    merged = scenarios.apply_scenario(scenario, baseline)
    return {
        "assumptions": merged,
        "work": work,
        "from_week": intervals.central_phones_from_week(merged),
        "queue": intervals.central_phone_queue(merged, sites, work),
        "shaped": intervals.onsite_intervals(merged, sites, work)["intervals"],
        "flat": intervals.onsite_intervals(baseline, sites, work)["intervals"],
        "report": scenarios.central_phones_report(baseline, sites, work),
    }


def test_centralizing_never_needs_more_phone_staff_than_six_queues(central):
    """Spec Section 9, with the +5% uplift OFF so it is pooling alone.

    Erlang C's spare capacity is what pooling shares: six sites each
    keeping somebody free in case the phone rings keep six people free,
    and one queue keeps fewer. The comparison has to hold at the same goal
    AND the same handle time, or it is measuring two things at once.
    """
    queue = central["queue"]
    assert len(queue) > 0
    assert (queue["required_capped"] <= queue["required_separate"]).all()
    assert (queue["required_no_cap"] <= queue["required_separate"]).all()
    # And strictly fewer somewhere, or pooling bought nothing.
    assert (queue["required_capped"] < queue["required_separate"]).any()

    # The separate figure is itself computed at the base handle time.
    assert (queue["handle_seconds_base"] < queue["handle_seconds_central"]).all()
    uplift = central["assumptions"]["scenario"]["central_phones"]["handle_time_uplift"]
    np.testing.assert_allclose(
        queue["handle_seconds_central"],
        queue["handle_seconds_base"] * (1 + uplift),
        rtol=1e-12,
    )


def test_with_the_uplift_off_the_central_answer_is_the_capped_one(baseline, sites):
    """The decomposition's middle term, isolated.

    Turn the uplift off and the final staffing must be exactly the capped
    service-level answer - which is what makes the three reported lines
    add up rather than merely look plausible.
    """
    merged = copy.deepcopy(baseline)
    merged["scenario"]["central_phones"].update(enabled=True, handle_time_uplift=0.0)
    merged["scenario"]["starts_week"] = 13
    work = work_volume.build(baseline, sites)["work"]
    queue = intervals.central_phone_queue(merged, sites, work)
    np.testing.assert_array_equal(
        queue["required_staff"].to_numpy(), queue["required_capped"].to_numpy()
    )
    assert (queue["required_staff"] <= queue["required_separate"]).all()


def test_the_three_reported_lines_add_up(central):
    """pooling gain - cap give-back - uplift = the net, exactly."""
    n = central["report"]["network"]
    assert n["pooling_gain"] == pytest.approx(
        n["phones_six_queues"] - n["phones_central_service_level_only"]
    )
    assert n["cap_gives_back"] == pytest.approx(
        n["phones_central_capped"] - n["phones_central_service_level_only"]
    )
    assert n["uplift_costs"] == pytest.approx(
        n["phones_central_final"] - n["phones_central_capped"]
    )
    assert n["phones_net_gain"] == pytest.approx(
        n["pooling_gain"] - n["cap_gives_back"] - n["uplift_costs"]
    )
    assert n["pooling_gain"] > 0
    assert n["uplift_costs"] > 0
    # The 85% cap never binds here, which is worth recording: Session 3
    # expected it to bite hardest in large queues, and this network's
    # pooled queue is still not large enough for it to reach.
    assert n["cap_gives_back"] == 0


def test_calls_are_conserved_interval_by_interval(central, sites):
    """The central queue answers exactly the calls the sites received.

    Pooling moves WHO answers, never how many call. If the network total
    drifted from the sum of the sites, the pooling gain would partly be
    calls going missing.
    """
    queue = central["queue"].set_index(["week_index", "day_of_week", "interval"])
    shaped = central["shaped"]
    after = shaped[shaped["week_index"] >= central["from_week"]]
    by_interval = after.groupby(["week_index", "day_of_week", "interval"])[
        "call_arrivals"
    ].sum()

    # The queue spans the whole 208-week timeline; only the weeks from
    # the changeover on are used, so the join is against those.
    joined = queue[["call_arrivals"]].join(
        by_interval.to_frame("site_sum"), how="inner"
    )
    assert len(joined) == len(by_interval)
    assert len(joined) == 40 * 5 * 18
    np.testing.assert_allclose(
        joined["call_arrivals"], joined["site_sum"], rtol=1e-9
    )

    # And each site's share of the queue is its share of the calls.
    shares = after.groupby(["week_index", "day_of_week", "interval"])[
        "central_call_share"
    ].sum()
    np.testing.assert_allclose(shares.to_numpy(), 1.0, rtol=1e-9)


def test_the_weeks_before_the_changeover_match_the_baseline(central):
    """Spec 7: the scenario starts in week 13, so weeks 1 to 12 are blended.

    Bit for bit, which is the strongest statement available: those weeks
    come out of the same builder the baseline uses, not out of a special
    case that happens to agree.
    """
    shaped, flat = central["shaped"], central["flat"]
    assert len(shaped) == len(flat)
    before = shaped["week_index"] < central["from_week"]
    assert before.sum() > 0

    for column in (
        "required_counter", "required_phones", "required_staff",
        "scheduled_staff", "scheduled_hours", "away_share", "traffic_pooled",
    ):
        np.testing.assert_array_equal(
            shaped.loc[before, column].to_numpy(),
            flat.loc[before, column].to_numpy(),
            err_msg=column,
        )

    # And no central columns are filled in before the changeover.
    assert shaped.loc[before, "central_required_staff"].isna().all()
    assert shaped.loc[~before, "central_required_staff"].notna().all()


def test_the_counter_loses_its_blend_and_needs_more_than_walk_ins_alone(central):
    """The effect a pooling argument on its own would miss.

    A counter taking a handful of walk-ins an hour still needs somebody
    standing at it. Today that somebody answers the phone between
    requesters; afterwards they do not, so the walk-in-only requirement is
    far from zero even though the calls have gone.
    """
    n = central["report"]["network"]
    # Walk-ins alone still need most of what the blended team needed for
    # BOTH channels - which is the whole reason the blend was worth having.
    assert n["counter_walk_ins_only"] > 0
    assert n["counter_walk_ins_only"] / n["counter_blended_baseline"] > 0.35

    # Two separate queues per site would have cost more than the blend.
    separate_per_site = n["counter_walk_ins_only"] + n["phones_six_queues"]
    assert separate_per_site > n["counter_blended_baseline"]

    # And centralizing still comes out ahead of the blend overall.
    central_total = n["counter_walk_ins_only"] + n["phones_central_final"]
    assert central_total < n["counter_blended_baseline"]
    assert n["counter_extra"] == pytest.approx(
        central_total - n["counter_blended_baseline"]
    )
    assert n["counter_extra"] < 0, "centralizing nets a saving"


def test_every_site_needs_fewer_staff_intervals_than_its_blended_team(central):
    """The gain is shared, because the central team is charged back by calls.

    Allocating the central requirement pro rata by each site's calls is
    what stops one site capturing the pooling gain while another pays for
    it - and it is why every site comes out ahead rather than on average.
    """
    per_site = central["report"]["per_site"]
    assert len(per_site) == 6
    assert (per_site["change_vs_blended"] < 0).all()
    assert per_site["call_share"].sum() == pytest.approx(1.0, rel=1e-6)
    np.testing.assert_allclose(
        per_site["total_staff_intervals"],
        per_site["counter_only_staff_intervals"]
        + per_site["central_share_staff_intervals"],
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# The FTE glossary (reconciled 10/27)
# ---------------------------------------------------------------------------


def test_the_fte_glossary_names_three_definitions_with_their_weeks(comparison):
    """Four figures got quoted across tasks 5b to 5e; three were right.

    They differ on weeks, on floor-versus-net-of-help, and on which
    role's shrinkage divides them. The glossary exists so no table can be
    read against the wrong one.
    """
    assert len(scenarios.FTE_GLOSSARY) == 3
    names = [name for name, _meaning, _weeks in scenarios.FTE_GLOSSARY]
    assert names == ["floor FTE", "onsite required FTE", "network required FTE"]
    for _name, meaning, weeks in scenarios.FTE_GLOSSARY:
        assert meaning and weeks

    # The comparison table carries the two it reports, under those names.
    base = comparison.loc["Baseline"]
    assert base["onsite_required_fte"] == pytest.approx(24.03, abs=0.01)
    assert base["network_required_fte"] == pytest.approx(107.94, abs=0.01)


def test_the_three_definitions_reconcile_to_the_help_and_the_shrinkage(
    baseline, sites
):
    """Each difference is one named thing, and the test says which.

    floor to onsite required  = the Records Specialist help, exactly
    onsite to network         = the back office
    and every figure divides by ITS OWN role's productive hours.
    """
    shape = scenarios.onsite_shape(baseline, sites)
    basis_onsite = hiring.fte_basis("onsite_associate", baseline)
    basis_back = hiring.fte_basis("tier_1", baseline)

    assert basis_onsite == pytest.approx(40 * (1 - 0.2175))
    assert basis_back == pytest.approx(40 * (1 - 0.28))
    assert basis_onsite != basis_back, "one basis would misprice the counter"

    # Floor less help plus spillover IS the onsite figure, to the cent.
    assert shape["floor_fte"] == pytest.approx(25.14, abs=0.01)
    assert shape["required_fte"] == pytest.approx(24.03, abs=0.01)
    gap = shape["floor_fte"] - shape["required_fte"]
    plan_weeks, _mondays = supply.plan_window(baseline)
    assert gap == pytest.approx(
        (shape["records_specialist_help_hours"] - shape["email_spillover_hours"])
        / basis_onsite
        / len(plan_weeks),
        rel=1e-9,
    )
    assert gap == pytest.approx(1.11, abs=0.01)


# ---------------------------------------------------------------------------
# Deflection and the 7.2 worked example (spec Section 9)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ai_on(baseline):
    scenario = scenarios.load_scenario(
        scenarios.scenario_dir(baseline) / "ai_and_automation.yaml"
    )
    return scenarios.apply_scenario(scenario, baseline)


@pytest.fixture(scope="module")
def ai_half(baseline):
    scenario = scenarios.load_scenario(
        scenarios.scenario_dir(baseline) / "ai_and_automation_50.yaml"
    )
    return scenarios.apply_scenario(scenario, baseline)


def test_the_seven_two_worked_example_reproduces_4800_minutes(ai_on):
    """Spec Section 9 and 7.2, revised 10/28, worked by hand.

    1,000 calls at 5 minutes is 5,000 minutes of workload. Deflect
    1,000 x 40% x (15% + 5%) = 80 of them, and those 80 averaged 2.5
    minutes - half an average call, because a status call a portal can
    answer is the short one - so 200 minutes leave with them.

    The remaining 920 calls carry 4,800 minutes, an average of 5.217 each:
    a mix uplift of +4.35%. Workload FALLS 4.0%.

    The old version of this test asserted 5,060 minutes and a 1.2% RISE,
    from a flat +10% uplift. That was impossible as a mix effect: it
    implied the deflected calls had taken less than zero minutes.
    """
    share = automation.deflected_share(ai_on)
    assert share == pytest.approx(0.08)
    easy = ai_on["scenario"]["deflection"]["easy_handle_time_relative"]
    assert easy == 0.50

    calls, handle = 1000, 5.0
    removed = calls * share
    assert removed == pytest.approx(80.0)
    assert removed * handle * easy == pytest.approx(200.0)

    mix = automation.deflection_mix_uplift(ai_on)
    assert mix == pytest.approx(0.0434783, abs=1e-7)
    assert mix == pytest.approx(0.0435, abs=0.00005)

    multiple = automation.deflection_handle_time_multiple("W20", 10**6, ai_on)
    assert multiple == pytest.approx(1 + mix), "escalation is zero by default"
    minutes = (calls - removed) * handle * multiple
    assert minutes == pytest.approx(4800.0)
    assert minutes / (calls * handle) - 1 == pytest.approx(-0.040, abs=1e-9)


def test_the_break_even_escalation_is_four_point_one_seven_percent(ai_on, ai_half):
    """Spec Section 9 and 7.2, revised 10/28.

    Break-even is where the hours deflection saved are used up again:

        (1 - removed share x easy ratio) x (1 + escalation) = 1

    At the defaults workload falls to 96%, so break-even escalation is
    1/0.96 - 1 = 4.17% - the 200 minutes saved over the 4,800 remaining.
    The 7.1 default escalation is 0%, so deflection as configured SAVES
    hours and the question is how much escalation it can absorb.

    At 50% realization only 4% is deflected, workload falls to 98%, and
    break-even tightens to 2.04%: half the benefit, half the room.
    """
    assert automation.deflection_break_even_uplift(ai_on) == pytest.approx(
        1 / 0.96 - 1, rel=1e-12
    )
    assert automation.deflection_break_even_uplift(ai_on) == pytest.approx(
        0.0417, abs=0.00005
    )
    escalation = ai_on["scenario"]["deflection"]["escalation_uplift"]
    assert escalation == 0.0
    assert escalation < automation.deflection_break_even_uplift(ai_on)

    assert automation.deflected_share(ai_half) == pytest.approx(0.04)
    assert automation.deflection_break_even_uplift(ai_half) == pytest.approx(
        1 / 0.98 - 1, rel=1e-12
    )
    assert automation.deflection_break_even_uplift(ai_half) == pytest.approx(
        0.0204, abs=0.00005
    )
    # Half the realization, half the room before it stops paying.
    assert automation.deflection_break_even_uplift(
        ai_half
    ) < automation.deflection_break_even_uplift(ai_on)


def test_rates_over_one_hundred_percent_in_a_group_are_refused(ai_on):
    """Spec Section 9, both groups. Refused, not clipped.

    A scenario asking for 60% by AI and 60% by self-service has a typo in
    it, and silently capping at 100% would hide the typo while the plan
    was argued from a number nobody chose.
    """
    broken = copy.deepcopy(ai_on)
    broken["scenario"]["deflection"].update(ai_deflection=0.60, self_service=0.60)
    with pytest.raises(ValueError, match="over 100%"):
        automation.deflected_share(broken)

    broken = copy.deepcopy(ai_on)
    broken["automation"]["rates"].update(ai_handling=0.60, automation_displacement=0.60)
    with pytest.raises(ValueError, match="over 100%"):
        automation.target_share(broken)

    # Exactly 100% is allowed: it is extreme, not wrong.
    edge = copy.deepcopy(ai_on)
    edge["scenario"]["deflection"].update(ai_deflection=0.60, self_service=0.40)
    assert automation.deflected_share(edge) == pytest.approx(0.40)


def test_deflection_removes_contacts_and_touches_nothing_else(
    baseline, sites, ai_on
):
    """Only status calls and emails, only from the starts week.

    Not walk-ins - somebody at the counter has already arrived - and not
    the request pipeline, which is work that was never a contact.
    """
    base = work_volume.build(baseline, sites)["work"]
    after = work_volume.build(ai_on, sites)["work"]
    first = baseline["calendar"]["history_weeks"]
    from_week = automation.deflection_from_week(ai_on)
    assert from_week == first + 12

    deflected = set(automation.deflection_work_types(ai_on))
    assert deflected == {"W20", "W21"}
    share = automation.deflected_share(ai_on)

    for code in ("W19", "W20", "W21", "W1", "W8"):
        was = base[(base["work_type"] == code) & (base["week_index"] >= from_week)][
            "work_items"
        ].sum()
        now = after[(after["work_type"] == code) & (after["week_index"] >= from_week)][
            "work_items"
        ].sum()
        expected = was * (1 - share) if code in deflected else was
        assert now == pytest.approx(expected, rel=1e-9), code

    # The history and the weeks before the changeover are untouched.
    early = base["week_index"] < from_week
    np.testing.assert_allclose(
        base.loc[early, "work_items"].to_numpy(),
        after.loc[early, "work_items"].to_numpy(),
        rtol=0,
        atol=0,
    )


def test_the_uplift_reaches_the_queues_and_the_email_but_not_the_history(
    baseline, sites, ai_on
):
    """Spec 7.1's demand uplift, where it lands.

    Calls and email cost 4.35% more per contact from the starts week - the
    calculated mix uplift, with escalation at its 7.1 default of zero. The
    uplift is on HANDLE TIME, not volume, because Erlang C needs the two
    apart: a longer call changes the staffing a service goal demands in a
    way a higher arrival rate does not.
    """
    first = baseline["calendar"]["history_weeks"]
    from_week = automation.deflection_from_week(ai_on)
    lift = 1 + automation.deflection_mix_uplift(ai_on)
    assert lift == pytest.approx(1.0434783, abs=1e-7)
    for code in ("W20", "W21"):
        assert automation.deflection_handle_time_multiple(
            code, from_week, ai_on
        ) == pytest.approx(lift)
        assert automation.deflection_handle_time_multiple(
            code, from_week - 1, ai_on
        ) == 1.0
        assert automation.deflection_handle_time_multiple(code, 0, ai_on) == 1.0
    # Walk-ins never carry it.
    assert automation.deflection_handle_time_multiple("W19", from_week, ai_on) == 1.0
    # And neither does the baseline, at any week.
    assert automation.deflection_handle_time_multiple("W20", from_week, baseline) == 1.0

    # Email hours per contact rise by the uplift from the changeover.
    work = work_volume.build(baseline, sites)["work"]
    plain = intervals.support_work_hours(work, baseline)
    lifted = intervals.support_work_hours(work, ai_on)
    before = [(week, site) for week, site in plain.index if week < from_week]
    after = [(week, site) for week, site in plain.index if week >= from_week]
    np.testing.assert_allclose(plain.loc[before], lifted.loc[before], rtol=0, atol=0)
    np.testing.assert_allclose(
        lifted.loc[after], plain.loc[after] * lift, rtol=1e-12
    )


def test_the_supply_uplift_only_touches_eligible_work_from_go_live(baseline, ai_on):
    """Spec 7.1's supply uplift, and the rows it leaves alone.

    W11 and W24 are pinned to 0% by Section 1, so no machine took their
    simple items and nothing about their mix changed - they carry no
    uplift. Neither do the demand-side work types, and neither does any
    week before go-live.
    """
    go_live = automation.go_live_week_index(ai_on)
    # Called without a share, the function prices the NOMINAL target of 8%,
    # which is the full-ramp mix uplift of 4.35%. The weekly build passes
    # the share actually removed instead, and a test downstream says so.
    assert automation.supply_handle_time_multiple(
        "W1", go_live, ai_on
    ) == pytest.approx(1.0434783, abs=1e-7)
    assert automation.supply_handle_time_multiple("W1", go_live - 1, ai_on) == 1.0
    for code in ("W11", "W24", "W19", "W20", "W21"):
        assert automation.supply_handle_time_multiple(code, go_live, ai_on) == 1.0, code
    # And nothing at all in the baseline.
    for code in ("W1", "W8", "W14"):
        assert automation.supply_handle_time_multiple(code, go_live, baseline) == 1.0


def test_raising_the_exception_rate_never_lowers_human_hours_or_cost(
    baseline, sites, ai_on
):
    """Spec Section 9's direction check on 5.5.

    An exception is work coming BACK off the machine, so more of them can
    only add human hours and can only make the automated item cost more.
    Obvious, and exactly the kind of thing a sign error reverses.
    """
    work = work_volume.build(ai_on, sites)["work"]
    previous_hours = None
    previous_cost = None
    for rate in (0.0, 0.10, 0.25, 0.50):
        config_at = copy.deepcopy(ai_on)
        config_at["automation"]["work_types"]["W1"]["exception_rate"] = rate
        built = backlog.required_hours(config_at, sites, work)
        loop = built["human_in_the_loop"]
        hours = loop[
            (loop["work_type"] == "W1") & (loop["work_kind"] == "exception")
        ]["required_hours"].sum()
        cost = plan.automated_cost_per_item("W1", "tier_1", 6, config_at)["total"]

        if previous_hours is not None:
            assert hours >= previous_hours - 1e-9, rate
            assert cost >= previous_cost - 1e-12, rate
        previous_hours, previous_cost = hours, cost

    # And it genuinely moved, so the check is not vacuous.
    assert previous_hours > 0
    assert previous_cost > plan.automated_cost_per_item("W1", "tier_1", 6, ai_on)[
        "total"
    ]


# ---------------------------------------------------------------------------
# Realization scales the uplifts too (spec 7.1, decided 10/27)
# ---------------------------------------------------------------------------


def test_the_uplift_scales_linearly_with_realization(baseline, ai_on):
    """Spec 7.1, decided 10/27: effective uplift = stated x realization.

    An uplift is a CONSEQUENCE of the mix changing, not a promise anybody
    made, so half the simple work removed shifts the mix half as much.
    Scaling the rates while leaving the uplifts at full strength made a
    half-delivered programme worse than a whole one - an artefact of the
    arithmetic rather than a finding.
    """
    go_live = automation.go_live_week_index(ai_on)
    # The ESCALATION and EXTRA uplifts scale with realization (10/27).
    for realization, expected in ((1.0, 0.10), (0.5, 0.05), (0.25, 0.025), (0.0, 0.0)):
        at = copy.deepcopy(ai_on)
        at["automation"]["rates"]["realization"] = realization
        assert automation.effective_uplift(0.10, at) == pytest.approx(expected)

    # The MIX uplift needs no scaling: it is calculated from the share
    # actually removed, which already carries realization (10/28). So it
    # follows realization down on its own - but NOT linearly, which is
    # exactly why it would have been wrong to scale it by hand as well.
    shares, mixes = [], []
    for realization in (1.0, 0.5, 0.25, 0.0):
        at = copy.deepcopy(ai_on)
        at["automation"]["rates"]["realization"] = realization
        shares.append(automation.deflected_share(at))
        mixes.append(automation.deflection_mix_uplift(at))
    assert shares == [pytest.approx(x) for x in (0.08, 0.04, 0.02, 0.0)]
    assert mixes[0] > mixes[1] > mixes[2] > 0
    assert mixes[-1] == 0.0
    # Half the share is LESS than half the mix uplift. The formula is
    # share x (1 - easy) / (1 - share), convex in the share, so the
    # denominator shrinks the effect as well as the numerator: 2.08%
    # against half of 4.35%. Asserting linearity here would be wrong, and
    # asserting it the other way round would be wrong too.
    assert mixes[1] < mixes[0] / 2
    assert mixes[1] == pytest.approx(0.0208333, abs=1e-7)

    # And the two combine multiplicatively on the handle time.
    with_escalation = copy.deepcopy(ai_on)
    with_escalation["scenario"]["deflection"]["escalation_uplift"] = 0.10
    assert automation.deflection_handle_time_multiple(
        "W20", go_live, with_escalation
    ) == pytest.approx((1 + mixes[0]) * 1.10)


def test_zero_realization_is_the_baseline_exactly(baseline, sites, ai_on):
    """Nothing delivered, nothing changed - bit for bit.

    The strongest check on the whole AI path: every rate and every uplift
    collapses to zero together, so the run has to come out as the baseline
    rather than merely near it.
    """
    nothing = copy.deepcopy(ai_on)
    nothing["automation"]["rates"]["realization"] = 0.0

    assert automation.deflected_share(nothing) == 0.0
    assert automation.target_share(nothing) == 0.0
    assert automation.enabled(nothing) is False
    assert automation.deflection_enabled(nothing) is False
    assert automation.effective_uplift(0.10, nothing) == 0.0
    assert automation.deflection_mix_uplift(nothing) == 0.0

    base_work = work_volume.build(baseline, sites)["work"]
    none_work = work_volume.build(nothing, sites)["work"]
    assert len(base_work) == len(none_work)
    np.testing.assert_array_equal(
        base_work["work_items"].to_numpy(), none_work["work_items"].to_numpy()
    )

    base_req = backlog.required_hours(baseline, sites, base_work)
    none_req = backlog.required_hours(nothing, sites, none_work)
    assert "work_kind" not in none_req["hours"].columns
    np.testing.assert_array_equal(
        base_req["hours"]["required_hours"].to_numpy(),
        none_req["hours"]["required_hours"].to_numpy(),
    )


def test_the_seven_two_check_survives_the_realization_change(ai_on):
    """At 100% realization nothing moves, so 7.2 still reproduces exactly.

    The whole point of scaling by realization is that it is invisible at
    full delivery. If this ever drifts, the change has reached further
    than it was meant to.
    """
    assert ai_on["automation"]["rates"]["realization"] == 1.0
    share = automation.deflected_share(ai_on)
    multiple = automation.deflection_handle_time_multiple("W20", 10**6, ai_on)
    assert share == pytest.approx(0.08)
    assert multiple == pytest.approx(1.0434783, abs=1e-7)
    assert (1000 - 1000 * share) * 5.0 * multiple == pytest.approx(4800.0)
    assert automation.deflection_break_even_uplift(ai_on) == pytest.approx(
        0.0416667, abs=1e-7
    )


def test_with_zero_escalation_deflection_never_raises_workload(ai_on):
    """Spec 7.2, revised 10/28: the mix alone cannot cost hours.

    Workload after deflection is (1 - removed share x easy ratio) of what
    it was, which is at most 1 - and equals 1 only when the removed items
    were free. So however harsh the easy ratio, zero escalation can never
    leave the operation with MORE work than it started with.

    That is the error the flat +10% uplift was making, and the reason this
    test exists rather than a margin test.
    """
    for realization in (1.0, 0.75, 0.5, 0.25, 0.0):
        for easy in (0.0, 0.25, 0.5, 0.75, 1.0):
            at = copy.deepcopy(ai_on)
            at["automation"]["rates"]["realization"] = realization
            at["scenario"]["deflection"]["easy_handle_time_relative"] = easy
            at["scenario"]["deflection"]["escalation_uplift"] = 0.0
            share = automation.deflected_share(at)
            multiple = automation.deflection_handle_time_multiple("W20", 10**6, at)
            workload = (1 - share) * multiple
            assert workload <= 1.0 + 1e-12, (realization, easy)
            assert workload == pytest.approx(1 - share * easy, rel=1e-12)

    # And with a positive escalation it can, which is the whole point of
    # separating them.
    costly = copy.deepcopy(ai_on)
    costly["scenario"]["deflection"]["escalation_uplift"] = 0.10
    share = automation.deflected_share(costly)
    multiple = automation.deflection_handle_time_multiple("W20", 10**6, costly)
    assert (1 - share) * multiple > 1.0


def test_the_mix_uplift_never_exceeds_its_ceiling(ai_on):
    """Spec 7.2, revised 10/28: the ceiling is 1/(1 - removed share) - 1.

    Reached only when the removed items were free, and at that point total
    workload is exactly FLAT. Any uplift above it is not a mix effect, and
    the flat +10% this model used until 10/28 was above it at an 8%
    deflected share - which is how the error was found.
    """
    for share in (0.0, 0.02, 0.08, 0.20, 0.50, 0.90):
        ceiling = automation.mix_uplift_ceiling(share)
        for easy in (0.0, 0.1, 0.5, 0.9, 1.0):
            mix = automation.mix_uplift(share, easy)
            assert mix <= ceiling + 1e-12, (share, easy)
            assert mix >= -1e-12
        # The ceiling IS the free-items case, and it leaves workload flat.
        assert automation.mix_uplift(share, 0.0) == pytest.approx(ceiling)
        assert (1 - share) * (1 + ceiling) == pytest.approx(1.0, rel=1e-12)
        # Items worth exactly the average shift nothing.
        assert automation.mix_uplift(share, 1.0) == pytest.approx(0.0, abs=1e-12)

    # At the 7.1 defaults: 4.35% against an 8.70% ceiling.
    share = automation.deflected_share(ai_on)
    assert automation.deflection_mix_uplift(ai_on) == pytest.approx(0.043478, abs=1e-6)
    assert automation.mix_uplift_ceiling(share) == pytest.approx(0.086957, abs=1e-6)
    assert automation.deflection_mix_uplift(ai_on) < automation.mix_uplift_ceiling(share)

    # Removing everything is refused rather than returning nonsense.
    with pytest.raises(ValueError, match="cannot remove"):
        automation.mix_uplift(1.0, 0.5)


def test_the_supply_mix_uplift_follows_the_share_actually_removed(ai_on, sites):
    """Not the nominal target: the ramp and uptime mean the two differ.

    In the first live week the machine is at 25% ramp and 98% uptime, so it
    takes about 2% of the items rather than 8% - and the mix uplift on what
    is left has to be the 2% figure, or the model would charge a full-ramp
    penalty for a quarter-ramp machine.
    """
    work = work_volume.build(ai_on, sites)["work"]
    table = automation.build(ai_on, sites, work)
    go_live = automation.go_live_week_index(ai_on)
    first = table[(table["week_index"] == go_live) & (table["work_type"] == "W1")]
    assert len(first) == 1
    fraction = float(first["automated_fraction"].iloc[0])
    assert 0 < fraction < automation.target_share(ai_on)

    ramped = automation.supply_handle_time_multiple(
        "W1", go_live, ai_on, removed_share=fraction
    )
    nominal = automation.supply_handle_time_multiple("W1", go_live, ai_on)
    assert ramped < nominal, "a quarter-ramp machine must not be charged full"
    easy = ai_on["automation"]["easy_handle_time_relative"]
    assert ramped == pytest.approx(1 + automation.mix_uplift(fraction, easy))


# ---------------------------------------------------------------------------
# Human-in-the-loop work does not weight turnaround (spec 4.4, 10/27)
# ---------------------------------------------------------------------------


def test_loop_work_carries_hours_but_not_turnaround(baseline, sites, ai_on):
    """Spec 4.4, decided 10/27: each item counts once.

    A QA look at an automated intake item is part of THAT item's journey,
    not a second wait on top of it. Counted as its own queue line it let
    8% of the volume in at a fifth of the inventory - 0.216 days against
    the human line's 0.595 at intake - and flattered the stage average.

    The hours and the cost still land on the role. Only the turnaround
    weighting changes.
    """
    work = work_volume.build(ai_on, sites)["work"]
    required = backlog.required_hours(ai_on, sites, work)
    queues = backlog.build_queues(ai_on, sites, required["hours"])

    assert "work_kind" in queues.columns
    kinds = set(queues["work_kind"])
    assert kinds == {"human", "exception", "qa_sample", "sign_off"}

    # The loop lines are real queue lines with real hours.
    loop = queues[queues["work_kind"] != "human"]
    assert len(loop) > 0
    assert loop["required_hours"].sum() > 0

    # And they are filtered out of the turnaround path.
    on_path = backlog.on_the_turnaround_path(queues)
    assert set(on_path["work_kind"]) == {"human"}
    assert len(on_path) < len(queues)

    # The filter has to BITE, which is the half I got wrong first time:
    # dropping the loop lines by hand must change nothing, AND leaving
    # them in by force must change something. Without the second half the
    # test passes on a frame that never carried the lines at all.
    week = queues["week_index"].max()
    with_loop = queues[queues["week_index"] == week]
    _stages, _steps, filtered = backlog.turnaround(with_loop, ai_on)
    _s2, _b2, by_hand = backlog.turnaround(
        with_loop[with_loop["work_kind"] == "human"], ai_on
    )
    assert filtered == pytest.approx(by_hand, rel=1e-12)

    forced = with_loop.drop(columns="work_kind")
    _s3, _b3, unfiltered = backlog.turnaround(forced, ai_on)
    assert unfiltered < filtered - 1e-6, (
        "loop lines must drag the stage averages down when counted"
    )
    # In any ONE week the drag is small - about 0.01 days here, since the
    # loop lines are 8% of the volume at a fraction of the inventory. It
    # is the rolling projection that compounds it: build_plan carries each
    # queue forward, so a stage average that starts low keeps a backlog
    # low, and over 52 weeks the gap reached 0.78 days.
    assert filtered - unfiltered < 0.10


def test_the_baseline_turnaround_path_is_unfiltered(baseline, sites):
    """With automation off there is no work_kind column and nothing to drop."""
    work = work_volume.build(baseline, sites)["work"]
    required = backlog.required_hours(baseline, sites, work)
    queues = backlog.build_queues(baseline, sites, required["hours"])
    assert "work_kind" not in queues.columns
    assert backlog.on_the_turnaround_path(queues) is queues


def test_the_ai_runs_report_a_turnaround_without_the_loop_lines(result):
    """End to end, through the comparison table, in BOTH views (4.4, 10/28).

    This test has carried the history of three decisions, and it is worth
    keeping them in order because each one moved the number:

      before 10/27  4.01 days against the baseline's 4.67 - an apparent
        0.66-day improvement that was the human-in-the-loop lines diluting
        the stage weighting. Not a shorter wait: the same wait, counted
        twice at a fifth of the inventory.

      10/27 to 10/28  each item counted once, which reversed the sign to
        4.84. Read as "automation makes turnaround worse", and reported as
        such - wrongly.

      10/28  the headline became the REQUESTER's view, and the sign
        reversed again to 4.59. The 4.84 was an artefact of the
        denominator: take 8% of the items out of the arrival rate and days
        of inventory rises about 5% whatever the service did.

    So the headline now IMPROVES on the baseline and the planner's view
    still worsens, and both are true of the same plan. The planner's
    number describes the queues a person staffs; the headline describes
    what a requester waits.
    """
    table = result["comparison"].set_index("scenario")
    base = table.loc["Baseline", "mean_projected_turnaround_days"]
    full = table.loc["AI and automation", "mean_projected_turnaround_days"]
    half = table.loc["AI and automation, 50%", "mean_projected_turnaround_days"]

    assert base == pytest.approx(4.671, abs=0.005)

    # The requester's view: better than the baseline, on less work.
    #
    # Repinned 10/28 from 4.595 after the queue_key fix - the handle time
    # was in the key the rolling backlog is stored under, so queues
    # restarted at target whenever the mix uplift moved their handle time.
    # See test_the_backlog_still_rolls_in_every_scenario.
    assert full < base
    assert full == pytest.approx(4.551, abs=0.01)
    # And fewer weeks past the goal, not more.
    assert (
        table.loc["AI and automation", "weeks_over_turnaround_goal"]
        <= table.loc["Baseline", "weeks_over_turnaround_goal"]
    )

    # The planner's view of the SAME run: worse, and by a lot. Keeping
    # both is the point of the decision - a model that reported only one
    # of these would mislead one of its two readers.
    planner = table.loc["AI and automation", "mean_human_queue_turnaround_days"]
    assert planner == pytest.approx(4.796, abs=0.01)
    assert planner > base
    assert table.loc["AI and automation", "weeks_over_goal_human_queue"] > (
        table.loc["AI and automation", "weeks_over_turnaround_goal"]
    )

    # Half the programme improves turnaround by about half as much, which
    # is what it should always have done.
    #
    # It read WORSE THAN THE BASELINE until the queue_key fix on 10/28 -
    # 4.82 against 4.67 - and that was entirely the broken backlog roll:
    # the half-realized run suffered 820 of its 3,519 week-to-week rolls
    # resetting to target, against the full run's 196. The "half a
    # programme is worse than none" finding was an artefact and is
    # withdrawn.
    assert half < base
    assert half > full
    assert half == pytest.approx(4.616, abs=0.01)
    # Monotone in realization now, in both views.
    assert base - half == pytest.approx((base - full) / 2, abs=0.02)


# ---------------------------------------------------------------------------
# Two queues on one role must not collide (found 10/28)
# ---------------------------------------------------------------------------


def test_the_backlog_still_rolls_in_every_scenario(result):
    """Found 10/28, and the worst bug this model has had.

    queue_key is what the rolling backlog is stored under, so every field
    in it has to be a property of the QUEUE and not of the week. It
    included the handle time - harmless until 5h, when the mix uplift
    started being priced from the share ACTUALLY removed, which moves with
    the ramp and with uptime.

    From then on a queue got a new key whenever its handle time changed,
    `state.get(key, row.target_backlog)` missed, and the queue silently
    RESTARTED at its target with its real position thrown away. No error,
    no warning: just a backlog model that stopped being a backlog model in
    exactly the runs it was being used to judge.

    It broke 196 of 3,519 week-to-week rolls in the full AI run and 820 in
    the half-realized one, which is why that run reported worse turnaround
    than doing nothing at all - a finding that was entirely an artefact.

    So: for every queue, in every scenario, this week's opening backlog
    must be last week's ending backlog. That is what "the backlog rolls"
    means, and nothing else in the suite was checking it.
    """
    ident = ["role", "scope", "work_type", "media"]
    checked = 0
    for name, built in result["runs"].items():
        q = backlog.on_the_turnaround_path(built["queues"]).copy()
        if "work_kind" in built["queues"].columns:
            ident_here = ident + ["work_kind"]
            q = built["queues"].copy()
        else:
            ident_here = ident
        q = q.sort_values(ident_here + ["week_index"])
        prev_end = q.groupby(ident_here)["ending_backlog"].shift(1)
        prev_week = q.groupby(ident_here)["week_index"].shift(1)
        consecutive = prev_end.notna() & (prev_week == q["week_index"] - 1)
        rolled = q[consecutive]
        assert len(rolled) > 0, name
        gap = (rolled["opening_backlog"] - prev_end[consecutive]).abs()
        assert gap.max() < 1e-6, (
            f"{name}: {int((gap > 1e-6).sum())} of {len(rolled)} rolls broken, "
            f"worst {gap.max():,.2f} items"
        )
        checked += len(rolled)
    assert checked > 50_000, "every scenario's queues must be covered"


def test_the_queue_key_holds_no_property_of_the_week(baseline, sites, ai_on):
    """The structural half of the same bug.

    Two things have to be true at once, and they pull against each other:

      the key must SEPARATE every line the projection carries, or two
        queues collide and one vanishes (the 10/28 collision bug);

      the key must not contain anything that CHANGES from week to week, or
        one queue becomes two and the roll breaks (this bug).

    The handle time fails the second test, and it turns out not to be
    needed for the first: work kind already separates the human line from
    the exception and sign-off lines, and the rest of the key is unique
    without it. That is what this test pins, in both runs that have loop
    work and in the baseline that has none.
    """
    for assumptions in (baseline, ai_on):
        work = work_volume.build(assumptions, sites)["work"]
        required = backlog.required_hours(assumptions, sites, work)
        queues = backlog.build_queues(assumptions, sites, required["hours"])
        keys = ["week_index", "role", "scope", "work_type", "media"]
        if "work_kind" in queues.columns:
            keys.append("work_kind")
        counts = queues.groupby(keys).size()
        assert (counts == 1).all(), (
            f"{int((counts > 1).sum())} queue identities are not unique "
            "without the handle time"
        )

    # And the key itself carries no handle time.
    Row = collections.namedtuple(
        "Row", "role scope work_type media work_kind handle_time_minutes"
    )
    common = dict(
        role="tier_1", scope="NETWORK", work_type="W1", media="",
        work_kind="human",
    )
    slow = Row(**common, handle_time_minutes=6.0)
    fast = Row(**common, handle_time_minutes=6.26)
    assert plan.queue_key(slow) == plan.queue_key(fast), (
        "the same queue at two handle times must be ONE queue"
    )

    # While the kinds still separate, which is the other half.
    exception = Row(
        role="tier_1", scope="NETWORK", work_type="W1", media="",
        work_kind="exception", handle_time_minutes=0.45,
    )
    assert plan.queue_key(slow) != plan.queue_key(exception)


def test_every_distinct_queue_survives_the_projection(baseline, sites, ai_on):
    """The bug this test exists for, stated as a count.

    build_plan keyed each queue by (role, scope, work type, medium). Once
    automation is on, the human line and the exception or sign-off line
    for the same work type sit on the SAME owning role - so they collided,
    the later row replaced the earlier one, and whole human queues
    vanished from the projection without any error.

    Every distinct line that goes in must come out, week by week.
    """
    work = work_volume.build(ai_on, sites)["work"]
    required = backlog.required_hours(ai_on, sites, work)
    expected = backlog.build_queues(ai_on, sites, required["hours"])
    roles = set(plan.back_office_roles(ai_on))
    plan_weeks, _mondays = supply.plan_window(ai_on)
    expected = expected[
        expected["role"].isin(roles) & expected["week_index"].isin(plan_weeks)
    ]

    built = plan.build_plan(ai_on, sites, work)["queues"]
    key = ["week_index", "role", "scope", "work_type", "media", "work_kind",
           "handle_time_minutes"]
    assert "work_kind" in built.columns

    # The projection carries a handle time per line; rebuild it from the
    # hours so the two frames can be compared on the same key.
    built = built.copy()
    built["handle_time_minutes"] = np.where(
        built["arrivals"] > 0, built["required_hours"] / built["arrivals"] * 60, 0.0
    )
    assert len(built) == len(expected), (
        f"{len(expected) - len(built)} queue lines lost in the projection"
    )

    # And the collision case specifically: a work type whose human line and
    # loop line share one role must show BOTH.
    together = built[built["work_type"] == "W1"]
    week = together["week_index"].max()
    tier_1 = together[
        (together["week_index"] == week) & (together["role"] == "tier_1")
    ]
    kinds = set(tier_1["work_kind"])
    assert "human" in kinds, "the human line was replaced by its own loop line"
    assert len(kinds) > 1, "W1 at tier_1 should carry loop work too"


def test_the_queue_key_separates_the_kinds(ai_on):
    """The key itself, on two rows that differ only by kind.

    A unit check beside the end-to-end one, so a future change to the key
    fails here with a one-line diagnosis rather than as a row count.
    """
    class Row:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    common = dict(
        role="tier_1", scope="NETWORK", work_type="W1", media="",
    )
    human = Row(**common, work_kind="human", handle_time_minutes=6.0)
    exception = Row(**common, work_kind="exception", handle_time_minutes=9.0)
    assert plan.queue_key(human) != plan.queue_key(exception)

    # With automation off there is no work_kind, and the key still works.
    plain = Row(**common, handle_time_minutes=6.0)
    assert plan.queue_key(plain) == plan.queue_key(human)


# ---------------------------------------------------------------------------
# The two turnaround views (spec 4.4, decided 10/28)
# ---------------------------------------------------------------------------


def test_the_requester_view_counts_every_request_once(baseline, sites, ai_on):
    """Spec 4.4, decided 10/28, and the reason the headline moved.

    Two views of the same backlog, differing only in the denominator:

      REQUESTER (headline): days of inventory over EVERY request that
        arrived, with an automated item counted as completed by the
        machine. This is the number that answers "are requests going out
        on time".

      HUMAN QUEUE (planner): the same backlog over only the arrivals a
        person handled.

    The test is that the arrival rates really are the two different
    things, and that the human-queue view is the one missing the automated
    items - 8% of the eligible volume, which inflates its days of
    inventory by about 5% whatever the service did.
    """
    built = plan.build_all(ai_on, sites)
    turnaround = built["turnaround"]
    assert "turnaround_days" in turnaround.columns
    assert "turnaround_days_human_queue" in turnaround.columns

    # Rebuild both denominators from the queue rows and check which is
    # which. The human rows carry arrivals NET of what the machine took.
    human_rows = backlog.on_the_turnaround_path(built["queues"])
    in_path = {w["code"] for w in ai_on["work_types"] if w.get("in_turnaround")}
    human_rows = human_rows[human_rows["work_type"].isin(in_path)]

    work = built["prepared"]["work"]
    auto = built["prepared"]["automated"]
    assert auto is not None and len(auto) > 0
    shares = automation.automated_fraction_by_week(auto)
    assert shares, "automation is on, so there are shares to put back"

    human_arrivals = human_rows["arrivals"].sum()
    full_arrivals = sum(
        arrivals / (1.0 - shares.get((week, code), 0.0))
        for arrivals, week, code in zip(
            human_rows["arrivals"], human_rows["week_index"],
            human_rows["work_type"],
        )
    )
    assert full_arrivals > human_arrivals
    # Every request counted once: the gap IS the automated volume on the
    # request path, and nothing is counted twice.
    assert full_arrivals - human_arrivals == pytest.approx(
        sum(
            row.automated_items
            for row in auto.itertuples()
            if row.work_type in in_path
        ),
        rel=1e-6,
    )

    # And the headline is the better of the two, because its denominator
    # is the bigger one over the same backlog.
    plan_weeks = set(supply.plan_window(ai_on)[0])
    live = turnaround[turnaround["week_index"].isin(plan_weeks)]
    assert live["turnaround_days"].mean() < live[
        "turnaround_days_human_queue"
    ].mean()


def test_the_human_queue_view_excludes_the_automated_items(baseline, sites, ai_on):
    """The other half of the same check, from the loop lines' side.

    The human-queue view must exclude the automated items from BOTH ends:
    they are not in its backlog, because a machine finished them, and they
    are not in its arrival rate either. The loop lines (5.5) are excluded
    too, which is the 10/27 "each item counts once" rule - an exception on
    an automated intake item is part of that item's journey, not a second
    wait on top of it.
    """
    built = plan.build_all(ai_on, sites)
    queues = built["queues"]
    assert "work_kind" in queues.columns
    kept = backlog.on_the_turnaround_path(queues)
    assert set(kept["work_kind"]) == {backlog.HUMAN_WORK_KIND}
    assert len(kept) < len(queues), "loop lines must be dropped"

    # The human rows' items are net: human + automated adds back to the
    # full volume, with nothing lost and nothing double counted.
    hours = built["prepared"]["required"]
    auto = built["prepared"]["automated"]
    plan_weeks = set(supply.plan_window(ai_on)[0])
    human_items = hours[
        (hours["week_index"].isin(plan_weeks))
        & (hours.get("work_kind", backlog.HUMAN_WORK_KIND) == backlog.HUMAN_WORK_KIND)
    ]["items"].sum()
    automated_items = auto[auto["week_index"].isin(plan_weeks)][
        "automated_items"
    ].sum()

    plain = work_volume.build(baseline, sites)["work"]
    handled_elsewhere = backlog.onsite_work_types(baseline)
    total = plain[
        (plain["week_index"].isin(plan_weeks))
        & (~plain["work_type"].isin(handled_elsewhere))
    ]["work_items"].sum()
    assert human_items + automated_items == pytest.approx(total, rel=1e-9)


def test_the_baseline_turnaround_has_one_view_only(baseline, sites):
    """No machine, no second view - and so no change to the baseline file.

    With automation off the two denominators are the same number, so a
    second column would carry nothing but a duplicate. The byte-identity
    of the baseline's turnaround output is the whole reason this matters.

    The scenario-table side of this - the baseline's planner cells coming
    back empty - is checked in
    test_the_scenario_table_carries_both_views.
    """
    built = plan.build_all(baseline, sites)
    turnaround = built["turnaround"]
    assert "turnaround_days" in turnaround.columns
    assert "turnaround_days_human_queue" not in turnaround.columns
    # The automation plan is built either way - it is how the model proves
    # nothing was automated - but it is EMPTY, so there are no shares to
    # put back and no second view to report.
    assert len(built["prepared"]["automated"]) == 0
    assert automation.automated_fraction_by_week(
        built["prepared"]["automated"]
    ) == {}


def test_the_scenario_table_carries_both_views(result):
    """End to end: the headline is the requester's, the planner's is named.

    And the AI runs show the gap that made the decision: measured on human
    queues the full programme looks worse than the baseline in many more
    weeks than the requester ever experiences.
    """
    table = result["comparison"].set_index("scenario")
    assert "mean_projected_turnaround_days" in table.columns
    assert "mean_human_queue_turnaround_days" in table.columns

    base = table.loc["Baseline"]
    full = table.loc["AI and automation"]

    # The baseline has no second view, so its planner cells are empty.
    assert pd.isna(base["mean_human_queue_turnaround_days"])

    # The AI run has both, and the planner's is the pessimistic one.
    assert full["mean_human_queue_turnaround_days"] > full[
        "mean_projected_turnaround_days"
    ]
    assert full["weeks_over_goal_human_queue"] > full["weeks_over_turnaround_goal"]


# ---------------------------------------------------------------------------
# AI targeted (spec 7, added 10/28)
# ---------------------------------------------------------------------------

# Re-picked 10/28 after the easy-item decision (spec 6.6). Was six rows:
# W1, W5, W7, W8, W12, W14. Valuing the items a machine takes at the
# easy-item handle time rather than the average halved the by-hand saving
# on every row, and only the two highest-volume ones still clear a $10,000
# setup. Confirmed 11/03 against the live ranking, and kept at two rows by
# decision on the same date (spec 7, "Combined"): W14 crosses over at +15%
# volume but frees only QA hours already inside paid time.
TARGETED = ["W5", "W8"]


@pytest.fixture(scope="module")
def ai_targeted(baseline):
    scenario = scenarios.load_scenario(
        scenarios.scenario_dir(baseline) / "ai_targeted.yaml"
    )
    return scenarios.apply_scenario(scenario, baseline)


def test_only_the_listed_work_types_are_automated(ai_targeted, sites):
    """Spec 7's "AI targeted": the rows that pay back after fixed costs.

    Read off the Next Dollar ranking in the AI 100% run as the rows whose
    volume-weighted saving after fixed costs is positive. Since the 10/28
    easy-item decision that is two rows, not six (figures re-measured
    11/03; the list is unchanged):

        W8  digital retrieval        +$1.58 an item, 156,532 items
        W5  authorization validation +$0.378 an item, 107,119 items

    Both survive on VOLUME - they are the two highest-volume eligible rows
    with a handle time worth automating - which is the finding: at $10,000
    of setup a work type, automation is a volume play.

    W14 quality review misses by seven cents (-$0.069) on the
    second-highest volume in the network, so it is the row to revisit if
    the setup fee ever moves - and it DOES cross over at +15% volume, for
    no money, which test_combined.py measures.

    Nothing here asserts those per-item figures: they are the live output
    of a ranking that moves with volume, the setup fee and the easy-item
    relative time, so pinning them in a comment is a note and pinning
    them in an assertion would be a trap. What is asserted is the LIST,
    which is the decision.
    """
    assert automation.targeted_work_types(ai_targeted) == TARGETED

    aimed = {
        code
        for code in automation.eligible_codes(ai_targeted)
        if automation.row_target_share(code, ai_targeted) > 0
    }
    assert aimed == set(TARGETED)

    work = work_volume.build(ai_targeted, sites)["work"]
    table = automation.build(ai_targeted, sites, work)
    automated = table[table["automated_items"] > 0]
    assert set(automated["work_type"]) == set(TARGETED)
    # And the rejected rows are present in the table at zero, not missing -
    # a row nobody automated still has eligible items.
    idle = table[table["automated_items"] <= 0]
    assert set(idle["work_type"]) >= {"W2", "W4", "W16", "W18"}
    assert idle["platform_cost"].abs().max() == 0.0
    assert idle["setup_cost"].abs().max() == 0.0


def test_the_targeted_run_pays_only_the_listed_setups(ai_targeted, sites):
    """Which is the point: setup was 75.7% of the full run's spend.

    $7,692 a work type a year, whether the row automates nine thousand
    items or a hundred and ten. Two rows instead of nineteen is $15,385 of
    setup instead of $146,154 - and that, not the per-item fee, is what
    turns the programme from a loss into a saving.
    """
    work = work_volume.build(ai_targeted, sites)["work"]
    table = automation.build(ai_targeted, sites, work)
    plan_weeks, _mondays = supply.plan_window(ai_targeted)
    live = table[table["week_index"].isin(plan_weeks)]

    weeks_live = live[live["automated_items"] > 0]["week_index"].nunique()
    assert weeks_live == 40, "live from plan week 13"
    weekly_setup = 10000 / 52
    assert live["setup_cost"].sum() == pytest.approx(
        weekly_setup * 40 * len(TARGETED), rel=1e-9
    )
    # The platform fee is still one shared $500 a week, not six.
    assert live["platform_cost"].sum() == pytest.approx(500 * 40, rel=1e-9)


def test_the_targeted_run_has_no_deflection(ai_targeted, baseline, sites):
    """Spec 7, added 10/28: deflection off.

    It is a net loss at any realization by construction, so a run about
    doing only what pays has no business keeping it.
    """
    assert automation.deflection_enabled(ai_targeted) is False
    assert automation.deflected_share(ai_targeted) == 0.0

    base = work_volume.build(baseline, sites)["work"]
    targeted = work_volume.build(ai_targeted, sites)["work"]
    for code in ("W19", "W20", "W21"):
        was = base[base["work_type"] == code]["work_items"].sum()
        now = targeted[targeted["work_type"] == code]["work_items"].sum()
        assert now == pytest.approx(was, rel=1e-12), code

    # And no demand-side uplift reaches the live queues either.
    from_week = automation.deflection_from_week(ai_targeted)
    for code in ("W20", "W21"):
        assert automation.deflection_handle_time_multiple(
            code, from_week, ai_targeted
        ) == 1.0


def test_the_targeted_run_leaves_the_first_twelve_weeks_alone(
    ai_targeted, baseline, sites
):
    """Spec 7: week 13. Bit for bit before it, on both sides of the chain."""
    from_week = automation.go_live_week_index(ai_targeted)
    assert from_week == baseline["calendar"]["history_weeks"] + 12

    base_work = work_volume.build(baseline, sites)["work"]
    work = work_volume.build(ai_targeted, sites)["work"]
    np.testing.assert_array_equal(
        base_work["work_items"].to_numpy(), work["work_items"].to_numpy()
    )

    base_req = backlog.required_hours(baseline, sites, base_work)["hours"]
    req = backlog.required_hours(ai_targeted, sites, work)
    human = req["human"]
    early = human["week_index"] < from_week
    base_early = base_req["week_index"] < from_week
    np.testing.assert_allclose(
        base_req.loc[base_early, "required_hours"].to_numpy(),
        human.loc[early, "required_hours"].to_numpy(),
        rtol=0,
        atol=0,
    )
    # No loop work before go-live.
    loop = req["human_in_the_loop"]
    assert (loop["week_index"] >= from_week).all()


def test_targeting_beats_automating_everything_on_money_not_on_hours(
    ai_targeted, ai_on, sites
):
    """Does doing only what pays turn the programme positive?

    Compared on hours and on spend, which are the two questions - and
    since the 10/28 uplift split they have DIFFERENT answers.

    Under the flat +10% uplift, automating a low-volume row charged that
    10% against all of its human hours while removing almost nothing, so
    leaving 13 such rows alone saved hours. The mix uplift is calculated
    from the share actually removed, so a row that gives up few items now
    earns a correspondingly small uplift - there is no longer a penalty to
    avoid, and automating MORE rows removes MORE hours.

    So the full programme now wins on hours and the targeted one wins on
    money, by a margin that is not close: the extra hours cost far less
    than the setup fees they take to buy.
    """
    plan_weeks, _mondays = supply.plan_window(ai_on)

    def totals(assumptions):
        work = work_volume.build(assumptions, sites)["work"]
        req = backlog.required_hours(assumptions, sites, work)
        auto = req["automation"]
        live = auto[auto["week_index"].isin(plan_weeks)]
        hours = req["hours"]
        return {
            "hours": hours[hours["week_index"].isin(plan_weeks)][
                "required_hours"
            ].sum(),
            "spend": live["total_cost"].sum(),
            "setups": live["setup_cost"].sum(),
        }

    full, targeted = totals(ai_on), totals(ai_targeted)
    assert targeted["spend"] < full["spend"]
    assert targeted["setups"] < full["setups"]
    assert targeted["setups"] == pytest.approx(
        full["setups"] * len(TARGETED) / 19, rel=1e-6
    )

    # Hours: the full programme wins, because 17 more rows give up items
    # and none of them now carries an uplift it did not earn.
    assert full["hours"] < targeted["hours"]
    extra_hours = targeted["hours"] - full["hours"]
    assert extra_hours == pytest.approx(1_348, rel=0.02)

    # Money: and those extra hours are bought dearly. The targeted run
    # spends $150K less for 1,348 more hours of human work - about $111 an
    # hour, against a productive hour that costs under $40. So targeting is
    # the better programme even though it saves fewer hours, and since the
    # 10/28 easy-item decision it is the only one that saves money at all.
    saving = full["spend"] - targeted["spend"]
    assert saving == pytest.approx(149_844, rel=0.01)
    assert saving / extra_hours > 100
    assert saving / extra_hours > 2.5 * plan.productive_hour_cost("tier_1", ai_on)
