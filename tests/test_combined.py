"""Spec Section 9, Session 5: the Combined scenario (spec Section 7).

Task 5l. Combined is the run with no new mechanism in it at all - 15%
more volume from week 1 and AI targeted from week 13, both already built
and both already tested. So the checks here are not about machinery, they
are about the two things a combination can quietly get wrong:

    THE DATES. Two levers with two start weeks, in one scenario that
      holds a single starts week. The volume has to land from week 1 and
      the machine from week 13, which means the first twelve weeks must
      be Volume +15% EXACTLY - demand arrives a quarter before the relief
      does, and that quarter is where the backlog and the hiring trigger
      live. Dating both from week 13 would silently delete the hardest
      part of the year.

    THE ARITHMETIC OF THE ANSWER. "Can automation absorb the growth" is
      a ratio, and a ratio has two ways to flatter itself: measure the
      saving on a basis the money never reaches, or leave the machine's
      own invoice out. Both are checked, and both change the verdict.

The diff test and the baseline pins are the shared ones in
test_scenarios.py; what is pinned here is the finding.
"""

import numpy as np
import pandas as pd
import pytest

from capacity_model import automation, config, scenarios, supply


@pytest.fixture(scope="module")
def baseline():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def report(baseline, sites):
    return scenarios.combined_report(baseline, sites)


@pytest.fixture(scope="module")
def table(report):
    return report["comparison"]


# ---------------------------------------------------------------------------
# Two levers, two dates
# ---------------------------------------------------------------------------


def test_the_volume_starts_in_week_one_and_the_machine_in_week_thirteen(baseline):
    """Spec 7's table: "Week 1 / 13", and they come from different inputs.

    The scenario's starts week drives the demand multiplier; the machine
    has its own go-live week in the automation block. That separation is
    what lets one scenario hold two dates, and it is worth pinning
    because folding the machine onto the scenario's starts week would be
    an easy and invisible simplification.
    """
    merged = scenarios.apply_scenario(
        scenarios.load_scenario(scenarios.scenario_dir(baseline) / "combined.yaml"),
        baseline,
    )
    assert merged["scenario"]["starts_week"] == 1
    assert merged["scenario"]["demand_multiplier"] == 1.15
    assert merged["automation"]["go_live_week"] == 13

    history = merged["calendar"]["history_weeks"]
    assert config.scenario_starts_week(merged) == history
    assert automation.go_live_week_index(merged) == history + 12


def test_the_first_twelve_weeks_are_volume_plus_fifteen_exactly(report):
    """The quarter before the relief arrives, to the hour.

    This is the Combined equivalent of every other scenario's
    "weeks before the starts week match the baseline" test, except the
    reference is Volume +15% rather than the baseline - because the
    volume is already running.
    """
    go_live = automation.go_live_week_index(report["runs"]["combined"]["assumptions"])

    def before(key):
        required = report["runs"][key]["prepared"]["required"]
        mine = required[required["week_index"] < go_live]
        return (
            mine.groupby(["week_index", "site", "role", "work_type"])["required_hours"]
            .sum()
            .sort_index()
        )

    combined, volume = before("combined"), before("volume_only")
    assert len(combined) and len(volume)
    pd.testing.assert_series_equal(combined, volume)

    # And from go-live onward they must differ, or the test above would
    # be passing because the machine never started.
    def after(key):
        required = report["runs"][key]["prepared"]["required"]
        mine = required[required["week_index"] >= go_live]
        return float(mine["required_hours"].sum())

    assert after("combined") < after("volume_only")


def test_the_requests_are_the_volume_run_s_requests(table):
    """Supply-side automation moves HOURS, never items (spec 5.4).

    Combined must arrive at exactly the same request count as Volume
    +15%: a machine that did the work is not a request that never came.
    The full-programme run is the control - it switches deflection on,
    which IS demand-side, so its requests have to differ.
    """
    volume = table.loc["volume_only", "plan_year_requests"]
    assert table.loc["combined", "plan_year_requests"] == pytest.approx(volume)
    assert table.loc["combined_ai_100_no_deflection", "plan_year_requests"] == (
        pytest.approx(volume)
    )
    base = table.loc["baseline", "plan_year_requests"]
    assert volume / base == pytest.approx(1.15, rel=1e-3)


# ---------------------------------------------------------------------------
# Direction checks (spec Section 9)
# ---------------------------------------------------------------------------


def test_adding_automation_to_the_volume_never_makes_anything_worse(table):
    """Spec Section 9's direction check, pointed the other way.

    More volume never lowers required staff, and automation on top of it
    never raises them. Checked on every run that adds a machine, because
    a sign error in the 5.5 return terms would show up here as a machine
    that created work.
    """
    volume = table.loc["volume_only"]
    base = table.loc["baseline"]
    for key in (
        "combined",
        "combined_ai_100",
        "combined_ai_100_no_deflection",
        "combined_repicked",
    ):
        row = table.loc[key]
        for column in (
            "network_required_fte",
            "back_office_required_fte",
            "overtime_hours",
            "uncovered_hours",
            "mean_projected_turnaround_days",
        ):
            assert row[column] <= volume[column] + 1e-9, (key, column)
            # And never better than the baseline it is 15% above.
            assert row[column] >= base[column] - 1e-9, (key, column)


def test_the_peak_gap_is_before_the_machine_arrives_so_it_cannot_help(
    report, table
):
    """Why the hiring plan is identical in all four runs.

    The worst week of the year is plan week 10, and the machine goes live
    in week 13. The hiring trigger reads the peak (spec 6.5), so no
    amount of automation starting in week 13 can change a single class -
    which is the whole reason Combined hires the same 23 people as Volume
    +15% does.

    A lever that arrives after the peak cannot cut the peak. Obvious said
    out loud, and invisible in a table of annual averages.
    """
    volume = table.loc["volume_only"]
    history = report["runs"]["combined"]["assumptions"]["calendar"]["history_weeks"]

    gap = report["runs"]["volume_only"]["gap_fte"]
    worst = gap.sort_values("short_fte", ascending=False).iloc[0]
    peak_plan_week = int(worst["week_index"]) - history + 1
    assert peak_plan_week == 10
    assert peak_plan_week < 13

    for key in ("combined", "combined_ai_100", "combined_repicked"):
        assert table.loc[key, "peak_gap_fte"] == pytest.approx(volume["peak_gap_fte"])
    assert table.loc["combined", "hired_people"] == volume["hired_people"] == 23


# ---------------------------------------------------------------------------
# How much of the rise it absorbs
# ---------------------------------------------------------------------------


def test_the_targeted_machine_absorbs_about_a_twentieth_of_the_rise(report):
    """The task's question, answered as a ratio (measured 11/02).

    15% more volume needs 15.43 more FTE. The targeted machine removes
    0.74 of them - 4.8%. Pinned because it IS the answer, and because the
    gap between 4.8% and the 8% share the machine automates is the thing
    worth explaining: the machine takes 8% of the eligible items on two
    work types, and gives part of it back as exceptions and QA (spec
    5.5), and two work types are not the network.
    """
    fte = report["rise"]["network_required_fte"]
    assert fte["rise_from_volume"] == pytest.approx(15.4265, rel=1e-3)

    targeted = fte["absorbed"]["combined"]
    assert targeted["absorbed"] == pytest.approx(0.7392, rel=1e-3)
    assert targeted["share_of_rise"] == pytest.approx(0.0479, rel=1e-2)

    # The full programme absorbs three times as much, on nineteen rows
    # plus deflection - which is the comparison the spec asks for.
    full = fte["absorbed"]["combined_ai_100"]
    assert full["share_of_rise"] == pytest.approx(0.1499, rel=1e-2)
    assert full["absorbed"] > targeted["absorbed"]

    # Every one of them absorbs a part, and none of them absorbs it all.
    for key, entry in fte["absorbed"].items():
        assert 0.0 < entry["share_of_rise"] < 0.25, key


def test_breadth_and_deflection_are_separated_rather_than_netted(report):
    """"AI 100%" differs from "AI targeted" in two ways, not one.

    Breadth is supply-side and buys nineteen setup fees; deflection is
    demand-side and removes the contact altogether. Reporting one number
    for the gap would credit the wrong lever, so the report runs the full
    programme with deflection off and names both.

    Each is reported on labour and on all-in, and the PAIR is the
    finding (11/03). On the labour line breadth looks like the bigger
    prize - $51,217 against deflection's $27,511 - and all-in it is the
    only one of the two that loses money, because seventeen more rows buy
    seventeen more setup fees. Deflection reads the same either way: a
    contact that never arrives has no item to pay a fee on.

    So the cheaper lever is the one with no machine attached to it, and
    the labour line on its own would have pointed the other way.
    """
    split = report["breadth_against_deflection"]
    breadth, deflection = split["breadth_19_rows_vs_2"], split["deflection"]

    assert breadth["labour"] == pytest.approx(51217.37, rel=1e-4)
    assert deflection["labour"] == pytest.approx(27510.98, rel=1e-4)
    assert breadth["labour"] > deflection["labour"] > 0

    # Deflection is demand-side: it adds nothing to the invoice.
    assert deflection["machine"] == pytest.approx(0.0)
    assert deflection["all_in"] == pytest.approx(deflection["labour"])

    # Breadth is supply-side, and the invoice is bigger than the saving.
    assert breadth["machine"] < 0
    assert breadth["all_in"] == pytest.approx(-101487.90, rel=1e-4)
    assert breadth["all_in"] < 0 < deflection["all_in"]

    # Deflection touches the counter and phones; breadth does not.
    table = report["comparison"]
    assert table.loc["combined_ai_100", "onsite_required_fte"] < table.loc[
        "combined_ai_100_no_deflection", "onsite_required_fte"
    ]
    assert table.loc["combined_ai_100", "back_office_required_fte"] == pytest.approx(
        table.loc["combined_ai_100_no_deflection", "back_office_required_fte"]
    )


# ---------------------------------------------------------------------------
# The all-in cost, which reverses the answer
# ---------------------------------------------------------------------------


def test_the_headline_cost_is_all_in_with_both_halves_beside_it(report, baseline):
    """Spec 6.4, decided 11/03, and it is the easiest figure to misread.

    6.4's cost formula is LABOUR: regular pay plus overtime. The
    machine's own invoice - per item, setup, platform - lives in the
    automation table, and until 11/03 it was nowhere in the comparison,
    so an automation scenario's headline was its saving with none of its
    cost. Now all-in is the headline column and the two halves sit beside
    it, pinned here against the runs they came from.
    """
    table = report["comparison"]
    for key, built in report["runs"].items():
        labour = float(built["role_cost"]["total_cost"].sum())
        spend = report["machine_spend"][key]["total"]
        assert report["all_in_cost"][key]["labour"] == pytest.approx(labour)
        assert report["all_in_cost"][key]["all_in"] == pytest.approx(labour + spend)
        # And the comparison table agrees with the report.
        assert table.loc[key, "labour_cost"] == pytest.approx(labour)
        assert table.loc[key, "machine_cost"] == pytest.approx(spend)
        assert table.loc[key, "total_cost"] == pytest.approx(labour + spend)
        # No automation, no invoice.
        if not automation.enabled(built["assumptions"]):
            assert spend == 0.0
            assert table.loc[key, "total_cost"] == pytest.approx(
                table.loc[key, "labour_cost"]
            )
        else:
            assert spend > 0.0
            assert table.loc[key, "total_cost"] > table.loc[key, "labour_cost"]


def test_the_machines_invoice_is_the_same_order_as_the_labour_it_saves(report):
    """The finding, and the reason the verdict turns over (measured 11/02).

    On top of +15% volume the targeted machine saves 9,788 of labour and
    costs 44,465 to run, so all-in it is about 34,700 WORSE than doing
    nothing. The full programme saves more labour and costs far more:
    88,516 against 197,170, about 108,700 worse.
    """
    against = report["against_volume_only"]

    targeted = against["combined"]
    assert targeted["labour"] == pytest.approx(-9787.95, rel=1e-4)
    assert targeted["machine"] == pytest.approx(44465.0, rel=1e-3)
    assert targeted["all_in"] == pytest.approx(34677.0, rel=1e-3)
    assert targeted["all_in"] > 0

    full = against["combined_ai_100"]
    assert full["labour"] < targeted["labour"]  # saves more labour
    assert full["all_in"] > targeted["all_in"]  # and still loses more money
    assert full["all_in"] == pytest.approx(108654.0, rel=1e-3)

    # Every combined run is worse all-in than the volume rise alone.
    for key, entry in against.items():
        if key.startswith("combined"):
            assert entry["all_in"] > 0, key


def test_the_same_machine_does_pay_at_the_lower_volume(report):
    """Which is what makes the combined verdict a finding and not a bug.

    AI targeted on its own IS worth doing: 57,360 of labour against a
    43,280 invoice, about 14,080 to the good. The identical machine loses
    money on top of +15% volume, and the difference is not the machine -
    it is what the labour saving lands on. At baseline volume it cancels
    two hiring classes (8 hires against 6). At +15% the plan is already
    hiring 23 people and 0.74 FTE cancels none of them, so the saving
    collapses to overtime.
    """
    against = report["against_baseline"]
    alone = against["ai_targeted_only"]
    assert alone["labour"] == pytest.approx(-57360.0, rel=1e-3)
    assert alone["machine"] == pytest.approx(43280.0, rel=1e-3)
    assert alone["all_in"] == pytest.approx(-14080.0, rel=1e-2)
    assert alone["all_in"] < 0

    table = report["comparison"]
    assert table.loc["baseline", "hired_people"] == 8
    assert table.loc["ai_targeted_only", "hired_people"] == 6
    assert table.loc["volume_only", "hired_people"] == 23
    assert table.loc["combined", "hired_people"] == 23


# ---------------------------------------------------------------------------
# The list, re-picked at the higher volume
# ---------------------------------------------------------------------------


def test_the_targeted_list_would_be_longer_at_the_higher_volume(report):
    """Setup is FIXED and the saving SCALES, so the list moves with volume.

    A setup fee spread over 15% more items is 13% less per item, so every
    row's after-fixed saving improves by about twenty cents - and W14
    quality review, which misses by seven cents at baseline volume on the
    second-highest item count in the network, crosses over.

    The scenario file keeps the list Siri decided it builds on. This is
    the finding that the list is an input with a volume attached to it,
    not a property of the work types.
    """
    repick = report["repick"]
    assert repick["list_in_the_scenario"] == ["W5", "W8"]
    assert repick["pays_at_baseline_volume"] == ["W5", "W8"]
    assert repick["would_pay_at_plus_15"] == ["W14", "W5", "W8"]
    assert repick["newly_positive"] == ["W14"]

    before = repick["ranking_at_baseline_volume"].set_index("work_type")
    after = repick["ranking"].set_index("work_type")
    assert before.loc["W14", "savings_per_item_after_fixed"] == pytest.approx(
        -0.0692, rel=1e-2
    )
    assert after.loc["W14", "savings_per_item_after_fixed"] == pytest.approx(
        0.1113, rel=1e-2
    )

    # The mechanism: the fixed cost per item falls by about 13% on every
    # row, and the human cost per item does not move at all.
    for code in ("W5", "W8", "W14", "W1", "W2"):
        assert after.loc[code, "plan_year_items"] / before.loc[
            code, "plan_year_items"
        ] == pytest.approx(1.15, rel=0.02)
        assert after.loc[code, "savings_per_item_after_fixed"] > before.loc[
            code, "savings_per_item_after_fixed"
        ]


def test_re_picking_absorbs_more_hours_and_saves_no_more_money(report):
    """The sharpest thing in this scenario, and it is a WFM point.

    Adding W14 removes another 0.23 FTE of requirement - so the ranking
    was right, the row does pay per item. And the labour cost does not
    move by a cent, because the hours it frees sit INSIDE the QA team's
    existing paid hours: nobody is laid off and no overtime hour is
    avoided, so there is no cash behind the hours.

    A per-item saving is only money if the hour it frees is on the
    margin. The Next Dollar ranking prices a human hour at its loaded
    rate, which quietly assumes that hour can be released - and on a held
    roster it cannot.
    """
    table = report["comparison"]
    assert table.loc["combined_repicked", "back_office_required_fte"] < table.loc[
        "combined", "back_office_required_fte"
    ]
    absorbed = report["rise"]["back_office_required_fte"]["absorbed"]
    assert absorbed["combined_repicked"]["absorbed"] > absorbed["combined"]["absorbed"]

    # Identical LABOUR cost, identical overtime, identical uncovered -
    # the all-in column is deliberately not in this list, because the
    # bigger invoice is exactly what makes re-picking worse.
    for column in ("labour_cost", "overtime_hours", "uncovered_hours"):
        assert table.loc["combined_repicked", column] == pytest.approx(
            table.loc["combined", column]
        )
    assert table.loc["combined_repicked", "total_cost"] > table.loc[
        "combined", "total_cost"
    ]

    # And a bigger invoice, so re-picking is worse all-in than not.
    against = report["against_volume_only"]
    assert against["combined_repicked"]["machine"] > against["combined"]["machine"]
    assert against["combined_repicked"]["all_in"] > against["combined"]["all_in"]
    assert against["combined_repicked"]["labour"] == pytest.approx(
        against["combined"]["labour"]
    )


def test_the_variants_are_derived_and_not_hard_coded(report, baseline):
    """The re-picked list comes from the ranking, not from a literal.

    Which matters because the ranking moves: if the setup fee, the
    volumes or the easy-item relative time are ever revised, this variant
    follows them instead of pinning yesterday's answer into the code.
    """
    pays = set(report["repick"]["would_pay_at_plus_15"])
    variant = scenarios.combined_repick_variant(pays, baseline)
    assert variant["overrides"]["automation"]["rates"]["only_work_types"] == sorted(
        pays
    )
    merged = scenarios.apply_scenario(variant, baseline)
    assert set(automation.targeted_work_types(merged)) == pays
    # And it is a variant, never a registered scenario - the registered
    # Combined run keeps the list the spec says it builds on.
    assert "combined_repicked" not in {
        s.stem for s in scenarios.list_scenarios(baseline)
    }


def test_the_machine_spend_is_restricted_to_the_plan_year(report):
    """Setup, platform and per-item fees, from go-live onward only.

    The machine does not exist in the 156 weeks of history, and a setup
    fee spread over a year is a plan-year cost. Checked by rebuilding the
    three components by hand from the scenario's own inputs.
    """
    built = report["runs"]["combined"]
    spend = report["machine_spend"]["combined"]
    cost = automation.settings(built["assumptions"])["cost"]

    plan_weeks, _mondays = supply.plan_window(built["assumptions"])
    table = built["prepared"]["automated"]
    live = table[
        table["week_index"].isin(plan_weeks) & (table["automated_items"] > 0)
    ]
    weeks_live = live["week_index"].nunique()
    assert weeks_live == 40, "live from plan week 13"

    rows = len(automation.targeted_work_types(built["assumptions"]))
    assert spend["setup_cost"] == pytest.approx(
        float(cost["setup_per_work_type"])
        / float(cost["setup_spread_weeks"])
        * weeks_live
        * rows
    )
    # One shared platform fee, not one per row (spec 5.4, 10/27).
    assert spend["platform_cost"] == pytest.approx(
        float(cost["platform_per_week"]) * weeks_live
    )
    assert spend["item_cost"] == pytest.approx(
        spend["automated_items"] * float(cost["ai_per_item"])
    )
    assert spend["total"] == pytest.approx(
        spend["item_cost"] + spend["setup_cost"] + spend["platform_cost"]
    )
    # Nothing from the history leaked in.
    history = table[~table["week_index"].isin(plan_weeks)]
    assert float(history["total_cost"].sum()) == 0.0


def test_the_baseline_is_untouched_by_any_of_it(report):
    """The baseline knows nothing about volume, machines or lists."""
    built = report["runs"]["baseline"]
    assert not automation.enabled(built["assumptions"])
    assert built["assumptions"]["scenario"]["demand_multiplier"] == 1.0
    assert report["machine_spend"]["baseline"]["total"] == 0.0
    assert "work_kind" not in built["prepared"]["required"].columns
    assert report["all_in_cost"]["baseline"]["all_in"] == pytest.approx(
        report["all_in_cost"]["baseline"]["labour"]
    )
