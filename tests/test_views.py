"""The browse view: does the page say what the model said? (7b)

Spec Section 8 and Section 9's Session 6-7 checks. The question these
tests exist to answer is narrow and it is the only one that matters for a
reporting layer: can a figure on the dashboard differ from the same
figure in the workbook? The tie-out
(test_every_scenario_s_cards_tie_to_the_comparison_table) checks all six
headline measures for every saved run against the comparison table to the
cent, and the AppTest below checks the numbers the page actually renders,
not just the ones the view layer would hand it.

These tests read the SAVED results. That the saved results match a fresh
run of the model is tests/test_api.py's job, and it re-runs the whole
model to prove it - so nothing here needs eight minutes.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

from capacity_model import api, export, scenarios, views

APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"

# A cent, as a tolerance. Money ties to the cent; the hours and FTE
# figures beside it are held to the same absolute bar, which is far
# tighter than anything those columns are reported at.
A_CENT = 0.005


@pytest.fixture(scope="module")
def results():
    return api.load_results()


@pytest.fixture(scope="module")
def dimensions():
    return api.dimensions()


@pytest.fixture(scope="module")
def comparison_rows(results):
    """The comparison table keyed by scenario name, as the workbook has it."""
    table = results["comparison"]
    return {row["scenario"]: row for _index, row in table.iterrows()}


def app_test():
    """The real dashboard, run headlessly.

    from_file needs an absolute path, because it resolves a relative one
    against the caller.
    """
    from streamlit.testing.v1 import AppTest

    if str(APP.parents[1] / "src") not in sys.path:
        sys.path.insert(0, str(APP.parents[1] / "src"))
    return AppTest.from_file(str(APP), default_timeout=300)


def widget(app, kind, key):
    """One widget by its key, never by its position.

    The second view was added above the run picker and every index-based
    lookup in these tests moved by one - which is a thing a test should
    not be able to notice.
    """
    found = [element for element in getattr(app, kind) if element.key == key]
    assert len(found) == 1, f"{kind} {key!r}: {len(found)} found"
    return found[0]


def run_picker(app):
    return widget(app, "radio", "run")


def view_picker(app):
    return widget(app, "radio", "view")


def scope_picker(app):
    return widget(app, "selectbox", "scope")


def role_picker(app):
    return widget(app, "multiselect", "roles")


# ---------------------------------------------------------------------------
# The tie-out: can the page and the workbook disagree?
# ---------------------------------------------------------------------------


def test_every_card_is_a_measure_the_workbook_also_reports(results):
    """A card the comparison table has no column for cannot be tied out."""
    measures = {key for _header, key in api.COMPARISON_MEASURES}
    for measure, _format, _direction in views.CARD_MEASURES:
        assert measure in measures, measure


def test_the_cards_carry_the_workbook_s_own_headers():
    """A measure is called one thing in this project, not two."""
    headers = {key: header for header, key in api.COMPARISON_MEASURES}
    for measure, _format, _direction in views.CARD_MEASURES:
        assert views.MEASURE_HEADERS[measure] == headers[measure]


def test_every_scenario_s_cards_tie_to_the_comparison_table(
    results, comparison_rows
):
    """THE TIE-OUT (7b item 6).

    Every card, for every saved run, against the row the workbook's
    Capacity Scenarios tab is built from - to the cent. If this passes,
    the dashboard cannot quote a headline figure the workbook does not
    carry, because there is only one place either of them gets it.
    """
    baseline_row = comparison_rows["Baseline"]
    for key, run in results["scenarios"].items():
        row = comparison_rows[run["name"]]
        for card in views.headline_cards(results, key):
            expected = float(row[card["measure"]])
            assert card["value"] == pytest.approx(expected, abs=A_CENT), (
                f"{run['name']}: {card['label']}"
            )
            if key == views.BASELINE_KEY:
                assert card["difference"] is None
                continue
            expected_difference = expected - float(
                baseline_row[card["measure"]]
            )
            assert card["difference"] == pytest.approx(
                expected_difference, abs=A_CENT
            ), f"{run['name']}: {card['label']} difference"


def test_the_page_renders_the_same_figures_it_was_handed(
    results, comparison_rows
):
    """The rendered cards, run by run, through the real app.

    The tie-out above checks the view layer. This checks the page: a
    label or a format could still go astray between the two, and a
    reader only ever sees the page.
    """
    app = app_test().run()
    assert not app.exception

    for key, run in results["scenarios"].items():
        run_picker(app).set_value(key).run()
        assert not app.exception, run["name"]
        rendered = {card.label: (card.value, card.delta) for card in app.metric}
        for card in views.headline_cards(results, key):
            value, delta = rendered[card["label"]]
            assert value == views.format_value(
                float(comparison_rows[run["name"]][card["measure"]]),
                card["format"],
            ), f"{run['name']}: {card['label']}"
            assert delta == card["difference_text"]


def test_the_cost_split_adds_up_to_the_all_in_cost(results):
    """Labour plus machine is the headline, or the chart is a decoration."""
    for key, run in results["scenarios"].items():
        split = views.cost_split(results, key)
        total = float(split.loc[run["name"]].sum())
        assert total == pytest.approx(
            float(run["summary"]["total_cost"]), abs=A_CENT
        ), run["name"]


# ---------------------------------------------------------------------------
# Picking a run
# ---------------------------------------------------------------------------


def test_every_saved_run_is_offered_exactly_once(results):
    """A run that is in the file and not in the menu cannot be read."""
    keys = [entry["key"] for entry in views.menu_items(results)]
    assert sorted(keys) == sorted(results["scenarios"])
    assert len(keys) == len(set(keys))


def test_the_eight_headlines_come_first_in_the_summary_s_own_order(results):
    """The order is Siri's curation, not alphabetical and not re-decided."""
    groups = dict(views.scenario_menu(results))
    headline_names = [entry["name"] for entry in groups[views.HEADLINE_GROUP]]
    assert headline_names == [name for name, _finding in export.HEADLINES]

    items = views.menu_items(results)
    assert items[0]["key"] == views.BASELINE_KEY
    assert [entry["name"] for entry in items[1:9]] == headline_names


def test_every_headline_run_shows_its_approved_finding(results):
    """The one sentence that says what the run found, reused verbatim."""
    approved = dict(export.HEADLINES)
    for entry in views.menu_items(results):
        if entry["name"] in approved:
            assert entry["finding"] == approved[entry["name"]]
        else:
            assert entry["finding"] is None


def test_the_menu_shows_each_run_s_declared_kind(results):
    """Declared in the scenario file (11/09), not inferred here.

    The marking used to be worked out from the scenario keys - a run
    whose key was another run's key plus a suffix was taken to be a
    variant of it. That is a guess about file naming standing in for a
    fact about the run, and it had nothing to say about a risk at all.
    """
    marked = {
        entry["key"]: (entry["kind"], entry["kind_label"])
        for entry in views.menu_items(results)
    }
    assert marked["site_shortfall_n1_borrow_2h"] == ("sensitivity", "Sensitivity")
    assert marked["after_proof_of_concept_true_05"] == ("sensitivity", "Sensitivity")
    assert marked["volume_plus_15"] == ("risk", "Risk")
    assert marked["attrition_doubles"] == ("risk", "Risk")
    assert marked[views.BASELINE_KEY] == ("choice", "Choice")


def test_every_kind_the_model_allows_has_a_name_on_screen(results):
    """A kind added to the YAML cannot reach the page unnamed."""
    for kind in scenarios.KINDS:
        assert views.kind_label(kind) == kind.capitalize()
    for run in results["scenarios"].values():
        assert run["kind"] in scenarios.KINDS


# ---------------------------------------------------------------------------
# Compare all scenarios (the second view)
# ---------------------------------------------------------------------------


def test_the_comparison_view_ties_to_the_comparison_table(
    results, comparison_rows
):
    """THE TIE-OUT, again, for the ranked table (11/09 item 2).

    Every measure of every run against the row the workbook's tab is
    built from, to the cent - including the ranking column, which is the
    one figure this view computes rather than reads, and so the one that
    could drift.
    """
    table = views.comparison_table(results)
    assert len(table) == len(results["scenarios"])

    baseline_cost = float(comparison_rows["Baseline"]["total_cost"])
    headers = {key: header for header, key in api.COMPARISON_MEASURES}
    for _index, shown in table.iterrows():
        row = comparison_rows[shown["Scenario"]]
        for measure, header in headers.items():
            if measure == "scenario":
                continue
            assert shown[header] == pytest.approx(
                float(row[measure]), abs=A_CENT
            ), f"{shown['Scenario']}: {header}"
        assert shown[views.DIFFERENCE_HEADER] == pytest.approx(
            float(row["total_cost"]) - baseline_cost, abs=A_CENT
        ), shown["Scenario"]


def test_the_comparison_view_is_ranked_by_what_it_costs(results):
    """Biggest saving first, baseline at zero, most expensive last."""
    table = views.comparison_table(results)
    differences = list(table[views.DIFFERENCE_HEADER])
    assert differences == sorted(differences)
    assert differences[0] < 0 < differences[-1]

    baseline_at = list(table["Scenario"]).index("Baseline")
    assert table[views.DIFFERENCE_HEADER].iloc[baseline_at] == 0.0
    # The known shape of the answer: pooling the phones is the saving,
    # the volume rise is the risk.
    assert table["Scenario"].iloc[0] == "Centralized phones"
    assert table["Scenario"].iloc[-1] == "Combined, AI 100%"


def test_the_comparison_view_shows_each_run_s_kind(results):
    """A ranking puts a risk and a proposal side by side."""
    table = views.comparison_table(results)
    kind_of = dict(zip(table["Scenario"], table[views.KIND_HEADER]))
    assert kind_of["Volume +15%"] == "Risk"
    assert kind_of["Centralized phones"] == "Choice"
    assert kind_of["AI and automation, 50%"] == "Sensitivity"
    assert set(table[views.KIND_HEADER]) <= {
        views.kind_label(kind) for kind in scenarios.KINDS
    }


def test_the_comparison_view_carries_the_workbook_s_headers(results):
    """Named as the tab names them, with the two this view adds."""
    table = views.comparison_table(results)
    expected = [header for header, _key in api.COMPARISON_MEASURES]
    for header in expected:
        assert header in table.columns
    assert list(table.columns)[:4] == [
        "Scenario", views.KIND_HEADER, views.ALL_IN_HEADER,
        views.DIFFERENCE_HEADER,
    ]


def test_the_comparison_view_rounds_only_for_the_screen(results):
    table = views.comparison_table(results)
    shown = views.comparison_display(table)
    assert shown[views.ALL_IN_HEADER].equals(table[views.ALL_IN_HEADER].round(0))
    assert list(shown["Scenario"]) == list(table["Scenario"])
    assert list(shown[views.KIND_HEADER]) == list(table[views.KIND_HEADER])


def test_the_page_offers_both_views_and_renders_the_comparison(results):
    """The second view, through the real app."""
    app = app_test().run()
    views_offered = view_picker(app).options
    assert set(views_offered) == {"One run", "Compare all scenarios"}

    view_picker(app).set_value("Compare all scenarios").run()
    assert not app.exception
    shown = app.dataframe[0].value
    assert len(shown) == len(results["scenarios"])
    assert list(shown["Scenario"]) == list(
        views.comparison_table(results)["Scenario"]
    )
    # The banner is on this view too: a screenshot of either travels.
    assert results["meta"]["label"] in [banner.value for banner in app.warning]


# ---------------------------------------------------------------------------
# Scope: a back-office role has no site
# ---------------------------------------------------------------------------


def test_the_scope_split_comes_from_the_yaml_not_this_module(dimensions):
    """supply.per_site_roles is the model's own rule (spec 4.4)."""
    per_site = set(api.config.load_assumptions()["supply"]["per_site_roles"])
    for role in dimensions["roles"]:
        expected = api.SITE_SCOPE if role["code"] in per_site else api.NETWORK_SCOPE
        assert role["scope"] == expected, role["code"]


def test_a_site_view_offers_no_back_office_role(dimensions):
    """The defect this guards against is a line drawn at zero.

    A Tier 1 Specialist is not absent from N1 - the role was never based
    there, and the work it does for N1 is counted at NETWORK scope. A
    chart that showed it as zero at every site would be reporting a
    measurement where there is none.
    """
    choices = {choice["code"]: choice for choice in views.scope_choices(dimensions)}
    for code in ("N1", "north"):
        offered = {role["code"] for role in views.roles_in_scope(dimensions, choices[code])}
        network = {role["code"] for role in views.network_roles(dimensions)}
        assert offered.isdisjoint(network), code
        assert offered  # the site-based roles are still there


def test_a_site_view_says_what_it_leaves_out(dimensions):
    """And says it in words, naming the roles."""
    choices = {choice["code"]: choice for choice in views.scope_choices(dimensions)}
    note = views.scope_note(dimensions, choices["N1"])
    for role in views.network_roles(dimensions):
        assert role["name"] in note
    assert dimensions["network_code"] in note
    assert views.scope_note(dimensions, choices[views.NETWORK_KIND]) == ""


def test_the_network_view_covers_every_scope_and_role(results, dimensions):
    """Fifteen (role, scope) pairs over 52 weeks: the whole plan (6.7)."""
    plan = results["scenarios"][views.BASELINE_KEY]["plan"]
    network = views.scope_choices(dimensions)[0]
    assert network["kind"] == views.NETWORK_KIND
    assert network["scopes"] is None
    filtered = views.filter_plan(plan, None, network["scopes"])
    assert len(filtered) == len(plan)
    assert len(filtered[["role", "scope"]].drop_duplicates()) == 15


def test_filtering_to_a_site_keeps_only_that_site_s_rows(results, dimensions):
    plan = results["scenarios"][views.BASELINE_KEY]["plan"]
    choices = {choice["code"]: choice for choice in views.scope_choices(dimensions)}
    roles = [role["code"] for role in views.roles_in_scope(dimensions, choices["N1"])]
    filtered = views.filter_plan(plan, roles, choices["N1"]["scopes"])
    assert set(filtered["scope"]) == {"N1"}
    assert set(filtered["role"]) == set(roles)
    weeks = plan["plan_week"].nunique()
    assert len(filtered) == len(roles) * weeks


def test_a_cluster_view_keeps_its_own_sites(results, dimensions):
    plan = results["scenarios"][views.BASELINE_KEY]["plan"]
    choices = {choice["code"]: choice for choice in views.scope_choices(dimensions)}
    north = choices["north"]
    filtered = views.filter_plan(plan, None, north["scopes"])
    assert set(filtered["scope"]) == set(north["scopes"])


# ---------------------------------------------------------------------------
# The charts
# ---------------------------------------------------------------------------


def test_the_gap_chart_holds_both_runs_over_every_plan_week(results):
    chart = views.weekly_gap(results, "central_phones")
    assert list(chart.columns) == [views.BASELINE_LABEL, "Centralized phones"]
    assert len(chart) == results["meta"]["plan_weeks"]
    assert chart.index.name == views.PLAN_WEEK


def test_the_gap_chart_is_the_plan_s_own_short_fte(results):
    """Summed over the filtered rows, and nothing else done to it."""
    plan = results["scenarios"][views.BASELINE_KEY]["plan"]
    chart = views.weekly_gap(results, views.BASELINE_KEY)
    assert list(chart.columns) == [views.BASELINE_LABEL]
    assert chart[views.BASELINE_LABEL].sum() == pytest.approx(
        float(plan[views.GAP_COLUMN].sum()), abs=1e-9
    )


def test_the_gap_chart_does_not_net_surplus_against_shortfall(results):
    """The netting mistake, guarded.

    A spare hour in one team does not staff another's counter. If the
    chart ever starts subtracting surplus, this fails - because the
    plan's surplus is real and large enough that the two series cannot
    coincide by accident.
    """
    plan = results["scenarios"][views.BASELINE_KEY]["plan"]
    netted = (plan[views.GAP_COLUMN] - plan["surplus_fte"]).sum()
    chart = views.weekly_gap(results, views.BASELINE_KEY)
    assert chart[views.BASELINE_LABEL].sum() != pytest.approx(netted, abs=1.0)


def test_the_gap_chart_follows_the_filters(results, dimensions):
    choices = {choice["code"]: choice for choice in views.scope_choices(dimensions)}
    chart = views.weekly_gap(
        results, "central_phones", ["onsite_associate"], choices["N1"]["scopes"]
    )
    plan = views.filter_plan(
        results["scenarios"]["central_phones"]["plan"], ["onsite_associate"], ["N1"]
    )
    assert chart["Centralized phones"].sum() == pytest.approx(
        float(plan[views.GAP_COLUMN].sum()), abs=1e-9
    )


def test_the_cost_chart_keeps_the_machine_half_even_when_it_is_zero(results):
    """Two bars have to measure the same thing to be comparable."""
    split = views.cost_split(results, "central_phones")
    assert list(split.columns) == ["Labour", "Machine"]
    assert list(split.index) == ["Baseline", "Centralized phones"]
    assert float(split.loc["Baseline", "Machine"]) == 0.0


# ---------------------------------------------------------------------------
# The detail table and its download
# ---------------------------------------------------------------------------


def test_the_detail_table_carries_the_workbook_s_capacity_plan_headers(results):
    view = views.plan_view(results["scenarios"][views.BASELINE_KEY]["plan"])
    expected = [
        header for header, _source, _format, _width, _total
        in export.CAPACITY_PLAN_COLUMNS
    ]
    assert list(view.columns) == expected


def test_the_hours_gap_is_the_subtraction_the_workbook_shows(results):
    """Available minus required: the workbook's live formula, computed."""
    plan = results["scenarios"][views.BASELINE_KEY]["plan"]
    view = views.plan_view(plan)
    expected = plan["available_hours"] - plan["required_hours"]
    assert view[views.HOURS_GAP_HEADER].equals(expected)


def test_the_screen_table_is_rounded_the_way_the_workbook_prints(results):
    """Rounded for reading, and only for reading."""
    plan = views.filter_plan(
        results["scenarios"][views.BASELINE_KEY]["plan"], ["tier_1"], None
    )
    view = views.plan_view(plan)
    shown = views.plan_display(view)
    assert shown["Labour cost"].equals(view["Labour cost"].round(0))
    assert shown["Required hours"].equals(view["Required hours"].round(1))
    # The exact figures are untouched, because the download uses them.
    assert view["Labour cost"].sum() != pytest.approx(
        shown["Labour cost"].sum(), abs=1e-9
    )


def test_every_format_the_plan_uses_has_a_way_to_be_written(results):
    """A format with no entry would silently print a raw float."""
    for _header, _source, number_format, _width, _total in (
        export.CAPACITY_PLAN_COLUMNS
    ):
        if number_format is None:
            continue
        assert number_format in views.VALUE_FORMATS
        assert number_format in views.DISPLAY_DECIMALS


def test_the_download_is_the_filtered_rows_at_full_precision(results):
    """A CSV is something a reader does arithmetic on."""
    plan = views.filter_plan(
        results["scenarios"]["volume_plus_15"]["plan"], ["qa_specialist"], None
    )
    view = views.plan_view(plan)
    back = pd.read_csv(io.BytesIO(views.csv_bytes(view)))
    assert list(back.columns) == list(view.columns)
    assert len(back) == len(view)
    assert back["Labour cost"].sum() == pytest.approx(
        float(view["Labour cost"].sum()), abs=1e-6
    )


def test_the_download_is_labelled_synthetic_in_its_file_name(results):
    """The only label that travels with a downloaded file."""
    name = views.csv_name("volume_plus_15", "N1")
    assert "SYNTHETIC" in name
    assert name.endswith(".csv")
    assert "volume_plus_15" in name and "N1" in name


# ---------------------------------------------------------------------------
# How a difference is written
# ---------------------------------------------------------------------------


def test_a_difference_is_signed_and_a_non_difference_says_so():
    assert views.format_difference(-51568.94, export.MONEY) == "-$51,569"
    assert views.format_difference(1062703.25, export.MONEY) == "+$1,062,703"
    assert views.format_difference(0.0, export.DAYS) == views.NO_CHANGE
    assert views.format_difference(None, export.MONEY) == ""


def test_required_fte_is_not_judged_better_or_worse():
    """A scenario that needs more people is not failing.

    It is reporting what the work asks for. Colouring that red would be a
    verdict the model never made, so the direction is neutral and the
    page shows the change without a judgement.
    """
    directions = {
        measure: direction
        for measure, _format, direction in views.CARD_MEASURES
    }
    assert directions["network_required_fte"] == views.NEUTRAL
    assert directions["total_cost"] == views.BETTER_LOWER
    assert directions["uncovered_hours"] == views.BETTER_LOWER


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------


def test_the_page_says_it_is_synthetic(results):
    """On the page, not in a footnote: screenshots travel."""
    app = app_test().run()
    assert not app.exception
    banners = [banner.value for banner in app.warning]
    assert results["meta"]["label"] in banners


def test_the_page_shows_the_approved_finding_for_a_headline_run():
    app = app_test().run()
    run_picker(app).set_value("central_phones").run()
    assert not app.exception
    approved = dict(export.HEADLINES)["Centralized phones"]
    assert approved in [note.value for note in app.info]


def test_the_page_drops_the_back_office_roles_in_a_site_view(dimensions):
    """And puts the reason in the sidebar where the filter is."""
    app = app_test().run()
    site = next(
        choice for choice in views.scope_choices(dimensions)
        if choice["code"] == "N1"
    )
    scope_picker(app).set_value(site["label"]).run()
    assert not app.exception
    offered = set(role_picker(app).options)
    network = {role["name"] for role in views.network_roles(dimensions)}
    assert offered.isdisjoint(network)
    assert any(
        role["name"] in note.value
        for note in app.sidebar.info
        for role in views.network_roles(dimensions)
    )


def test_the_page_lists_the_filtered_plan_rows(dimensions):
    """What is on screen is the filtered slice, not the whole plan."""
    app = app_test().run()
    whole = app.dataframe[0].value
    site = next(
        choice for choice in views.scope_choices(dimensions)
        if choice["code"] == "N1"
    )
    scope_picker(app).set_value(site["label"]).run()
    filtered = app.dataframe[0].value
    assert set(filtered["Site or scope"]) == {"N1"}
    assert len(filtered) < len(whole)


def test_the_page_offers_the_csv_download(dimensions):
    app = app_test().run()
    buttons = app.get("download_button")
    assert len(buttons) == 1
    assert "synthetic" in buttons[0].label.lower()
