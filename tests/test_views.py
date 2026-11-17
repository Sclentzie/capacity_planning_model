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

import html
import io
import json
import re
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


CARD_LABEL = re.compile(r'<div class="cm-card-label">(.*?)</div>')
CARD_VALUE = re.compile(r'<div class="cm-card-value">(.*?)</div>')
CARD_DELTA = re.compile(r'<div class="cm-card-delta"[^>]*>(.*?)</div>')


def rendered_cards(app):
    """The cards as the page sends them: label -> (value, difference).

    The arrow is stripped back off, because it is drawn beside the
    difference rather than being part of it, and this compares figures.
    """
    cards = {}
    for element in app.main:
        markup = str(getattr(element, "value", ""))
        label = CARD_LABEL.search(markup)
        value = CARD_VALUE.search(markup)
        if not (label and value):
            continue
        delta = CARD_DELTA.search(markup)
        written = html.unescape(delta.group(1)).strip() if delta else ""
        for arrow in (views.UP_ARROW, views.DOWN_ARROW):
            written = written.replace(arrow, "").strip()
        cards[html.unescape(label.group(1))] = (
            html.unescape(value.group(1)),
            written,
        )
    return cards


def test_the_page_renders_the_same_figures_it_was_handed(
    results, comparison_rows
):
    """The rendered cards, run by run, through the real app.

    The tie-out above checks the view layer. This checks the page: a
    label or a format could still go astray between the two, and a
    reader only ever sees the page.

    It used to read app.metric. The cards are drawn by hand since
    11/16, because st.metric cannot say "no change" in blue, so this
    reads the markup the page sends instead - the same guarantee, one
    layer closer to the screen.
    """
    app = app_test().run()
    assert not app.exception

    for key, run in results["scenarios"].items():
        run_picker(app).set_value(key).run()
        assert not app.exception, run["name"]
        rendered = rendered_cards(app)
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


def test_the_page_offers_every_view_and_renders_the_comparison(results):
    """The views on offer, and the comparison one, through the real app.

    It asserted exactly two views until 7c added the third, and failed
    for a change that was correct. Kept as a set rather than a count, so
    a view added later still has to be named here deliberately - the
    point of the assertion is that the page offers what it is meant to,
    not that nothing ever changes.
    """
    app = app_test().run()
    views_offered = view_picker(app).options
    assert set(views_offered) == {
        "One run", "Compare all scenarios", "Build your own"
    }

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
# The charts are locked (11/16, after Siri's walkthrough)
# ---------------------------------------------------------------------------


def test_neither_chart_can_be_zoomed_or_panned(results):
    """The finding behind this: a leader scrolled a chart out of scale.

    Vega-Lite pans and zooms only through a parameter bound to "scales",
    which is exactly what st.line_chart and st.bar_chart add and what
    .interactive() adds in Altair. A spec with no such parameter cannot
    be rescaled by a mouse or a trackpad, however hard anybody leans on
    it, and the axes stay the ones this code chose.
    """
    gap = views.weekly_gap(results, "combined")
    cost = views.cost_split(results, "combined")
    for chart in (views.gap_chart(gap, 52), views.cost_chart(cost)):
        spec = chart.to_dict()
        assert views._scale_bindings(spec) == []
        assert views.chart_is_locked(chart)


def test_the_shortfall_chart_is_fixed_to_the_whole_plan_year(results):
    """1 to 52, from the saved run's own plan length, never autoscaled.

    Fixed across, so two runs are read against the same calendar and a
    scenario that only bites in week 40 still shows its quiet start.
    nice=False matters as much as the domain: left to itself Vega-Lite
    rounds a domain outwards to its own taste, and the axis stops being
    the plan year.
    """
    plan_weeks = results["meta"]["plan_weeks"]
    assert plan_weeks == 52
    spec = views.gap_chart(views.weekly_gap(results, "combined"), plan_weeks).to_dict()
    scale = spec["encoding"]["x"]["scale"]
    assert scale["domain"] == [1, 52]
    assert scale["nice"] is False


def test_both_charts_fix_the_vertical_axis_to_a_round_number(results):
    """Fixed up as well as across, or the mouse is not the only thing
    that can rescale a chart - a filter change would do it too.

    The top is a round number above the tallest value in the current
    selection, so it clears the data and lands where a reader would have
    put it.
    """
    gap = views.weekly_gap(results, "combined")
    gap_scale = views.gap_chart(gap, 52).to_dict()["encoding"]["y"]["scale"]
    assert gap_scale["domain"][0] == 0
    assert gap_scale["domain"][1] >= float(gap.max().max())
    assert gap_scale["nice"] is False

    cost = views.cost_split(results, "combined")
    cost_scale = views.cost_chart(cost).to_dict()["encoding"]["y"]["scale"]
    # The bars are stacked, so the top has to clear the whole bar.
    assert cost_scale["domain"][1] >= float(cost.sum(axis=1).max())
    assert cost_scale["nice"] is False


def test_the_axis_top_is_a_number_a_reader_would_have_chosen():
    """1, 2 or 5 times a power of ten, and never a degenerate scale."""
    assert views.nice_ceiling(1473.6) == 2000
    assert views.nice_ceiling(1_000_000) == 1_000_000
    assert views.nice_ceiling(0.4) == 0.5
    # An all-zero selection still needs an axis to draw a flat line on.
    assert views.nice_ceiling(0) == 1
    assert views.nice_ceiling(float("nan")) == 1


def rendered_chart_specs(app):
    """The chart specs as they reach the browser.

    Streamlit's test harness has no typed element for a Vega-Lite chart,
    so the spec is read off the protobuf. Worth the awkwardness: this is
    the spec the page actually sends, which is the thing that has to be
    locked - the chart object being right would not help if something
    between here and the page added a binding back.
    """
    return [
        json.loads(element.proto.spec)
        for element in app.main
        if getattr(element, "type", None) == "vega_lite_chart"
    ]


def test_the_charts_reach_the_page_locked(results):
    """The lock, checked on the rendered page rather than the object."""
    app = app_test().run()
    assert not app.exception
    specs = rendered_chart_specs(app)
    assert len(specs) == 2

    for spec in specs:
        assert views._scale_bindings(spec) == []
        # Fixed up: every chart on the page has a vertical axis that a
        # filter change cannot quietly rescale.
        assert spec["encoding"]["y"]["scale"]["domain"][0] == 0

    gap_x = specs[0]["encoding"]["x"]["scale"]
    assert gap_x["domain"] == [1, results["meta"]["plan_weeks"]]
    assert gap_x["nice"] is False


def test_the_charts_still_answer_a_hover(results):
    """Locking the axes must not cost the one interaction they need.

    Reading a week's value off a line is what this chart is for; what
    was removed is the gesture that moves the picture.
    """
    gap_spec = views.gap_chart(views.weekly_gap(results, "combined"), 52).to_dict()
    titles = [tip["title"] for tip in gap_spec["encoding"]["tooltip"]]
    assert views.PLAN_WEEK in titles and views.SHORT_FTE_HEADER in titles

    cost_spec = views.cost_chart(views.cost_split(results, "combined")).to_dict()
    assert cost_spec["encoding"]["tooltip"]


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


def test_the_detail_table_fits_a_page_without_scrolling_sideways(results):
    """The second thing the walkthrough turned up (11/16).

    Seventeen columns under the workbook's own headers run off the side
    of a laptop screen, and the columns a reader most wants to compare -
    available against required, short against surplus - end up behind a
    sideways gesture. The screen gets short headers and measured widths;
    the budget is what those widths have to add up to inside.
    """
    specs = views.plan_column_specs()
    assert sum(spec["width"] for spec in specs) <= views.PLAN_WIDTH_BUDGET
    # Every column keeps a width of its own: a column left to Streamlit
    # would size itself to its header and undo the budget.
    assert all(spec["width"] > 0 for spec in specs)


def test_every_plan_column_has_a_short_header_and_keeps_its_full_one(results):
    """A short header is only honest while the full one is a hover away.

    The tooltip leads with the workbook's own header, so a reader who
    wonders what "Uncov" is called on the tab can find out without
    leaving the page - and so the two names can never drift apart,
    because one is built from the other.
    """
    expected = [
        header for header, _source, _format, _width, _total
        in export.CAPACITY_PLAN_COLUMNS
    ]
    specs = views.plan_column_specs()
    assert [spec["column"] for spec in specs] == expected
    for spec in specs:
        assert spec["help"].startswith(spec["column"])
        assert len(spec["label"]) <= len(spec["column"])


def test_the_download_is_not_shortened_with_the_screen(results):
    """Short headers are done to a screen, not to the data.

    A downloaded row has to match a workbook row, and a reader does
    arithmetic on a CSV, so the file keeps the full headers and the full
    precision whatever the table on screen is doing.
    """
    plan = views.filter_plan(
        results["scenarios"][views.BASELINE_KEY]["plan"], ["tier_1"], None
    )
    view = views.plan_view(plan)
    first_line = views.csv_bytes(view).decode("utf-8").splitlines()[0]
    assert first_line.split(",")[0] == "Plan week"
    assert views.HOURS_GAP_HEADER in first_line
    shortened = {spec["label"] for spec in views.plan_column_specs()}
    assert not shortened & set(first_line.split(",")) - {"Role"}


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


def test_every_measure_is_read_the_same_way(results):
    """All six are better lower, required FTE among them (11/16).

    Required FTE used to be neutral, on the reasoning that a scenario
    needing more people is being honest rather than failing. Siri's call
    is that a planner reads it the way they read cost, and that one card
    among six with no colour reads as a measure nobody has an opinion
    about - which is its own claim, and not one the model is making.
    """
    for _measure, _format, direction in views.CARD_MEASURES:
        assert direction == views.BETTER_LOWER

    # And the colour follows: fewer people than the baseline is green.
    fewer = next(
        card for card in views.headline_cards(results, "central_phones")
        if card["measure"] == "network_required_fte"
    )
    assert fewer["difference"] < 0
    assert views.card_colour(fewer) == views.BETTER_COLOUR

    more = next(
        card for card in views.headline_cards(results, "combined")
        if card["measure"] == "network_required_fte"
    )
    assert more["difference"] > 0
    assert views.card_colour(more) == views.WORSE_COLOUR


def test_every_card_on_every_run_follows_the_colour_rule(results):
    """One rule, stated here independently of the code that applies it.

    Green moved the way a planner wants, red the other way, blue did not
    move at all, and the baseline has nothing to compare against. Checked
    on all six cards of all 21 runs, because a rule that holds for the
    run someone looked at is not a rule.
    """
    for key in results["scenarios"]:
        for card in views.headline_cards(results, key):
            colour = views.card_colour(card)
            difference = card["difference"]
            where = f"{key}/{card['measure']}"

            if difference is None:
                assert key == views.BASELINE_KEY, where
                assert colour is None, where
                assert card["difference_text"] == "", where
                continue

            if round(difference, views.DISPLAY_DECIMALS[card["format"]]) == 0:
                assert colour == views.SAME_COLOUR, where
                assert card["difference_text"] == views.NO_CHANGE, where
                # No arrow: an arrow is a claim about direction, and a
                # difference that rounds away has none to report.
                assert views.card_arrow(card) == "", where
            elif difference < 0:
                assert colour == views.BETTER_COLOUR, where
                assert views.card_arrow(card) == views.DOWN_ARROW, where
            else:
                assert colour == views.WORSE_COLOUR, where
                assert views.card_arrow(card) == views.UP_ARROW, where


def test_a_difference_that_rounds_away_says_so_instead_of_showing_zero(results):
    """The card the walkthrough caught: FTE on Wider lunch window.

    It moved by four thousandths of a person. The card printed "0.00"
    with an up arrow, which is a rounding artefact wearing a verdict -
    a reader would take it as "this needs more people". It now says so
    in words, in blue, with no arrow, and the exact figure is on the
    hover so nothing is hidden by being rounded.
    """
    card = next(
        card for card in views.headline_cards(results, "wider_lunch_window")
        if card["measure"] == "network_required_fte"
    )
    # Genuinely not zero - this is a rounding case, not an identical run.
    assert card["difference"] != 0
    assert round(card["difference"], 2) == 0

    assert card["no_change"] is True
    assert card["difference_text"] == views.NO_CHANGE
    assert views.card_colour(card) == views.SAME_COLOUR
    assert views.card_arrow(card) == ""

    # Nothing hidden: the hover carries the figure the card rounded.
    tooltip = views.card_tooltip(card)
    assert views.exact_difference_text(card["difference"]) in tooltip
    assert "rounds to nothing" in tooltip

    markup = views.card_html(card)
    assert views.SAME_COLOUR in markup
    assert views.NO_CHANGE in markup
    assert views.UP_ARROW not in markup and views.DOWN_ARROW not in markup


def test_the_rounding_rule_is_the_precision_the_card_shows(results):
    """Each measure rounds at its own precision, not at a shared epsilon.

    Half a cent is no change on a card that prints dollars; half a
    person is not, on a card that prints hundredths.
    """
    assert views.rounds_to_nothing(0.4, export.MONEY) is True
    assert views.rounds_to_nothing(0.6, export.MONEY) is False
    assert views.rounds_to_nothing(0.004, export.FTE) is True
    assert views.rounds_to_nothing(0.006, export.FTE) is False
    assert views.rounds_to_nothing(None, export.FTE) is False


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------


def rendered_text(app):
    """Everything a reader can read on the page, as one block of text.

    Labels, captions, headings, widget options and the table's own
    cells. Widget options are the formatted labels Streamlit sends to
    the browser, not the codes behind them, which is the point: this
    looks at what is on the screen rather than at what the page knows.
    """
    chunks = []
    for element in list(app.main) + list(app.sidebar):
        for attribute in ("value", "label", "body", "help"):
            found = getattr(element, attribute, None)
            if isinstance(found, str):
                chunks.append(found)
            elif isinstance(found, pd.DataFrame):
                chunks.append(found.to_string())
        options = getattr(element, "options", None)
        if options is not None:
            chunks.extend(str(option) for option in options)
    return "\n".join(chunks)


def test_no_role_is_called_by_its_code_name_anywhere_on_the_page(results):
    """A code name on a page is the model talking to itself in public.

    "records_specialist" is how the model refers to a role internally.
    A reader should see "Records Specialist", and "Tier 1" rather than
    "tier_1", everywhere - the sidebar, the cards, the captions and the
    table. The codes stay in the CSV, which is a different audience.
    """
    app = app_test().run()
    assert not app.exception
    on_screen = rendered_text(app)

    codes = [role["code"] for role in api.dimensions()["roles"]]
    assert codes, "no roles to check"
    for code in codes:
        assert code not in on_screen, code

    # And the plain names really are there, rather than nothing at all.
    for role in api.dimensions()["roles"]:
        assert role["name"] in on_screen or role["short_name"] in on_screen


def test_the_page_shows_a_no_change_card_in_blue(results):
    """The rule, checked where a reader meets it.

    Wider lunch window leaves required FTE alone to two decimals, so
    its card has to reach the page saying so, in blue, with no arrow.
    """
    app = app_test()
    app.run()
    run_picker(app).set_value("wider_lunch_window").run()
    assert not app.exception

    cards = [
        element.value for element in app.main
        if type(element).__name__ == "Markdown"
        and "cm-card-value" in str(element.value)
    ]
    blue = [markup for markup in cards if views.SAME_COLOUR in markup]
    assert blue, "no card reached the page in blue"
    assert all(views.NO_CHANGE in markup for markup in blue)
    assert all(
        views.UP_ARROW not in markup and views.DOWN_ARROW not in markup
        for markup in blue
    )
    # The exact difference travels with the card, in its hover.
    assert any("rounds to nothing" in markup for markup in blue)


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
