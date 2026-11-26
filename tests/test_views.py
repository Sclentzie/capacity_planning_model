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
import html as html_module
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from capacity_model import api, download, export, scenarios, views

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



def page_html(app):
    """Everything the page wrote as markup, in one string.

    THE TABLES ARE OUR OWN HTML NOW (11/24), so there is no app.dataframe
    to read. This is the equivalent handle: what actually went to the
    browser, which is also what a reader sees.
    """
    return "\n".join(block.value for block in app.markdown)


def table_rows(markup):
    """Every body row of the first table in some markup, cell by cell.

    A small parser rather than a dependency: the markup is written by
    views.table_html three lines away, so a regex that only has to read
    what this project writes is honest about its scope.
    """
    body = re.search(r"<tbody>(.*?)</tbody>", markup, re.S)
    if not body:
        return []
    rows = []
    for row in re.findall(r"<tr>(.*?)</tr>", body.group(1), re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        rows.append([
            html_module.unescape(re.sub(r"<[^>]+>", "", cell)) for cell in cells
        ])
    return rows


def table_headers(markup):
    """The header row of the first table, as (label, tooltip) pairs."""
    head = re.search(r"<thead>(.*?)</thead>", markup, re.S)
    if not head:
        return []
    return [
        (html_module.unescape(re.sub(r"<[^>]+>", "", cell)),
         html_module.unescape(title))
        for title, cell in re.findall(
            r'<th[^>]*title="([^"]*)"[^>]*>(.*?)</th>', head.group(1), re.S
        )
    ]


def page_style(app):
    """The stylesheet the page sent, which is an st.html element.

    st.html and not st.markdown, and for a reason worth keeping in the
    test: markdown ends an HTML block at the line holding its closing
    tag, so four sheets joined together lost everything after the first
    "</style><style>" and the page silently had no stylesheet at all.
    """
    found = app.get("html")
    assert found, "the page sent no stylesheet"
    return "\n".join(str(element.proto.body) for element in found)


def banner_texts(app, tone):
    """The banners of one tone the page wrote."""
    # The opening tag, not the class name: the page's own stylesheet
    # mentions every class it styles, and matching that would return a
    # block of CSS as if it were a banner.
    opening = f'<div class="cm-banner cm-{tone}">'
    return [
        block.value for block in app.markdown if opening in block.value
    ]


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
# The delta line is a Streamlit colour directive now (11/20), not a div:
# ":green[▼ -$14,369]". The name and the words are captured apart,
# because one is the colour and the other is what a reader who cannot see
# the colour reads instead.
CARD_DELTA = re.compile(r"^:([a-z]+)\[(.*)\]$", re.S)


def rendered_cards(app):
    """The cards as the page sends them: label -> (value, difference).

    The arrow is stripped back off, because it is drawn beside the
    difference rather than being part of it, and this compares figures.
    """
    cards = {}
    waiting = None
    # IN ORDER, because a card is two elements now (11/20): the label and
    # the value as one block of HTML, then the delta as its own markdown
    # so that the colour in it is a name the browser resolves live. The
    # delta belongs to the card immediately before it.
    for element in app.main:
        markup = str(getattr(element, "value", ""))
        label = CARD_LABEL.search(markup)
        value = CARD_VALUE.search(markup)
        if label and value:
            waiting = html.unescape(label.group(1))
            cards[waiting] = (html.unescape(value.group(1)), "")
            continue
        delta = CARD_DELTA.match(markup.strip())
        if delta and waiting:
            written = html.unescape(delta.group(2)).strip()
            for arrow in (views.UP_ARROW, views.DOWN_ARROW):
                written = written.replace(arrow, "").strip()
            cards[waiting] = (cards[waiting][0], written)
            waiting = None
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
    """Labor plus machine is the headline, or the chart is a decoration."""
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
    # The three kinds, plus the baseline, which is named rather than
    # classified (11/25) because it is the line the others are measured
    # from and not something anybody is proposing.
    assert set(table[views.KIND_HEADER]) <= (
        {views.kind_label(kind) for kind in scenarios.KINDS}
        | {views.BASELINE_KIND_LABEL}
    )
    assert kind_of[views.BASELINE_LABEL] == views.BASELINE_KIND_LABEL


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
    rows = table_rows(page_html(app))
    assert len(rows) == len(results["scenarios"])
    assert [row[0] for row in rows] == list(
        views.comparison_table(results)["Scenario"]
    )
    # The banner is on this view too: a screenshot of either travels.
    assert any(
        views.SYNTHETIC_LEAD in banner
        for banner in banner_texts(app, "synthetic")
    )

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
    # The workbook's own names for the two halves since 7d, so the
    # legend here and the columns of the comparison table agree.
    assert list(split.columns) == ["Labor cost", "Machine cost"]
    assert list(split.index) == ["Baseline", "Centralized phones"]
    assert float(split.loc["Baseline", "Machine cost"]) == 0.0


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
    assert shown["Labor cost"].equals(view["Labor cost"].round(0))
    assert shown["Required hours"].equals(view["Required hours"].round(1))
    # The exact figures are untouched, because the download uses them.
    assert view["Labor cost"].sum() != pytest.approx(
        shown["Labor cost"].sum(), abs=1e-9
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
    assert back["Labor cost"].sum() == pytest.approx(
        float(view["Labor cost"].sum()), abs=1e-6
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

    NO LONGER RUN ONCE PER PALETTE (11/20), and the reason is the point
    of that day's work: a card names its colour and never picks a shade,
    so there is one answer here rather than two that have to agree. The
    meaning cannot drift between the palettes because it never meets
    them.
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

            # A NAME, never a hex. A "#" here would be a shade this
            # module chose, and a shade chosen here is stale the moment
            # a reader switches Light/Dark without a rerun.
            assert colour in views.CARD_COLOURS, where
            assert not colour.startswith("#"), where

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

    # The delta line is markdown now, not HTML (11/20), so the colour
    # and the words are read off that rather than off the card's block.
    delta = views.card_markdown(card)
    assert delta == f":{views.SAME_COLOUR}[{views.NO_CHANGE}]"
    assert views.UP_ARROW not in delta and views.DOWN_ARROW not in delta
    # And no colour at all in the block carrying the label and value.
    assert "color:" not in views.card_html(card)


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

    markdown = [
        str(element.value) for element in app.main
        if type(element).__name__ == "Markdown"
    ]

    # The delta lines reach the page as markdown now (11/20), carrying a
    # colour NAME rather than a hex, so this looks for the directive the
    # browser resolves rather than for a shade.
    blue = [line for line in markdown if line.startswith(f":{views.SAME_COLOUR}[")]
    assert blue, "no card reached the page in blue"
    assert all(views.NO_CHANGE in line for line in blue)
    assert all(
        views.UP_ARROW not in line and views.DOWN_ARROW not in line
        for line in blue
    )

    # The card bodies are there too, and carry no colour of their own.
    bodies = [line for line in markdown if "cm-card-value" in line]
    assert bodies, "no card body reached the page"
    for body in bodies:
        assert "color:" not in body, body
    # The exact difference travels with the card, in its hover.
    assert any("rounds to nothing" in body for body in bodies)


def test_the_page_says_it_is_synthetic(results):
    """On the page, not in a footnote: screenshots travel.

    A banner this page draws itself since 11/24, so the statement is
    checked in the markup rather than in st.warning's text. The whole
    label has to be there: the lead word is lifted out and coloured, and
    the rest of the sentence must still be said.
    """
    app = app_test().run()
    assert not app.exception
    banners = banner_texts(app, "synthetic")
    assert banners, "the synthetic banner is not on the page"
    shown = " ".join(
        html_module.unescape(re.sub(r"<[^>]+>", "", banners[0])).split()
    )
    label = " ".join(results["meta"]["label"].split())
    assert shown.startswith(views.SYNTHETIC_LEAD)
    # The WHOLE label, word for word: the lead is lifted out and
    # coloured, and nothing else about the statement may change.
    assert shown == label, (shown, label)

def test_the_page_shows_the_approved_finding_for_a_headline_run():
    """The workbook's own sentence for the run, in the note banner."""
    app = app_test().run()
    run_picker(app).set_value("central_phones").run()
    assert not app.exception
    approved = dict(export.HEADLINES)["Centralized phones"]
    shown = [
        html_module.unescape(re.sub(r"<[^>]+>", "", banner))
        for banner in banner_texts(app, "note")
    ]
    assert any(approved in line for line in shown), shown

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
    whole = table_rows(page_html(app))
    site = next(
        choice for choice in views.scope_choices(dimensions)
        if choice["code"] == "N1"
    )
    scope_picker(app).set_value(site["label"]).run()
    filtered = table_rows(page_html(app))
    scope_at = [spec["column"] for spec in views.plan_column_specs()].index(
        views.SCOPE_HEADER
    )
    assert {row[scope_at] for row in filtered} == {"N1"}
    assert len(filtered) < len(whole)

def test_the_page_offers_excel_first_and_csv_beside_it(dimensions):
    """Two downloads under the detail table, in that order (Siri, 11/23).

    ORDER IS THE POINT, not just presence. Almost everyone who takes this
    table off the page opens it in Excel and shows it to somebody, so the
    formatted file is the main button and the raw CSV is the one beside
    it. A test that only counted the buttons would pass with them the
    wrong way round.
    """
    app = app_test().run()
    buttons = app.get("download_button")
    assert [button.label for button in buttons] == [
        download.EXCEL_LABEL, download.CSV_LABEL
    ]
    assert "excel" in buttons[0].label.lower()
    assert "csv" in buttons[1].label.lower()


def test_both_downloads_carry_synthetic_in_the_file_name():
    """The only label that travels with a file is its name."""
    names = [
        download.plan_name("baseline", "network"),
        download.custom_plan_name("network"),
        download.comparison_name(),
        download.comparison_csv_name(),
        views.csv_name("baseline", "network"),
        views.custom_csv_name("network"),
    ]
    for name in names:
        assert "SYNTHETIC" in name, name
    assert all("CUSTOM" in name for name in names if "custom" in name.lower())


def test_compare_all_offers_the_same_two_downloads():
    """The ranked view is a table somebody will want to forward too."""
    app = app_test().run()
    view_picker(app).set_value(views.COMPARE_VIEW).run()
    buttons = app.get("download_button")
    assert [button.label for button in buttons] == [
        download.EXCEL_LABEL, download.CSV_LABEL
    ]


# ---------------------------------------------------------------------------
# The landing intro (7d, 11/17)
# ---------------------------------------------------------------------------


def test_the_intro_is_three_lines_and_says_what_this_is(results, dimensions):
    """Three, and no more: a landing paragraph a visitor has to read
    before they can start is a landing paragraph they skip.
    """
    lines = views.intro_lines(results, dimensions)
    assert len(lines) == 3
    what, finding, how = lines

    # What this is, including that none of it is real.
    assert "Northbrook Records Services" in what
    assert "fictional" in what
    assert "synthetic" in what.lower()
    assert f"{len(dimensions['sites'])}-site" in what
    # And how to use it names the three views and the number of runs.
    assert str(results["meta"]["scenario_count"]) in how
    assert finding


def test_the_intro_s_headline_figures_are_the_saved_results(results):
    """A headline typed into a page goes stale the first time the model
    is rebuilt, and the stale version is the one a visitor quotes. Both
    figures are checked against the saved summary rows here, and against
    Siri's approved sentence for that run - if a rebuild moved the cost,
    this fails, which is the loudest way for it to say that the finding
    needs rewriting too.
    """
    risk = views.headline_risk(results)
    summary = results["scenarios"][risk["key"]]["summary"]
    baseline = results["scenarios"][views.BASELINE_KEY]["summary"]

    expected_rise = float(summary["total_cost"]) - float(baseline["total_cost"])
    assert risk["cost_rise"] == pytest.approx(expected_rise, abs=A_CENT)
    assert risk["weeks_over_goal"] == int(summary["weeks_over_turnaround_goal"])

    money = views.format_millions(risk["cost_rise"])
    weeks = str(risk["weeks_over_goal"])
    line = views.intro_lines(results)[1]
    assert risk["name"] in line
    assert money in line
    assert f"{weeks} weeks" in line
    # The same two figures as the workbook's approved finding for the run.
    assert money.lstrip("+") in risk["finding"]
    assert weeks in risk["finding"]


def test_the_headline_risk_is_the_run_the_summary_leads_with(results):
    """The claim "biggest risk" is Siri's, not the page's: it is read
    from the order of her approved findings, and the word "risk" is the
    run's own declared kind.
    """
    risk = views.headline_risk(results)
    assert risk["name"] == export.HEADLINES[0][0]
    assert results["scenarios"][risk["key"]]["kind"] == scenarios.KIND_RISK


def test_the_intro_is_the_first_thing_on_every_view(results, dimensions):
    """Above the picker, on all three views - a visitor who arrives on
    the comparison view needs the same three lines.
    """
    expected = views.intro_lines(results, dimensions)
    app = app_test().run()
    for view in views.VIEW_ORDER:
        view_picker(app).set_value(view).run()
        assert not app.exception
        shown = [str(block.value) for block in app.markdown]
        for line in expected:
            assert line in shown, f"{view}: intro line missing"


def test_the_intro_names_the_views_the_sidebar_offers(results):
    """The sentence points at the radio buttons, so the names have to be
    the radio buttons' names. They come from one place for that reason.
    """
    how = views.intro_lines(results)[2]
    app = app_test().run()
    offered = set(view_picker(app).options)
    assert offered == set(views.VIEW_ORDER)
    for name in views.VIEW_ORDER:
        assert name.lower() in how.lower()


# ---------------------------------------------------------------------------
# One name, one format, one explanation per measure (7d, 11/17)
# ---------------------------------------------------------------------------


def test_the_stated_precisions_are_the_ones_used(results):
    """Siri 11/17: dollars whole, FTE to two decimals, hours to one,
    days to two. Pinned here so a later change to a format table has to
    be deliberate.
    """
    assert views.DISPLAY_DECIMALS[export.MONEY] == 0
    assert views.DISPLAY_DECIMALS[export.FTE] == 2
    assert views.DISPLAY_DECIMALS[export.HOURS] == 1
    assert views.DISPLAY_DECIMALS[export.DAYS] == 2

    assert views.format_value(1234567.89, export.MONEY) == "$1,234,568"
    assert views.format_value(12.345, export.FTE) == "12.35"
    assert views.format_value(1234.56, export.HOURS) == "1,234.6"
    assert views.format_value(4.567, export.DAYS) == "4.57"


def test_one_measure_is_written_one_way_in_every_view(results):
    """The card, the comparison table and the detail table all write a
    figure through the same format table, so the all-in cost cannot read
    "$7,456,636" on a card and "7456636" in the view beside it.
    """
    table = views.comparison_table(results)
    comparison = {
        spec["column"]: spec["format"]
        for spec in views.comparison_column_specs(table)
    }
    plan = {
        spec["column"]: spec["format"] for spec in views.plan_column_specs()
    }

    # Every column of both tables is written, not left as a raw float.
    assert all(comparison.values())
    assert all(plan.values())

    # The measures the two tables share are written identically.
    shared = set(comparison) & set(plan)
    assert shared, "the two tables share no measure, so nothing is checked"
    for header in shared:
        assert comparison[header] == plan[header], header

    # And a card's own format is the one the comparison table uses.
    for measure, number_format, _direction in views.CARD_MEASURES:
        header = views.MEASURE_HEADERS[measure]
        assert comparison[header] == views.SCREEN_NUMBER_FORMATS[number_format]


def test_one_measure_is_called_one_thing_in_every_view(results):
    """The cards take the workbook's headers; so do the comparison
    table's columns, the chart axes and the two halves of the cost bar.
    """
    headers = {header for header, _key in api.COMPARISON_MEASURES}
    for measure, _format, _direction in views.CARD_MEASURES:
        assert views.MEASURE_HEADERS[measure] in headers

    # The chart axes are measures too, and they are named as the
    # workbook names them.
    plan_headers = {
        header for header, _source, _f, _w, _t in export.CAPACITY_PLAN_COLUMNS
    }
    assert views.SHORT_FTE_HEADER in plan_headers
    assert views.COST_HEADER == views.ALL_IN_HEADER
    assert views.COST_HEADER in headers
    assert views.LABOUR_HEADER in headers
    assert views.MACHINE_HEADER in headers
    assert [label for label, _measure in views.COST_HALVES] == [
        views.LABOUR_HEADER, views.MACHINE_HEADER
    ]


def test_every_card_says_what_its_measure_is(results):
    """Plain-language help on every card (7d). "Network required FTE" is
    a term of art, and a label alone cannot explain it.
    """
    for key in results["scenarios"]:
        for card in views.headline_cards(results, key):
            note = views.measure_note(card["label"])
            assert note, f"{card['label']}: no note"
            assert note.endswith("."), card["label"]
            tooltip = views.card_tooltip(card)
            assert note in tooltip
            # The exact difference is still there, as 7b promised.
            if card["difference"] is not None:
                assert views.exact_difference_text(card["difference"]) in tooltip


def test_every_comparison_column_says_what_its_measure_is(results):
    """The same note as the card's, so the two views cannot explain one
    figure two different ways.
    """
    table = views.comparison_table(results)
    for spec in views.comparison_column_specs(table):
        assert spec["help"], f"{spec['column']}: no note"
        # The measure's own sentence, with the workbook's header in front
        # of it wherever the column had to be shortened to fit the page
        # (11/25) - so the note is the same sentence either way.
        assert spec["help"].endswith(views.MEASURE_NOTES[spec["column"]])
    assert views.KIND_HEADER in views.MEASURE_NOTES
    assert views.DIFFERENCE_HEADER in views.MEASURE_NOTES

    # A card and a column about the same measure share one sentence.
    cost = next(
        card for card in views.headline_cards(results, "volume_plus_15")
        if card["label"] == views.ALL_IN_HEADER
    )
    note = views.MEASURE_NOTES[views.ALL_IN_HEADER]
    assert note in views.card_tooltip(cost)
    assert note in next(
        spec["help"] for spec in views.comparison_column_specs(table)
        if spec["column"] == views.ALL_IN_HEADER
    )


def test_every_plan_column_says_what_it_is(results):
    """The detail table had these from 7b; checked here with the rest so
    one test covers the promise across all three views.
    """
    for spec in views.plan_column_specs():
        assert spec["help"], spec["column"]
        assert spec["column"] in spec["help"]


def test_the_chart_axes_are_written_like_the_cards(results):
    """A cost axis in SI units - "1.5M" against a card reading
    "$1,456,636" - is one figure in two notations on one screen.
    """
    gap = views.gap_chart(views.weekly_gap(results, "volume_plus_15"), 52)
    cost = views.cost_chart(views.cost_split(results, "volume_plus_15"))
    assert gap.to_dict()["encoding"]["y"]["axis"]["format"] == ",.2f"
    assert cost.to_dict()["encoding"]["y"]["axis"]["format"] == "$,.0f"
    # Still locked: adding an axis format must not add an interaction.
    assert views.chart_is_locked(gap)
    assert views.chart_is_locked(cost)


HELP_KINDS = (
    "radio", "selectbox", "multiselect", "slider", "toggle", "number_input",
    "time_input", "text_input", "button",
)


def test_every_card_filter_and_control_has_help_text(results):
    """Siri 11/17, across all three views: nothing a reader can click or
    drag is left unexplained.

    Walked on the real page rather than listed here, so a widget added
    later arrives with this test already pointed at it.
    """
    app = app_test().run()
    missing = []
    for view in views.VIEW_ORDER:
        view_picker(app).set_value(view).run()
        assert not app.exception
        if view == views.BUILD_VIEW:
            # The Advanced box draws three more widgets once something is
            # searched for, so they are searched for.
            widget(app, "text_input", "advanced_search").set_value(
                "uptime"
            ).run()
        for kind in HELP_KINDS:
            for element in getattr(app, kind):
                if not getattr(element, "help", ""):
                    missing.append((view, kind, element.label))
        for element in app.main:
            if type(element).__name__ == "DownloadButton":
                if not getattr(element, "help", ""):
                    missing.append((view, "download_button", element.label))
    assert not missing, f"no help text: {missing}"


# ---------------------------------------------------------------------------
# The two palettes, the scope tints and the kinds (7d, 11/19)
# ---------------------------------------------------------------------------

THEME_FILE = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


def theme():
    """The committed theme file, as a dict."""
    import tomllib

    with open(THEME_FILE, "rb") as handle:
        return tomllib.load(handle)["theme"]


def palette(mode):
    """One mode's section of that file: [theme.light] or [theme.dark]."""
    return theme()[mode]


def test_the_theme_is_committed_with_the_app(results):
    """Committed on purpose: Streamlit reads this file out of the
    repository, so the hosted app is themed by the same file a local run
    is. A theme set in the host's settings dialog instead would be
    invisible here.
    """
    assert THEME_FILE.exists(), "the hosted app would fall back to white"
    settings = theme()
    for mode in views.MODES:
        assert mode in settings, f"no [theme.{mode}] section"
        for option, pair in views.THEME_COLOURS.items():
            assert settings[mode][option] == views.shade(pair, mode), (
                f"theme.{mode}.{option}"
            )
        # The sidebar is the neighbouring shade, which is what makes it
        # read as a panel beside the page rather than part of it.
        assert settings[mode]["sidebar"]["backgroundColor"] == (
            views.sidebar_colour(mode)
        )


def test_both_palettes_exist_and_nothing_is_shared_between_them():
    """Siri 11/19, and the trap this avoids.

    Streamlit merges a bare [theme] section FIRST and then lays the
    active variant over it, so a colour left in [theme] applies to both
    palettes until the other one happens to override it. A cream page up
    there would leak into dark mode. So every colour lives in its own
    section, and this fails if one creeps back up.

    It also checks that both sections carry colours at all, because that
    - and only that - is what makes Streamlit offer its System / Light /
    Dark switch. One palette and the switch is hidden.
    """
    settings = theme()
    shared = {
        name: value for name, value in settings.items()
        if not isinstance(value, dict)
    }
    assert not shared, f"these would apply to BOTH palettes: {shared}"

    for mode in views.MODES:
        colours = [
            name for name in palette(mode) if name.lower().endswith("color")
        ]
        assert colours, f"[theme.{mode}] sets no colours"


def test_streamlit_can_see_both_palettes(results):
    """Checked through Streamlit's own serialiser, not just the TOML.

    Parsing the file ourselves proves the file is right; it does not
    prove Streamlit agrees. This runs the config through the same
    function app_session uses to fill the message the browser is sent,
    so a section Streamlit silently drops fails here.
    """
    from streamlit import config
    from streamlit.proto.NewSession_pb2 import NewSession
    from streamlit.runtime.app_session import _populate_theme_msg

    config.get_config_options()
    message = NewSession().custom_theme
    _populate_theme_msg(message, "theme")
    for mode in views.MODES:
        variant = getattr(message, mode)
        _populate_theme_msg(variant, f"theme.{mode}")
        _populate_theme_msg(variant.sidebar, f"theme.{mode}.sidebar")
        assert variant.background_color == views.page_colour(mode), mode
        assert variant.primary_color == views.accent_colour(mode), mode
        assert variant.text_color == views.text_colour(mode), mode
        assert variant.sidebar.background_color == views.sidebar_colour(mode)

    # And nothing in the shared message, which is the leak above again,
    # seen from the other side.
    assert message.background_color == ""
    assert message.primary_color == ""


def test_every_option_in_the_theme_file_is_one_streamlit_accepts():
    """A typo in this file is silent: Streamlit ignores what it does not
    know, the app deploys, and the theme is simply absent.
    """
    from streamlit import config

    config.get_config_options()
    unknown = []

    def walk(prefix, mapping):
        for name, value in mapping.items():
            full = f"{prefix}.{name}"
            if isinstance(value, dict):
                walk(full, value)
            elif full not in config._config_options:
                unknown.append(full)

    walk("theme", theme())
    assert not unknown, f"Streamlit does not know: {unknown}"


@pytest.mark.parametrize("mode", views.MODES)
def test_the_theme_passes_wcag_aa(results, mode):
    """Measured, not eyeballed, in BOTH palettes. AA is 4.5:1 for
    ordinary text and 3:1 for a control's own edge.

    Run once per mode because a ratio is a fact about a colour AND its
    background, and the two palettes have different backgrounds: the
    light shades were measured on cream and are meaningless on charcoal.
    """
    page = views.page_colour(mode)
    sidebar = views.sidebar_colour(mode)
    text = views.text_colour(mode)
    primary = views.accent_colour(mode)

    assert views.contrast_ratio(text, page) >= views.AA_TEXT
    assert views.contrast_ratio(text, sidebar) >= views.AA_TEXT
    # The Run button is a primary button, so the teal carries a white
    # label - in both modes, which is what pins how light the dark teal
    # is allowed to be.
    assert views.contrast_ratio("#ffffff", primary) >= views.AA_TEXT
    # And its own edge against the page, which is a graphic, not text.
    assert views.contrast_ratio(primary, page) >= views.AA_LARGE

    # The mode's own body text on each scope tint. A tint is a translucent
    # wash now (11/20), so it has no colour of its own until the browser
    # puts it over the page: composite it first and measure what a reader
    # actually meets. ONE set of three washes serves both modes, and this
    # is the check that says both results are readable.
    for region in views.REGION_TINTS:
        wash = views.region_tint(region)
        assert views.is_wash(wash), (region, wash)
        tint = views.composite_over(wash, page)
        assert views.contrast_ratio(text, tint) >= views.AA_TEXT, (region, tint)

    # Every kind label, on both backgrounds it is written on - as
    # Streamlit paints it (11/20), not as the theme file declares it.
    # The two differ: our #5b3a8c violet is painted #382456.
    for kind in views.KIND_COLOURS:
        painted = views.rendered_colour(views.KIND_MARKDOWN[kind], mode)
        assert views.contrast_ratio(painted, page) >= views.AA_TEXT, kind
        assert views.contrast_ratio(painted, sidebar) >= views.AA_TEXT, kind

        # And the hex the theme file declares, which is what Streamlit
        # derives that shade FROM, and what the Kind column's wash is
        # mixed from. Both have to be readable in their own right.
        declared = views.declared_kind_colour(kind, mode)
        assert views.contrast_ratio(declared, page) >= views.AA_TEXT, kind
        assert views.contrast_ratio(declared, sidebar) >= views.AA_TEXT, kind

    # Every kind's wash, composited the way the browser composites it,
    # with this mode's body text on top - the Compare-all Kind column.
    for kind in views.KIND_TINTS:
        wash = views.kind_tint(kind)
        assert views.is_wash(wash), (kind, wash)
        washed = views.composite_over(wash, page)
        assert views.contrast_ratio(text, washed) >= views.AA_TEXT, (kind, washed)


# The pairs where what Streamlit paints is closer together than this
# project's own rules allow. MEASURED, in a real browser, in both
# palettes - see test_the_derived_shades_that_are_too_close below, which
# is a strict xfail so the gap is recorded rather than argued away.
#
# THERE WERE SIX OF THESE ON 11/20, AND THERE ARE NOW TWO. Four were
# kind-against-kind pairs, and they went two different ways:
#   - Siri moved the three kinds to the DIFFERENT bar, because a kind is
#     never colour alone - every kind label is printed as its own word
#     beside its own colour. Three of the four cleared 20 already.
#   - The fourth did not, and was the worst of the six: dark Choice
#     #e4d0f5 against dark Sensitivity #dadde1, 19.3 apart, one above the
#     other in the sidebar. So the gray was re-hued (see the theme file)
#     and that pair is now 26.2. It is pinned by a test that passes -
#     test_the_kinds_clear_the_bar_a_labelled_colour_answers_to.
#
# What was left was a card verdict against the teal accent, in one
# palette each. Those stay on the strict bar, because that is the one
# pairing where colour IS the only signal: the accent is a control's
# fill and it carries no word at all, so a slider that read as "no
# change" would be a lie about a number with nothing to correct it.
#
# ONE OF THOSE TWO IS NOW FIXED (11/23). The light pair was the better
# green against the teal accent, 29.8 apart, and the reason was that
# both of them were teal: Streamlit's inherited green derived to
# #0c5a44, hue 167. Setting a real green in the theme file moved the
# rendered shade to #246b27, hue 140, and the pair to 52.6 - so it was
# never a bar that needed lowering, it was a hue that needed choosing.
# It came off this list by being measured again, which is the only way
# anything should come off it.
#
# The dark no-change blue against the dark accent is still open at 32.8.
TOO_CLOSE = {
    ("dark", "no change", "accent"),
}


def painted_shades(mode):
    """Every meaning on the page, as the shade a reader actually sees.

    THE RENDERED SHADE, NOT THE DECLARED ONE (11/20). Each of these
    reaches the page as a colour NAME now, and Streamlit derives the hex
    from the live theme rather than using the configured one - our
    #5b3a8c violet is painted #382456. Measuring views.KIND_COLOURS here
    would be measuring a number nobody sees.
    """
    names = {
        "better": views.BETTER_COLOUR,
        "worse": views.WORSE_COLOUR,
        "no change": views.SAME_COLOUR,
        **{kind: views.KIND_MARKDOWN[kind] for kind in views.KIND_COLOURS},
    }
    shades = {
        label: views.rendered_colour(name, mode)
        for label, name in names.items()
    }
    shades["accent"] = views.accent_colour(mode)
    return shades


@pytest.mark.parametrize("mode", views.MODES)
def test_no_colour_on_the_page_means_two_things(results, mode):
    """Red, green and blue are spoken for by the cards (11/16), so no
    kind, no scope tint and not the accent may wear one of them - a
    tinted cell or a kind label in red would read as a verdict.

    Not just "is a different hex", which two near-identical colours pass
    while looking the same: held apart by delta-E, so the question asked
    is the reader's one - could these be confused?

    Measured on the SHADES STREAMLIT PAINTS (11/20), not on the hexes
    this project declares. The six meanings reach the page as colour
    names, so a declared hex is an input to the browser's choice and not
    the answer to it.
    """
    shades = painted_shades(mode)
    page = views.page_colour(mode)
    cards = {name: shades[name] for name in ("better", "worse", "no change")}
    kinds = {kind: shades[kind] for kind in views.KIND_COLOURS}
    # Composited over this mode's page, because that is the colour on
    # screen - the wash string itself is the same in both modes and would
    # answer this question once instead of twice.
    regions = {
        region: views.composite_over(views.region_tint(region), page)
        for region in views.REGION_TINTS
    }
    kind_washes = {
        kind: views.composite_over(views.kind_tint(kind), page)
        for kind in views.KIND_TINTS
    }
    accent = shades["accent"]

    # Distinct hexes first - the cheap check, and the one whose failure
    # message is easiest to read.
    assert len(set(cards.values())) == 3
    assert len(set(kinds.values())) == len(views.KIND_COLOURS)
    assert len(set(regions.values())) == len(views.REGION_TINTS)
    assert len(set(kind_washes.values())) == len(views.KIND_TINTS)
    assert not set(kinds.values()) & set(cards.values())
    assert not set(regions.values()) & set(cards.values())
    assert not set(kinds.values()) & set(regions.values())

    # Then far enough apart to be TOLD apart, which distinct hexes do not
    # guarantee. Two bars, for the two things the page asks of a reader -
    # views.DISTINCT and views.DIFFERENT carry the reasoning. Scope tints
    # and kind washes are left out: they are pale backgrounds in one
    # column rather than labels, and they are already held off the card
    # colours above.
    def apart(one, other, bar, group):
        if (mode, one, other) in TOO_CLOSE or (mode, other, one) in TOO_CLOSE:
            return  # recorded, and failed on purpose, in the xfail below
        distance = views.colour_distance(labels[one], labels[other])
        assert distance >= bar, (
            f"{mode}, {group}: {one} {labels[one]} and {other} "
            f"{labels[other]} are only {distance:.1f} apart (needs {bar})"
        )

    labels = {**cards, **kinds, "accent": accent}
    card_names = sorted(cards)
    kind_names = sorted(kinds)

    # Unmistakable: colour is the only thing separating these.
    for at, one in enumerate(card_names):
        for other in card_names[at + 1:]:
            apart(one, other, views.DISTINCT, "the three verdicts")
    for at, one in enumerate(kind_names):
        for other in kind_names[at + 1:]:
            # DIFFERENT, not DISTINCT (Siri, 11/20): see the bars in
            # views.py. A kind always carries its own word, so colour is
            # its second signal and not its only one.
            apart(one, other, views.DIFFERENT, "the three kinds")
    for one in card_names:
        apart(one, "accent", views.DISTINCT, "a verdict against the accent")

    # Obviously another colour: these never stand in for each other, and
    # each one carries its own word as well as its colour.
    for one in kind_names:
        apart(one, "accent", views.DIFFERENT, "a kind against the accent")
        for other in card_names:
            apart(one, other, views.DIFFERENT, "a kind against a verdict")

    # The kinds are written in Streamlit's own palette, and the three
    # names it offers outside red, green and blue are these. The NAME is
    # the same in both modes; the theme file gives it a hex per mode.
    assert set(views.KIND_MARKDOWN.values()) <= {"violet", "orange", "gray"}


# Only the palettes that still have a recorded pair. Parametrising over
# both modes would hand the light one a test with nothing to check,
# which would fail on "no recorded pair" and be reported as an EXPECTED
# failure - a strict xfail passing for a reason that has nothing to do
# with colour.
@pytest.mark.parametrize("mode", sorted({mode for mode, _, _ in TOO_CLOSE}))
@pytest.mark.xfail(
    strict=True,
    reason="the dark no-change blue is under DISTINCT against the dark "
           "accent; open, and not to be traded away cheaply",
)
def test_the_derived_shades_that_are_too_close(mode):
    """The gap, written down as a failing test rather than as a comment.

    STRICT XFAIL ON PURPOSE. Every pair below is one this project's own
    rules say a reader must be able to tell apart, and cannot. Prose
    rots; a test that is expected to fail does not, and the day somebody
    moves the accent or the card hues and fixes it, pytest reports an
    unexpected pass and this marker comes off.

    WHAT IS LEFT, AND WHAT WENT. This started 11/20 with six pairs, four
    of them kind-against-kind. Those were the ones a reader could at
    least read their way out of, since a kind is printed as its own word,
    and Siri moved them to the DIFFERENT bar and had the gray re-hued to
    clear it. A fifth went on 11/23: the better green against the light
    accent was 29.8 because Streamlit's inherited green was a teal, and
    choosing a real one took the pair to 52.6.

    ONE REMAINS - the no-change blue against the dark accent, 32.8 - and
    it stays strict because the accent is the one thing on this page
    that wears a colour and no word.
    """
    shades = painted_shades(mode)
    checked = 0
    for a_mode, one, other in sorted(TOO_CLOSE):
        if a_mode != mode:
            continue
        checked += 1
        distance = views.colour_distance(shades[one], shades[other])
        assert distance >= views.DISTINCT, (
            f"{mode}: {one} {shades[one]} and {other} {shades[other]} "
            f"are only {distance:.1f} apart (needs {views.DISTINCT})"
        )
    assert checked, f"no recorded pair for {mode}"


# What a browser measured between the three kinds, as Streamlit paints
# them, after the 11/20 re-hue of the dark gray. Recorded rather than
# merely bounded: a pair that only has to beat 20 can rot to 20.1 without
# a test noticing, and the dark Choice/Sensitivity pair is the one this
# project has already had to fix once.
KIND_DISTANCES = {
    "light": {
        ("choice", "risk"): 51.8,
        ("choice", "sensitivity"): 36.0,
        ("risk", "sensitivity"): 25.8,
    },
    "dark": {
        ("choice", "risk"): 40.4,
        ("choice", "sensitivity"): 26.2,   # was 19.3 before the re-hue
        ("risk", "sensitivity"): 32.9,     # was 27.9
    },
}


@pytest.mark.parametrize("mode", views.MODES)
def test_the_kinds_clear_the_bar_a_labelled_colour_answers_to(mode):
    """The three kinds, at the bar Siri set for them on 11/20.

    THE FIX, PINNED. Four kind pairs used to sit in TOO_CLOSE above and
    fail on purpose. They are here instead now, and they pass - which is
    only worth something if they pass for the right reason, so this
    checks both halves of that reason:

      - the bar is DIFFERENT (20) and not DISTINCT (40), because every
        kind label on this page is printed as its own word beside its own
        colour. That is Siri's call, and views.py carries the argument.
      - the DISTANCES ARE THE MEASURED ONES. The worst pair was 19.3 -
        dark Choice #e4d0f5 against dark Sensitivity #dadde1, which at
        that distance was one pale colour in the sidebar - and the dark
        grayColor was re-hued from a cool #afb6bf to a slate #88bdbe to
        move it. Asserting only "beats 20" would let that work be undone
        by a hex that scrapes 20.1, and would have let the original 19.3
        pass the day somebody lowered the bar instead of fixing it.
    """
    shades = painted_shades(mode)
    for (one, other), expected in sorted(KIND_DISTANCES[mode].items()):
        distance = views.colour_distance(shades[one], shades[other])
        assert distance >= views.DIFFERENT, (
            f"{mode}: {one} {shades[one]} and {other} {shades[other]} "
            f"are only {distance:.1f} apart (needs {views.DIFFERENT})"
        )
        # And still the distance a browser measured. A drift either way
        # means the theme file moved and nobody re-measured.
        assert abs(distance - expected) < 1.0, (
            f"{mode}: {one} to {other} measures {distance:.1f}, and "
            f"KIND_DISTANCES records {expected} - re-measure in a "
            "browser, do not edit the number"
        )

    # Margin, not a coin toss, on the pair that had to be fixed. Siri
    # chose hue 200 over a warm gray precisely because the warm one
    # cleared the violet at 25.2 but fell to 20.2 against the orange.
    worst = min(
        views.colour_distance(shades[one], shades[other])
        for one, other in KIND_DISTANCES[mode]
    )
    assert worst >= views.DIFFERENT + 5, (
        f"{mode}: the closest two kinds are {worst:.1f} apart, which "
        f"clears {views.DIFFERENT} with nothing to spare"
    )


@pytest.mark.parametrize("mode", views.MODES)
def test_every_card_colour_passes_wcag_aa(results, mode):
    """The three card colours, as Streamlit actually paints them.

    THE HISTORY IS WORTH KEEPING. These began as Streamlit's own three,
    and measuring them on 11/19 showed two never reached AA as small
    text - green at 2.8:1, red at 3.4:1 - so Siri replaced them with
    hand-picked hexes, one per palette, measured against the page each
    was drawn on. On 11/20 the hexes went away again, because a shade
    chosen in Python is stranded when a reader switches Light/Dark
    without a rerun, and the cards went back to naming their colour.

    So this measures Streamlit's choices now, not ours. The point of
    measuring rather than trusting is that we no longer pick: if a
    Streamlit release changes how it derives a shade, this is what says
    so. As it stands its derivation is comfortably better than the
    hand-picked hexes it replaced - the worst of the six is 6.7:1 where
    the hand-picked worst was 5.6:1.

    Held here rather than in a comment, because the next person to reach
    for a brighter green should have to argue with a test.
    """
    page = views.page_colour(mode)
    for name in views.CARD_COLOURS:
        colour = views.rendered_colour(name, mode)
        assert colour, name
        ratio = views.contrast_ratio(colour, page)
        assert ratio >= views.AA_TEXT, (
            f"{mode}: {name} {colour} is {ratio:.2f}:1 on {page}"
        )

    # Still the three a reader was taught: a lighter green is a green,
    # and a darker one is too. Checked in both modes, because a pastel
    # that drifted into cyan would still pass every contrast check above
    # while quietly changing what "better" looks like. This is the check
    # that matters most now that the shade is Streamlit's and not ours -
    # it is the one thing we still assert about a colour we do not pick.
    for name in views.CARD_COLOURS:
        colour = views.rendered_colour(name, mode)
        red, green, blue = (int(colour[at:at + 2], 16) for at in (1, 3, 5))
        strongest = {"green": green, "red": red, "blue": blue}[name]
        assert max(red, green, blue) == strongest, f"{mode}: {name} {colour}"


def test_a_card_never_leans_on_colour_alone(results):
    """Whatever the contrast of the three card colours, the card also
    says which way it moved in words or with an arrow - so a reader who
    cannot separate the colours loses nothing.
    """
    for key in results["scenarios"]:
        for card in views.headline_cards(results, key):
            if card["difference"] is None:
                continue
            if card["no_change"]:
                assert card["difference_text"] == views.NO_CHANGE
            else:
                assert card["difference_text"][0] in "+-"
                assert views.card_arrow(card) in (views.UP_ARROW, views.DOWN_ARROW)


def test_the_theme_file_and_the_kind_colours_agree():
    """Two copies of a hex is one copy too many, and here they cannot
    disagree: the sidebar writes a kind as markdown (":violet[Choice]"),
    which resolves the NAME against this file, while the comparison
    table paints the cell with the hex from views.
    """
    for mode in views.MODES:
        settings = palette(mode)
        for kind in views.KIND_COLOURS:
            colour = views.declared_kind_colour(kind, mode)
            name = views.KIND_MARKDOWN[kind]
            assert settings[f"{name}Color"] == colour, (mode, kind)
            assert settings["sidebar"][f"{name}Color"] == colour, (mode, kind)


def test_every_kind_has_a_colour_a_label_and_one_line_of_meaning():
    for kind in scenarios.KINDS:
        assert views.declared_kind_colour(kind), kind
        assert views.kind_tint(kind), kind
        assert views.kind_markdown(kind) == (
            f":{views.KIND_MARKDOWN[kind]}[{views.kind_label(kind)}]"
        )
        meaning = views.KIND_MEANINGS[kind]
        assert meaning.endswith("."), kind
        assert meaning[0].isupper(), kind
        # And the pair the table writes the word with (11/24), which is
        # read out of the theme file rather than copied beside it.
        pair = views.KIND_WORD_COLOURS[kind]
        assert pair == (
            views.KIND_COLOURS[kind][views.LIGHT_MODE],
            views.KIND_COLOURS[kind][views.DARK_MODE],
        )
        assert views.kind_word_class(kind) == f"cm-kind-{kind}"

    # A kind nobody declared gets no colour rather than a wrong one.
    assert views.declared_kind_colour("proposal") is None
    assert views.kind_tint("proposal") is None
    assert views.kind_markdown("proposal") == "proposal"
    assert views.kind_word_class("proposal") is None

def test_every_run_s_kind_is_a_kind_with_a_colour(results):
    """Checked against the saved runs, not against a list here: a run
    added with a new kind has to be given a colour before it can be
    shown.
    """
    for key, run in results["scenarios"].items():
        assert views.kind_tint(run["kind"]), f"{key}: {run['kind']}"


# ---------------------------------------------------------------------------
# The Scope cell, tinted by region
# ---------------------------------------------------------------------------


def test_every_scope_in_every_run_has_a_region_and_a_tint(results, dimensions):
    """Walked over every saved run's plan, so a scope the model produces
    and this does not know about shows up here rather than as an
    untinted cell on the page.
    """
    for key, run in results["scenarios"].items():
        for scope in set(run["plan"]["scope"]):
            region = views.region_of(scope, dimensions)
            assert region is not None, f"{key}: no region for {scope!r}"
            assert views.region_tint(region), f"{scope}: no tint"


def test_the_region_is_read_from_the_clusters_not_from_the_code(dimensions):
    """A site belongs to the cluster sites.yaml puts it in, not to the
    letter its code starts with, so a site added later lands in its real
    cluster.
    """
    for cluster in dimensions["clusters"]:
        for site in cluster["sites"]:
            assert views.region_of(site, dimensions) == cluster["code"]
    assert views.region_of(dimensions["network_code"], dimensions) == (
        views.NETWORK_REGION
    )
    assert views.region_of("Z9", dimensions) is None


def test_the_back_office_is_its_own_region(dimensions):
    """Not a site and not a cluster (spec 6.7), and the one line under
    the table says so in its own colour.
    """
    line = views.scope_line_html(dimensions)
    assert views.scope_word_class(views.NETWORK_REGION) in line
    assert dimensions["network_code"] in line
    assert "back office" in line
    for cluster in dimensions["clusters"]:
        assert views.scope_word_class(cluster["code"]) in line, cluster["code"]
        assert cluster["name"] in line, cluster["name"]
    # One word per region and no more: a key with four entries for three
    # regions would mean a scope was being explained twice.
    assert line.count("cm-word") == len(dimensions["clusters"]) + 1

def test_the_scope_word_is_coloured_by_the_region_it_sits_in(
    results, dimensions
):
    """Row by row, against the scope in that row (11/24).

    The colour used to be a wash on the cell; the reviewers asked for the
    word itself, so what is checked is the class the word wears. Through
    views.scope_class_of rather than a copy of the mapping here, so a
    site moved to another cluster in sites.yaml moves its colour too.
    """
    view = views.plan_view(results["scenarios"]["baseline"]["plan"])
    class_of = views.scope_class_of(dimensions)
    for scope in view[views.SCOPE_HEADER].unique():
        region = views.region_of(scope, dimensions)
        assert class_of(scope) == views.scope_word_class(region), scope

    # All three colours really are in use on a network-wide view.
    assert {class_of(scope) for scope in view[views.SCOPE_HEADER]} == {
        views.scope_word_class(region) for region in views.SCOPE_WORD_COLOURS
    }

    markup = views.table_html(
        views.plan_display(view.head(60)),
        views.plan_column_specs(),
        views.SCOPE_HEADER,
        class_of,
    )
    for region in views.SCOPE_WORD_COLOURS:
        assert views.scope_word_class(region) in markup, region
    # BOLD, which is the other half of what was asked for, and said once
    # in the stylesheet rather than on every one of 780 rows.
    assert "cm-word" in markup
    assert ".cm-word { font-weight: 700; }" in views.table_style()

    # No hex in the markup at all: the shades are in the stylesheet, as
    # pairs, and the row only names the class.
    assert "#" not in markup.replace("&#x27;", ""), "a hex reached a row"

def test_the_scope_words_have_no_mode_and_read_in_both_modes():
    """Siri 11/20, restated for the words that replaced the washes.

    The bug was never a wrong colour, it was a colour chosen at the wrong
    TIME: Streamlit's Light/Dark menu re-themes the browser WITHOUT
    rerunning the script, so a hex this module picked from the mode was
    still on the page after the reader had switched away from it. A
    light-dark() pair cannot do that, because the browser resolves it on
    every repaint - but only if nothing here ever picks one end of it.

    Four things, then. That no colour function takes a mode; that every
    word colour goes out as a PAIR; that both ends clear AA on their own
    page; and that the three regions are still told apart in both
    palettes, which is the part a prettier palette could quietly cost.
    """
    import inspect

    # 1. Not a mode in sight, and no way to pass one.
    for function in (
        views.live_colour, views.scope_word_class, views.kind_word_class,
        views.scope_class_of, views.kind_class_of, views.table_html,
        views.scope_line_html, views.banner_html, views.table_style,
    ):
        assert "mode" not in inspect.signature(function).parameters, (
            f"{function.__name__} takes a mode again"
        )

    # 2. A pair, not a pick, and both ends really are different.
    for region, pair in views.SCOPE_WORD_COLOURS.items():
        light, dark = pair
        assert light != dark, region
        assert views.live_colour(pair) == f"light-dark({light}, {dark})"

    # 3. Readable on its own page, measured against the real backgrounds.
    for name, pair in (
        list(views.SCOPE_WORD_COLOURS.items())
        + list(views.KIND_WORD_COLOURS.items())
    ):
        for mode, shade in zip(views.MODES, pair):
            ratio = views.contrast_ratio(shade, views.page_colour(mode))
            assert ratio >= views.AA_TEXT, (
                f"{mode} {name}: {shade} is only {ratio:.2f}:1"
            )

    # 4. And still three regions a reader can tell apart, in both.
    for mode, end in enumerate(views.MODES):
        shades = [pair[mode] for pair in views.SCOPE_WORD_COLOURS.values()]
        for first in range(len(shades)):
            for second in range(first + 1, len(shades)):
                apart = views.colour_distance(shades[first], shades[second])
                assert apart >= views.DISTINCT, (
                    f"{end}: {shades[first]} and {shades[second]} are only "
                    f"{apart:.1f} apart"
                )

def test_the_colour_lands_on_the_scope_word_and_nowhere_else(
    results, dimensions
):
    """Siri 11/19, still true of the word: one cell, not the whole row.

    Counted, not spot-checked. Every coloured span in the table is found
    and the column it sits in is worked out from its position in the row,
    so a colour that reached a figure would be caught rather than hidden
    by a test that only asks "is the colour in there?".
    """
    view = views.plan_view(results["scenarios"]["baseline"]["plan"]).head(40)
    specs = views.plan_column_specs()
    scope_at = [spec["column"] for spec in specs].index(views.SCOPE_HEADER)
    markup = views.table_html(
        views.plan_display(view), specs, views.SCOPE_HEADER,
        views.scope_class_of(dimensions),
    )

    coloured = set()
    for row_at, row in enumerate(re.findall(r"<tr>(.*?)</tr>", markup, re.S)):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        for column_at, cell in enumerate(cells):
            if "cm-word" in cell:
                coloured.add((row_at, column_at))
    assert coloured, "nothing was coloured at all"
    assert {column for _row, column in coloured} == {scope_at}, (
        "a colour reached a column that is not Scope: "
        f"{sorted({column for _row, column in coloured})}"
    )
    assert len(coloured) == len(view), (
        f"{len(coloured)} coloured words for {len(view)} rows"
    )

def test_the_colours_leave_the_figures_and_the_download_alone(
    results, dimensions
):
    """A colour is a thing done to a screen. The figures on screen and
    the full-precision bytes in the CSV are the same as before it.
    """
    view = views.plan_view(results["scenarios"]["volume_plus_15"]["plan"])
    specs = views.plan_column_specs()
    plain = views.plan_display(view)
    text = views.csv_bytes(view).decode("utf-8")

    # Every figure in the table is the display frame's own value, written
    # the way the workbook writes it - nothing is rounded, scaled or
    # dropped on the way into a cell.
    rows = table_rows(
        views.table_html(
            plain.head(25), specs, views.SCOPE_HEADER,
            views.scope_class_of(dimensions),
        )
    )
    for row, (_at, record) in zip(rows, plain.head(25).iterrows()):
        for cell, spec in zip(row, specs):
            assert cell == views.format_value(
                record[spec["column"]], spec.get("number_format")
            ), spec["column"]

    assert views.csv_bytes(view) == views.csv_bytes(view)
    assert "cm-word" not in text
    assert "light-dark" not in text

def test_the_scope_line_sits_beside_the_download(results, dimensions):
    """One line of coloured words, where the key with swatches used to be.

    The key is gone (11/24) because the colour it explained is gone: the
    scope IS the word now, so the line names the word and colours it from
    the same CSS rule the column does.
    """
    app = app_test().run()
    assert not app.exception
    blocks = [str(block.value) for block in app.markdown]
    # The opening tag, not the class name: the stylesheet mentions every
    # class it styles, and matching that would find CSS, not a line.
    line = [block for block in blocks if '<div class="cm-scope-line">' in block]
    assert line, "no scope line reached the page"
    for cluster in dimensions["clusters"]:
        assert views.scope_word_class(cluster["code"]) in line[0]
        assert cluster["name"] in line[0]
    assert dimensions["network_code"] in line[0]

    # The keys it replaced are gone rather than both being on the page.
    assert not any("cm-key-item" in block for block in blocks)
    assert not any("cm-key-table" in block for block in blocks)

def test_the_kind_column_carries_what_the_key_used_to_say(results):
    """The key is gone and its three sentences are on the column (11/24).

    A key under a table is a second place to look. The tooltip is on the
    header a reader is already pointing at when they wonder what the word
    beside a run means.
    """
    app = app_test().run()
    view_picker(app).set_value(views.COMPARE_VIEW).run()
    assert not app.exception

    markup = page_html(app)
    headers = dict(table_headers(markup))
    assert views.KIND_HEADER in headers, "no Kind column on the page"
    note = headers[views.KIND_HEADER]
    for kind in scenarios.KINDS:
        assert views.kind_label(kind).upper() in note.upper(), kind

    # Every run's kind is a coloured bold word in its own row.
    for kind in scenarios.KINDS:
        assert views.kind_word_class(kind) in markup, kind
    # And no hex beside it: the shades are in the stylesheet, as pairs.
    body = markup[markup.index("<tbody>"):]
    assert "color: #" not in body, "a kind label is written in a hex again"

    # The paragraph it replaced is gone, rather than both being shown.
    captions = [str(caption.value) for caption in app.caption]
    assert not any("A CHOICE is" in caption for caption in captions)

def test_the_kind_beside_a_run_in_the_picker_is_coloured(results):
    """In the picker, a kind is markdown, so the colour arrives as the
    name Streamlit resolves against the theme file.
    """
    app = app_test().run()
    captions = run_picker(app).captions
    assert captions
    for kind, name in views.KIND_MARKDOWN.items():
        wanted = f":{name}[{views.kind_label(kind)}]"
        assert any(wanted in caption for caption in captions), kind


def test_the_detail_table_writes_every_figure_the_way_the_cards_do():
    """One format per measure (7d), now that the table writes its own.

    The column settings used to do this and the grid obeyed them. An
    HTML cell is a string, so views.table_html formats every figure
    through format_value - the same function the cards use - and this is
    what checks the page still gets money with a dollar sign and hours to
    one decimal rather than raw floats.
    """
    app = app_test().run()
    assert not app.exception
    specs = views.plan_column_specs()
    rows = table_rows(page_html(app))
    assert rows, "no detail table on the page"

    at = {spec["column"]: index for index, spec in enumerate(specs)}
    money = rows[0][at[views.LABOUR_HEADER]]
    assert money.startswith("$") and "," in money, money
    hours = rows[0][at["Required hours"]]
    assert re.fullmatch(r"[\d,]+\.\d", hours), hours
    # And the headers carry the workbook's own name in their tooltip.
    headers = dict(table_headers(page_html(app)))
    assert any("Required hours" in note for note in headers.values())

def test_the_shade_lookup_answers_with_the_right_mode():
    """The one function every custom colour goes through.

    Worth a test of its own because every other colour test leans on it:
    if shade() ever returned the light hex for dark mode, each AA check
    would still pass - it would be measuring the wrong colour against the
    right background and getting a true answer to the wrong question.
    """
    pair = {views.LIGHT_MODE: "#111111", views.DARK_MODE: "#eeeeee"}
    assert views.shade(pair, views.LIGHT_MODE) == "#111111"
    assert views.shade(pair, views.DARK_MODE) == "#eeeeee"

    # No mode, or a mode nobody has heard of, is light - never an
    # exception, because a theme is not worth a blank page.
    assert views.DEFAULT_MODE == views.LIGHT_MODE
    for nonsense in (None, "", "Dark", "auto", "system", 0, object()):
        assert views.resolve_mode(nonsense) == views.LIGHT_MODE, nonsense
        assert views.shade(pair, nonsense) == "#111111", nonsense
    for mode in views.MODES:
        assert views.resolve_mode(mode) == mode


def per_mode_hexes():
    """Every hex this project declares twice, once per palette.

    The list the page is checked against. Each of these is a real colour
    that something needs - the page background, the accent, the hue a
    kind is derived from - and not one of them may be written into the
    page by this code, because a hex chosen here is chosen for whichever
    palette was showing when the script last ran.
    """
    pairs = {
        "accent": views.ACCENT_COLOURS,
        "page": views.PAGE_COLOURS,
        "sidebar": views.SIDEBAR_COLOURS,
        "text": views.TEXT_COLOURS,
        **{f"{kind} kind": pair for kind, pair in views.KIND_COLOURS.items()},
        **{f"{name} rendered": pair
           for name, pair in views.RENDERED_COLOURS.items()},
    }
    return {
        f"{name}/{mode}": pair[mode]
        for name, pair in pairs.items() for mode in views.MODES
    }


def test_every_custom_colour_has_both_shades():
    """Walked over the colours themselves, so a colour added later cannot
    be half a pair - which would be a KeyError on somebody's dark-mode
    screen and nowhere else.
    """
    pairs = {
        "accent": views.ACCENT_COLOURS,
        "page": views.PAGE_COLOURS,
        "sidebar": views.SIDEBAR_COLOURS,
        "text": views.TEXT_COLOURS,
        **{f"{kind} kind": pair for kind, pair in views.KIND_COLOURS.items()},
        **{f"{name} rendered": pair
           for name, pair in views.RENDERED_COLOURS.items()},
    }
    for name, pair in pairs.items():
        assert set(pair) == set(views.MODES), f"{name}: {sorted(pair)}"
        for mode in views.MODES:
            colour = pair[mode]
            assert re.fullmatch(r"#[0-9a-f]{6}", colour), f"{name}/{mode}"
        # A pair whose two shades are the same hex is a pair that was
        # never thought about. The no-change blue is the one colour that
        # could legitimately repeat, and it does not.
        assert pair[views.LIGHT_MODE] != pair[views.DARK_MODE], name

    # What is NOT on that list is the point of 11/20. The three card
    # colours are names, the scope tints and the kind tints are washes,
    # and none of the three has a mode to be half a pair of.
    for name in views.CARD_COLOURS:
        assert isinstance(name, str) and not name.startswith("#"), name
    for label, tints in (
        ("scope", views.REGION_TINTS), ("kind", views.KIND_TINTS)
    ):
        for key, tint in tints.items():
            assert views.is_wash(tint), f"{label} {key} is not a wash: {tint}"

    # The kind washes are mixed from the kind's own declared hue rather
    # than chosen beside it, so the theme file and the table cannot drift
    # into two different violets.
    for kind, pair in views.KIND_COLOURS.items():
        expected = "rgba(%d, %d, %d, %s)" % (
            *views._parts(pair[views.LIGHT_MODE]), views.TINT_ALPHA
        )
        assert views.kind_tint(kind) == expected, kind


def test_a_per_mode_hex_only_reaches_the_page_inside_a_pair(results):
    """THE GUARD (Siri, 11/20), restated for 11/24.

    The rule has never been "no hexes on the page". It is that nothing
    on the page may depend on which palette was showing when the script
    last ran - because switching Light/Dark re-themes the browser WITHOUT
    rerunning it, so a shade chosen here is correct when it is chosen and
    wrong the moment the reader switches.

    A light-dark(a, b) pair does not break that rule: BOTH shades go to
    the browser and the browser picks, every repaint. One of the two
    shades on its own does break it. So this test takes every pair out of
    what reached the page and then looks for a per-mode hex in what is
    left - which is the rule exactly as stated, rather than a ban on the
    character "#".
    """
    def drawn(app):
        blocks = [str(block.value) for block in app.markdown]
        blocks += [str(caption.value) for caption in app.caption]
        # The stylesheet, which is where every colour this page draws
        # itself with lives.
        blocks.append(page_style(app))
        # The sidebar's run captions carry the kinds, and a radio keeps
        # its captions in its proto rather than sending them as markdown.
        blocks += [str(radio.proto.captions) for radio in app.radio]
        return " ".join(blocks)

    page = ""
    # Three runs, chosen because between them they draw all three card
    # colours: the baseline draws none at all (nothing to compare
    # against). Centralized phones saves people (green), Volume +15%
    # costs them (red), and the wider lunch window moves the FTE by four
    # thousandths, which rounds away (blue).
    for key in ("central_phones", "volume_plus_15", "wider_lunch_window"):
        app = app_test().run()
        run_picker(app).set_value(key).run()
        assert not app.exception, key
        page += " " + drawn(app)

    app = app_test().run()
    view_picker(app).set_value(views.COMPARE_VIEW).run()
    assert not app.exception
    page += " " + drawn(app)

    # Both shades really are being sent, or the rest of this test would
    # pass for the wrong reason.
    assert "light-dark(" in page, "no pair reached the page at all"
    for pair in list(views.SCOPE_WORD_COLOURS.values()) + [views.EXCEL_COLOUR]:
        assert views.live_colour(pair) in page, pair

    outside_a_pair = re.sub(r"light-dark\([^)]*\)", " ", page)
    stray = {
        name: colour for name, colour in per_mode_hexes().items()
        if colour.lower() in outside_a_pair.lower()
    }
    assert not stray, (
        "a hex chosen per mode reached the page outside a light-dark() "
        f"pair, and will be the wrong one the moment a reader switches: "
        f"{stray}"
    )

    # And the live forms really are there too.
    assert f":{views.SAME_COLOUR}[" in page, "no card colour name drawn"
    assert views.kind_markdown(scenarios.KIND_CHOICE) in page, "no kind drawn"

def test_nothing_coloured_is_drawn_with_a_mode():
    """Read off the page's own source, which is the only way to catch a
    call site that quietly passed one.

    THE OPPOSITE OF WHAT THIS TEST USED TO ASK (11/20). It used to
    require every coloured call to be handed the mode, which was the
    right discipline for a page that reruns - and this page does not.
    Now nothing may be handed one, because there is no longer anything on
    the page that a mode could correctly answer.
    """
    source = (
        Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
    ).read_text(encoding="utf-8")

    def arguments_of(call):
        found = re.findall(re.escape(call) + r"([^)]*)\)", source, re.S)
        assert found, f"{call} is not called by the page at all"
        return found

    for call in (
        "views.card_html(",
        "views.card_markdown(",
        "views.table_html(",
        "views.table_style(",
        "views.scope_line_html(",
        "views.scope_class_of(",
        "views.kind_class_of(",
        "views.banner_html(",
        "views.banner_style(",
        "views.download_style(",
    ):
        for arguments in arguments_of(call):
            assert "mode" not in arguments, (
                f"{call}{arguments}) takes a mode again - nothing drawn by "
                "this page may depend on which palette is showing"
            )

    # The page no longer asks which palette is showing, because there is
    # nothing left for the answer to decide. A reader of this file should
    # not have to take that on trust.
    assert "active_mode" not in source.replace(
        "# There used to be an active_mode() here", ""
    ), "the page is asking which palette is showing again"

    # Nor may any of the painting functions still accept one.
    import inspect

    for name in (
        "card_colour", "card_markdown", "card_html", "kind_tint",
        "kind_markdown", "region_tint", "live_colour", "live_wash",
        "table_html", "table_style", "scope_line_html", "banner_html",
        "banner_style", "download_style", "scope_class_of", "kind_class_of",
    ):
        taken = inspect.signature(getattr(views, name)).parameters
        assert "mode" not in taken, f"views.{name} takes a mode again"

def dark_mode_context(monkeypatch):
    """Make the real page believe the reader is in dark mode.

    Patched at Streamlit's own context - the one thing the app used to
    ask - rather than at views.resolve_mode. Patching the context leaves
    every default in place, so anything that did still choose a shade
    from the mode would really change, and be caught.
    """
    from streamlit.runtime.context import ContextProxy, StreamlitTheme

    monkeypatch.setattr(
        ContextProxy,
        "theme",
        property(lambda self: StreamlitTheme({"type": views.DARK_MODE})),
        raising=True,
    )


def test_a_dark_mode_reader_gets_exactly_the_same_page(results, monkeypatch):
    """The end of the chain, on the real page: ONE page serves both.

    The strongest form the 11/20 rule can be stated in. Nothing this app
    draws depends on which palette is showing, so the markup a dark-mode
    reader is sent must be IDENTICAL to the markup a light-mode reader is
    sent - the same colour names, the same washes, byte for byte. The
    difference between the two is made in the browser, by Streamlit and
    by the compositor, which is the only place that can make it at the
    moment the reader asks for it.

    Before 11/20 this test checked the opposite: that a dark reader got
    the dark shades and not one light one. That was the right check for a
    page that picked shades, and the reason it passed while the page was
    broken is that AppTest reruns on every interaction - which is exactly
    what the theme menu does not do.
    """
    def drawn(app):
        # EVERYTHING this page draws is markup now (11/24): the tables as
        # well as the banners and the cards, so the comparison is the
        # markup itself with nothing normalised away. It used to need a
        # Styler's class hashes normalised out, and those are gone with
        # the Styler.
        return "\n".join(str(block.value) for block in app.markdown)

    def render(key=None, view=None):
        app = app_test().run()
        if view is not None:
            view_picker(app).set_value(view).run()
        if key is not None:
            run_picker(app).set_value(key).run()
        assert not app.exception, (key, view)
        return drawn(app)

    light = [render(key="central_phones"), render(view=views.COMPARE_VIEW)]
    assert light[0].strip(), "nothing drawn at all"

    dark_mode_context(monkeypatch)
    dark = [render(key="central_phones"), render(view=views.COMPARE_VIEW)]

    for at, (one, other) in enumerate(zip(light, dark)):
        assert one == other, (
            f"view {at}: the page a dark reader is sent differs from the "
            "page a light reader is sent, so something on it is still "
            "being chosen from the mode"
        )

def test_the_contrast_and_distance_arithmetic_is_right():
    """The two yardsticks every colour test leans on, against known
    values, so a bug in the maths cannot quietly pass the palette.
    """
    assert views.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert views.contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0)
    # The textbook pair: #767676 is the lightest gray that passes AA on
    # white, at 4.54:1.
    assert views.contrast_ratio("#767676", "#ffffff") == pytest.approx(
        4.54, abs=0.01
    )
    # Delta-E runs 0 to 100 over the lightness axis, and is symmetric.
    assert views.colour_distance("#000000", "#ffffff") == pytest.approx(100.0)
    assert views.colour_distance("#3a7a8c", "#3a7a8c") == pytest.approx(0.0)
    assert views.colour_distance("#0f7034", "#bc2626") == pytest.approx(
        views.colour_distance("#bc2626", "#0f7034")
    )
    # A nudge of one step per channel is below the threshold anybody can
    # see (the textbook figure is about 2.3), which is the property that
    # makes this a better test of "could these be confused?" than
    # comparing hex strings: two colours can differ and still be the
    # same colour to look at.
    assert views.colour_distance("#3a7a8c", "#3b7b8d") < 2.3

    # And it orders the palette the way an eye does: the teal accent is
    # further from the no-change blue than from the better-green, which
    # is why the blue was the colour that had to move.
    assert views.colour_distance("#3a7a8c", "#2048cc") > views.colour_distance(
        "#3a7a8c", "#0f7034"
    )


def test_the_sidebar_says_where_light_and_dark_are_changed(results):
    """Streamlit 1.67 has no API for setting the theme from page code -
    st.context.theme is read-only - so the app points at Streamlit's own
    switch instead of faking one. The switch is easy to miss, and it only
    exists because this app ships two palettes.
    """
    app = app_test().run()
    assert not app.exception
    captions = [str(caption.value) for caption in app.sidebar.caption]
    note = [
        caption for caption in captions
        if "Light or dark" in caption
    ]
    assert note, f"no light/dark note in the sidebar: {captions}"
    assert len(note) == 1, note
    # It has to say WHERE, and that the default follows the machine.
    assert "top right" in note[0]
    assert "System" in note[0]
    # One line, as asked.
    assert "\n" not in note[0]


def test_a_filtered_view_colours_each_row_by_its_own_scope(
    results, dimensions
):
    """The bug this guards against, which a full-table test cannot see.

    Filtering hands the table a view whose index is whatever survived the
    filter - 418, 419, 423 - not 0, 1, 2. Anything that lined colours up
    with POSITIONS while reading values by LABEL would give every row
    some other row's colour, and on the unfiltered table the two agree,
    so it would look perfect until somebody used a filter.

    The table writer reads each record whole, so the two cannot come
    apart by construction - and this is what says so, row by row, on a
    view whose labels are deliberately not its positions.
    """
    view = views.plan_view(results["scenarios"]["baseline"]["plan"])
    sliced = view.iloc[10:30]
    assert list(sliced.index) != list(range(len(sliced))), (
        "this test needs a view whose labels are not its positions"
    )

    specs = views.plan_column_specs()
    scope_at = [spec["column"] for spec in specs].index(views.SCOPE_HEADER)
    markup = views.table_html(
        views.plan_display(sliced), specs, views.SCOPE_HEADER,
        views.scope_class_of(dimensions),
    )

    body = markup[markup.index("<tbody>"):]
    rows = re.findall(r"<tr>(.*?)</tr>", body, re.S)
    assert len(rows) == len(sliced), (len(rows), len(sliced))
    for row, scope in zip(rows, sliced[views.SCOPE_HEADER]):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        cell = cells[scope_at]
        expected = views.scope_word_class(views.region_of(scope, dimensions))
        assert expected in cell, (scope, cell)
        assert scope in cell, (scope, cell)


# ---------------------------------------------------------------------------
# The rest of the outside reviewers' fixes (11/24)
# ---------------------------------------------------------------------------


def test_the_sort_control_reorders_the_rows():
    """What replaced clicking a column header.

    An HTML table cannot sort itself, so the sort happens in pandas and
    the table is written again. Both ends are checked, because a control
    that only ever sorted one way would pass a test that only looked at
    one of them.
    """
    app = app_test().run()
    specs = views.plan_column_specs()
    at = {spec["column"]: index for index, spec in enumerate(specs)}
    column = views.LABOUR_HEADER

    sorter = next(box for box in app.selectbox if box.label == "Sort by")
    assert sorter.options[0] == views.SORT_AS_SHOWN
    assert column in sorter.options

    sorter.set_value(column).run()
    assert not app.exception
    money = [
        float(row[at[column]].replace("$", "").replace(",", ""))
        for row in table_rows(page_html(app))
    ]
    assert money == sorted(money), "low to high is not sorted"

    app.toggle[0].set_value(True).run()
    assert not app.exception
    money = [
        float(row[at[column]].replace("$", "").replace(",", ""))
        for row in table_rows(page_html(app))
    ]
    assert money == sorted(money, reverse=True), "high to low is not sorted"

    # And the view's own order is still on offer, and still the default
    # a reader lands on.
    app.toggle[0].set_value(False).run()
    sorter = next(box for box in app.selectbox if box.label == "Sort by")
    sorter.set_value(views.SORT_AS_SHOWN).run()
    weeks = [int(row[at["Plan week"]]) for row in table_rows(page_html(app))]
    assert weeks == sorted(weeks), "the view's own order is not by week"


def test_changing_the_run_sends_the_reader_back_to_the_top():
    """The scroll, as the only thing a headless test can see of it.

    Streamlit has no scroll API, so the page sends a one-line script in
    an iframe of a single collapsed pixel. What is checkable here is
    WHEN it is sent: on a change of run and not otherwise. The scroll
    itself is measured in a browser, in test_live_colours.

    Not on first load, because there is nothing to scroll back from, and
    not on a filter change, because a reader who moves a filter is
    looking at the thing they just filtered.
    """
    app = app_test().run()
    assert len(app.get("iframe")) == 0, "the page scrolled on first load"

    run_picker(app).set_value("volume_plus_15").run()
    assert not app.exception
    assert len(app.get("iframe")) == 1, "picking a run did not scroll"

    app.get("multiselect")[0].set_value(["records_specialist"]).run()
    assert not app.exception
    assert len(app.get("iframe")) == 0, "a filter change scrolled the page"

    # The script scrolls the element Streamlit actually scrolls, which is
    # the main block and not the window - measured in the browser on
    # 11/24, and the one thing about it that is easy to get wrong.
    run_picker(app).set_value("central_phones").run()
    script = app.get("iframe")[0].proto.srcdoc
    assert 'data-testid="stMain"' in script
    assert "scrollTo" in script

    # AND THE SCRIPT NAMES THE RUN, which is not decoration: an iframe
    # whose srcdoc is unchanged is the same DOM node after a rerun, and a
    # browser does not re-run a script it has already run - so one fixed
    # script scrolled exactly once per session and then silently stopped.
    assert "central_phones" in script
    run_picker(app).set_value("volume_plus_15").run()
    second = app.get("iframe")[0].proto.srcdoc
    assert second != script, (
        "two runs sent the same frame, so the second one will not scroll"
    )


def test_the_two_downloads_are_each_in_their_own_keyed_container():
    """How a stylesheet reaches a widget this page does not own.

    Streamlit writes a container's key onto the element as a class, and
    that class is the only handle the CSS gets. If a key is renamed in
    one place and not the other, nothing raises - the buttons just go
    back to being Streamlit's blue and gray - so the two are checked
    against each other here.
    """
    app = app_test().run()
    assert not app.exception
    sheet = page_style(app)
    for key in (views.EXCEL_BUTTON_KEY, views.CSV_BUTTON_KEY):
        assert f".st-key-{key} button" in sheet, key

    # Both hues are really in the sheet, as pairs, and they differ.
    assert views.live_colour(views.EXCEL_COLOUR) in sheet
    assert views.live_colour(views.CSV_COLOUR) in sheet
    assert views.EXCEL_COLOUR != views.CSV_COLOUR

    # And the label is left to the theme rather than given a shade.
    assert "color: inherit !important" in sheet


def test_every_colour_the_page_draws_itself_with_goes_out_as_a_pair():
    """The 11/20 rule as a property of the stylesheets themselves.

    Each sheet is built from the recorded pairs, so the check is that no
    single shade escaped into one: every hex in every sheet has to sit
    inside a light-dark() pair, because a lone hex is a shade chosen for
    whichever palette happened to be showing.
    """
    sheets = {
        "table": views.table_style(),
        "banner": views.banner_style(),
        "download": views.download_style(),
        "collapse": views.collapse_style(),
    }
    for name, sheet in sheets.items():
        outside = re.sub(r"light-dark\([^)]*\)", " ", sheet)
        assert "#" not in outside, (
            f"the {name} stylesheet has a colour outside a pair: "
            f"{outside[max(0, outside.index('#') - 40):][:80]}"
        )

    # And the pairs really are there, rather than a sheet with no colour
    # in it passing for the wrong reason.
    assert views.live_colour(views.SYNTHETIC_COLOUR) in sheets["banner"]
    for region, pair in views.SCOPE_WORD_COLOURS.items():
        assert views.live_colour(pair) in sheets["table"], region


# ---------------------------------------------------------------------------
# Round 2 of the screenshot review (Siri, 11/25)
# ---------------------------------------------------------------------------


def test_a_negative_amount_puts_the_sign_before_the_dollar():
    """-$51,569, never $-51,569 (Siri, 11/25).

    "$-51,569" is not how anybody writes money, and for the moment it
    takes to read it, it is a dollar amount of minus something. The sign
    belongs to the quantity, so it comes first - the same way the cards
    have always written a difference.
    """
    assert views.money(-51568.94) == "-$51,569"
    assert views.money(51568.94) == "$51,569"
    assert views.money(-51568.94, 2) == "-$51,568.94"
    # A figure that rounds to nothing gets no minus sign: a zero with a
    # sign on it is a rounding artefact wearing a direction.
    assert views.money(-0.4) == "$0"
    assert views.money(0.0) == "$0"

    for number_format in (export.MONEY, export.MONEY_CENTS):
        assert views.format_value(-1234.5, number_format).startswith("-$")
        assert "$-" not in views.format_value(-1234.5, number_format)


def test_no_figure_anywhere_on_the_page_reads_dollar_minus():
    """The property, on the real page, in every view that shows money.

    The comparison table is where it showed - its "vs base" column is
    negative for every run that saves anything - but the rule is about
    the page, so the page is what is checked.
    """
    for view in (views.ONE_RUN_VIEW, views.COMPARE_VIEW):
        app = app_test().run()
        if view != views.ONE_RUN_VIEW:
            view_picker(app).set_value(view).run()
        assert not app.exception, view
        markup = page_html(app)
        assert "$-" not in markup, f"{view} writes a dollar before the sign"
        assert "$ -" not in markup, view

    # And the figures really are negative, or this passes for the wrong
    # reason: the ranked table opens with the biggest saving.
    app = app_test().run()
    view_picker(app).set_value(views.COMPARE_VIEW).run()
    rows = table_rows(page_html(app))
    at = [spec["column"] for spec in views.comparison_column_specs(
        views.comparison_table(api.load_results())
    )].index(views.DIFFERENCE_HEADER)
    assert rows[0][at].startswith("-$"), rows[0][at]


def test_the_comparison_headers_are_short_with_the_full_name_on_hover():
    """Eleven full workbook headers did not fit the page (Siri, 11/25).

    The last column was cut off at 1440 with the sidebar open, so the
    headers are shortened the way the detail table's have been since 7d.
    What matters is that nothing is LOST by shortening: every shortened
    header still carries the workbook's own name at the front of its
    tooltip, so a reader can find the column on the tab.
    """
    table = views.comparison_table(api.load_results())
    specs = views.comparison_column_specs(table)
    for spec in specs:
        assert len(spec["label"]) <= 14, spec["label"]
        if spec["label"] != spec["column"]:
            assert spec["help"].startswith(spec["column"]), spec["column"]
        assert spec["help"], spec["column"]

    # Every column of the table has a spec, and every shortened name is
    # read out of the one table rather than invented twice.
    assert [spec["column"] for spec in specs] == list(table.columns)
    for header, short in views.COMPARISON_SHORT.items():
        assert header in table.columns, header
        assert short != header

    # On the page, the headers are the short ones and the tooltips the
    # long ones.
    app = app_test().run()
    view_picker(app).set_value(views.COMPARE_VIEW).run()
    headers = dict(table_headers(page_html(app)))
    assert "Turnaround" in headers
    assert headers["Turnaround"].startswith("Requester turnaround, mean days")


def test_the_baseline_is_named_rather_than_classified(results):
    """The baseline's Kind reads "Baseline", in no colour (Siri, 11/25).

    Its file declares a kind because every run must, but on a table
    ranked AGAINST the baseline, calling the baseline a choice reads as
    if somebody were proposing it. It is the line the others are
    measured from, so it is named - and it takes no kind colour, which
    falls out of the lookup rather than being special-cased in the CSS.
    """
    table = views.comparison_table(results)
    kinds = dict(zip(table["Scenario"], table[views.KIND_HEADER]))
    assert kinds[views.BASELINE_LABEL] == views.BASELINE_KIND_LABEL
    assert views.BASELINE_KIND_LABEL not in [
        views.kind_label(kind) for kind in scenarios.KINDS
    ]
    assert views.kind_class_of()(views.BASELINE_KIND_LABEL) is None

    # Every other run still declares one of the three kinds.
    for name, kind in kinds.items():
        if name == views.BASELINE_LABEL:
            continue
        assert kind in [views.kind_label(one) for one in scenarios.KINDS], name

    # And on the page, the baseline's cell carries no kind class.
    app = app_test().run()
    view_picker(app).set_value(views.COMPARE_VIEW).run()
    markup = page_html(app)
    body = markup[markup.index("<tbody>"):]
    row = next(
        line for line in re.findall(r"<tr>(.*?)</tr>", body, re.S)
        if f">{views.BASELINE_LABEL}<" in line
    )
    assert "cm-kind-" not in row, "the baseline row is coloured as a kind"
    assert views.BASELINE_KIND_LABEL in row


def test_both_downloads_are_strong_enough_to_see_in_either_palette():
    """Dark mode barely showed a colour at all (Siri, 11/25).

    The same fraction of a hue does not read the same over both pages: a
    10% violet over cream is a visible tint, and 10% of a pale violet
    over charcoal is almost the charcoal. So the fill takes two alphas,
    and the dark one is the stronger.

    Checked by compositing, which is what the browser does: each fill is
    mixed over its own page and has to land a measurable distance from
    it. The browser test measures the painted pixels; this one says what
    the stylesheet is asking for.
    """
    # ONE ALPHA PER BUTTON, and the louder hue takes the lower of the
    # two: they are not equally loud, so one shared strength leaves
    # either slate invisible or violet shouting.
    assert views.EXCEL_FILL_ALPHA != views.CSV_FILL_ALPHA
    assert views.HOVER_BOOST > 0

    def over(pair, mode, alpha):
        shade = pair[0] if mode == views.LIGHT_MODE else pair[1]
        return views.composite_over(
            f"rgba({', '.join(str(part) for part in views._parts(shade))}"
            f", {alpha})",
            views.page_colour(mode),
        )

    for mode in views.MODES:
        page = views.page_colour(mode)
        excel = over(views.EXCEL_COLOUR, mode, views.EXCEL_FILL_ALPHA)
        csv = over(views.CSV_COLOUR, mode, views.CSV_FILL_ALPHA)
        # Each stands out from the page it sits on...
        for name, fill in (("excel", excel), ("csv", csv)):
            apart = views.colour_distance(fill, page)
            assert apart >= views.DIFFERENT, (
                f"{mode}: the {name} fill is only {apart:.1f} from the page"
            )
            ratio = views.contrast_ratio(views.text_colour(mode), fill)
            assert ratio >= views.AA_TEXT, f"{mode}: {name} label {ratio:.2f}:1"
        # ...and from each other, which is the half that failed first.
        assert views.colour_distance(excel, csv) >= views.DIFFERENT, (
            f"{mode}: the two buttons are "
            f"{views.colour_distance(excel, csv):.1f} apart"
        )
        # Hover is stronger and still readable - the one state a reader
        # is looking straight at.
        for pair, alpha in (
            (views.EXCEL_COLOUR, views.EXCEL_FILL_ALPHA),
            (views.CSV_COLOUR, views.CSV_FILL_ALPHA),
        ):
            hovered = over(pair, mode, alpha + views.HOVER_BOOST)
            assert views.colour_distance(hovered, page) > views.colour_distance(
                over(pair, mode, alpha), page
            )
            assert views.contrast_ratio(
                views.text_colour(mode), hovered
            ) >= views.AA_TEXT, f"{mode}: hover label"

    sheet = views.download_style()
    assert views.BUTTON_EDGE in sheet
    assert str(views.EXCEL_FILL_ALPHA) in sheet
    assert str(views.CSV_FILL_ALPHA) in sheet


def test_the_note_banner_is_a_wash_in_both_palettes():
    """The soft MIRROR of the dark design, not its inverse (Siri, 11/25).

    The first attempt read "reversed" as swapping the two ends, which on
    a cream page is a solid blue slab with white text - louder than the
    finding inside it. A mirror keeps the design and changes only what
    has to change: the same translucent blue over whichever page is
    showing, the text in the blue that reads on that page, a thin edge.
    """
    sheet = views.banner_style()
    assert views.live_wash(
        views.NOTE_COLOUR, views.PANEL_ALPHA, views.PANEL_DARK_ALPHA
    ) in sheet, "the note panel is not a wash in both palettes"
    assert views.live_colour(views.NOTE_COLOUR) in sheet
    assert "3px solid" in sheet, "the note edge is not thin"
    # No pale text colour left behind from the inverted version.
    assert "#f8f9fc" not in sheet
    assert not hasattr(views, "NOTE_REVERSED_TEXT")

    # Readable both ways, composited the way the browser will.
    for mode, shade in zip(views.MODES, views.NOTE_COLOUR):
        alpha = (
            views.PANEL_ALPHA if mode == views.LIGHT_MODE
            else views.PANEL_DARK_ALPHA
        )
        page = views.page_colour(mode)
        over = views.composite_over(
            f"rgba({', '.join(str(part) for part in views._parts(shade))}"
            f", {alpha})",
            page,
        )
        ratio = views.contrast_ratio(shade, over)
        assert ratio >= views.AA_TEXT, f"{mode}: {ratio:.2f}:1"


def test_the_sort_control_and_the_download_note_say_it_plainly():
    """Wording (Siri, 11/25): "Default order", and a caption in plain words."""
    assert views.SORT_AS_SHOWN == "Default order"
    assert views.sort_options(views.plan_column_specs())[0] == "Default order"
    assert "Default order" in views.SORT_HELP

    app = app_test().run()
    sorter = next(box for box in app.selectbox if box.label == "Sort by")
    assert sorter.value == "Default order"

    captions = [str(caption.value) for caption in app.caption]
    note = next(
        caption for caption in captions if "SYNTHETIC is in the file name"
        in caption
    )
    assert "full-precision" in note
    assert "current filters" in note
    # Short enough to be read at a glance, which is the point of the
    # rewrite: the sentence it replaced ran to 34 words.
    assert len(note.split()) <= 20, note


# ---------------------------------------------------------------------------
# The chart labels read left to right (Siri, 11/25, from the live app)
# ---------------------------------------------------------------------------


def test_the_cost_chart_writes_its_run_names_flat(results):
    """Found in the live app: the run names were drawn on their side.

    Vega-Lite rotates a band axis's labels the moment it decides they
    do not fit, and it decided that here - measured in the browser,
    both labels came out at rotate(270) AND cut short by the 180px
    default limit, so the longest read "Site shortfall N1, ..."
    sideways. Setting the angle turns the auto-rotation off for good.
    """
    spec = views.cost_chart(views.cost_split(results, "combined")).to_dict()
    axis = spec["encoding"]["x"]["axis"]
    assert axis["labelAngle"] == 0
    assert views.LABEL_FLAT == 0

    # Wide enough for the longest run name this model can produce, so
    # no name is handed to a reader with its end cut off.
    assert axis["labelLimit"] >= 200, axis["labelLimit"]
    longest = max(
        len(run["name"]) for run in results["scenarios"].values()
    )
    assert longest <= 32, f"a run name grew to {longest} characters"

    # A hidden label is worse than a cramped one: on a two-bar chart an
    # unlabelled bar is a bar nobody can identify.
    assert axis["labelOverlap"] is False


def test_both_legends_read_along_the_bottom(results):
    """Stacked up the right-hand side, each entry on its own line.

    Along the bottom they read as one line of text under the picture
    they describe, in the direction the rest of the page is read.
    """
    gap = views.gap_chart(views.weekly_gap(results, "combined"), 52).to_dict()
    cost = views.cost_chart(views.cost_split(results, "combined")).to_dict()
    for spec, what in ((gap, "shortfall"), (cost, "cost")):
        legend = spec["encoding"]["color"]["legend"]
        assert legend["orient"] == "bottom", what
        assert legend["direction"] == "horizontal", what
        # The longest run name drew at 159px against Vega-Lite's 160px
        # default: the name that fits today was one character from an
        # ellipsis. These clear it.
        assert legend["labelLimit"] >= 240, what
        assert legend["titleLimit"] >= 280, what
        # And the legend still says what it is keyed on.
        assert legend["title"], what


def test_moving_the_labels_left_both_charts_locked(results):
    """The one thing that must not change while the labels do.

    Axis and legend settings add no parameter to a spec, so neither
    chart can have picked up a scale binding on the way - but that is
    the sort of thing that is true until it isn't, and this chart was
    locked on purpose after a leader scrolled one out of scale.
    """
    gap = views.gap_chart(views.weekly_gap(results, "combined"), 52)
    cost = views.cost_chart(views.cost_split(results, "combined"))
    for chart in (gap, cost):
        assert views.chart_is_locked(chart)
        assert views._scale_bindings(chart.to_dict()) == []

    # And the axes are still the ones this code chose.
    gap_scale = gap.to_dict()["encoding"]["x"]["scale"]
    assert gap_scale["domain"] == [1, 52]
    assert gap_scale["nice"] is False


def test_the_chart_labels_are_readable_in_both_palettes():
    """Siri's "both modes readable", measured rather than trusted.

    Streamlit's Vega theme paints every axis and legend label #808495,
    which on this page's cream is 3.48:1 - under AA, and under the bar
    every other word on the page is held to. Dark was never in doubt.
    The replacement is Streamlit's own gray darkened the smallest
    number of steps that clears AA with a margin.
    """
    sheet = views.chart_style()
    # A PAIR, like every other colour this page draws: nothing in the
    # markup depends on which palette is showing (the 11/20 rule).
    assert views.live_colour(views.CHART_LABEL_COLOUR) in sheet
    assert "role-axis-label" in sheet and "role-legend-label" in sheet
    # Vega writes fill as an attribute, so an ordinary rule beats it.
    assert "!important" not in sheet

    for mode, shade in zip(views.MODES, views.CHART_LABEL_COLOUR):
        ratio = views.contrast_ratio(shade, views.page_colour(mode))
        assert ratio >= views.AA_TEXT, f"{mode}: {ratio:.2f}:1"

    # Still a muted label gray, not body text promoted onto the axis:
    # the light shade stays close to the colour it replaces.
    assert views.colour_distance(views.CHART_LABEL_COLOUR[0], "#808495") < 20


def test_the_chart_stylesheet_reaches_the_page():
    """The sheet is no use sitting in views if the page never sends it."""
    app = app_test().run()
    assert not app.exception
    assert views.live_colour(views.CHART_LABEL_COLOUR) in page_style(app)
