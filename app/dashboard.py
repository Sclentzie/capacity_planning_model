"""The capacity model dashboard: the browse view (spec Section 8, 7b).

Two views. ONE RUN: pick a run, filter it, and see what it did to the
plan. COMPARE ALL SCENARIOS: every run at once, ranked by what it does to
the all-in cost, with each run's declared kind beside it.

Every figure on the page comes from the saved results that
capacity_model.api built, and is selected and labelled by
capacity_model.views - so the page holds no arithmetic of its own, and a
number here is the same number the workbook prints on the tab of the same
name.

WHY IT READS A FILE. Running the 21 scenarios takes about eight minutes.
A hosted app that did that on every page view would be unusable, so the
runs are saved by `python -m capacity_model.api`, committed, and loaded
here in well under a second. A build-your-own run that goes through the
doorway on demand is 7c.

WHAT IS NOT HERE YET: the design pass (7d). This is the functional view.

Run it:
    .venv\\Scripts\\python -m streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

import streamlit as st

# The model lives under src/ and is not installed as a package, so the
# path is added here rather than relying on PYTHONPATH - a hosted app has
# no shell to set it in.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from capacity_model import api, views  # noqa: E402  (after the path is set)

TITLE = "Northbrook Records Services - capacity model (SYNTHETIC)"

# st.metric colours a rise green and a fall red by default. For cost,
# overtime, uncovered hours and turnaround that is backwards, and for
# required FTE it is a verdict the model has not made - so the direction
# recorded in views decides, in one place, rather than per card.
DELTA_COLOURS = {views.BETTER_LOWER: "inverse", views.NEUTRAL: "off"}

CARDS_PER_ROW = 3

# Widget keys, so a test can ask for the run picker by name rather than
# by position. Adding the view switcher above the run picker renumbered
# every index-based lookup in the tests, which is a thing a test should
# not be able to notice.
VIEW_KEY = "view"
RUN_KEY = "run"
SCOPE_KEY = "scope"
ROLES_KEY = "roles"

ONE_RUN = "One run"
COMPARE_ALL = "Compare all scenarios"
VIEWS = (ONE_RUN, COMPARE_ALL)


@st.cache_data
def saved_results():
    """The saved runs, read once per session rather than per rerun."""
    return api.load_results()


@st.cache_data
def filter_dimensions():
    """Sites, clusters and roles, with the names the YAML gives them."""
    return api.dimensions()


@st.cache_data
def current_inputs():
    """A fingerprint of the input files as they are on disk right now."""
    return api.inputs_fingerprint()


def synthetic_banner(meta):
    """The synthetic statement, on the page itself.

    Not a footnote: a screenshot of a dashboard travels further than the
    repository it came from, and whoever sees the screenshot has to be
    able to tell that Northbrook Records Services is invented.
    """
    st.warning(meta["label"], icon=":material/science:")


def staleness_banner(meta):
    """Say so if the inputs on disk are not the ones that were run.

    Cheap honesty: it compares hashes, so it catches the ordinary case of
    an assumption changed without a rebuild. It cannot catch a changed
    formula, which is what the test that recomputes every cell is for.
    """
    saved = meta.get("inputs") or {}
    if not saved or saved == current_inputs():
        return
    changed = sorted(
        name for name, mark in saved.items() if current_inputs().get(name) != mark
    )
    st.error(
        "These saved results were built from different inputs ("
        + ", ".join(changed)
        + "). Rebuild with `python -m capacity_model.api` before trusting "
        "the figures below.",
        icon=":material/warning:",
    )


def caption_for(entry):
    """The one line under a run's name in the picker.

    The kind is the run's own declared field, so a risk cannot be read as
    a proposal just because it sits in a list of them.
    """
    parts = [entry["kind_label"]]
    if entry["key"] == views.BASELINE_KEY:
        parts = [views.BASELINE_GROUP]
    elif entry["group"] == views.HEADLINE_GROUP:
        parts.insert(0, "Headline")
    return " - ".join(parts)


def pick_run(results):
    """The sidebar run picker: the eight headlines first, then the rest."""
    items = views.menu_items(results)
    keys = [entry["key"] for entry in items]
    label_of = {entry["key"]: entry["name"] for entry in items}
    captions = [caption_for(entry) for entry in items]

    st.sidebar.subheader("The run")
    return st.sidebar.radio(
        "Scenario",
        keys,
        format_func=lambda key: label_of[key],
        captions=captions,
        label_visibility="collapsed",
        key=RUN_KEY,
    )


def pick_filters(dimensions):
    """The sidebar filters: one scope grain, then the roles inside it."""
    st.sidebar.subheader("What to look at")

    choices = views.scope_choices(dimensions)
    choice = st.sidebar.selectbox(
        "Network, cluster or site",
        choices,
        format_func=lambda option: option["label"],
        key=SCOPE_KEY,
    )

    offered = views.roles_in_scope(dimensions, choice)
    codes = [role["code"] for role in offered]
    name_of = {role["code"]: role["name"] for role in offered}
    picked = st.sidebar.multiselect(
        "Roles",
        codes,
        default=codes,
        format_func=lambda code: name_of[code],
        key=ROLES_KEY,
    )

    note = views.scope_note(dimensions, choice)
    if note:
        st.sidebar.info(note, icon=":material/info:")

    return choice, picked


def show_run_header(entry):
    """What this run is and when it starts, before any figure."""
    st.subheader(entry["name"])
    st.markdown(f"**Changes:** {entry['what_it_changes']}")
    if entry["key"] != views.BASELINE_KEY:
        st.caption(
            f"The change starts in plan week {entry['starts_week']}; the "
            "figures below are the whole 52-week plan year, which is why a "
            "late-starting scenario can look mild."
        )
    if entry["finding"]:
        # The workbook's own approved sentence for this run. A difference
        # column says a scenario is cheaper; only the finding says what
        # the cheapness cost.
        st.info(entry["finding"], icon=":material/lightbulb:")


def show_cards(results, key):
    """The six headline measures, each against the baseline."""
    st.markdown("#### Against the baseline")
    st.caption(
        "Network-wide, as the workbook's Summary reports them. The filters "
        "in the sidebar apply to the chart and the table below, not to "
        "these cards."
    )
    cards = views.headline_cards(results, key)
    for start in range(0, len(cards), CARDS_PER_ROW):
        row = cards[start:start + CARDS_PER_ROW]
        for column, card in zip(st.columns(len(row)), row):
            column.metric(
                card["label"],
                card["value_text"],
                delta=card["difference_text"] or None,
                delta_color=DELTA_COLOURS.get(card["direction"], "normal"),
            )


def show_charts(results, key, choice, roles):
    """The weekly shortfall, and where the money goes."""
    scopes = choice["scopes"]

    st.markdown("#### Weekly shortfall, scenario against baseline")
    gap = views.weekly_gap(results, key, roles, scopes)
    # Built as a locked spec rather than drawn by st.line_chart, which
    # hands the chart a pan-and-zoom binding: one stray trackpad gesture
    # and the axes are nobody's choice, with no way back but a reload.
    st.altair_chart(views.gap_chart(gap, results["meta"]["plan_weeks"]), width="stretch")
    st.caption(
        "Short FTE added up over the roles and scopes selected, by plan "
        "week. Surplus is NOT netted off: a spare hour in one team does "
        "not staff another team's counter, so a netted line would hide "
        "exactly the weeks a planner is looking for. The axes are fixed - "
        "the whole plan year across, and a round number above the tallest "
        "value in this selection up - so the chart cannot be scrolled out "
        "of scale. Hover a line to read a week off it."
    )

    st.markdown("#### All-in cost, labour and machine")
    st.altair_chart(views.cost_chart(views.cost_split(results, key)), width="stretch")
    st.caption(
        "All-in cost is labour plus the automation's own invoice (spec "
        "6.4). The machine half is zero in any run that buys no "
        "automation, and is shown anyway so the two bars measure the same "
        "thing. Network-wide, like the cards. Fixed axes, as above."
    )


def plan_column_config():
    """The Capacity Plan columns as Streamlit column settings.

    The widths, the short headers and the tooltips are decided in views,
    as plain data, so a test can check that the table fits the page
    without starting one. All this does is put them in the shapes
    Streamlit wants.
    """
    config = {}
    for spec in views.plan_column_specs():
        shared = {"label": spec["label"], "width": spec["width"], "help": spec["help"]}
        if spec["format"] == views.TEXT_COLUMN:
            config[spec["column"]] = st.column_config.TextColumn(**shared)
        else:
            config[spec["column"]] = st.column_config.NumberColumn(
                format=spec["format"], **shared
            )
    return config


def show_detail(results, key, choice, roles):
    """The Capacity Plan rows behind it all, and a CSV of them."""
    st.markdown("#### The rows behind the figures")
    plan = views.filter_plan(
        results["scenarios"][key]["plan"], roles, choice["scopes"]
    )
    view = views.plan_view(plan)
    st.caption(
        f"{len(view):,} rows of the Capacity Plan tab, under its own column "
        "headers. One row per plan week, role and scope. Blanks are "
        "deliberate: a service level exists only where somebody is waiting "
        "in real time, and days of inventory only for work that is allowed "
        "to wait - a zero in either place would read as a measurement. "
        "The headers are shortened to fit the page; hover one to see what "
        "the workbook calls that column and what it means."
    )
    st.dataframe(
        views.plan_display(view),
        hide_index=True,
        width="stretch",
        column_config=plan_column_config(),
    )
    st.download_button(
        "Download these rows (CSV, synthetic)",
        data=views.csv_bytes(view),
        file_name=views.csv_name(key, choice["code"]),
        mime="text/csv",
        icon=":material/download:",
    )
    st.caption(
        "The download carries the figures at full precision, not the "
        "rounded ones on screen, and SYNTHETIC in its file name - the only "
        "label that travels with a file."
    )


def show_comparison(results):
    """Every run at once, ranked by what it does to the all-in cost."""
    st.subheader("Every run, ranked by all-in cost against the baseline")
    st.caption(
        "The biggest saving first. The workbook's Capacity Scenarios tab "
        "lists the same runs alphabetically on purpose - a reference table "
        "that ranks is editorialising - but a reader who opens this view is "
        "asking which runs save money, so here the answer is the first row. "
        "Click a column header to sort by it."
    )
    table = views.comparison_table(results)
    st.dataframe(
        views.comparison_display(table), hide_index=True, width="stretch"
    )
    st.caption(
        "KIND IS DECLARED BY THE RUN, not worked out here. A CHOICE is "
        "something the operation could decide to do; a RISK is something "
        "that could happen to it, which nobody is proposing; a SENSITIVITY "
        "varies one input of another run to test whether its finding holds "
        "and is not a proposal of its own. The ranking puts all three next "
        "to each other, and the cheapest row is not always something anyone "
        "is offering to do."
    )


def main():
    st.set_page_config(page_title="Capacity model (synthetic)", layout="wide")
    st.title(TITLE)

    results = saved_results()
    meta = results["meta"]

    # The banner is drawn before anything else and on every view, because
    # a screenshot of either one travels on its own.
    synthetic_banner(meta)
    staleness_banner(meta)

    st.sidebar.subheader("View")
    view = st.sidebar.radio(
        "View", VIEWS, label_visibility="collapsed", key=VIEW_KEY
    )

    if view == COMPARE_ALL:
        # No filters: this view is the network figures for every run, the
        # same measures the workbook's Summary carries.
        st.sidebar.caption(
            f"{meta['scenario_count']} saved runs over {meta['plan_weeks']} "
            "plan weeks, all synthetic."
        )
        show_comparison(results)
        return

    dimensions = filter_dimensions()
    key = pick_run(results)
    choice, roles = pick_filters(dimensions)
    st.sidebar.caption(
        f"{meta['scenario_count']} saved runs over {meta['plan_weeks']} plan "
        "weeks, all synthetic."
    )

    entry = next(
        item for item in views.menu_items(results) if item["key"] == key
    )
    show_run_header(entry)
    show_cards(results, key)

    if not roles:
        st.warning(
            "No roles selected, so there is nothing to chart or list. Pick "
            "at least one role in the sidebar.",
            icon=":material/filter_alt_off:",
        )
        return

    show_charts(results, key, choice, roles)
    show_detail(results, key, choice, roles)


main()
