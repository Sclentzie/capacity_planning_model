"""The capacity model dashboard (spec Section 8, Sessions 7b and 7c).

Three views. ONE RUN: pick a saved run, filter it, and see what it did to
the plan. COMPARE ALL SCENARIOS: every run at once, ranked by what it
does to the all-in cost, with each run's declared kind beside it. BUILD
YOUR OWN: set the levers yourself and put the result through the model on
demand.

Every figure on the page comes from the saved results that
capacity_model.api built, and is selected and labelled by
capacity_model.views - so the page holds no arithmetic of its own, and a
number here is the same number the workbook prints on the tab of the same
name.

WHY IT READS A FILE. Running the 21 scenarios takes about eight minutes.
A hosted app that did that on every page view would be unusable, so the
runs are saved by `python -m capacity_model.api`, committed, and loaded
here in well under a second. A build-your-own run goes through the
doorway instead, on demand, and takes about twenty-five seconds - which
is why that view has a Run button rather than running as controls move.

WHAT IS NOT HERE YET: the design pass (7d). This is the functional view.

Run it:
    .venv\\Scripts\\python -m streamlit run app/dashboard.py
"""

import sys
from datetime import time
from pathlib import Path

import pandas as pd
import streamlit as st

# The model lives under src/ and is not installed as a package, so the
# path is added here rather than relying on PYTHONPATH - a hosted app has
# no shell to set it in.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from capacity_model import api, controls, views  # noqa: E402  (after the path)

TITLE = "Northbrook Records Services - capacity model (SYNTHETIC)"

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
BUILD_YOUR_OWN = "Build your own"
VIEWS = (ONE_RUN, COMPARE_ALL, BUILD_YOUR_OWN)

# The panel's own state, which has to outlive a rerun. Streamlit reruns
# the whole script on every widget change, so what was run last is kept
# here rather than recomputed - that is what stops the model running on
# every keystroke.
SIGNATURE_STATE = "custom_signature"
ADVANCED_STATE = "custom_advanced"
CONTROL_PREFIX = "ctl_"
ADVANCED_PICK = "advanced_pick"
ADVANCED_VALUE = "advanced_value"
ADVANCED_SEARCH = "advanced_search"
RUN_BUTTON = "run_custom"
RESET_BUTTON = "reset_custom"
CONTROLS_PER_ROW = 3


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
        "these cards. Green moved the way a planner wants, red the other "
        "way, blue did not move at all - and a difference that rounds to "
        "nothing at the precision shown says so in words rather than "
        "wearing an arrow. Hover a card for the exact difference."
    )
    cards = views.headline_cards(results, key)
    st.markdown(views.CARD_STYLE, unsafe_allow_html=True)
    for start in range(0, len(cards), CARDS_PER_ROW):
        row = cards[start:start + CARDS_PER_ROW]
        for column, card in zip(st.columns(len(row)), row):
            # Drawn rather than st.metric, which has green, red and off
            # and no way to say "this did not move" - the third thing a
            # card has to be able to say. Hover a card for the exact
            # difference, which the card itself rounds.
            column.markdown(views.card_html(card), unsafe_allow_html=True)


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


def show_detail(results, key, choice, roles, changes=None):
    """The Capacity Plan rows behind it all, and a CSV of them.

    `changes` is the plain list of what a custom run changed. Given,
    the download carries it; left out, this is a saved run and the
    download is the one 7b settled on.
    """
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
    if changes is None:
        data = views.csv_bytes(view)
        name = views.csv_name(key, choice["code"])
        note = (
            "The download carries the figures at full precision, not the "
            "rounded ones on screen, and SYNTHETIC in its file name - the "
            "only label that travels with a file."
        )
    else:
        # A custom run's inputs exist nowhere else, so they travel with
        # the file. Above the header row, on comment lines, so the data
        # rows still match the workbook's.
        data = views.custom_csv_bytes(view, changes)
        name = views.custom_csv_name(choice["code"])
        note = (
            "The download carries the figures at full precision and, above "
            "the header row, the inputs this run changed - a custom run's "
            "inputs exist nowhere else, so a file without them would be "
            "unreadable a week later. The lines start with # so a reader "
            "and pandas both skip them."
        )
    st.download_button(
        "Download these rows (CSV, synthetic)",
        data=data,
        file_name=name,
        mime="text/csv",
        icon=":material/download:",
    )
    st.caption(note)


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


# ---------------------------------------------------------------------------
# Build your own (7c)
#
# WHY THERE IS A RUN BUTTON. Streamlit reruns the whole script on every
# widget change. A panel that ran the model on each of those would spend
# twenty-five seconds answering a question nobody had finished asking, so
# nothing runs until the button is pressed, and the signature of what was
# run last is kept in session state - a later rerun redraws that result
# from the cache instead of computing it again.
# ---------------------------------------------------------------------------


@st.cache_data
def control_catalogue():
    """The levers, with their baseline values, read once per session."""
    return controls.catalogue()


def advanced_changes():
    """The Advanced box's queued changes: dotted key -> value."""
    return st.session_state.setdefault(ADVANCED_STATE, {})


def control_values(catalogue):
    """What every control is set to right now.

    Read out of session state rather than from the widgets' return
    values, because the Run button is drawn ABOVE the controls - it has
    to be, or a reader would scroll past thirty-six of them to find it -
    so the values are not in hand yet when the button is read.
    """
    values = {}
    for control in catalogue:
        name = CONTROL_PREFIX + control["key"]
        value = st.session_state.get(name, control["baseline"])
        if control["kind"] == controls.TIME and hasattr(value, "strftime"):
            # The model wants "HH:MM"; st.time_input hands back a time.
            value = value.strftime("%H:%M")
        if control["kind"] == controls.MULTISELECT:
            value = list(value)
        values[control["key"]] = value
    return values


def draw_control(column, control):
    """One lever, as the widget its catalogue record asks for.

    The baseline is written into every control's LABEL, not only into its
    tooltip. A reader changing a rate needs to know what it was, and a
    value you have to hover to find is a value you will not look up.
    """
    name = CONTROL_PREFIX + control["key"]
    shown = controls.write(control, control["baseline"])
    label = f"{control['label']} - baseline {shown}"
    note = f"{control['help']}\n\nBASELINE: {shown}"
    kind = control["kind"]

    if kind == controls.TOGGLE:
        column.toggle(label, value=control["baseline"], help=note, key=name)
    elif kind == controls.SELECT:
        options = list(control["options"])
        column.selectbox(
            label,
            options,
            index=options.index(control["baseline"]),
            format_func=lambda option: option or "none",
            help=note,
            key=name,
        )
    elif kind == controls.MULTISELECT:
        column.multiselect(
            label,
            control["options"],
            default=control["baseline"],
            help=note,
            key=name,
        )
    elif kind == controls.TIME:
        hour, minute = (int(part) for part in control["baseline"].split(":"))
        column.time_input(
            label, value=time(hour, minute), step=1800, help=note, key=name
        )
    elif kind == controls.NUMBER:
        column.number_input(
            label,
            min_value=control["minimum"],
            max_value=control["maximum"],
            value=control["baseline"],
            step=control["step"],
            help=note,
            key=name,
        )
    else:
        column.slider(
            label,
            min_value=control["minimum"],
            max_value=control["maximum"],
            value=control["baseline"],
            step=control["step"],
            help=note,
            key=name,
        )


def draw_groups(catalogue):
    """The levers, grouped the way a reader would look for one."""
    for group in controls.GROUP_ORDER:
        mine = [control for control in catalogue if control["group"] == group]
        if not mine:
            continue
        with st.expander(
            f"{group}  ({len(mine)})", expanded=group == controls.DEMAND
        ):
            st.caption(controls.GROUP_NOTES[group])
            for start in range(0, len(mine), CONTROLS_PER_ROW):
                row = mine[start:start + CONTROLS_PER_ROW]
                for column, control in zip(st.columns(len(row)), row):
                    draw_control(column, control)


def draw_advanced():
    """Any other input, by name.

    The controls above are the levers the 21 scenario files pull. This is
    everything else - handle times, shrinkage, hiring lead times, the
    automation cost lines - searched by key and by the sentence
    assumptions.yaml carries above it. Every change still goes through
    the doorway, so a key that does not exist, or a value the model
    refuses, is refused here with the model's own message.
    """
    with st.expander("Advanced: change any other input by name"):
        st.caption(
            "Search by key or by what the input is for. The description is "
            "the comment assumptions.yaml carries above that input, so it "
            "is the project's own sentence rather than one invented for "
            "this box. Changes queue up below and go through the same "
            "validator as a scenario file."
        )
        query = st.text_input(
            "Search the inputs",
            key=ADVANCED_SEARCH,
            placeholder="shrinkage, handle time, lead week, platform fee",
        )
        found = controls.search_inputs(query) if query else []

        if query and not found:
            st.info("No input matches that.", icon=":material/search_off:")
        if found:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Input": row["path"],
                            "Baseline": str(row["baseline"]),
                            "What it is": row["description"] or "-",
                        }
                        for row in found
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            paths = [row["path"] for row in found]
            baseline_of = {row["path"]: row["baseline"] for row in found}
            picked = st.selectbox("Input to change", paths, key=ADVANCED_PICK)
            st.text_input(
                f"New value (baseline {baseline_of[picked]!r})",
                key=ADVANCED_VALUE,
            )
            if st.button("Add this change", icon=":material/add:"):
                typed = st.session_state.get(ADVANCED_VALUE, "")
                try:
                    value = controls.coerce(typed, baseline_of[picked])
                except (TypeError, ValueError) as refused:
                    st.error(
                        f"{picked}: {refused}. The baseline is "
                        f"{baseline_of[picked]!r}, so this input wants a "
                        f"{type(baseline_of[picked]).__name__}.",
                        icon=":material/block:",
                    )
                else:
                    advanced_changes()[picked] = value
                    st.rerun()

        queued = advanced_changes()
        if queued:
            st.markdown("**Changes queued here**")
            for path, value in sorted(queued.items()):
                line, drop = st.columns([5, 1])
                line.write(f"`{path}` = {value}")
                if drop.button("Remove", key=f"drop_{path}"):
                    queued.pop(path)
                    st.rerun()


def run_custom(catalogue):
    """Assemble what the panel says, validate it, and run it.

    Returns the signature of the run, or None if the inputs were refused.
    The refusal comes from the doorway - the same validator a scenario
    file goes through - so this panel cannot accept a change a file
    could not.
    """
    mapping = controls.overrides_from_values(
        catalogue, control_values(catalogue), advanced_changes()
    )
    try:
        # Validated before spending twenty-five seconds on it. It has to
        # be check_overrides: scenario_from_overrides builds the run
        # record without touching the YAML, so it would accept a key
        # that does not exist and the refusal would arrive at the END of
        # the run.
        api.check_overrides(mapping)
    except (KeyError, TypeError, ValueError) as refused:
        st.error(
            f"These inputs were refused: {refused}", icon=":material/block:"
        )
        return None

    signature = controls.signature(mapping)
    hits_before = controls.run_signature.cache_info().hits
    with st.spinner(
        "Running the model on your inputs. A full plan year takes about "
        "twenty-five seconds; inputs already run come back at once."
    ):
        try:
            controls.run_signature(signature)
        except (KeyError, TypeError, ValueError) as refused:
            st.error(
                f"The model refused these inputs: {refused}",
                icon=":material/block:",
            )
            return None
    if controls.run_signature.cache_info().hits > hits_before:
        st.toast("These inputs had been run already, so that was instant.")
    return signature


def reset_panel(catalogue):
    """Every control back to the baseline, and the results forgotten."""
    for control in catalogue:
        st.session_state.pop(CONTROL_PREFIX + control["key"], None)
    st.session_state.pop(SIGNATURE_STATE, None)
    st.session_state[ADVANCED_STATE] = {}
    controls.forget_runs()


def show_custom_result(saved, catalogue, signature, choice, roles):
    """The result, through the browse view's own cards, charts and table.

    The same functions the One run view calls, handed a two-run bundle.
    A custom run and a saved run are therefore reported by one piece of
    code, and cannot disagree about what a card means.
    """
    run = controls.run_signature(signature)
    bundle = views.custom_bundle(saved, run)
    key = run["key"]
    changes = controls.describe_changes(
        catalogue, control_values(catalogue), advanced_changes()
    )

    st.subheader(run["name"])
    if changes:
        st.markdown("**Changed from the baseline:**")
        for line in changes:
            st.markdown(f"- {line}")
    else:
        st.info(
            "Nothing is changed, so this IS the baseline: every card below "
            "says no change, which is the panel working rather than a "
            "fault.",
            icon=":material/info:",
        )
    st.caption(
        f"The change starts in plan week {run['starts_week']}; the figures "
        "below are the whole plan year, as everywhere else on this page."
    )

    show_cards(bundle, key)
    if not roles:
        st.warning(
            "No roles selected, so there is nothing to chart or list. Pick "
            "at least one role in the sidebar.",
            icon=":material/filter_alt_off:",
        )
        return
    show_charts(bundle, key, choice, roles)
    show_detail(bundle, key, choice, roles, changes=changes)


def show_build_your_own(saved, dimensions):
    """The panel: controls, a Run button, and the result below them."""
    catalogue = control_catalogue()

    st.subheader("Build your own scenario")
    st.caption(
        "Every lever the 21 scenario files pull, plus an Advanced box for "
        "anything else in assumptions.yaml. Nothing runs until you press "
        "Run: a full plan year takes about twenty-five seconds, and a "
        "panel that ran on every change would spend it answering a "
        "question you had not finished asking. Inputs you have already "
        "run this session come back at once."
    )

    go, reset, _rest = st.columns([1, 1, 4])
    pressed = go.button(
        "Run", type="primary", icon=":material/play_arrow:", key=RUN_BUTTON
    )
    if reset.button(
        "Reset to baseline", icon=":material/restart_alt:", key=RESET_BUTTON
    ):
        reset_panel(catalogue)
        st.rerun()

    draw_groups(catalogue)
    draw_advanced()

    if pressed:
        signature = run_custom(catalogue)
        if signature is None:
            return
        st.session_state[SIGNATURE_STATE] = signature

    signature = st.session_state.get(SIGNATURE_STATE)
    if not signature:
        st.info(
            "Set the controls above and press Run.", icon=":material/tune:"
        )
        return

    st.divider()
    choice, roles = pick_filters(dimensions)
    show_custom_result(saved, catalogue, signature, choice, roles)


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

    if view == BUILD_YOUR_OWN:
        # The filters are drawn by the panel itself, under the controls
        # and above the result, because there is nothing to filter until
        # a run exists.
        show_build_your_own(results, filter_dimensions())
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
