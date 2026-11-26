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

import re
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

from capacity_model import (  # noqa: E402  (after the path)
    api,
    controls,
    download,
    views,
)

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

# The view names come from views, because the intro tells a visitor what
# the views are called and the sidebar draws them (7d). Two copies would
# have drifted by a word, and the sentence pointing at the radio button
# would have stopped pointing at it.
ONE_RUN = views.ONE_RUN_VIEW
COMPARE_ALL = views.COMPARE_VIEW
BUILD_YOUR_OWN = views.BUILD_VIEW
VIEWS = views.VIEW_ORDER

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
SPEC_BUTTON = "load_spec_defaults"
ADVANCED_ADD = "advanced_add"
CONTROLS_PER_ROW = 3

# ONE SENTENCE, WRITTEN ONCE. Both views that can be emptied by the role
# filter say this, and they were two copies of it until 11/25 - which is
# two chances for them to drift by a word while a reviewer reads one of
# them. It is a note rather than a warning: nothing is wrong, the reader
# has simply unticked everything, and the second half says what to do
# about it.
NO_ROLES_NOTE = (
    "No roles selected, so there is nothing to chart or list. Pick at "
    "least one role in the sidebar."
)


@st.cache_data
def saved_results():
    """The saved runs, read once per session rather than per rerun."""
    return api.load_results()


@st.cache_data
def filter_dimensions():
    """Sites, clusters and roles, with the names the YAML gives them."""
    return api.dimensions()


# NOBODY ASKS WHICH PALETTE IS SHOWING ANY MORE (11/20).
#
# There used to be an active_mode() here, reading st.context.theme.type
# and handing the answer to every custom-coloured thing on the page. It
# was the right shape for a page that reruns, and this page does not:
# switching Light/Dark in the menu re-themes the browser without sending
# a rerun, so the answer this function gave was correct when it was asked
# and wrong by the time anyone read it.
#
# Everything it used to serve now works without it. A card's delta is
# markdown with a colour NAME in it; a kind in the sidebar is the same; a
# scope tint and a kind's cell are translucent washes the browser mixes
# against whichever page is showing. None of them can be caught in the
# wrong palette, because none of them was ever told which palette to use.


@st.cache_data
def current_inputs():
    """A fingerprint of the input files as they are on disk right now."""
    return api.inputs_fingerprint()


def page_styles():
    """Every stylesheet this page draws itself with, once per rerun.

    Built in views from recorded colour pairs, and sent as light-dark()
    so the browser picks the shade on every repaint. Nothing here knows
    which palette is showing, which is the 11/20 rule and still holds.
    """
    sheets = (
        views.collapse_style(), views.table_style(), views.banner_style(),
        views.download_style(), views.chart_style(),
    )
    # st.html AND NOT st.markdown, which is not a style preference (11/24).
    # st.markdown runs its argument through markdown first, and a markdown
    # HTML block ENDS at the line holding its closing tag - so four sheets
    # joined together put "</style><style>" on one line, ended the block
    # there, and the rest arrived as text with its tags stripped. Nothing
    # raised: the page simply had no stylesheet, which is how it was found
    # (in a browser, reading back the colour of a word that should have
    # been amber and was the body text instead).
    with st.container(key=views.STYLE_KEY):
        st.html("\n".join(sheets))


def sort_controls(specs, prefix):
    """The control that replaced clicking a column header (11/24).

    An HTML table cannot sort itself - that was the grid's one trick
    this page gives up - so the sort happens in pandas and the table is
    written again. Two small widgets rather than one, because "by which
    column" and "which end first" are two questions, and a reader who
    wants the worst weeks first should not have to know that sorting
    descending is a second click on the same header.
    """
    left, right = st.columns([3, 3], vertical_alignment="bottom")
    column = left.selectbox(
        "Sort by", views.sort_options(specs), key=f"{prefix}-sort",
        help=views.SORT_HELP,
    )
    descending = right.toggle(
        "High to low", key=f"{prefix}-order", help=views.ORDER_HELP,
        disabled=column == views.SORT_AS_SHOWN,
    )
    return column, descending


def show_intro(results, dimensions):
    """The three lines a visitor reads before anything else (7d).

    What this is, what it found, how to use it - and then out of the way.
    The figures in it are read from the saved results, so the headline
    cannot go stale while the page still prints it.
    """
    for line in views.intro_lines(results, dimensions):
        st.markdown(line)


def notice(text, tone="note", lead=None, sidebar=False):
    """One notice, in the Finding banner's design (Siri, 11/25, round 3).

    EVERY NOTICE ON THIS PAGE GOES THROUGH HERE. They were st.info,
    st.warning and st.error - Streamlit's own alert blocks, which are
    filled panels in a palette that was never this page's, and in light
    mode they read as something pasted in from another application. The
    reviewers said so about the last one of them a reader meets ("Set
    the controls above and press Run"), and it was true of all twelve.

    The design is the one already approved on this page for the
    workbook's findings: a wash of the hue, the sentence written in that
    hue, a thin left edge. views.NOTICE_TONES carries the three hues and
    why there are three rather than one.

    WHAT WE GAVE UP. Streamlit drew a Material icon beside each of these
    and this draws none: the Finding banner has no icon either, and one
    design with two different ornaments is two designs. What the icon
    said, the lead word says in a word a screen reader can also read.
    """
    where = st.sidebar if sidebar else st
    where.markdown(
        views.banner_html(text, tone, lead=lead), unsafe_allow_html=True
    )


def synthetic_banner(meta):
    """The synthetic statement, on the page itself.

    Not a footnote: a screenshot of a dashboard travels further than the
    repository it came from, and whoever sees the screenshot has to be
    able to tell that Northbrook Records Services is invented.
    """
    # NOT st.warning ANY MORE (reviewers, 11/24). Streamlit's warning
    # is a filled yellow block that reads as "something is wrong here",
    # and nothing is wrong: the page is doing what it says. This is the
    # same statement in the amber the scope words use, as a wash with an
    # edge, with SYNTHETIC itself as the coloured lead.
    st.markdown(
        views.synthetic_banner_html(meta["label"]), unsafe_allow_html=True
    )


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
    notice(
        "These saved results were built from different inputs ("
        + ", ".join(changed)
        + "). Rebuild with `python -m capacity_model.api` before trusting "
        "the figures below.",
        tone="refused",
        lead="Out of date",
    )


def caption_for(entry):
    """The one line under a run's name in the picker.

    The kind is the run's own declared field, so a risk cannot be read as
    a proposal just because it sits in a list of them.
    """
    # The kind in its own colour (11/19). A caption takes markdown and
    # not HTML, so the colour comes from the name Streamlit resolves
    # against .streamlit/config.toml - the same hex the comparison table
    # and the key below it use.
    parts = [views.kind_markdown(entry["kind"])]
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
        help=views.RUN_HELP,
        key=RUN_KEY,
    )


LAST_RUN_KEY = "last-run-shown"


def scroll_to_top(key):
    """Put the reader back at the top when they pick a different run.

    A RUN IS A WHOLE PAGE (reviewers, 11/24). Picking one from the
    sidebar rewrites everything in the main column, but the main column
    keeps whatever scroll position the last run was being read at - so a
    reader who was looking at week 40 of the detail table picks a new
    run and lands in the middle of a table, with the name of the run
    they just chose off the top of the screen.

    Streamlit has no scroll API, so this is an iframe of a single
    collapsed pixel whose script scrolls the parent document's main
    element. THAT element and not the window, because Streamlit scrolls
    the main block and the window itself never moves - measured in the
    browser on 11/24.

    Only on a CHANGE, remembered in session state: scrolling on every
    rerun would yank the page back every time a filter moved.

    THE RUN'S NAME IS IN THE SCRIPT, and it is not decoration. An iframe
    whose srcdoc is unchanged is the same DOM node after a rerun, and the
    browser does not re-run a script it has already run - so with one
    fixed script this worked exactly once per session and then silently
    stopped. Naming the run makes each rerun's srcdoc different, so the
    frame really is reloaded. Found in a browser, because nothing about
    it fails in Python.
    """
    first_visit = LAST_RUN_KEY not in st.session_state
    if st.session_state.get(LAST_RUN_KEY) == key and not first_visit:
        return
    st.session_state[LAST_RUN_KEY] = key
    if first_visit:
        # Nothing to scroll back from on the first render, and an iframe
        # nobody needs is still an element on the page.
        return
    # Letters, digits and underscores only: a scenario key is already
    # one of those, and a key that was not cannot become script.
    safe = re.sub(r"\W", "", str(key))
    with st.container(key=views.SCROLL_KEY):
        st.iframe(
            f"""<script>
            /* run: {safe} */
            const doc = window.parent.document;
            const main = doc.querySelector('[data-testid="stMain"]');
            if (main) {{ main.scrollTo({{top: 0, behavior: 'instant'}}); }}
            </script>""",
            height=1,
        )


def pick_filters(dimensions):
    """The sidebar filters: one scope grain, then the roles inside it."""
    st.sidebar.subheader("What to look at")

    choices = views.scope_choices(dimensions)
    choice = st.sidebar.selectbox(
        "Network, cluster or site",
        choices,
        format_func=lambda option: option["label"],
        help=views.SCOPE_HELP,
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
        help=views.ROLES_HELP,
        key=ROLES_KEY,
    )

    note = views.scope_note(dimensions, choice)
    if note:
        # In the sidebar, which is a step off the page colour - so the
        # wash is measured against that background too, not only
        # against the page.
        notice(note, sidebar=True)

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
        st.markdown(
            views.banner_html(entry["finding"], "note", lead="Finding"),
            unsafe_allow_html=True,
        )


def show_cards(results, key):
    """The six headline measures, each against the baseline."""
    st.markdown("#### Against the baseline")
    st.caption(
        "Network-wide, as the workbook's Summary reports them. The filters "
        "in the sidebar apply to the chart and the table below, not to "
        "these cards. Green moved the way a planner wants, red the other "
        "way, blue did not move at all - and a difference that rounds to "
        "nothing at the precision shown says so in words rather than "
        "wearing an arrow. Hover a card for what the measure is and the "
        "exact difference behind the rounding."
    )
    cards = views.headline_cards(results, key)
    st.markdown(views.CARD_STYLE, unsafe_allow_html=True)
    for start in range(0, len(cards), CARDS_PER_ROW):
        row = cards[start:start + CARDS_PER_ROW]
        for column, card in zip(st.columns(len(row)), row):
            # Drawn rather than st.metric, which has green, red and off
            # and no way to say "this did not move" - the third thing a
            # card has to be able to say.
            #
            # TWO ELEMENTS, NOT ONE (11/20). The label and the value are
            # HTML, because the card's sizes are its own; the delta line
            # is a separate st.markdown, because the colour in it is a
            # NAME that Streamlit resolves in the browser against the
            # live theme. They cannot be one string: a colour directive
            # inside raw HTML renders as the literal text ":green[...]",
            # which was checked in a browser rather than assumed.
            #
            # The container's key is what the stylesheet closes the gap
            # between the two with, and what sizes the delta line.
            with column.container(key=views.card_key(card)):
                # The hover rides on this block, in a title attribute, as
                # it did before. NOT on the delta line below: st.markdown
                # takes a help= argument, but Streamlit renders it as a
                # "?" button beside the text, and six of those across a
                # card row is a change to the layout rather than to a
                # colour. The sentence is the same one, on the label and
                # the value, which is most of the card.
                st.markdown(views.card_html(card), unsafe_allow_html=True)
                # ALWAYS this second element, even when there is nothing
                # to say (11/21). Emitting it only when the card has a
                # delta made a card one element on the baseline and two
                # everywhere else, and Streamlit rebuilds a container
                # whose child count changes instead of updating it - so
                # the previous run's whole row of colours stayed on
                # screen while the new one was built. views.CARD_BLANK
                # carries the reasoning and the measurement.
                st.markdown(views.card_markdown(card) or views.CARD_BLANK)


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

    st.markdown("#### All-in cost, labor and machine")
    st.altair_chart(views.cost_chart(views.cost_split(results, key)), width="stretch")
    st.caption(
        "All-in cost is labor plus the automation's own invoice (spec "
        "6.4). The machine half is zero in any run that buys no "
        "automation, and is shown anyway so the two bars measure the same "
        "thing. Network-wide, like the cards. Fixed axes, as above."
    )


EXCEL_HELP = (
    "The rows above as an Excel file, formatted the way the workbook is: "
    "the same headers, the same number formats, a frozen header row and "
    "a filter on it, and the hours gap and the total still live formulas. "
    "Roles are named as a person would say them. SYNTHETIC is in the file "
    "name, because a file travels further than the page it came from."
)

CSV_HELP = (
    "The same rows as raw data, at full precision and under the "
    "workbook's own column headers, with the model's own role codes - so "
    "a downloaded row can be matched or joined to a workbook row. "
    "SYNTHETIC is in the file name."
)


# BUILT ONCE PER SET OF INPUTS, NOT ONCE PER RERUN. Writing the plan
# workbook takes about half a second for 780 rows, and Streamlit reruns
# this script on every widget change - so without this, moving any slider
# on the page would spend half a second producing a file nobody had asked
# for. Keyed on the view itself, so a filter change rebuilds it and
# nothing else does.
@st.cache_data(show_spinner=False)
def plan_excel(view, run_name, filters, changes=None):
    book = download.plan_book(
        view, run_name=run_name, filters=filters, changes=changes,
    )
    return download.excel_bytes(book)


@st.cache_data(show_spinner=False)
def comparison_excel(table):
    return download.excel_bytes(download.comparison_book(table))


def show_downloads(excel, csv, help_text=CSV_HELP, legend=None):
    """The two download buttons, Excel first.

    EXCEL IS THE MAIN BUTTON AND CSV IS THE OTHER ONE (Siri, 11/23).
    Almost everyone who takes a table off this page is going to open it
    in Excel and show it to somebody, and what they used to get was an
    unformatted CSV that each of them then spent ten minutes making
    readable - so the file that got forwarded was nobody's version of the
    model. The CSV stays, because raw data at full precision under the
    model's own codes is what anybody doing arithmetic on it actually
    wants, and that is a different job.

    `excel` and `csv` are (label, data, file name) triples, and `legend`
    is anything that belongs on the same line - the region key sits
    beside the detail table's buttons, where a reader who has just
    noticed the tints is already looking.
    """
    widths = [2, 2, 4] if legend is not None else [1, 1]
    # IN ITS OWN KEYED CONTAINER, so the stylesheet can put clear space
    # between the table above and these buttons (Siri, 11/25, round 3).
    # The columns go inside it: the margin belongs to the row, not to
    # either button, or the two would drift apart from each other.
    row = st.container(key=views.DOWNLOAD_ROW_KEY)
    columns = row.columns(widths, vertical_alignment="center")
    # EACH IN ITS OWN KEYED CONTAINER (11/24). Streamlit puts the key on
    # the element as a class, and that class is the only handle a
    # stylesheet gets on a widget it does not own - which is how each
    # button wears its own hue instead of Streamlit's primary blue and
    # secondary gray. Both are "secondary", so nothing underneath is
    # fighting the colour on top.
    label, data, name = excel
    with columns[0].container(key=views.EXCEL_BUTTON_KEY):
        st.download_button(
            label,
            data=data,
            file_name=name,
            mime=download.EXCEL_MIME,
            help=EXCEL_HELP,
            icon=":material/download:",
            type="secondary",
            width="stretch",
        )
    label, data, name = csv
    with columns[1].container(key=views.CSV_BUTTON_KEY):
        st.download_button(
            label,
            data=data,
            file_name=name,
            mime="text/csv",
            help=help_text,
            icon=":material/download:",
            type="secondary",
            width="stretch",
        )
    if legend is not None:
        columns[2].markdown(legend, unsafe_allow_html=True)


def show_detail(results, key, choice, roles, changes=None):
    """The Capacity Plan rows behind it all, and two downloads of them.

    `changes` is the plain list of what a custom run changed. Given, the
    Excel file carries it as an INPUTS CHANGED section and the CSV as
    comment lines above the header; left out, this is a saved run, which
    the workbook already describes.
    """
    st.markdown("#### The rows behind the figures")
    dimensions = filter_dimensions()
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
    # THE SCOPE IS A COLOURED BOLD WORD NOW (reviewers, 11/24), which is
    # why this is our own HTML and not st.dataframe: the grid paints its
    # cells on a canvas, where no CSS and no markdown can reach, so the
    # only colour it could carry was a background wash. The word says
    # the same thing in less ink, and the one line under the table says
    # what the three colours mean.
    #
    # NO MODE IS PASSED, which is the 11/20 rule unchanged. Every colour
    # in the stylesheet is a light-dark() pair the browser resolves on
    # each repaint, so a theme switch - which does not rerun this script
    # - lands on the right shade by itself.
    specs = views.plan_column_specs()
    shown = views.plan_display(view)
    column, descending = sort_controls(specs, "detail")
    st.markdown(
        views.table_html(
            views.sorted_view(shown, column, descending),
            specs,
            views.SCOPE_HEADER,
            views.scope_class_of(dimensions),
        ),
        unsafe_allow_html=True,
    )
    filters = views.filter_lines(dimensions, choice, roles)
    if changes is None:
        data = views.csv_bytes(view)
        name = views.csv_name(key, choice["code"])
        excel = plan_excel(
            view, results["scenarios"][key]["name"], tuple(filters),
        )
        excel_name = download.plan_name(key, choice["code"])
        note = (
            "Downloads include full-precision figures and your current "
            "filters. SYNTHETIC is in the file name."
        )
    else:
        # A custom run's inputs exist nowhere else, so they travel with
        # the file. Above the header row, on comment lines, so the data
        # rows still match the workbook's.
        data = views.custom_csv_bytes(view, changes)
        name = views.custom_csv_name(choice["code"])
        excel = plan_excel(
            view, api.CUSTOM_RUN_NAME, tuple(filters), tuple(changes),
        )
        excel_name = download.custom_plan_name(choice["code"])
        note = (
            "Downloads include full-precision figures, your current "
            "filters, and the inputs you changed - they exist nowhere "
            "else, so the file carries them. SYNTHETIC is in the file "
            "name."
        )
    # The scope line sits beside the buttons, where a reader who has just
    # noticed the coloured words is already looking.
    show_downloads(
        (download.EXCEL_LABEL, excel, excel_name),
        (download.CSV_LABEL, data, name),
        legend=views.scope_line_html(dimensions),
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
        "Sort by any column with the control below, or hover a header to "
        "see what the measure is - the same sentence the cards in the "
        "other view give."
    )
    table = views.comparison_table(results)
    # EACH RUN'S KIND AS A COLOURED BOLD WORD (reviewers, 11/24), in the
    # same three shades Streamlit paints the kind with in the sidebar -
    # read out of the theme file, so the word beside a run in the picker
    # and the word in this table are one colour rather than two that
    # nearly match.
    #
    # THE KIND KEY IS GONE with the swatches it explained. What it said
    # is on the Kind column's own tooltip, which is where a reader who
    # does not already know the word is looking.
    specs = views.comparison_column_specs(table)
    column, descending = sort_controls(specs, "compare")
    st.markdown(
        views.table_html(
            views.sorted_view(views.comparison_display(table), column,
                              descending),
            specs,
            views.KIND_HEADER,
            views.kind_class_of(),
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "The ranking puts all three kinds next to each other, and the "
        "cheapest row is not always something anyone is offering to do - "
        "hover the Kind header for what each word means."
    )

    show_downloads(
        (download.EXCEL_LABEL, comparison_excel(table),
         download.comparison_name()),
        (download.CSV_LABEL, views.csv_bytes(table),
         download.comparison_csv_name()),
        help_text=(
            "The same rows as raw data, at full precision, under the "
            "workbook's own column headers and with the model's own role "
            "codes. SYNTHETIC is in the file name."
        ),
    )
    st.caption(
        "Downloads include every run in the order above, ranked rather "
        "than alphabetical. SYNTHETIC is in the file name."
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
    value you have to hover to find is a value you will not look up. The
    same goes for spec 7.1's default where there is one: a reader looking
    at the AI rates is usually asking how far from 7.1 they are, and the
    answer belongs on the screen rather than in the spec.
    """
    name = CONTROL_PREFIX + control["key"]
    shown = controls.write(control, control["baseline"])
    label = f"{control['label']} - baseline {shown}"
    note = f"{control['help']}\n\nBASELINE: {shown}"
    if control["spec"]:
        spec_shown = controls.write(control, control["spec_default"])
        label += f", {controls.SPEC_LABEL} {spec_shown}"
        note += f"\n\n{controls.SPEC_LABEL.upper()}: {spec_shown}"
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
        # Plain names on the options, the way the roles are shown (11/16).
        # Streamlit cannot give one option of a multiselect its own
        # tooltip, so the codes are listed in the control's own tooltip -
        # which is the hover a reader has, and keeps the code available
        # for anybody tracing a row back to the spec.
        names = control["option_names"] or {}
        if names:
            note += "\n\nCODES: " + "; ".join(
                f"{code} {names.get(code, code)}" for code in control["options"]
            )
        column.multiselect(
            label,
            control["options"],
            default=control["baseline"],
            format_func=lambda code: names.get(code, code),
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


def load_spec_defaults(catalogue):
    """Put the AI and automation group at spec 7.1's defaults.

    WHY A BUTTON. The 7.1 defaults are thirteen values across two sides
    of the model, and setting them by hand is thirteen chances to set one
    of them wrong - after which the panel reports a run nobody meant and
    nothing on screen says so. One button, and the figures are the ones
    the spec's own AI run reports.

    It writes into the widgets' session state and leaves the rerun to the
    caller, because a widget already drawn in this script run cannot have
    its value changed underneath it.
    """
    for control in catalogue:
        if control["spec"]:
            st.session_state[CONTROL_PREFIX + control["key"]] = (
                control["spec_default"]
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
            if group == controls.AUTOMATION:
                # Drawn before this group's controls, so none of them has
                # been instantiated yet when the click is handled.
                if st.button(
                    f"Load the {controls.SPEC_LABEL}s",
                    icon=":material/auto_awesome:",
                    key=SPEC_BUTTON,
                    help=(
                        "Sets every control in this group to the default "
                        "spec 7.1 gives it, which is the programme the "
                        "saved AI and automation run reports. The "
                        "proof-of-concept pair is left alone: it is a "
                        "Section 7 addition and 7.1 names no default for "
                        "it."
                    ),
                ):
                    load_spec_defaults(catalogue)
                    st.rerun()
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
            help=(
                "Type part of a key - automation.uptime - or part of what "
                "the input is for - shrinkage, handle time. The "
                "description in the results is the comment "
                "assumptions.yaml carries above that input, so it is the "
                "project's own sentence rather than one written for this "
                "box."
            ),
        )
        found = controls.search_inputs(query) if query else []

        if query and not found:
            notice("No input matches that.")
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
            picked = st.selectbox(
                "Input to change",
                paths,
                help=(
                    "One of the inputs found above. Only single-value "
                    "inputs are offered: a list is a table keyed by code, "
                    "which a text box cannot change safely."
                ),
                key=ADVANCED_PICK,
            )
            st.text_input(
                f"New value (baseline {baseline_of[picked]!r})",
                key=ADVANCED_VALUE,
                help=(
                    "Typed in the same units the baseline is in, so a "
                    "share is 0.15 rather than 15. It has to be the kind "
                    "of number the input is: a share or a rate between 0% "
                    "and 100%, hours, counts and costs not negative, a "
                    "multiplier not negative. A value more than three "
                    "times the baseline, or under a third of it, is "
                    "allowed and warned about."
                ),
            )
            if st.button(
                "Add this change",
                icon=":material/add:",
                help=(
                    "Queues the change. It is checked now, so a value the "
                    "model would refuse costs a message rather than a "
                    "run, and it joins the next Run."
                ),
                key=ADVANCED_ADD,
            ):
                typed = st.session_state.get(ADVANCED_VALUE, "")
                try:
                    value = controls.coerce(typed, baseline_of[picked])
                except (TypeError, ValueError) as refused:
                    notice(
                        f"{picked}: {refused}. The baseline is "
                        f"{baseline_of[picked]!r}, so this input wants a "
                        f"{type(baseline_of[picked]).__name__}.",
                        tone="refused",
                        lead="Refused",
                    )
                else:
                    try:
                        # A rule about what KIND of number this input is
                        # (11/17). A slider has ends; a text box has
                        # none, and uptime at 500% was accepted before
                        # this because no formula complains about it.
                        controls.guard(picked, value)
                    except ValueError as refused:
                        notice(str(refused), tone="refused",
                               lead="Refused")
                    else:
                        advanced_changes()[picked] = value
                        st.rerun()

        queued = advanced_changes()
        if queued:
            st.markdown("**Changes queued here**")
            for path, value in sorted(queued.items()):
                line, drop = st.columns([5, 1])
                line.write(f"`{path}` = {value}")
                if drop.button(
                    "Remove",
                    help="Drop this change; the input goes back to its "
                         "baseline.",
                    key=f"drop_{path}",
                ):
                    queued.pop(path)
                    st.rerun()
            # Allowed, and warned about: a value three times its baseline
            # or under a third of it is outside anything this model has
            # been run at, and the warning stays on screen next to the
            # change rather than appearing once when it was typed.
            for path, value in sorted(queued.items()):
                warning = controls.tested_range_warning(
                    path, value, controls.baseline_of_input(path)
                )
                if warning:
                    # The amber tone, not the red one: this change IS in
                    # the run. The lead says so in the one word a reader
                    # needs, because the red of a refusal next to a
                    # change that was accepted would be a lie.
                    notice(warning, tone="caution", lead="Queued")


def run_custom(catalogue):
    """Assemble what the panel says, validate it, and run it.

    Returns the signature of the run, or None if the inputs were refused.
    The refusal comes from the doorway - the same validator a scenario
    file goes through - so this panel cannot accept a change a file
    could not.
    """
    values = control_values(catalogue)
    mapping = controls.overrides_from_values(
        catalogue, values, advanced_changes()
    )
    try:
        # Validated before spending twenty-five seconds on it. It has to
        # be check_overrides: scenario_from_overrides builds the run
        # record without touching the YAML, so it would accept a key
        # that does not exist and the refusal would arrive at the END of
        # the run.
        api.check_overrides(mapping)
    except (KeyError, TypeError, ValueError) as refused:
        # No lead word: the sentence already says it was refused, and a
        # lead that repeats the sentence is an ornament.
        notice(f"These inputs were refused: {refused}", tone="refused")
        return None

    # The break windows, in the panel's own words, against the opening
    # hours as THIS run has them. The model does refuse an impossible
    # window, but it refuses it as a sentence about interval masks
    # twenty-five seconds in - and it accepts a lunch window that starts
    # before the doors open, because intervals inside it still exist.
    problems = controls.window_problems(
        catalogue, values, controls.merged_config(mapping)
    )
    if problems:
        for problem in problems:
            notice(problem, tone="refused", lead="Refused")
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
            notice(
                f"The model refused these inputs: {refused}", tone="refused"
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
        st.markdown(
            views.banner_html(
                "Nothing is changed, so this IS the baseline: every card "
                "below says no change, which is the panel working rather "
                "than a fault.",
                "note",
                lead="Note",
            ),
            unsafe_allow_html=True,
        )
    st.caption(
        f"The change starts in plan week {run['starts_week']}; the figures "
        "below are the whole plan year, as everywhere else on this page."
    )

    show_cards(bundle, key)
    if not roles:
        notice(NO_ROLES_NOTE)
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
        "Run",
        type="primary",
        icon=":material/play_arrow:",
        help=(
            "Puts the controls below through the model: a full 52-week "
            "plan year, about twenty-five seconds. Inputs you have "
            "already run this session come back at once. Nothing runs "
            "until you press this."
        ),
        key=RUN_BUTTON,
    )
    if reset.button(
        "Reset to baseline",
        icon=":material/restart_alt:",
        help=(
            "Every control back to the value assumptions.yaml gives it, "
            "the Advanced changes cleared, and the runs remembered this "
            "session forgotten."
        ),
        key=RESET_BUTTON,
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
        # THE ONE THE REVIEWERS NAMED (Siri, 11/25). It was an st.info:
        # a filled blue-gray block at the bottom of a cream page, which
        # is the last thing a reader meets on this view and the thing
        # they said looked out of place.
        notice("Set the controls above and press Run.")
        return

    st.divider()
    choice, roles = pick_filters(dimensions)
    show_custom_result(saved, catalogue, signature, choice, roles)


def main():
    st.set_page_config(page_title="Capacity model (synthetic)", layout="wide")
    st.title(TITLE)

    results = saved_results()
    meta = results["meta"]
    dimensions = filter_dimensions()

    # The stylesheets first: every banner and table below is drawn with
    # them, and a page that sent its markup before its CSS would flash
    # unstyled on every rerun.
    page_styles()

    # The intro first, above the picker: a visitor arrives knowing
    # neither what this operation is nor that it is invented (7d).
    show_intro(results, dimensions)

    # The banner is drawn before anything else and on every view, because
    # a screenshot of either one travels on its own.
    synthetic_banner(meta)
    staleness_banner(meta)

    # Where light and dark are changed (11/19). Streamlit 1.67 has no API
    # for setting the theme from the page - st.context.theme is read-only
    # - so there is no honest on-page toggle to offer, and a fake one
    # that only recoloured half the page would be worse than a pointer to
    # the real control. The switch Streamlit draws is easy to miss, so
    # the sidebar says where it is. It only exists at all because this
    # app ships both palettes.
    st.sidebar.caption(
        "Light or dark: the menu at the top right of the page. *System* "
        "follows your own machine, and is the default."
    )

    st.sidebar.subheader("View")
    view = st.sidebar.radio(
        "View",
        VIEWS,
        label_visibility="collapsed",
        help=views.VIEW_HELP,
        key=VIEW_KEY,
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
        show_build_your_own(results, dimensions)
        return

    key = pick_run(results)
    scroll_to_top(key)
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
        notice(NO_ROLES_NOTE)
        return

    show_charts(results, key, choice, roles)
    show_detail(results, key, choice, roles)


main()
