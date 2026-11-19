"""What the browse view shows (spec Section 8, Session 7b).

NOTHING HERE COMPUTES A FIGURE. Every number comes from a run that
capacity_model.api built and saved; this module only chooses which of
them to show, calls them what the workbook calls them, and subtracts the
baseline. That claim is worth no more than the test behind it, so
tests/test_views.py checks every scenario's cards against the comparison
table to the cent.

WHY IT IS IN THE PACKAGE and not in app/. The dashboard and the tests
have to agree about what a card says, and the only way to be sure of that
is for both to import the same function. A view layer that lived beside
the Streamlit page would be testable only through the page.

WHAT IT DELIBERATELY DOES NOT DO: design. Session 7b is the browse view -
the right figures, grouped sensibly, labelled honestly. Spacing, colour
and typography are 7d.
"""

import functools
import html
import math

import altair as alt
import pandas as pd

from . import api, export, scenarios

BASELINE_KEY = "baseline"
BASELINE_LABEL = "Baseline"

# Excel number format -> how the same figure is written on the page. The
# dashboard looks the format up instead of choosing its own precision,
# because a cost rounded to the dollar on the tab and to the cent on the
# screen is the start of two sets of books.
VALUE_FORMATS = {
    export.MONEY: lambda value: f"${value:,.0f}",
    export.MONEY_CENTS: lambda value: f"${value:,.2f}",
    export.HOURS: lambda value: f"{value:,.1f}",
    export.FTE: lambda value: f"{value:,.2f}",
    export.DAYS: lambda value: f"{value:,.2f}",
    export.PERCENT: lambda value: f"{value:.1%}",
    export.COUNT: lambda value: f"{value:,.0f}",
}

# The same precisions again, as the printf strings Streamlit's column
# settings take, so a figure in a TABLE is written the way the same
# figure is written on a CARD (7d, 11/17). One table, used by the
# Capacity Plan columns and by the comparison table: before this, the
# comparison table printed raw numbers, so the all-in cost read
# "$7,456,636" on a card and "7456636" in the other view.
TEXT_COLUMN = "text"

SCREEN_NUMBER_FORMATS = {
    export.COUNT: "%d",
    export.HOURS: "%,.1f",
    export.FTE: "%,.2f",
    export.DAYS: "%,.2f",
    export.PERCENT: "percent",
    export.MONEY: "$%,.0f",
    export.MONEY_CENTS: "$%,.2f",
}

# How many decimals the workbook prints, so the table on screen can be
# rounded to the same precision and still be a number a reader can sort.
DISPLAY_DECIMALS = {
    export.MONEY: 0,
    export.MONEY_CENTS: 2,
    export.HOURS: 1,
    export.FTE: 2,
    export.DAYS: 2,
    export.PERCENT: 3,
    export.COUNT: 0,
}

NO_CHANGE = "no change"

# ---------------------------------------------------------------------------
# One name per thing (7d, 11/17)
#
# The three views are named here rather than in the page, because the
# intro tells a visitor what the views are called and the sidebar draws
# them. Two copies of "Compare all scenarios" would eventually differ by
# a word, and the sentence pointing at the radio button would stop
# pointing at it.
# ---------------------------------------------------------------------------

ONE_RUN_VIEW = "One run"
COMPARE_VIEW = "Compare all scenarios"
BUILD_VIEW = "Build your own"
VIEW_ORDER = (ONE_RUN_VIEW, COMPARE_VIEW, BUILD_VIEW)

# What each measure IS, in plain words (7d, 11/17). One note per measure,
# used by the card tooltips and by the comparison table's column
# tooltips, so the two views cannot explain the same figure differently.
# Keyed by the workbook's own column header, which is what the cards and
# that table both label the figure.
MEASURE_NOTES = {
    "Scenario": "The run's name, as the workbook's Capacity Scenarios tab "
                "lists it.",
    # "Kind" is KIND_HEADER below; written out here because that constant
    # belongs to the comparison section and a test checks the two agree.
    "Kind": "Declared by the run itself, never worked out here. A CHOICE is "
        "something the operation could decide to do; a RISK is something "
        "that could happen to it, which nobody is proposing; a "
        "SENSITIVITY varies one input of another run to test whether its "
        "finding holds.",
    "All-in cost": "Labour plus the automation's own invoice, over the "
                   "whole 52-week plan year (spec 6.4). Dollars.",
    "All-in vs baseline": "This run's all-in cost minus the baseline's. "
                          "Negative is a saving.",
    "Labour cost": "Pay for the hours worked over the plan year, overtime "
                   "included.",
    "Machine cost": "What the automation is invoiced: a fee per item, the "
                    "shared platform fee, and setup spread over a year. "
                    "Zero in a run that buys no automation, and shown "
                    "anyway so every run reports the same measure.",
    "Network required FTE": "The people the arriving work needs, network "
                            "wide, as whole-time equivalents averaged "
                            "across the plan year. Each role is divided "
                            "by its own productive hours, because the "
                            "counter carries 21.75% shrinkage against the "
                            "back office's 28%.",
    "Overtime hours": "Overtime hours worked over the plan year to close "
                      "gaps the roster could not.",
    "Uncovered hours": "Required work nobody did, after overtime. Never "
                       "netted against another team's spare capacity: a "
                       "spare hour in one team does not staff another "
                       "team's counter.",
    "Requester turnaround, mean days": "How long a request takes from the "
                                       "requester's point of view, "
                                       "averaged over the plan year. Every "
                                       "request counts once, and an item a "
                                       "machine finished counts as "
                                       "finished when the machine finished "
                                       "it.",
    "Weeks over the 5-day goal": "How many of the 52 plan weeks end with "
                                 "projected turnaround above the goal in "
                                 "assumptions.yaml. A count of weeks, not "
                                 "a measure of how far over.",
}

# Which way is better, recorded once.
#
# ALL SIX MEASURES ARE BETTER LOWER (decided 11/16). Required FTE used to
# be neutral here, on the reasoning that a scenario needing more people
# is not failing, only honest about the work. Siri's call is that a
# planner reads required FTE the same way they read cost - fewer is
# better, more is worse - and that a card alone among six with no colour
# reads as a measure nobody has an opinion about, which is its own claim.
# The run's finding is where the nuance belongs; a card is a direction.
BETTER_LOWER = "lower"

CARD_MEASURES = (
    ("total_cost", export.MONEY, BETTER_LOWER),
    ("network_required_fte", export.FTE, BETTER_LOWER),
    ("overtime_hours", export.HOURS, BETTER_LOWER),
    ("uncovered_hours", export.HOURS, BETTER_LOWER),
    ("mean_projected_turnaround_days", export.DAYS, BETTER_LOWER),
    ("weeks_over_turnaround_goal", export.COUNT, BETTER_LOWER),
)

# Each card's label is the workbook's own column header, looked up rather
# than retyped, so a measure is called one thing in this project.
MEASURE_HEADERS = {key: header for header, key in api.COMPARISON_MEASURES}


def format_value(value, number_format):
    """One figure, written the way the workbook writes it."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    formatter = VALUE_FORMATS.get(number_format)
    return formatter(value) if formatter else str(value)


def rounds_to_nothing(difference, number_format):
    """True when a difference disappears at the precision shown.

    ONE RULE FOR EVERY CARD (decided 11/16). A difference of four tenths
    of a person against a card that prints two decimals is not "+0.00
    with an up arrow" - it is a rounding artefact wearing a verdict. If
    the figure on the card would read as zero, the card says so in words
    instead, and the exact difference stays one hover away, so nothing
    is hidden by being rounded.
    """
    if difference is None:
        return False
    decimals = DISPLAY_DECIMALS.get(number_format)
    if decimals is None:
        return abs(difference) < 1e-9
    return round(difference, decimals) == 0


def format_difference(difference, number_format):
    """A difference, signed, or the words for no difference at all."""
    if difference is None:
        return ""
    if rounds_to_nothing(difference, number_format):
        return NO_CHANGE
    sign = "+" if difference > 0 else "-"
    return sign + format_value(abs(difference), number_format)


# ---------------------------------------------------------------------------
# The landing intro (7d, 11/17)
#
# WHAT IT IS FOR. A visitor arrives knowing nothing: not what operation
# this is, not that it is invented, not which of three views to open.
# Three lines answer those in the order they are asked - what this is,
# what it found, how to use it - and then get out of the way.
#
# EVERY FIGURE IN IT IS READ FROM THE SAVED RESULTS. A headline typed
# into a page is a headline that goes stale the first time the model is
# rebuilt, and the stale version is the one a visitor quotes. So the
# biggest risk is FOUND - the risk-kind run whose all-in cost rises most
# against the baseline - and its two figures come off that run's own
# summary row.
# ---------------------------------------------------------------------------

MILLION = 1_000_000


def format_millions(value):
    """A signed figure in millions, to two decimals: "+$1.06M".

    For the intro only, where a full-precision dollar figure would be
    nine characters of noise in a sentence. Everywhere a reader might do
    arithmetic on it, the figure is written in full.
    """
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value) / MILLION:,.2f}M"


def headline_risk(results):
    """The run the Summary tab leads with, and its two headline figures.

    WHOSE CLAIM "BIGGEST RISK" IS. Siri's. export.HEADLINES is her
    approved finding per run, in the order the Summary tab puts them, and
    the first of them is the one the workbook leads with: "The biggest
    risk in the plan: +$1.06M, and turnaround misses the 5-day goal in 27
    weeks."

    An earlier draft of this worked "biggest" out instead - the
    risk-kind run whose all-in cost rose most - and it answered
    Combined, which is +15% volume AND an automation programme together.
    Arithmetically true, editorially wrong: Combined is a compound run,
    and the project's own finding names the volume rise as the single
    biggest risk. So the SELECTION is read from Siri's ordered findings,
    the FIGURES are read from the saved results, and the word "risk" is
    the run's own declared kind - checked here, because a sentence that
    called a choice a risk would be the page editorialising.
    """
    runs = results["scenarios"]
    lead_name = export.HEADLINES[0][0]
    found = [
        (key, run) for key, run in runs.items() if run["name"] == lead_name
    ]
    if not found:
        raise ValueError(
            f"the Summary's lead finding is about {lead_name!r}, which is "
            "not among the saved runs"
        )
    key, run = found[0]
    if run["kind"] != scenarios.KIND_RISK:
        raise ValueError(
            f"{lead_name} declares itself a {run['kind']}, so the intro "
            "cannot call it the biggest risk in the plan"
        )
    baseline = float(runs[BASELINE_KEY]["summary"]["total_cost"])
    return {
        "key": key,
        "name": run["name"],
        "finding": findings().get(run["name"], ""),
        "cost_rise": float(run["summary"]["total_cost"]) - baseline,
        "weeks_over_goal": int(run["summary"]["weeks_over_turnaround_goal"]),
    }


def intro_lines(results, dimensions=None):
    """The three lines above everything: what, what it found, how to use.

    Three, and no more: a landing paragraph a visitor has to read before
    they can start is a landing paragraph they skip.
    """
    dimensions = dimensions if dimensions is not None else api.dimensions()
    risk = headline_risk(results)
    sites = len(dimensions["sites"])
    return (
        f"**What this is:** a workforce capacity model of Northbrook "
        f"Records Services, a fictional {sites}-site medical-records "
        f"operation. Every figure on this page is synthetic - generated "
        f"by the model from the inputs in this repository.",
        f"**The headline finding:** {risk['name']} is the biggest risk in "
        f"the plan - {format_millions(risk['cost_rise'])} over the plan "
        f"year, and {risk['weeks_over_goal']} weeks over the turnaround "
        f"goal.",
        f"**How to use it:** {ONE_RUN_VIEW.lower()} shows one of the "
        f"{results['meta']['scenario_count']} saved runs against the "
        f"baseline, {COMPARE_VIEW.lower()} ranks them all by cost, and "
        f"{BUILD_VIEW.lower()} puts your own inputs through the model.",
    )


# ---------------------------------------------------------------------------
# Picking a run
# ---------------------------------------------------------------------------

HEADLINE_GROUP = "The eight headline runs"
OTHER_GROUP = "The other runs"
BASELINE_GROUP = "The plan as it stands"


def findings():
    """Scenario name -> its one-line finding (export.HEADLINES).

    The findings are Siri's approved wording for the workbook's Summary
    tab, reused rather than rewritten: a dashboard that said something
    slightly different about the same run would leave a reader to decide
    which sentence to believe.
    """
    return dict(export.HEADLINES)


# What each kind is called on screen, derived from the model's own three
# so a kind added to the YAML cannot go unnamed here.
KIND_LABELS = {kind: kind.capitalize() for kind in scenarios.KINDS}


def kind_label(kind):
    """'Risk', 'Sensitivity', 'Choice' - the run's own declared kind.

    Declared in the scenario file (11/09), not worked out here. This used
    to be inferred from the scenario keys, which was a guess about file
    naming doing the work of a fact about the run.
    """
    return KIND_LABELS.get(kind, kind)


def scenario_menu(results):
    """Every run, grouped the way a reader would look for one.

    The baseline on its own, because it is what everything else is
    measured against. Then the eight a leader would act on, in the order
    the Summary tab puts them. Then the rest, with each variant marked
    for the run it varies - a sensitivity read as a scenario in its own
    right is how "AI pays back" gets quoted from the 50% run.
    """
    runs = results["scenarios"]
    finding_of = findings()
    key_of_name = {run["name"]: key for key, run in runs.items()}

    def item(key):
        run = runs[key]
        return {
            "key": key,
            "name": run["name"],
            "label": run["name"],
            "kind": run["kind"],
            "kind_label": kind_label(run["kind"]),
            "finding": finding_of.get(run["name"]),
            "what_it_changes": run["what_it_changes"],
            "starts_week": run["starts_week"],
        }

    headline_keys = [
        key_of_name[name]
        for name, _finding in export.HEADLINES
        if name in key_of_name
    ]
    taken = set(headline_keys) | {BASELINE_KEY}
    others = sorted(
        (key for key in runs if key not in taken),
        key=lambda key: runs[key]["name"],
    )

    groups = []
    if BASELINE_KEY in runs:
        groups.append((BASELINE_GROUP, [item(BASELINE_KEY)]))
    groups.append((HEADLINE_GROUP, [item(key) for key in headline_keys]))
    groups.append((OTHER_GROUP, [item(key) for key in others]))
    return groups


def menu_items(results):
    """The menu flattened, group kept, in the order it is shown."""
    return [
        dict(entry, group=group)
        for group, entries in scenario_menu(results)
        for entry in entries
    ]


# ---------------------------------------------------------------------------
# The headline cards
# ---------------------------------------------------------------------------


def headline_cards(results, key):
    """The six measures for one run, each against the baseline.

    The difference is the scenario's full plan year minus the baseline's,
    the same subtraction the workbook's Summary makes - even for a run
    that starts in plan week 13, because the question a leader asks is
    what the year costs, not what the last forty weeks cost.
    """
    scenario = results["scenarios"][key]
    baseline = results["scenarios"][BASELINE_KEY]
    is_baseline = key == BASELINE_KEY

    cards = []
    for measure, number_format, direction in CARD_MEASURES:
        value = float(scenario["summary"][measure])
        difference = (
            None if is_baseline else value - float(baseline["summary"][measure])
        )
        cards.append(
            {
                "measure": measure,
                "label": MEASURE_HEADERS[measure],
                "value": value,
                "value_text": format_value(value, number_format),
                "difference": difference,
                "difference_text": format_difference(difference, number_format),
                "no_change": rounds_to_nothing(difference, number_format),
                "direction": direction,
                "format": number_format,
            }
        )
    return cards


# ---------------------------------------------------------------------------
# What a card looks like (decided 11/16, after the second walkthrough)
#
# WHY THESE CARDS ARE NOT st.metric. st.metric can colour a difference
# green or red, or turn the colour off - and that is the whole palette.
# It has no way to say "this measure did not move", which is a third
# thing a card has to be able to say and the one a reader most often
# misreads: "+0.00" with an up arrow is a rounding artefact wearing a
# verdict. So the six cards are drawn here, in the same layout st.metric
# used, with a third colour for no change at all.
# ---------------------------------------------------------------------------

# Green, red and blue - DARKENED FOR CONTRAST (Siri, 11/19). These began
# as Streamlit's own three, so that a card looked like the rest of the
# page. Measuring them for the theme pass showed that two of the three
# never reached WCAG AA as small text: Streamlit's green was 2.8:1 and
# its red 3.4:1 against the page, and had been 3.0 and 3.7 against the
# white page before it. Looking like the rest of the page is worth less
# than being readable, so the green and the red are darker and the blue,
# which already passed at 5.0:1, is untouched. A test holds all three to
# AA on the page background.
BETTER_COLOUR = "#077d2a"   # green: moved the way a planner wants, 4.8:1
WORSE_COLOUR = "#c62828"    # red:   moved the other way, 5.1:1
SAME_COLOUR = "#0068c9"     # blue:  did not move at all, 5.0:1

DOWN_ARROW = "▼"
UP_ARROW = "▲"

CARD_STYLE = """<style>
.cm-card { padding: 0.25rem 0 0.75rem 0; }
.cm-card-label { font-size: 0.875rem; line-height: 1.4; opacity: 0.65; }
.cm-card-value { font-size: 2.25rem; font-weight: 600; line-height: 1.25; }
.cm-card-delta { font-size: 0.875rem; line-height: 1.4; }
.cm-card-delta-blank { visibility: hidden; }
</style>"""


def card_colour(card):
    """Green, red or blue - the one rule every card follows.

    Blue first, because "did not move" is a different statement from
    "moved a little the wrong way" and the rounding must not turn one
    into the other. Then the direction: all six measures are better
    lower (11/16), so down is green and up is red.
    """
    if card["difference"] is None:
        return None
    if card["no_change"]:
        return SAME_COLOUR
    better = card["difference"] < 0 if card["direction"] == BETTER_LOWER else (
        card["difference"] > 0
    )
    return BETTER_COLOUR if better else WORSE_COLOUR


def card_arrow(card):
    """Which way the figure moved, or nothing when it did not move.

    No arrow on a no-change card: an arrow is a claim about direction,
    and a difference that rounds away has no direction to report.
    """
    if card["difference"] is None or card["no_change"]:
        return ""
    return UP_ARROW if card["difference"] > 0 else DOWN_ARROW


def exact_difference_text(difference):
    """The difference as it actually is, for the hover.

    The card rounds; this does not. Trailing zeros are trimmed so a
    difference of exactly two people reads as 2 rather than 2.000000.
    """
    if difference is None:
        return ""
    written = f"{difference:+,.6f}"
    if "." in written:
        written = written.rstrip("0").rstrip(".")
    return written


def measure_note(label):
    """What a measure IS, in plain words, or "" if nothing is written.

    One note per measure (7d), shared by the card hovers and the
    comparison table's column hovers. A measure with no note would be a
    figure a reader has to guess at, so a test requires one for every
    card and every column.
    """
    return MEASURE_NOTES.get(label, "")


def card_tooltip(card):
    """What the hover says: what the measure is, then the exact difference.

    TWO THINGS, in that order (7d, 11/17). The note says what the figure
    means, which is the question a visitor has first and which the card's
    label alone cannot answer - "Network required FTE" is a term of art.
    Then the difference as it actually is, so nothing a card rounds away
    is lost: a reader who wants to know whether "no change" means nothing
    moved or merely nothing that shows can find out without leaving the
    page.
    """
    note = measure_note(card["label"])
    lead = f"{card['label']}: {note}" if note else f"{card['label']}."
    if card["difference"] is None:
        return (
            f"{lead} This is the baseline, so there is nothing to compare "
            "it against."
        )
    exact = exact_difference_text(card["difference"])
    if card["no_change"]:
        return (
            f"{lead} Against the baseline: {exact}, which rounds to "
            "nothing at the precision shown."
        )
    return f"{lead} Against the baseline: {exact}."


def card_html(card):
    """One card, in the layout st.metric drew.

    Built as a string here rather than in the page so a test can read a
    card's colour without starting one.
    """
    colour = card_colour(card)
    delta = card["difference_text"]
    if delta:
        arrow = card_arrow(card)
        shown = f"{arrow} {delta}" if arrow else delta
        line = f'<div class="cm-card-delta" style="color: {colour};">{html.escape(shown)}</div>'
    else:
        # The baseline has nothing to compare against. The line is still
        # drawn, invisibly, so six cards keep one height and a row does
        # not jump when the run changes.
        line = '<div class="cm-card-delta cm-card-delta-blank">&nbsp;</div>'
    return (
        f'<div class="cm-card" title="{html.escape(card_tooltip(card))}">'
        f'<div class="cm-card-label">{html.escape(card["label"])}</div>'
        f'<div class="cm-card-value">{html.escape(card["value_text"])}</div>'
        f"{line}</div>"
    )


# ---------------------------------------------------------------------------
# Compare all scenarios (the second view, 11/09)
# ---------------------------------------------------------------------------

KIND_HEADER = "Kind"
ALL_IN_HEADER = "All-in cost"
DIFFERENCE_HEADER = "All-in vs baseline"

# The Capacity Scenarios tab's own number format per column, so this
# table is rounded exactly where the workbook rounds. It carries the
# difference column too, which the cards do not.
SCENARIO_FORMATS = {
    header: number_format
    for header, _source, number_format, _width in export.SCENARIO_COLUMNS
}


def comparison_table(results):
    """Every run, ranked by what it does to the all-in cost.

    RANKED, not alphabetical, and that is the difference between this and
    the workbook's Capacity Scenarios tab. The tab is deliberately
    neutral - alphabetical, because ranking a reference table smuggles in
    an opinion. A browse view is where the opinion belongs: a reader
    opening "compare all" is asking which runs save money, and the answer
    should be the first row rather than something to be found by eye.

    The kind is shown beside the name because the ranking puts a risk and
    a proposal next to each other, and they are not the same claim: the
    cheapest row might be something nobody is proposing to do.
    """
    comparison = results["comparison"]
    baseline = comparison[comparison["scenario"] == BASELINE_LABEL]
    if baseline.empty:
        raise ValueError(
            "the comparison table has no Baseline row to rank against"
        )
    baseline_cost = float(baseline["total_cost"].iloc[0])

    table = api.comparison_view(comparison)
    table.insert(1, KIND_HEADER, [kind_label(kind) for kind in comparison["kind"]])
    position = list(table.columns).index(ALL_IN_HEADER) + 1
    table.insert(
        position,
        DIFFERENCE_HEADER,
        comparison["total_cost"].astype(float) - baseline_cost,
    )
    return table.sort_values(DIFFERENCE_HEADER).reset_index(drop=True)


def comparison_column_specs(table):
    """One spec per column of the comparison table (7d, 11/17).

    Same shape as plan_column_specs, and for the same two reasons. The
    figures are written the way the cards write them - before this the
    table printed raw numbers, so the all-in cost read "$7,456,636" on a
    card and "7456636" here. And every column carries the plain-language
    note for its measure, which is the SAME note the card's hover gives,
    so the two views cannot explain one figure two ways.
    """
    specs = []
    for header in table.columns:
        number_format = SCENARIO_FORMATS.get(header)
        note = MEASURE_NOTES.get(header, "")
        specs.append(
            {
                "column": header,
                "label": header,
                "help": note,
                "format": SCREEN_NUMBER_FORMATS.get(number_format, TEXT_COLUMN),
            }
        )
    return specs


def comparison_display(table):
    """The same table, rounded the way the workbook's tab prints it."""
    rounded = table.copy()
    for header in rounded.columns:
        decimals = DISPLAY_DECIMALS.get(SCENARIO_FORMATS.get(header))
        if decimals is None:
            continue
        if pd.api.types.is_numeric_dtype(rounded[header]):
            rounded[header] = rounded[header].round(decimals)
    return rounded



# ---------------------------------------------------------------------------
# What a kind looks like (7d final pass, 11/19)
#
# WHY THESE THREE COLOURS AND NOT ANY OTHERS. Red, green and blue are
# spoken for: on a card they mean worse, better and no change (11/16), so
# a kind wearing one of them would be read as a verdict on the run. That
# leaves violet, orange and gray - which happen to be the three colours
# Streamlit's own markdown palette offers outside red, green and blue, so
# the kind beside a run in the sidebar (where only markdown is allowed)
# and the kind in the comparison table (where the Styler takes a hex) can
# be the SAME colour rather than two colours that nearly match. The hexes
# live in .streamlit/config.toml as violetColor, orangeColor and
# grayColor, and a test checks the file and this table agree.
#
# Gray for a sensitivity is not an accident either: a sensitivity is not
# a proposal, and a muted colour says so before the sentence does.
# ---------------------------------------------------------------------------

KIND_COLOURS = {
    scenarios.KIND_CHOICE: "#5b3a8c",       # violet
    scenarios.KIND_RISK: "#8a4b0a",         # orange
    scenarios.KIND_SENSITIVITY: "#4f555e",  # gray
}

# The same three as Streamlit markdown colour names, for the places that
# take markdown and not HTML - a radio's captions, above all.
KIND_MARKDOWN = {
    scenarios.KIND_CHOICE: "violet",
    scenarios.KIND_RISK: "orange",
    scenarios.KIND_SENSITIVITY: "gray",
}

# One line each, which is what the key table carries. These are the three
# sentences the Compare-all caption used to spell out in a paragraph.
KIND_MEANINGS = {
    scenarios.KIND_CHOICE:
        "Something the operation could decide to do.",
    scenarios.KIND_RISK:
        "Something that could happen to it, which nobody is proposing.",
    scenarios.KIND_SENSITIVITY:
        "Varies one input of another run, to test whether its finding "
        "holds - not a proposal of its own.",
}


def kind_colour(kind):
    """The hex a kind is written in, or None for a kind nobody declared."""
    return KIND_COLOURS.get(kind)


def kind_markdown(kind):
    """A kind's label in Streamlit markdown, coloured: ":violet[Choice]".

    For the sidebar captions, which take markdown and not HTML. The
    colour name resolves to the hex in .streamlit/config.toml, so this
    and the comparison table are the same colour by construction rather
    than by two people choosing the same swatch.
    """
    label = kind_label(kind)
    name = KIND_MARKDOWN.get(kind)
    return f":{name}[{label}]" if name else label


def kind_key_rows():
    """The kind key: one row per kind, in the order a reader meets them."""
    return [
        {
            "kind": kind,
            "label": kind_label(kind),
            "colour": KIND_COLOURS[kind],
            "meaning": KIND_MEANINGS[kind],
        }
        for kind in scenarios.KINDS
    ]


def comparison_styler(table):
    """The comparison table with each run's kind in its kind's colour.

    The Kind column only. Streamlit takes the colours from a Styler and
    its own column_config still decides every number's format, which is
    what keeps 7d's one-format-per-measure rule true here.
    """
    colour_of = {kind_label(kind): colour for kind, colour in KIND_COLOURS.items()}

    def paint(column):
        return [
            f"color: {colour_of[value]}; font-weight: 600"
            if value in colour_of else ""
            for value in column
        ]

    shown = comparison_display(table)
    if KIND_HEADER not in shown.columns:
        return shown.style
    return shown.style.apply(paint, subset=[KIND_HEADER])


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

NETWORK_KIND = "network"
CLUSTER_KIND = "cluster"
SITE_KIND = "site"
NETWORK_LABEL = "Network (every site and the back office)"

# The hover text for the view switcher and the two filters (7d, 11/17).
# Written here beside the choices they explain, so the page cannot
# describe a filter one way while this module builds it another.
VIEW_HELP = (
    f"Three ways in. {ONE_RUN_VIEW}: one saved run against the baseline, "
    f"with its chart and its rows. {COMPARE_VIEW}: every run at once, "
    f"ranked by what it does to the all-in cost. {BUILD_VIEW}: set the "
    "inputs yourself and run the model on them."
)
RUN_HELP = (
    "Which saved run to look at. The baseline first, because everything "
    "else is measured against it; then the eight the workbook's Summary "
    "headlines; then the rest. The line under each name is the kind the "
    "run declares itself to be."
)
SCOPE_HELP = (
    "How much of the network the chart and the table below cover: "
    "everything, one cluster of sites, or one site. The cards above are "
    "always network-wide, because that is what the workbook's Summary "
    "reports."
)
ROLES_HELP = (
    "Which roles the chart and the table below count. Shortfall is added "
    "up over the roles picked and never netted against another role's "
    "spare capacity, so unticking a role removes its shortfall rather "
    "than moving it somewhere else."
)


def scope_choices(dimensions):
    """The three grains the plan is made at: network, cluster, site.

    A choice carries the scope codes it covers, so filtering is a lookup
    rather than the same rule written out again per chart.
    """
    choices = [
        {
            "kind": NETWORK_KIND,
            "code": NETWORK_KIND,
            "label": NETWORK_LABEL,
            "scopes": None,  # None means every scope, NETWORK included
        }
    ]
    for cluster in dimensions["clusters"]:
        sites = ", ".join(cluster["sites"])
        choices.append(
            {
                "kind": CLUSTER_KIND,
                "code": cluster["code"],
                "label": f"{cluster['name']} ({sites})",
                "scopes": list(cluster["sites"]),
            }
        )
    for site in dimensions["sites"]:
        choices.append(
            {
                "kind": SITE_KIND,
                "code": site["code"],
                "label": f"{site['code']} - {site['name']}",
                "scopes": [site["code"]],
            }
        )
    return choices


def network_roles(dimensions):
    """The back-office roles: one network pool, not based anywhere."""
    return [
        role
        for role in dimensions["roles"]
        if role["scope"] == api.NETWORK_SCOPE
    ]


def roles_in_scope(dimensions, choice):
    """The roles a site or cluster view can honestly show.

    A back-office role has no site. Offering it inside a site view would
    mean drawing a line at zero for a team that is not absent from the
    site - it was never there to be absent, and the work it does for that
    site is counted at NETWORK scope (spec 6.7).
    """
    if choice["kind"] == NETWORK_KIND:
        return list(dimensions["roles"])
    return [
        role for role in dimensions["roles"] if role["scope"] == api.SITE_SCOPE
    ]


def scope_note(dimensions, choice):
    """What a site or cluster view leaves out, said plainly."""
    if choice["kind"] == NETWORK_KIND:
        return ""
    names = ", ".join(role["name"] for role in network_roles(dimensions))
    kind = choice["kind"]
    return (
        f"{names} are one network team rather than site staff, so they are "
        f"not in a {kind} view. Their work for this {kind} is real, and it "
        f"is counted at {dimensions['network_code']} scope in the network "
        "view - showing them here as zero would read as a measurement "
        "instead of an absence of one."
    )


def filter_plan(plan, role_codes=None, scopes=None):
    """The Capacity Plan rows for the chosen roles and scopes."""
    table = plan
    if role_codes is not None:
        table = table[table["role"].isin(list(role_codes))]
    if scopes is not None:
        table = table[table["scope"].isin(list(scopes))]
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

GAP_COLUMN = "short_fte"
PLAN_WEEK = "Plan week"


def weekly_gap(results, key, role_codes=None, scopes=None):
    """Short FTE by plan week, this run against the baseline.

    SHORTFALL IS NOT NETTED AGAINST SURPLUS. A spare hour in the back
    office on Thursday does not staff a counter that was short on
    Tuesday, and the model keeps the two apart for that reason (6.2), so
    this chart shows what was short and the surplus table shows the rest.
    Netting them would flatten exactly the weeks a planner is looking
    for.
    """
    columns = {}
    picks = [(BASELINE_LABEL, BASELINE_KEY)]
    if key != BASELINE_KEY:
        picks.append((results["scenarios"][key]["name"], key))
    for label, scenario_key in picks:
        plan = filter_plan(
            results["scenarios"][scenario_key]["plan"], role_codes, scopes
        )
        columns[label] = plan.groupby("plan_week")[GAP_COLUMN].sum()
    chart = pd.DataFrame(columns)
    chart.index.name = PLAN_WEEK
    return chart


# The workbook's own names for the two halves (7d, 11/17): the Capacity
# Scenarios tab calls them Labour cost and Machine cost, and so does the
# comparison table, so the legend on this chart calls them that too.
LABOUR_HEADER = "Labour cost"
MACHINE_HEADER = "Machine cost"

COST_HALVES = (
    (LABOUR_HEADER, "labour_cost"),
    (MACHINE_HEADER, "machine_cost"),
)


def cost_split(results, key):
    """All-in cost as its two halves, this run and the baseline (6.4).

    Always both halves, even when the machine half is zero: the baseline
    buys no automation, and dropping the empty half would make the two
    bars look like two different measures.
    """
    rows = {}
    picks = [BASELINE_KEY] if key == BASELINE_KEY else [BASELINE_KEY, key]
    for scenario_key in picks:
        scenario = results["scenarios"][scenario_key]
        rows[scenario["name"]] = {
            label: float(scenario["summary"][measure])
            for label, measure in COST_HALVES
        }
    frame = pd.DataFrame(rows).T
    frame.index.name = "Run"
    return frame[[label for label, _measure in COST_HALVES]]


# ---------------------------------------------------------------------------
# The charts, built as locked specs (11/16, after Siri's walkthrough)
#
# WHY THE CHARTS ARE BUILT HERE AND NOT DRAWN BY STREAMLIT. st.line_chart
# and st.bar_chart hand Vega-Lite a pan-and-zoom binding, so a trackpad
# gesture over the chart rescales both axes. A leader who does that by
# accident has no way back short of reloading the page, and the chart
# they are looking at in the meantime is one whose axes nobody chose.
# These specs set both scales explicitly and bind nothing, so the picture
# cannot be moved. Hover tooltips are untouched - reading a value off a
# point is the interaction this chart is for.
#
# The axes are LOCKED, NOT AUTOSCALED-AND-FROZEN. The x axis is the plan
# year, always every week of it, so two runs are read against the same
# calendar and a scenario that only bites in week 40 still shows its
# quiet start. The y axis is fixed to a round number above the tallest
# value in the CURRENT SELECTION, so filtering to one site rescales it
# once, deliberately, rather than leaving a one-site line flat along the
# bottom of a network-sized axis.
# ---------------------------------------------------------------------------

RUN_HEADER = "Run"
SHORT_FTE_HEADER = "Short FTE"
COST_HEADER = ALL_IN_HEADER
COST_HALF_HEADER = "Half of the all-in cost"
CHART_HEIGHT = 320

# A chart that fits its data exactly puts the top point on the frame. A
# little headroom keeps the peak readable without inventing space.
HEADROOM = 1.08


def nice_ceiling(value):
    """The smallest round number at or above `value`, as an axis top.

    Round means 1, 2 or 5 times a power of ten - the steps an axis is
    normally labelled in - so the locked top lands somewhere a reader
    would have put it, and a chart that is redrawn after a filter change
    does not come back with a top of 1,473.6.
    """
    if not math.isfinite(value) or value <= 0:
        # An all-zero selection still needs an axis. One unit of whatever
        # is being measured gives a flat line along the bottom, which is
        # the truth, instead of a degenerate scale.
        return 1.0
    power = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 5, 10):
        if value <= step * power:
            return float(step * power)
    return float(10 * power)


def _locked_x(field, plan_weeks):
    """The plan year, every week of it, and not draggable."""
    return alt.X(
        f"{field}:Q",
        title=PLAN_WEEK,
        # nice=False, or Vega-Lite rounds the domain outwards to its own
        # taste and the axis stops being the plan year.
        scale=alt.Scale(domain=[1, plan_weeks], nice=False, zero=False, clamp=True),
        axis=alt.Axis(tickMinStep=1, format="d"),
    )


# The axis tick labels, in the same precision the cards and tables use
# (7d, 11/17). Vega-Lite's default would have written the cost axis in
# SI units - "1.5M" against a card reading "$1,456,636" - which is the
# same figure in two notations on one screen.
AXIS_FORMATS = {
    SHORT_FTE_HEADER: ",.2f",   # FTE to two decimals, as everywhere else
    COST_HEADER: "$,.0f",       # dollars to the dollar, as on the cards
}


def _locked_y(field, title, top):
    """Zero to a round top, fixed, not draggable, labelled as elsewhere."""
    return alt.Y(
        f"{field}:Q",
        title=title,
        scale=alt.Scale(domain=[0, top], nice=False, clamp=True),
        axis=alt.Axis(format=AXIS_FORMATS.get(title)),
    )


def gap_chart(table, plan_weeks):
    """The weekly shortfall, locked to the plan year.

    `table` is what weekly_gap returned: one column per run, indexed by
    plan week. It is reshaped to one row per run and week, because a
    chart that colours by run needs the run to be a value rather than a
    column name.
    """
    long = (
        table.reset_index()
        .melt(id_vars=PLAN_WEEK, var_name=RUN_HEADER, value_name=SHORT_FTE_HEADER)
    )
    top = nice_ceiling(float(long[SHORT_FTE_HEADER].max()) * HEADROOM)
    return (
        alt.Chart(long)
        .mark_line(point=False)
        .encode(
            x=_locked_x(PLAN_WEEK, plan_weeks),
            y=_locked_y(SHORT_FTE_HEADER, SHORT_FTE_HEADER, top),
            color=alt.Color(f"{RUN_HEADER}:N", title=RUN_HEADER),
            tooltip=[
                alt.Tooltip(f"{RUN_HEADER}:N", title=RUN_HEADER),
                alt.Tooltip(f"{PLAN_WEEK}:Q", title=PLAN_WEEK, format="d"),
                alt.Tooltip(
                    f"{SHORT_FTE_HEADER}:Q", title=SHORT_FTE_HEADER, format=",.2f"
                ),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )


def cost_chart(table):
    """All-in cost as its two halves, one locked bar per run."""
    long = (
        table.reset_index()
        .melt(id_vars=RUN_HEADER, var_name=COST_HALF_HEADER, value_name=COST_HEADER)
    )
    # The bars are stacked, so the axis has to clear the whole bar, not
    # the taller of its two halves.
    tallest = float(long.groupby(RUN_HEADER)[COST_HEADER].sum().max())
    top = nice_ceiling(tallest * HEADROOM)
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X(f"{RUN_HEADER}:N", title=None, sort=list(table.index)),
            y=_locked_y(COST_HEADER, COST_HEADER, top),
            color=alt.Color(f"{COST_HALF_HEADER}:N", title=COST_HALF_HEADER),
            tooltip=[
                alt.Tooltip(f"{RUN_HEADER}:N", title=RUN_HEADER),
                alt.Tooltip(f"{COST_HALF_HEADER}:N", title=COST_HALF_HEADER),
                alt.Tooltip(f"{COST_HEADER}:Q", title=COST_HEADER, format="$,.0f"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )


def chart_is_locked(chart):
    """True when nothing in this spec lets a mouse rescale it.

    Vega-Lite pans and zooms only through a parameter bound to "scales",
    which is what Altair's .interactive() and Streamlit's built-in charts
    add. No such parameter anywhere in the spec means the axes are the
    ones the code chose, and stay so.
    """
    return not _scale_bindings(chart.to_dict())


def _scale_bindings(node):
    """Every parameter in a spec that is bound to the scales."""
    found = []
    if isinstance(node, dict):
        for parameter in node.get("params", []) or []:
            if isinstance(parameter, dict) and parameter.get("bind") == "scales":
                found.append(parameter)
        for value in node.values():
            found.extend(_scale_bindings(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_scale_bindings(value))
    return found


# ---------------------------------------------------------------------------
# The detail table
# ---------------------------------------------------------------------------

HOURS_GAP_HEADER = "Hours gap (available - required)"


def plan_view(plan):
    """The Capacity Plan rows under the workbook's own column headers.

    The hours gap is a live formula in the workbook, so a reader can see
    the subtraction rather than trust it; here it is computed from the
    same two columns and given the same name. Role and scope stay as the
    model's own codes, which is what the tab and the CSV carry, so a
    downloaded row can be matched to a workbook row without translating
    anything.
    """
    columns = {}
    for header, source, _format, _width, _total in export.CAPACITY_PLAN_COLUMNS:
        if source is None:
            if header != HOURS_GAP_HEADER:
                raise ValueError(
                    f"Capacity Plan column {header!r} is a workbook formula "
                    "with no equivalent here; add it before showing it."
                )
            columns[header] = plan["available_hours"] - plan["required_hours"]
        else:
            columns[header] = plan[source]
    return pd.DataFrame(columns)


ROLE_HEADER = "Role"
SCOPE_HEADER = "Site or scope"


@functools.lru_cache(maxsize=1)
def role_short_names():
    """Role code -> the short name skills.yaml gives it.

    Cached because it reads the YAML and the answer cannot change while
    the page is open. The names come through the doorway like every
    other piece of reference data, so the page never reads a file itself
    and cannot start naming a role its own way.
    """
    return {role["code"]: role["short_name"] for role in api.dimensions()["roles"]}


def plan_display(view, names=None):
    """The same table, rounded and named for a reader (11/16).

    A copy for the screen only, and two things are done to it. The
    figures are rounded to the precision the workbook prints. The role
    is written as a person would say it - "Tier 1", not "tier_1" -
    because a code name on a page is the model talking to itself in
    front of a reader.

    THE DOWNLOAD IS NOT TOUCHED. It keeps the codes, because a
    downloaded row has to match a workbook row, and because the code is
    what a reader would filter or join on. A short name is a thing done
    to a screen.
    """
    rounded = view.copy()
    if ROLE_HEADER in rounded.columns:
        names = role_short_names() if names is None else names
        rounded[ROLE_HEADER] = rounded[ROLE_HEADER].map(
            lambda code: names.get(code, code)
        )
    decimals_of = {
        header: DISPLAY_DECIMALS.get(number_format)
        for header, _source, number_format, _width, _total in (
            export.CAPACITY_PLAN_COLUMNS
        )
    }
    decimals_of[HOURS_GAP_HEADER] = DISPLAY_DECIMALS[export.HOURS]
    for header, decimals in decimals_of.items():
        if decimals is None or header not in rounded.columns:
            continue
        if pd.api.types.is_numeric_dtype(rounded[header]):
            rounded[header] = rounded[header].round(decimals)
    return rounded


# A short header for every Capacity Plan column, the width it needs, and
# the full meaning as a tooltip (11/16, after Siri's walkthrough).
#
# WHY SHORT HEADERS AT ALL. Seventeen columns under the workbook's own
# headers do not fit a laptop screen, and a table that scrolls sideways
# hides the columns a reader most wants to compare - available against
# required, short against surplus - behind a gesture. So the screen gets
# short headers and the full workbook header sits in the tooltip, which
# is also where a column that needs explaining gets it.
#
# THE CSV IS NOT TOUCHED. It keeps the workbook's headers and the full
# precision, because a downloaded row has to match a workbook row and
# because a reader does arithmetic on a CSV. Short headers are a thing
# done to a screen, not to the data.
#
# WIDTHS ARE IN PIXELS and were measured against the widest value each
# column actually holds (Role is "records_specialist" at 18 characters,
# which is why it is the widest column on the page and cannot be made
# much narrower without cutting a role name in half).
PLAN_WIDTH_BUDGET = 1200

PLAN_COLUMNS = (
    # workbook header, short header, width, tooltip
    ("Plan week", "Week", 48, "Plan week, 1 to the end of the plan year."),
    ("Week starting", "Starting", 88, "Week starting (the Monday)."),
    (
        "Role",
        "Role",
        148,
        "Role, by its short name. The download carries the model's own "
        "code instead, so a downloaded row still matches a workbook row.",
    ),
    (
        "Site or scope",
        "Scope",
        68,
        "Site or scope. A site code for the two onsite roles, NETWORK for "
        "the back-office teams, whose work is not done at a site.",
    ),
    ("Required hours", "Req", 68, "Required hours: the work that arrived."),
    (
        "Available hours",
        "Avail",
        68,
        "Available hours: the roster after shrinkage, before overtime.",
    ),
    (
        "Hours gap (available - required)",
        "Gap",
        74,
        "Hours gap: available minus required. Negative is a shortfall.",
    ),
    ("Covered hours", "Cov", 68, "Covered hours: required work that got done."),
    (
        "Uncovered hours",
        "Uncov",
        68,
        "Uncovered hours: required work that did not get done, after "
        "overtime. Never netted against another team's surplus.",
    ),
    ("Overtime hours", "OT", 68, "Overtime hours worked to close the gap."),
    ("Paid headcount", "Paid HC", 64, "Paid headcount on the roster that week."),
    (
        "Short FTE",
        "Short",
        56,
        "Short FTE: the gap as whole people, before overtime. This is the "
        "number the hiring trigger reads.",
    ),
    (
        "Surplus FTE",
        "Surplus",
        64,
        "Surplus FTE: spare capacity, shown separately because it never "
        "offsets another team's shortfall.",
    ),
    (
        "Phone service level",
        "Phone",
        56,
        "Phone service level. Blank where nobody is waiting on a phone - "
        "a zero there would read as a measurement.",
    ),
    (
        "Walk-in service level",
        "Walk-in",
        64,
        "Walk-in service level. Blank for the back office, which has no "
        "counter.",
    ),
    (
        "Days of inventory",
        "Days",
        52,
        "Days of inventory: work waiting, in days. Blank for work that is "
        "not allowed to wait.",
    ),
    ("Labour cost", "Labour", 68, "Labour cost for the week, in dollars."),
)

# How each column is written on screen. Streamlit takes a printf string,
# so the precision is still the workbook's (hours to a tenth, FTE to a
# hundredth) - what changes is the thousands separator and the percent
# sign, which cost nothing and save a reader doing the conversion. The
# table is shared with the comparison view (7d), so one measure is
# written one way wherever it appears.
PLAN_NUMBER_FORMATS = SCREEN_NUMBER_FORMATS

def full_names_note():
    """'Tier 1 is Remote Fulfillment Specialist, Tier 1', and so on.

    Only the roles whose short name is not already the whole name, so
    the tooltip does not solemnly explain that QA Specialist is called
    QA Specialist.
    """
    shortened = [
        role
        for role in api.dimensions()["roles"]
        if role["short_name"] != role["name"]
    ]
    if not shortened:
        return ""
    said = "; ".join(
        f"{role['short_name']} is {role['name']}" for role in shortened
    )
    return f"In full: {said}."


def plan_column_specs():
    """One spec per Capacity Plan column, as plain data.

    Plain data rather than Streamlit objects on purpose: the dashboard
    turns these into st.column_config, and a test can read the widths and
    the headers without starting a page.
    """
    kind_of = {
        header: number_format
        for header, _source, number_format, _width, _total in (
            export.CAPACITY_PLAN_COLUMNS
        )
    }
    kind_of[HOURS_GAP_HEADER] = export.HOURS

    specs = []
    for header, short, width, note in PLAN_COLUMNS:
        number_format = kind_of[header]
        if header == ROLE_HEADER:
            # The full name on hover, for the two roles whose full name
            # does not fit a column. Built from the names themselves, so
            # a role renamed in skills.yaml cannot leave a stale sentence
            # behind here.
            note = f"{note} {full_names_note()}"
        specs.append(
            {
                "column": header,
                "label": short,
                "width": width,
                # The full workbook header leads the tooltip, so a reader
                # who wonders what "Uncov" is called on the tab can find
                # out without leaving the page.
                "help": f"{header}. {note}",
                "format": PLAN_NUMBER_FORMATS.get(number_format, TEXT_COLUMN),
            }
        )
    return specs



# ---------------------------------------------------------------------------
# Which region a row belongs to, and its tint (7d final pass, 11/19)
#
# WHY TINT THE ROWS AT ALL. The detail table is 780 rows of three
# different things stacked: the North sites, the South sites, and the
# back office, which is not at a site at all. Sorted by week, those
# interleave, and a reader scanning for "what happened at the South
# counter" has to read the Scope column on every line. A tint does that
# reading for them.
#
# WHY NOT RED, GREEN OR BLUE. Those three are spoken for on the cards -
# worse, better, no change - so a tinted row in one of them would look
# like a verdict on that row. These three are very pale (every one keeps
# charcoal text above 12:1, far past AA) and carry no meaning of their
# own beyond "same region".
#
# THE DOWNLOAD IS NOT TOUCHED. A tint is a thing done to a screen; the
# CSV is the same bytes it was, which a test checks.
# ---------------------------------------------------------------------------

NETWORK_REGION = "network"

REGION_TINTS = {
    "north": "#f3ecdf",          # warm sand
    "south": "#eae6f2",          # pale lilac
    NETWORK_REGION: "#e5e6e8",   # pale neutral gray
}

REGION_FALLBACK_TINT = "#ffffff"


def region_of(scope, dimensions=None):
    """Which region one plan row's scope sits in.

    A site code answers to its cluster; NETWORK answers to itself,
    because the back-office teams work every site's queue remotely and
    are not at any of them (spec 6.7). Read from the loaded dimensions
    rather than from the letter the code starts with, so a site added to
    sites.yaml lands in its real cluster.
    """
    dimensions = dimensions if dimensions is not None else api.dimensions()
    if scope == dimensions["network_code"]:
        return NETWORK_REGION
    for cluster in dimensions["clusters"]:
        if scope in cluster["sites"]:
            return cluster["code"]
    return None


def region_key_rows(dimensions=None):
    """The region key: label, tint and what the tint covers."""
    dimensions = dimensions if dimensions is not None else api.dimensions()
    rows = []
    for cluster in dimensions["clusters"]:
        rows.append(
            {
                "region": cluster["code"],
                "label": cluster["name"],
                "colour": REGION_TINTS.get(cluster["code"], REGION_FALLBACK_TINT),
                "meaning": ", ".join(cluster["sites"]),
            }
        )
    rows.append(
        {
            "region": NETWORK_REGION,
            "label": f"{dimensions['network_code']} (back office)",
            "colour": REGION_TINTS[NETWORK_REGION],
            "meaning": ", ".join(
                role["short_name"] for role in network_roles(dimensions)
            ),
        }
    )
    return rows


def row_tints(view, dimensions=None):
    """One tint per row of the plan view, in the view's own order."""
    dimensions = dimensions if dimensions is not None else api.dimensions()
    return [
        REGION_TINTS.get(region_of(scope, dimensions), REGION_FALLBACK_TINT)
        for scope in view[SCOPE_HEADER]
    ]


def plan_styler(view, dimensions=None, names=None):
    """The plan view, rounded and named for a reader, tinted by region.

    Streamlit takes the colours from a Styler and keeps its own
    column_config for every number's format - its words: "Text and
    number formatting from column_config always takes precedence over
    text and number formatting from pandas.Styler" - so the tints cannot
    quietly undo 7d's one-format-per-measure rule.
    """
    shown = plan_display(view, names=names)
    if SCOPE_HEADER not in view.columns:
        return shown.style
    # Keyed by the index LABEL rather than by position: pandas hands
    # `apply(axis=1)` a row whose .name is its label, and a view that
    # arrived with anything but a 0..n index would otherwise be tinted
    # off by however much the labels differ.
    tint_of = dict(zip(view.index, row_tints(view, dimensions)))

    def paint(row):
        return [f"background-color: {tint_of[row.name]}"] * len(row)

    return shown.style.apply(paint, axis=1)


# ---------------------------------------------------------------------------
# The two colour keys, and whether their colours can be read (11/19)
# ---------------------------------------------------------------------------

KEY_STYLE = """<style>
.cm-key { display: flex; flex-wrap: wrap; gap: 0.25rem 1.25rem;
          font-size: 0.8125rem; line-height: 1.5; }
.cm-key-item { display: flex; align-items: center; gap: 0.4rem; }
.cm-swatch { width: 0.85rem; height: 0.85rem; border-radius: 0.2rem;
             border: 1px solid rgba(0, 0, 0, 0.25); flex: none; }
.cm-key-table { border-collapse: collapse; font-size: 0.8125rem; }
.cm-key-table td { padding: 0.15rem 0.6rem 0.15rem 0; vertical-align: top; }
</style>"""


def region_key_html(rows):
    """The region key, as a line of swatches beside the download."""
    items = "".join(
        f'<div class="cm-key-item">'
        f'<span class="cm-swatch" style="background: {row["colour"]};"></span>'
        f'<span>{html.escape(row["label"])}</span></div>'
        for row in rows
    )
    return f'<div class="cm-key">{items}</div>'


def kind_key_html(rows):
    """The kind key: a swatch, the kind, and one line of what it means.

    A table rather than a paragraph (11/19). The paragraph said the same
    three things and nobody read it, because a reader scanning a ranked
    list wants to know what the word beside a run means, not to be
    taught a taxonomy.
    """
    body = "".join(
        f"<tr>"
        f'<td><span class="cm-swatch" style="background: {row["colour"]};">'
        f"</span></td>"
        f'<td style="color: {row["colour"]}; font-weight: 600;">'
        f'{html.escape(row["label"])}</td>'
        f"<td>{html.escape(row['meaning'])}</td>"
        f"</tr>"
        for row in rows
    )
    return f'<table class="cm-key-table">{body}</table>'


def _channel(value):
    value = value / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(colour):
    """WCAG relative luminance of a #rrggbb colour."""
    digits = colour.lstrip("#")
    red, green, blue = (int(digits[at:at + 2], 16) for at in (0, 2, 4))
    return (
        0.2126 * _channel(red)
        + 0.7152 * _channel(green)
        + 0.0722 * _channel(blue)
    )


def contrast_ratio(one, other):
    """WCAG contrast between two colours, 1 (same) to 21 (black on white).

    Here so the test suite can hold the theme to a number rather than to
    somebody's eye: AA wants 4.5 for ordinary text and 3 for a border or
    a large label.
    """
    first, second = relative_luminance(one), relative_luminance(other)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


AA_TEXT = 4.5
AA_LARGE = 3.0


# ---------------------------------------------------------------------------
# A run somebody built themselves (7c)
# ---------------------------------------------------------------------------


def custom_bundle(saved, run):
    """The saved bundle with one custom run spliced in beside the baseline.

    WHY SPLICE RATHER THAN BUILD A SECOND SET OF VIEWS. Every function
    above takes a bundle and a key, and a custom run is the same shape as
    a saved one - api.run returns what the saved file stores. Handing the
    cards, the charts and the table a two-run bundle means the panel
    shows its result through exactly the code the browse view uses, so a
    custom run and a saved run cannot be reported differently.

    The baseline comes from the saved file rather than being re-run: it
    is the same baseline, it is already on disk, and re-running it would
    add twenty-five seconds to every custom run to arrive at a number
    the file already holds.
    """
    baseline = saved["scenarios"][BASELINE_KEY]
    return {
        "meta": saved["meta"],
        "comparison": pd.DataFrame([baseline["summary"], run["summary"]]),
        "scenarios": {BASELINE_KEY: baseline, run["key"]: run},
    }


CSV_COMMENT = "#"


def custom_csv_bytes(view, changes):
    """The filtered rows, with what was changed written above them.

    WHY A COMMENT BLOCK AND NOT EXTRA COLUMNS. A downloaded row has to
    stay matchable against a workbook row, which is the rule the CSV has
    followed since 7b, so the data is left exactly as it is and the
    provenance goes above it on lines a reader and pandas both skip
    (`comment="#"`). A custom run is the one download where that
    provenance is not optional: the figures came from inputs that exist
    nowhere else, so a file that did not carry them would be unreadable
    a week later.
    """
    lines = [f"{CSV_COMMENT} {api.CUSTOM_RUN_NAME} - Northbrook Records "
             "Services is fictional and every figure here is generated."]
    if changes:
        lines.append(f"{CSV_COMMENT} Changed from the baseline:")
        lines.extend(f"{CSV_COMMENT}   {line}" for line in changes)
    else:
        lines.append(
            f"{CSV_COMMENT} Nothing changed: these are the baseline inputs."
        )
    head = "\n".join(lines) + "\n"
    return head.encode("utf-8") + csv_bytes(view)


def custom_csv_name(scope_code):
    """SYNTHETIC and CUSTOM both in the name, for a file that travels."""
    return f"capacity_plan_CUSTOM_{scope_code}_SYNTHETIC.csv"


def csv_bytes(view):
    """The filtered rows as a CSV, at full precision."""
    return view.to_csv(index=False).encode("utf-8")


def csv_name(key, scope_code):
    """SYNTHETIC in the file name, as every generated file here is.

    A downloaded file travels further than the page it came from, and the
    file name is the only label that travels with it.
    """
    return f"capacity_plan_{key}_{scope_code}_SYNTHETIC.csv"
