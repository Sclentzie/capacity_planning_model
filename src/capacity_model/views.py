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
import re

import altair as alt
import pandas as pd

from . import api, export, scenarios

BASELINE_KEY = "baseline"
BASELINE_LABEL = "Baseline"

# What the Kind column says for the baseline row (Siri, 11/25). Its file
# declares a kind like every other run, but on a table ranked against the
# baseline, the baseline is the line everything is measured from rather
# than one of the three kinds of run - so it is named, not classified,
# and it carries no kind colour.
BASELINE_KIND_LABEL = BASELINE_LABEL


# ---------------------------------------------------------------------------
# Light and dark: every custom colour is a PAIR (7d, 11/19)
#
# WHY TWO PALETTES. The page is read on other people's screens, and a
# reader whose machine is set to dark mode gets a white page thrown at
# them unless the app has a dark palette of its own. Streamlit 1.67 takes
# two, as [theme.light] and [theme.dark] in .streamlit/config.toml, and
# shows its own Light/Dark/System switch once both exist.
#
# WHY THE COLOURS CANNOT JUST BE REUSED. Streamlit swaps its own
# backgrounds, but the colours THIS model invented - the three card
# verdicts, the three scope tints, the three scenario kinds - are hexes
# in this file, and a hex that passes WCAG AA on a cream page fails on a
# charcoal one. A dark green on dark charcoal is 1.4:1: a reader sees
# nothing. So each one is a pair, and the shade is picked at render time
# from the palette the viewer is actually looking at.
#
# HOW THE MODE IS KNOWN. The dashboard reads st.context.theme.type and
# passes it in. This module stays free of Streamlit on purpose (a view
# layer that imported it could not be tested without starting a server),
# so nothing here reaches for the mode itself - it is always an argument,
# and an absent or unrecognised one falls back to light.
# ---------------------------------------------------------------------------

LIGHT_MODE = "light"
DARK_MODE = "dark"
MODES = (LIGHT_MODE, DARK_MODE)

# Light, not dark, when nobody said. Streamlit cannot always tell which
# mode a viewer is in - its own docs say the type "may be incorrect
# during a change in theme", including on the first load of a session -
# so this has to answer something sane for None rather than raise. One
# rerun later the real answer arrives and the colours follow it.
DEFAULT_MODE = LIGHT_MODE


def resolve_mode(mode=None):
    """Which palette to draw in: light unless told dark, clearly.

    Anything that is not one of the two known modes - None, "", a
    Streamlit version that starts calling it something else - is light,
    because a page that renders in the wrong palette is a bad day and a
    page that raises is a blank screen.
    """
    return mode if mode in MODES else DEFAULT_MODE


def shade(pair, mode=None):
    """One colour out of a light/dark pair.

    Every custom colour in this file is stored as {"light": ..., "dark":
    ...} and read through here, so there is exactly one place that knows
    how a pair turns into a hex.
    """
    return pair[resolve_mode(mode)]

# Excel number format -> how the same figure is written on the page. The
# dashboard looks the format up instead of choosing its own precision,
# because a cost rounded to the dollar on the tab and to the cent on the
# screen is the start of two sets of books.
def money(value, decimals=0):
    """Dollars, with the SIGN IN FRONT OF THE DOLLAR (Siri, 11/25).

    Python's own formatting puts it the other way round - "$-51,569" -
    which is not how anybody writes money and reads, for a moment, as a
    dollar amount of minus something. The sign belongs to the quantity,
    so it goes first.

    The sign is taken from the ROUNDED figure, so a difference of four
    tenths of a dollar against a column that prints whole dollars reads
    "$0" rather than "-$0", which is a minus sign on a zero.
    """
    rounded = round(float(value), decimals)
    sign = "-" if rounded < 0 else ""
    return f"{sign}${abs(rounded):,.{decimals}f}"


VALUE_FORMATS = {
    export.MONEY: money,
    export.MONEY_CENTS: lambda value: money(value, 2),
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
    "All-in cost": "Labor plus the automation's own invoice, over the "
                   "whole 52-week plan year (spec 6.4). Dollars.",
    "All-in vs baseline": "This run's all-in cost minus the baseline's. "
                          "Negative is a saving.",
    "Labor cost": "Pay for the hours worked over the plan year, overtime "
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

# Green, red and blue, AS NAMES RATHER THAN HEXES (Siri, 11/20).
#
# WHY A NAME AND NOT A SHADE. These were a light hex and a dark hex each,
# and this module picked between them from the mode when the card was
# drawn. Switching Light/Dark in the menu does not rerun the script, so
# the pick was made once and then stranded: a light-mode green sat on a
# charcoal card until something else happened to rerun the page. The
# scope tints had the same bug and were fixed on 11/20 by taking the mode
# away from them. These are fixed the same way, by a different route -
# the card names its colour and Streamlit resolves the name in the
# browser, on every repaint, from whichever theme is live.
#
# WHAT WE GAVE UP TO GET IT. The shade is no longer ours. Streamlit
# derives its own from the active theme, and it derives for contrast:
# measured in a real browser, its green is 7.7:1 against the page where
# the hand-picked one was 5.8:1, and every one of the six is further
# above AA than the hex it replaces. The cost is that a shade can change
# under us when Streamlit changes how it derives, which is why
# RENDERED_COLOURS below is measured rather than assumed and a browser
# test re-measures it.
BETTER_COLOUR = "green"   # moved the way a planner wants
WORSE_COLOUR = "red"      # moved the other way
SAME_COLOUR = "blue"      # did not move at all

# The three together, for the tests that hold every card colour to AA and
# check that no two of them could be mistaken for each other.
CARD_COLOURS = (BETTER_COLOUR, WORSE_COLOUR, SAME_COLOUR)

# WHAT STREAMLIT ACTUALLY PAINTS THESE NAMES AS, in this app's theme.
#
# MEASURED, NOT CHOSEN. Every hex here was read off a real browser with
# getComputedStyle, in both palettes, after driving the Light/Dark menu
# with no refresh. Nothing in the app reads this table to draw with - the
# page only ever sends the NAME - so it cannot go stale on the page. It
# is here so the tests can hold Streamlit's choices to the same WCAG AA
# bar the hand-picked hexes were held to, and so the next person can see
# what the names come out as without opening a browser.
#
# Worth knowing: for violet, orange and gray Streamlit takes the hue from
# .streamlit/config.toml and then DERIVES a readable text shade from it -
# it does not use our hex. Our #5b3a8c violet renders as #382456. The
# configured hex is used as-is only behind ":violet-background[]".
RENDERED_COLOURS = {
    "green":  {LIGHT_MODE: "#246b27", DARK_MODE: "#7ed5b0"},  # 6.1 / 9.9
    "red":    {LIGHT_MODE: "#9e303b", DARK_MODE: "#f2919a"},  # 6.7 / 7.6
    "blue":   {LIGHT_MODE: "#244fb4", DARK_MODE: "#85b8f8"},  # 6.9 / 8.4
    "violet": {LIGHT_MODE: "#382456", DARK_MODE: "#e4d0f5"},  # 12.6 / 12.0
    "orange": {LIGHT_MODE: "#422405", DARK_MODE: "#f2cda6"},  # 13.2 / 11.6
    "gray":   {LIGHT_MODE: "#343432", DARK_MODE: "#bad8d9"},  # 11.7 / 11.4
}

# The light green moved on 11/23 and the dark one did not, so they are
# not the same hue any more: #246b27 is hue 140 and #7ed5b0 is hue 164.
# That is deliberate (Siri's call). On cream, Streamlit's inherited green
# derived to #0c5a44 - a teal at hue 167 that read as dark gray against
# the page, and sat only 29.8 from the teal accent. On charcoal the same
# family reads as a clear mint, so the dark one was left alone rather
# than changed for symmetry nobody would see: the two palettes are never
# on screen together.


def rendered_colour(name, mode=None):
    """The hex Streamlit paints a colour NAME as, as measured.

    For the tests, and for nothing else. Painting from this would hand
    the page a per-mode hex again, which is the whole bug this change
    removed - so no caller outside the tests may use it, and a test
    checks that none does.
    """
    pair = RENDERED_COLOURS.get(name)
    return shade(pair, mode) if pair else None

DOWN_ARROW = "▼"
UP_ARROW = "▲"

# The card's layout, unchanged from the version it replaces - but it now
# has to style a Streamlit element it does not own. The delta line is
# st.markdown, so that the colour directive in it is resolved live; the
# container around each card carries a class of the form "st-key-cmcard-"
# (Streamlit builds that from the container's key), which is the only
# handle this module gets on it.
CARD_KEY = "cmcard"

CARD_STYLE = """<style>
.cm-card { padding: 0.25rem 0 0.25rem 0; }
.cm-card-label { font-size: 0.875rem; line-height: 1.4; opacity: 0.65; }
/* The number itself, and the room underneath it. THE MARGIN IS LOAD
   BEARING (11/23): with gap:0 below and Streamlit's own negative margin
   on a markdown block, the value's box and the change line's box
   measured 12px INTO each other, so a wide figure with commas and a
   dollar sign sat on top of the line below it. Measured in a browser at
   three window widths, sidebar open and collapsed, and held there by a
   test - a number that covers its own change line is a card that lies
   about two things at once. */
.cm-card-value {
    font-size: 2.25rem; font-weight: 600; line-height: 1.25;
    margin-bottom: 1.25rem;
}
/* Close the gap Streamlit puts between two elements in a container, so
   the delta sits under the value the way it did when one block of HTML
   drew all three lines. The space between the two is the value's own
   margin above, not this - gap would reopen on any card that happened
   to draw a third element. */
[class*="st-key-CARD_KEY"] { gap: 0 !important; }
/* The delta line: the card's own size, BOLD (11/23) so the change reads
   as the card's point rather than as a footnote to the number, and no
   colour set here - the colour arrives in the markdown as a name and
   Streamlit resolves it against the live theme. A colour in this
   stylesheet would be the stale-shade bug again, one layer further out. */
[class*="st-key-CARD_KEY"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.875rem; line-height: 1.4; font-weight: 600;
    /* ROOM UNDERNEATH (Siri, 11/25). The six cards are two rows of
       three, and with no margin here the first row's change line sat
       straight on top of the second row's label - two different cards'
       text reading as one block. The margin is below only: the space
       above is still the value's own, which is what keeps a wide figure
       off this line. */
    margin: 0 0 1.1rem 0;
}
</style>""".replace("CARD_KEY", CARD_KEY)


def card_colour(card):
    """Green, red or blue - the one rule every card follows.

    Blue first, because "did not move" is a different statement from
    "moved a little the wrong way" and the rounding must not turn one
    into the other. Then the direction: all six measures are better
    lower (11/16), so down is green and up is red.

    Returns the NAME of a colour, not a hex, and takes no mode (11/20).
    What the name looks like is Streamlit's business and it answers in
    the browser, every repaint, from the theme that is actually showing.
    This function decides only what the card MEANS.
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


def card_key(card):
    """The key for the container one card is drawn in.

    Streamlit turns a container's key into a CSS class ("st-key-" and
    then the key), which is the only handle the stylesheet gets on an
    element Streamlit owns. A class cannot contain a space, and every
    measure label does, so the label is slugged rather than used raw -
    "Network required FTE" would otherwise become four class names and
    the card's two lines would drift apart.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", card["label"].lower()).strip("-")
    return f"{CARD_KEY}-{slug}"


def card_text(card):
    """What the delta line says, before anything colours it.

    The arrow or the words, which is the half of the card that does NOT
    depend on colour. A reader who cannot separate green from red still
    gets the direction here, and a no-change card says "no change" in
    words rather than wearing an arrow it has not earned.
    """
    delta = card["difference_text"]
    if not delta:
        return ""
    arrow = card_arrow(card)
    return f"{arrow} {delta}" if arrow else delta


# A non-breaking space: what the delta line says when there is nothing to
# say. The BASELINE needs this - it has no run to compare itself against.
#
# WHY A BLANK LINE RATHER THAN NO LINE (11/21, Siri's on-screen bug).
# The page used to skip st.markdown entirely for a card with no delta,
# and draw an invisible spacer div inside the card's HTML instead to keep
# the row one height. Two mechanisms for one blank line, and the cost was
# not cosmetic: it meant a card held ONE element on the baseline and TWO
# on every other run. Streamlit addresses the children of a container by
# position, so a rerun that changes that count makes it rebuild the row
# rather than update it in place - and while it rebuilds, the PREVIOUS
# run's six delta lines are still on screen. Switching from a run where
# everything got worse to one where things improved therefore showed, for
# a moment, six red cards under the new run's numbers.
#
# One element always, carrying a space when it has nothing else, means
# the structure never changes and Streamlit updates the text where it
# stands. Measured: the mixed row - some cards repainted, the rest still
# showing the last run's colour - is gone.
#
# THE ENTITY, NOT THE CHARACTER (11/23). This held a bare non-breaking
# space from 11/21 until a browser was asked what it actually drew.
# Streamlit renders a markdown block whose whole content is whitespace as
# an EMPTY container: no paragraph at all, nothing on the page, no line.
# So the baseline's cards quietly lost the blank line that keeps a row
# one height, and the row jumped when the run changed - the very thing
# the old spacer div was there to stop. Written as the HTML entity,
# markdown passes it through and Streamlit draws "<p>&nbsp;</p>", a real
# line of the card's own height. Measured: 26px, the same as a card that
# has something to say.
CARD_BLANK = "&nbsp;"


def card_markdown(card):
    """The delta line as Streamlit markdown: ":green[▼ 1.20]".

    MARKDOWN AND NOT HTML, and that is the whole point of 11/20. A hex
    written into HTML here is painted once and then stranded when the
    reader switches Light/Dark, because the menu does not rerun this
    script. A colour NAME is resolved in the browser against the live
    theme, so the same string is right in both palettes and right the
    instant the reader switches.

    Returns "" for the baseline, which has nothing to compare against.
    """
    shown = card_text(card)
    if not shown:
        return ""
    name = card_colour(card)
    # The brackets a directive uses are the one thing that can break it,
    # and a measure label never contains them - but the figure is built
    # here, so this stays honest about what it is interpolating.
    return f":{name}[{shown}]" if name else shown


def card_html(card):
    """The card's label and value - NO COLOUR ANYWHERE IN IT.

    These two lines are drawn in the page's own text colour, which
    Streamlit keeps live, so nothing here can go stale. The third line,
    the coloured one, is card_markdown and is sent separately: a colour
    directive does not survive inside raw HTML (checked in a browser -
    ":green[x]" inside a div renders as that literal text), so the one
    line that needs a live colour cannot be in this string.

    The hover still hangs off this block, as it did before, carrying what
    the measure is and the exact difference behind the rounding. It stays
    here rather than moving to the delta line because Streamlit's own
    help= draws a "?" button beside the text, and the card is meant to
    look exactly as it did - so the hover covers the label and the value,
    which is most of the card, and not the one strip below them.
    """
    return (
        f'<div class="cm-card" title="{html.escape(card_tooltip(card))}">'
        f'<div class="cm-card-label">{html.escape(card["label"])}</div>'
        f'<div class="cm-card-value">{html.escape(card["value_text"])}</div>'
        "</div>"
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
    # THE BASELINE IS NOT A CHOICE (Siri, 11/25). Its file declares the
    # kind "choice" because every run must declare one, but on a table
    # ranked against the baseline, calling the baseline a choice reads as
    # if somebody were proposing it - and a reader looking for the line
    # everything else is measured against wants it named, not classified.
    # Neutral in colour too: "Baseline" is not one of the three kinds, so
    # it gets no kind colour, which falls out of the lookup by itself.
    table.insert(
        1,
        KIND_HEADER,
        [
            BASELINE_KIND_LABEL if name == BASELINE_LABEL else kind_label(kind)
            for name, kind in zip(comparison["scenario"], comparison["kind"])
        ],
    )
    position = list(table.columns).index(ALL_IN_HEADER) + 1
    table.insert(
        position,
        DIFFERENCE_HEADER,
        comparison["total_cost"].astype(float) - baseline_cost,
    )
    return table.sort_values(DIFFERENCE_HEADER).reset_index(drop=True)


# SHORT HEADERS, FULL NAMES ON HOVER (Siri, 11/25). Eleven columns of
# full workbook headers did not fit the page: "Requester turnaround, mean
# days" alone is wider than the figure under it, and the last column fell
# off the right-hand edge at 1440 with the sidebar open. The detail table
# has done this since 7d; this is the same treatment, and the tooltip
# still leads with the workbook's own header so a reader can find the
# column on the tab.
COMPARISON_SHORT = {
    "All-in cost": "All-in",
    "All-in vs baseline": "vs base",
    "Labor cost": "Labor",
    "Machine cost": "Machine",
    "Network required FTE": "Req FTE",
    "Overtime hours": "OT hrs",
    "Uncovered hours": "Uncov hrs",
    "Requester turnaround, mean days": "Turnaround",
    "Weeks over the 5-day goal": "Wks over goal",
}


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
        short = COMPARISON_SHORT.get(header, header)
        specs.append(
            {
                "column": header,
                "label": short,
                # The full workbook header leads the tooltip wherever the
                # column is shortened, so a reader who wonders what "OT
                # hrs" is called on the tab can find out without leaving
                # the page.
                "help": f"{header}. {note}" if short != header else note,
                "format": SCREEN_NUMBER_FORMATS.get(number_format, TEXT_COLUMN),
                "number_format": number_format,
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
# leaves violet, orange and gray - the three colours Streamlit's own
# markdown palette offers outside red, green and blue. The hexes live in
# .streamlit/config.toml as violetColor, orangeColor and grayColor, and a
# test checks the file and KIND_COLOURS below agree.
#
# ONE HUE PER KIND, SHOWN TWO WAYS (corrected 11/20). This comment used
# to claim the sidebar caption and the comparison table were "the SAME
# colour rather than two colours that nearly match". That was never true,
# and a browser said so: Streamlit does not use the configured hex for
# ":violet[Choice]", it DERIVES a more contrasty text shade from it. Our
# #5b3a8c renders as #382456 - 21.9 apart by delta-E, which is two
# colours, not one.
#
# What is true is one hue per kind, declared once in the theme file, and
# shown two ways because the two places will not take the same thing:
#   - the sidebar caption and anywhere else markdown is allowed get the
#     NAME, and Streamlit resolves it live against whichever palette is
#     showing;
#   - the comparison table's Kind cell is drawn on a canvas by the
#     dataframe, which renders no markdown at all (checked: a cell given
#     ":violet[Choice]" shows that literal text), so it gets a
#     translucent wash mixed from the same configured hue, over which the
#     cell's own live text sits.
# Neither carries a hex this module chose per mode, which is what makes
# both of them right the instant a reader switches Light/Dark.
#
# Gray for a sensitivity is not an accident either: a sensitivity is not
# a proposal, and a muted colour says so before the sentence does. The
# price of that is measured and worth knowing: gray is the kind whose
# derived text shade sits closest to the body text - delta-E 8.2 in
# light, and 7.3 in dark until the 11/20 re-hue moved it to 14.7 - so in
# the sidebar a sensitivity reads as muted rather than as a colour. The
# wash in the table and the key is what carries it there, and the word
# itself always says which kind it is.
# ---------------------------------------------------------------------------

KIND_COLOURS = {
    scenarios.KIND_CHOICE: {
        LIGHT_MODE: "#5b3a8c",   # violet, 8.1:1 on the cream page
        DARK_MODE: "#be8fe8",    #         6.8:1 on the charcoal page
    },
    scenarios.KIND_RISK: {
        LIGHT_MODE: "#8a4b0a",   # orange, 6.4:1
        DARK_MODE: "#e8a862",    #         8.4:1
    },
    scenarios.KIND_SENSITIVITY: {
        LIGHT_MODE: "#5c5b58",   # gray,   6.4:1 - a WARM gray on a cream
        DARK_MODE: "#88bdbe",    #         8.3:1   page, which also pulls it
    },                           #                 clear of the teal accent
    #                              The dark one is a COOL SLATE instead, and
    #                              re-hued on 11/20: see the theme file. A
    #                              gray in the violet's own hue neighbourhood
    #                              came out of Streamlit's derivation as a
    #                              second pale violet.
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


# The wash a kind wears where markdown is not allowed (Siri, 11/20).
#
# Mixed from the kind's own LIGHT hex, at the same alpha the region tints
# use, and the same string serves both palettes: the browser composites a
# translucent background against whatever page is showing, so there is no
# mode to pick and nothing to go stale.
#
# The light hex rather than the dark one, because that is the saturated
# end of each hue: at 30% the light violet still reads as violet over the
# charcoal page, where the dark pastel over cream washes out to nothing.
# Measured: worst contrast 8.5:1 against the 4.5 AA asks, worst pair 12.4
# apart by delta-E - closer than the regions manage, because these three
# hues are muted by design, but still above the 10.4 of the solid tints
# this approach replaced. The word in the cell says which kind it is
# regardless, so the wash never has to carry the meaning alone.
# Spelled out rather than mixed from KIND_COLOURS in a comprehension,
# because the alpha and the hex-splitter both live further down this file
# with the scope tints. A test mixes them the other way and fails if
# these three and the theme file ever drift apart.
KIND_TINTS = {
    scenarios.KIND_CHOICE: "rgba(91, 58, 140, 0.3)",    # #5b3a8c violet
    scenarios.KIND_RISK: "rgba(138, 75, 10, 0.3)",      # #8a4b0a orange
    scenarios.KIND_SENSITIVITY: "rgba(92, 91, 88, 0.3)",  # #5c5b58 gray
}


def declared_kind_colour(kind, mode=None):
    """The hex .streamlit/config.toml declares for a kind in a mode.

    A RECORD OF THE THEME FILE, not a colour to draw with. Streamlit
    resolves the kind's name against this hex itself, in the browser; if
    this module painted with it instead, the page would be carrying a
    shade chosen for whichever mode happened to be showing when the
    script last ran, which is exactly the bug 11/20 removed. The tests
    use it to check the file and this module agree and to hold both
    palettes to AA. Nothing on the page may, and a test checks that.
    """
    pair = KIND_COLOURS.get(kind)
    return shade(pair, mode) if pair else None


def kind_tint(kind):
    """The wash a kind wears - the same string in either palette.

    No mode argument, on purpose, for the reason region_tint has none.
    """
    return KIND_TINTS.get(kind)


def kind_markdown(kind):
    """A kind's label in Streamlit markdown, coloured: ":violet[Choice]".

    For the sidebar captions, and for anywhere else that takes markdown.
    The NAME goes to the browser and the browser picks the shade, every
    repaint, from the palette that is actually showing - so this is the
    one form of the kind that cannot be caught in the wrong mode.
    """
    label = kind_label(kind)
    name = KIND_MARKDOWN.get(kind)
    return f":{name}[{label}]" if name else label


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


def filter_lines(dimensions, choice, role_codes):
    """What the sidebar is filtering to, in plain lines for a download.

    A FILE IS OPENED COLD. The page says what it is filtered to in the
    sidebar beside the table; a file carries no sidebar, and a reader who
    sums a column of a filtered extract without knowing it was filtered
    gets a wrong number and no warning. So the filters travel with it.

    The roles are named the way the workbook names them, not by code, for
    the reason every other role on a page is.
    """
    scopes = choice["scopes"]
    covers = ", ".join(scopes) if scopes else dimensions["network_code"] + \
        " and every site"
    names = role_short_names()
    roles = [names.get(code, code) for code in role_codes]
    return [
        f"Scope: {choice['label']} - rows for {covers}.",
        "Roles: " + (", ".join(roles) if roles else "none selected") + ".",
    ]


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
# Scenarios tab calls them Labor cost and Machine cost, and so does the
# comparison table, so the legend on this chart calls them that too.
#
# The LABEL is US spelling (Siri, 11/23) and the identifier is not,
# deliberately: the measure behind it is stored as `labour_cost` in every
# saved scenario summary and in the results file on disk, so the name in
# the code still matches the name in the data. Renaming that would be a
# migration of a file, not a spelling fix.
LABOUR_HEADER = "Labor cost"
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


# LABELS READ LEFT TO RIGHT (Siri, 11/25, from the live app).
#
# Vega-Lite rotates a band axis's labels to vertical the moment it
# decides they do not fit, and it decided that here. Measured in the
# browser before this change, both run labels were drawn at
# rotate(270) AND cut short by the 180px default label limit: the
# longest read "Site shortfall N1, ..." sideways. A reader got a name
# turned on its side that was not even the whole name.
#
# Setting the angle explicitly turns the auto-rotation off for good.
# The room was always there: this chart draws only ever TWO bars - the
# baseline and the run being looked at - so one band measured 437px
# wide at 1440 and 677px at 1920, against a longest run name that
# draws 136px on the axis. Nothing has to be wrapped or shortened.
# The full name fits flat with three times the room it needs, which is
# why there is no abbreviation here to explain.
LABEL_FLAT = 0

# Wide enough that the longest run name is never given an ellipsis,
# and still a limit rather than none at all. On a window narrow enough
# to squeeze two of these, Vega-Lite shortens the name with "..."
# instead of letting two labels collide, and the bar's own tooltip
# carries the name in full either way - so the reader loses nothing a
# hover cannot give back.
RUN_LABEL_LIMIT = 200

# A label Vega-Lite has decided to hide is worse than a cramped one:
# an unlabelled bar on a two-bar chart is a bar nobody can identify.
# So overlapping labels are shortened, never dropped.
LABEL_OVERLAP = False

# The legend reads ALONG THE BOTTOM, not stacked up the right-hand
# side (Siri, 11/25). Stacked, each entry sat on its own line and the
# run names ran out to the right of the plot; along the bottom they
# read as one line of text under the picture they describe, in the
# direction everything else on the page is read.
#
# The limits are measured, not guessed: in the legend's own font the
# longest run name drew at 159px against Vega-Lite's 160px default, so
# the name that fits today was one character from an ellipsis. These
# clear it.
LEGEND_LABEL_LIMIT = 240
LEGEND_TITLE_LIMIT = 280


def _flat_legend(title):
    """A legend that reads along the bottom, left to right."""
    return alt.Legend(
        title=title,
        orient="bottom",
        direction="horizontal",
        labelLimit=LEGEND_LABEL_LIMIT,
        titleLimit=LEGEND_TITLE_LIMIT,
    )


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
            color=alt.Color(f"{RUN_HEADER}:N", legend=_flat_legend(RUN_HEADER)),
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
            # Flat, and long enough to say the whole name: this is the
            # axis Siri found rotated to vertical in the live app.
            x=alt.X(
                f"{RUN_HEADER}:N",
                title=None,
                sort=list(table.index),
                axis=alt.Axis(
                    labelAngle=LABEL_FLAT,
                    labelLimit=RUN_LABEL_LIMIT,
                    labelOverlap=LABEL_OVERLAP,
                ),
            ),
            y=_locked_y(COST_HEADER, COST_HEADER, top),
            color=alt.Color(
                f"{COST_HALF_HEADER}:N", legend=_flat_legend(COST_HALF_HEADER)
            ),
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
    ("Labor cost", "Labor", 68, "Labor cost for the week, in dollars."),
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

    Plain data rather than Streamlit objects on purpose: table_html
    writes the headers, the tooltips and the number formats out of these,
    and a test can read all three without starting a page. They fed
    st.column_config until 11/24, when the table became our own HTML so
    that a scope could be a coloured word; the widths are kept because
    they are still what the columns were sized against.
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
                # The workbook's own format for this measure, so the
                # table writer can print the figure the way the cards do.
                "number_format": number_format,
            }
        )
    return specs



# ---------------------------------------------------------------------------
# Which region a row belongs to, and its tint (7d final pass, 11/19)
#
# WHY TINT AT ALL. The detail table is 780 rows of three different things
# stacked: the North sites, the South sites, and the back office, which
# is not at a site at all. Sorted by week, those interleave, and a reader
# scanning for "what happened at the South counter" has to read the Scope
# column on every line. A tint does that reading for them.
#
# THE SCOPE CELL, NOT THE ROW (Siri, 11/19). The first build tinted the
# whole row. Seventeen columns of tinted background is a lot of colour
# for one fact, it competes with the figures, which are the thing worth
# looking at, and it reads as a judgement on the row rather than as a
# label on its scope. One tinted cell, in the column the colour is
# actually about, says the same thing and leaves the numbers alone.
#
# WHY NOT RED, GREEN OR BLUE. Those three are spoken for on the cards -
# worse, better, no change - so a tint in one of them would look like a
# verdict. These carry no meaning of their own beyond "same region", and
# a test checks none of them is a card colour or the accent.
#
# TRANSLUCENT, NOT ONE SHADE PER MODE (Siri, 11/20). The first build
# picked a solid hex per mode at render time, and that turned out to be
# broken in a way no test caught: switching Light/Dark in Streamlit's own
# menu does NOT rerun the script. The browser re-themes itself live, the
# Python that chose the hex never runs again, and the tint stays as it
# was - a pale sand sitting on a charcoal page under off-white text, at
# 1.2:1. Unreadable, in both directions, until something else happens to
# rerun the script.
#
# A WASH HAS NO MODE TO GET WRONG. Each region is one rgba colour at 30%
# over whatever the page background currently is, so the browser composites
# it afresh on every repaint and the arithmetic lands on the right side
# automatically: over cream it comes out a pale tint, over charcoal a deep
# one. Nothing here has to know which mode is on screen, so nothing here
# can be out of date. Measured, not assumed: a scratch browser test drove
# the real menu and read the composited pixels back off the canvas.
#
# AND THE TEXT IS LEFT ALONE. No colour is set on the cell, so the figures
# and the scope code keep the text colour Streamlit is painting with right
# now, which it does update live. Setting a text colour here would put the
# stale-colour bug straight back, one layer up.
#
# WHAT THE 30% BUYS. Composited, the three are further apart than the
# solid tints this replaces (worst pair 18.5 against 10.4 by CIE delta-E,
# in the worse of the two modes), and the body text clears AA on all six
# backgrounds by at least 8.3:1 against the 4.5 it needs. A test
# recomputes every one of those rather than trusting this paragraph.
#
# THE DOWNLOAD IS NOT TOUCHED. A tint is a thing done to a screen; the
# CSV is the same bytes it was, which a test checks.
# ---------------------------------------------------------------------------

NETWORK_REGION = "network"

# How much of the wash sits over the page. Low enough that the page still
# shows through and the figures stay the loudest thing in the row; high
# enough that three regions are told apart at a glance in both modes.
TINT_ALPHA = 0.30

REGION_TINTS = {
    # Warm amber, for the North cluster.
    "north": f"rgba(214, 150, 30, {TINT_ALPHA})",
    # Violet, for the South cluster.
    "south": f"rgba(142, 92, 214, {TINT_ALPHA})",
    # Slate gray, for the back office - which is not a region at all, and
    # a colour with no hue to speak of says that before the label does.
    NETWORK_REGION: f"rgba(112, 128, 144, {TINT_ALPHA})",
}

# No tint at all for a scope no cluster claims, rather than a white box
# on a cream page or a pale box on a charcoal one. A test walks every
# saved run, so in practice this is never reached - it is here so that a
# site added to sites.yaml without a cluster degrades to "plain" instead
# of to "wrong colour".
REGION_FALLBACK_TINT = None


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


def region_tint(region):
    """The wash one region wears - the same string in either palette.

    No mode argument on purpose. This is the one colour on the page that
    cannot go stale when a reader switches Light/Dark, and taking a mode
    it would then ignore is how it would quietly grow one again.
    """
    return REGION_TINTS.get(region, REGION_FALLBACK_TINT)


# ---------------------------------------------------------------------------
# ONE COLOUR MECHANISM FOR EVERYTHING THIS PAGE DRAWS ITSELF (11/24)
#
# The outside reviewers asked for the scope and the kind to be COLOURED
# BOLD WORDS rather than tinted cells, in both palettes. That is a harder
# request than it sounds, and the reason is worth writing down.
#
# A word needs a TEXT colour, and one text colour cannot serve both
# pages. Readable text needs 4.5:1. On the cream page (#faf7f2) that caps
# a colour at about 0.17 relative luminance; on the charcoal page
# (#191b1e) it needs at least 0.20. There is no colour in between, so any
# single hex is unreadable in one mode - which is why the cells were
# washed with a translucent background in the first place: a wash has no
# mode to get wrong, because the browser re-mixes it over whatever page
# is showing.
#
# THE MECHANISM: CSS light-dark(). Modern CSS resolves light-dark(a, b)
# against the color-scheme in force, and Streamlit sets color-scheme on
# .stApp - measured in a real browser on 11/24: "light" in light mode,
# "dark" in dark mode, flipping on a menu switch with NO rerun. So one
# string carries both shades and the browser picks, every repaint.
#
# This keeps the 11/20 rule exactly as it was, and a test still enforces
# it: nothing here chooses a shade from the mode. What goes to the
# browser is the SAME markup for both readers - a pair, not a pick.
#
# WHY THE TABLES ARE OUR OWN HTML NOW. st.dataframe paints on a canvas
# and renders no markdown and no CSS in its cells, so a coloured word
# cannot reach one. Siri chose the HTML table on 11/24, with a Sort by
# control to replace the grid's click-to-sort. What the grid gave and CSS
# gives back: a sticky header, a scrolling body, and a tooltip on every
# column header. What it does not give back is sorting by a click on the
# header, which is why the control exists.
# ---------------------------------------------------------------------------


def live_colour(pair):
    """Both shades in one string, for the browser to choose between.

    NEVER one shade chosen here. This function exists so that every
    colour this page draws goes out as a pair, and so that a test can
    find anything that does not.
    """
    light, dark = pair
    return f"light-dark({light}, {dark})"


def _rgba(colour, alpha):
    red, green, blue = _parts(colour)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def live_wash(pair, alpha, dark_alpha=None):
    """The same pair as a translucent fill, for a panel or a button.

    TWO ALPHAS, because the same fraction does not read the same over
    both pages (Siri, 11/25). A 10% violet over cream is a visible
    tint; the same 10% of a pale violet over charcoal is almost the
    charcoal, which is why the buttons barely showed a colour in dark
    mode. Where one number does for both, dark_alpha is left out.
    """
    light, dark = pair
    return (
        f"light-dark({_rgba(light, alpha)}, "
        f"{_rgba(dark, dark_alpha if dark_alpha is not None else alpha)})"
    )


# THE THREE SCOPE HUES, measured on 11/24 against both pages. They are
# the ones the workbook already tints a scope cell with - amber for the
# North cluster, violet for the South, slate for the back office - so a
# reader with the Excel file open beside the page sees the same three
# families.
#
# THIS RECORD IS THE SATURATED END of each hue, and it is what a PANEL is
# washed and edged with: the synthetic banner and the two download
# buttons. A panel asks more of a hue than a word does. It carries the
# colour at twelve percent and then writes IN that colour on top of it,
# so the hue has to be dark enough to read on its own wash - which the
# softer shades below are not: measured, the softened amber lead word
# came out at 4.37:1 on its own 12% amber, under AA, at every alpha
# strong enough for the wash to be visible at all.
#
#                          light          dark
#   north   amber/bronze   6.24:1         9.50:1
#   south   violet         8.09:1         6.81:1
#   network slate          6.89:1         8.20:1
SCOPE_HUES = {
    "north": ("#7d5410", "#e8b964"),
    "south": ("#5b3a8c", "#be8fe8"),
    NETWORK_REGION: ("#44586b", "#9fb6c6"),
}

# THE SAME THREE HUES AT THEIR SOFT END, which is what a scope WORD on
# the page is written in (Siri, 11/25, round 3: "keep bold in both modes,
# but use softer, less saturated light-mode shades").
#
# ONLY THE LIGHT END MOVES. Dark was not asked about and is not touched,
# so in dark mode this table and SCOPE_HUES above are the same three
# colours; in light mode these are the softer ones.
#
# SEARCHED, NOT PICKED. Each hue ANGLE is held exactly where it was - so
# amber is still amber - and the search walked saturation down and
# lightness up, which are the two directions "softer" means, keeping
# every bar this project already holds these words to:
#
#                       saturation      on the cream page
#   north   amber       0.77 -> 0.59    6.24:1 -> 5.12:1
#   south   violet      0.41 -> 0.32    8.09:1 -> 5.29:1
#   network slate       0.22 -> 0.17    6.89:1 -> 5.08:1
#
# A 23% cut in saturation is as far as the three can soften together:
# desaturating pulls every hue toward the same neutral, and the closest
# pair - violet against slate - was only 44.8 apart to begin with, so the
# bar that stops the cut is DISTINCT rather than AA. At these three the
# worst pair is 42.6 in light and 48.0 in dark, and the worst word reads
# at 5.08:1 against the 4.5 AA asks. Both have margin on purpose: this
# project has refused a shade at 4.53:1 before for having none.
#
# WORTH KNOWING, and not a regression this round introduced: the slate is
# a NEUTRAL, so it sits nearer the teal accent than a hue would - 17.1
# apart before this change and 13.3 after. Nothing on this page reads a
# scope by colour alone (the cell says N1, or S2, or NETWORK), and the
# same arithmetic is why the sensitivity gray is held to DIFFERENT rather
# than DISTINCT further up this file.
#
# A test recomputes all of it rather than trusting this table.
SCOPE_WORD_COLOURS = {
    "north": ("#896223", "#e8b964"),
    "south": ("#7757a8", "#be8fe8"),
    NETWORK_REGION: ("#596c7e", "#9fb6c6"),
}

# The kind words use the SAME hexes Streamlit paints a kind with
# everywhere else - the sidebar caption above all - so the word in the
# table and the word in the sidebar are one colour rather than two that
# nearly match. Read out of KIND_COLOURS rather than copied, so the theme
# file stays the single place a kind's shade is decided.
KIND_WORD_COLOURS = {
    kind: (pair[LIGHT_MODE], pair[DARK_MODE])
    for kind, pair in KIND_COLOURS.items()
}

# The two downloads, in two of the scope hues, because the reviewers
# asked for buttons that harmonise with the scope words rather than
# Streamlit's blue and gray. Violet leads and slate follows, which is the
# order the buttons are in: Excel is the one almost everybody wants, and
# the CSV is for whoever is doing arithmetic on it.
#
# THE HUE, not the softened word shade: these are panels, and the reason
# a panel takes the saturated end is written above SCOPE_HUES.
EXCEL_COLOUR = SCOPE_HUES["south"]
CSV_COLOUR = SCOPE_HUES[NETWORK_REGION]

# The synthetic banner takes the third scope hue. Softer than the warning
# yellow it replaces - a 12% wash and an edge rather than a filled block
# - and still the first coloured thing on the page.
SYNTHETIC_COLOUR = SCOPE_HUES["north"]

# The run-summary and instruction banners keep the blue they have always
# had, which is the one the reviewers liked in dark mode. Read out of
# RENDERED_COLOURS so there is one record of what blue is.
#
# NOT THE TEAL ACCENT: slate and teal are 10.3 apart by delta-E, and two
# quiet blue-grays on one page read as one colour used inconsistently.
NOTE_COLOUR = (
    RENDERED_COLOURS["blue"][LIGHT_MODE],
    RENDERED_COLOURS["blue"][DARK_MODE],
)

# EVERY NOTICE ON THE PAGE IS ONE OF THESE (Siri, 11/25, round 3: the
# "Set the controls" message "still looks out of place" in light mode).
#
# It was an st.info, and so were eleven other notices on this page:
# Streamlit's own alert blocks, which are filled panels in a palette of
# their own that was never this page's. The design they move to is the
# Finding banner's, unchanged - a wash of the hue, the sentence in that
# hue, a thin left edge - because that is the one a reviewer has already
# approved on this page.
#
# THREE HUES, NOT ONE, and the hue is the only thing that differs. A
# notice on this page is in one of three registers, and they are not
# interchangeable: "here is something to know" is not "this value was
# refused", and a refusal drawn in the Finding's calm blue would be a
# worse light mode than the filled block it replaces. So:
#
#   note      blue    something to know, or to do next
#   caution   amber   true, allowed, and worth reading twice
#   refused   red     the model or the validator said no
#
# All three hues are already on this page - the note blue is the
# Finding's, the amber is the synthetic banner's, and the red is the one
# Streamlit paints a card that moved the wrong way - so this adds a
# design, not a palette. Measured on every surface a notice is drawn on,
# which includes the SIDEBAR (the back-office note is written there, and
# the sidebar is a step off the page colour): the worst sentence reads at
# 4.77:1 against the 4.5 AA asks, and the three are 41.9 apart at the
# closest by delta-E. A test recomputes all of it.
CAUTION_COLOUR = SCOPE_HUES["north"]
REFUSED_COLOUR = (
    RENDERED_COLOURS["red"][LIGHT_MODE],
    RENDERED_COLOURS["red"][DARK_MODE],
)

# How much of a hue sits behind a panel or a button. Low, because the
# word or the label on top of it has to stay the loudest thing there.
PANEL_ALPHA = 0.12
PANEL_DARK_ALPHA = 0.18

# ONE ALPHA PER BUTTON, not one shared by both (Siri, 11/25). The two
# have to stand out from the page AND from each other, and they are not
# equally loud hues: at any shared strength that makes slate visible,
# violet is already far past it, and the two composited fills sat 14.6
# apart by delta-E - two muted boxes a reader tells apart by reading the
# labels.
#
# Searched rather than guessed: the lightest pair of alphas where, in
# BOTH palettes, each fill is at least 22 from its page, the two fills
# are at least 22 from each other, and the label still clears AA on
# either. That is violet at 48% and slate at 38% - worst page distance
# 22.5, worst pair 22.2, worst label 5.6:1 against the 4.5 AA asks.
#
# It leaves the main button the louder of the two, which is the order
# they are in: Excel is the one almost everybody wants.
EXCEL_FILL_ALPHA = 0.48
CSV_FILL_ALPHA = 0.38

# Hover adds six points. Not more: at twelve the violet's label fell to
# 4.4:1 in dark, which is under AA for the one state a reader is looking
# straight at.
HOVER_BOOST = 0.06

# The edge is the hue at full strength, and thick enough to see without
# turning the button into a box of colour.
BUTTON_EDGE = "2px"

# The lines a table draws itself with. Neutral on purpose: the only
# colour in a table should be the word that carries a meaning.
TABLE_LINE = ("#e2dcd2", "#32353a")
TABLE_HEAD_FILL = ("#f1ece4", "#23262b")

def scope_word_class(region):
    """The CSS class a scope word wears, or none for a scope with no region."""
    return f"cm-scope-{region}" if region in SCOPE_WORD_COLOURS else None


def kind_word_class(kind):
    """The CSS class a kind word wears."""
    return f"cm-kind-{kind}" if kind in KIND_WORD_COLOURS else None


def word_rules():
    """One CSS rule per coloured word, built from the recorded pairs."""
    rules = [
        f".cm-scope-{region} {{ color: {live_colour(pair)}; }}"
        for region, pair in SCOPE_WORD_COLOURS.items()
    ]
    rules += [
        f".cm-kind-{kind} {{ color: {live_colour(pair)}; }}"
        for kind, pair in KIND_WORD_COLOURS.items()
    ]
    return "\n".join(rules)


TABLE_STYLE = """<style>
.cm-table-wrap {{ max-height: {height}; overflow: auto;
                  border: 1px solid {line}; border-radius: 0.4rem; }}
.cm-table {{ border-collapse: collapse; width: 100%;
             font-size: 0.8125rem; font-variant-numeric: tabular-nums; }}
/* STICKY, so the headers stay put the way the grid's did. */
/* PADDING IS WIDTH (11/25). Seventeen columns at 0.6rem a side ran 79px
   past the page at 1440 with the sidebar open, which is the sideways
   scroll the reviewers found on the comparison table. 0.4rem a side
   gives back 108px and still leaves the figures room to breathe. */
.cm-table thead th {{ position: sticky; top: 0; z-index: 2;
                      background: {head}; color: inherit;
                      font-weight: 600; text-align: right;
                      padding: 0.35rem 0.4rem; white-space: nowrap;
                      border-bottom: 1px solid {line}; }}
.cm-table th.cm-left, .cm-table td.cm-left {{ text-align: left; }}
/* COLOUR: INHERIT on every figure, which is the 11/20 rule written in
   CSS. The page's own text colour is the one thing Streamlit keeps live,
   so a cell that sets no colour of its own is right in both palettes for
   nothing. */
.cm-table td {{ color: inherit; text-align: right;
                padding: 0.25rem 0.4rem; white-space: nowrap;
                border-bottom: 1px solid {line}; }}
.cm-table tbody tr:last-child td {{ border-bottom: none; }}
.cm-word {{ font-weight: 700; }}
.cm-scope-line {{ font-size: 0.8125rem; line-height: 1.6;
                  padding: 0.1rem 0 0.35rem 0; }}
{words}
</style>"""

TABLE_HEIGHT = "26rem"


def table_style(height=TABLE_HEIGHT):
    """The table stylesheet, with the height of the scrolling box."""
    return TABLE_STYLE.format(
        height=height,
        line=live_colour(TABLE_LINE),
        head=live_colour(TABLE_HEAD_FILL),
        words=word_rules(),
    )


def table_html(frame, specs, word_column=None, class_of=None):
    """The rows on screen as a table the page draws itself.

    `specs` are the same plain dicts the column settings were built from,
    so the headers, the tooltips and the number formats are decided in
    one place and merely written here. `word_column` is the one column
    whose value is a coloured bold word - Scope on the detail table, Kind
    on the comparison - and `class_of` turns that value into its class.

    EVERY FIGURE IS FORMATTED HERE, because an HTML cell is a string and
    there is no column setting left to do it. Through format_value, which
    is the function the cards use, so a figure reads the same wherever it
    appears on the page.
    """
    heads = []
    for spec in specs:
        align = ' class="cm-left"' if spec["format"] == TEXT_COLUMN else ""
        heads.append(
            f'<th{align} title="{html.escape(spec["help"])}">'
            f'{html.escape(spec["label"])}</th>'
        )
    body = []
    for record in frame.to_dict("records"):
        cells = []
        for spec in specs:
            column = spec["column"]
            value = record.get(column)
            text = html.escape(format_value(value, spec.get("number_format")))
            if spec["format"] == TEXT_COLUMN:
                if column == word_column and class_of is not None:
                    name = class_of(value)
                    if name:
                        text = f'<span class="cm-word {name}">{text}</span>'
                cells.append(f'<td class="cm-left">{text}</td>')
            else:
                cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="cm-table-wrap"><table class="cm-table"><thead><tr>'
        + "".join(heads)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def scope_line_html(dimensions=None):
    """One line of coloured bold words: what N, S and NETWORK mean.

    This replaces the region key and the kind key (11/24). A key with
    swatches explained a colour that no longer exists - the scope is the
    word now, so the line names the word and colours it the same way the
    column does, from the same CSS rule.

    Built from the loaded dimensions, so a cluster renamed in sites.yaml
    renames itself here.
    """
    dimensions = dimensions if dimensions is not None else api.dimensions()
    parts = [
        (scope_word_class(cluster["code"]), cluster["code"][0].upper(),
         cluster["name"])
        for cluster in dimensions["clusters"]
    ]
    parts.append(
        (scope_word_class(NETWORK_REGION), dimensions["network_code"],
         "back office")
    )
    shown = " &middot; ".join(
        f'<span class="cm-word {name}">{html.escape(word)}</span>'
        f" = {html.escape(meaning)}"
        for name, word, meaning in parts
    )
    return f'<div class="cm-scope-line">{shown}</div>'


def scope_class_of(dimensions=None):
    """How to colour a scope value: by the region its site sits in.

    Returned as a function rather than a dict, because the table writer
    should not have to know that a scope answers to a cluster.
    """
    dimensions = dimensions if dimensions is not None else api.dimensions()

    def class_of(scope):
        return scope_word_class(region_of(scope, dimensions))

    return class_of


def kind_class_of():
    """How to colour a kind value, which arrives as its own label."""
    by_label = {kind_label(kind): kind for kind in KIND_WORD_COLOURS}

    def class_of(label):
        return kind_word_class(by_label.get(label))

    return class_of


SORT_AS_SHOWN = "Default order"

SORT_HELP = (
    "Which column to order the rows by. Default order is the view's own: "
    "the detail table arrives by plan week, and the comparison by what "
    "each run does to the all-in cost."
)

ORDER_HELP = (
    "Which end first. Off is low to high, on is high to low - so the "
    "worst week, or the biggest saving, comes to the top."
)


def sort_options(specs):
    """What a reader may sort by: the view's own order, then its columns."""
    return [SORT_AS_SHOWN] + [spec["column"] for spec in specs]


def sorted_view(frame, column=None, descending=False):
    """The rows in the order the reader asked for.

    Stable, so the view's own order survives inside equal values: the
    detail table arrives sorted by week and the comparison by what each
    run does to the all-in cost, and sorting by a column full of ties
    should not shuffle what is underneath them.
    """
    if not column or column == SORT_AS_SHOWN or column not in frame.columns:
        return frame
    return frame.sort_values(
        column, ascending=not descending, kind="mergesort"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The banners and the two download buttons (11/24)
#
# All of it in light-dark() pairs, for the reason at the top of this
# section: these are panels a reader looks at in both palettes, and a
# panel that picked its shade when the script last ran would be wrong the
# moment they switched.
#
# THE NOTE BANNERS ARE THE DARK DESIGN MIRRORED, which is not the same as
# inverted (Siri, 11/25). The first attempt read "reversed" as swapping
# the two ends - the hue filling the panel with pale text on it - and on
# a cream page that is a solid blue slab that shouts louder than the
# finding inside it. The mirror of a wash is a wash: the same translucent
# blue over whichever page is showing, with the text in the blue that
# reads on that page, and a thin edge to carry the hue. One design, two
# palettes, nothing shouting.
# ---------------------------------------------------------------------------

# The three registers a notice can be in, by the class each one wears.
# ONE TABLE, so a tone cannot exist in the stylesheet without a hue or in
# the markup without a rule - the rules below are generated from this.
NOTICE_TONES = {
    "note": NOTE_COLOUR,
    "caution": CAUTION_COLOUR,
    "refused": REFUSED_COLOUR,
}

# Thin, which is the word Siri used for it: the edge carries the hue, it
# does not box the sentence in. The synthetic banner's own edge is a step
# thicker because it is the one statement that has to be seen before
# anything else on the page.
NOTICE_EDGE = "3px"
SYNTHETIC_EDGE = "4px"

BANNER_STYLE = """<style>
.cm-banner {{ border-radius: 0.5rem; padding: 0.7rem 0.9rem;
              margin: 0.25rem 0 0.75rem 0; font-size: 0.9rem;
              line-height: 1.55; }}
.cm-banner-lead {{ font-weight: 700; letter-spacing: 0.02em; }}
/* A link or a command inside a banner keeps the banner's own colour
   rather than bringing its own: the sentence is one colour, including
   the `python -m ...` in the one notice that names a command. */
.cm-banner a, .cm-banner code {{ color: inherit; }}
.cm-banner code {{ font-size: 0.95em; }}
/* SYNTHETIC: a wash and an edge, not a filled warning block. Prominent,
   because it leads with its own word in the hue and carries a 4px edge;
   soft, because nothing on this page is actually wrong. */
.cm-synthetic {{ background: {synth_fill}; color: inherit;
                 border: 1px solid {synth_edge};
                 border-left: {synth_width} solid {synth_edge}; }}
.cm-synthetic .cm-banner-lead {{ color: {synth_word}; }}
/* THE NOTICES: one design, three hues. A wash of the hue over whatever
   page is showing, the sentence written in the hue that reads on that
   page, and a thin left edge - the Finding banner, which is the note
   tone, and now every other notice this page writes as well. */
{notices}
</style>"""


def notice_rules():
    """One CSS rule per notice tone, built from the recorded pairs.

    A pair everywhere, never a pick: the wash and the text both go out as
    light-dark(), so a reader who switches palette - which does not rerun
    this script - gets the other end without anybody choosing it.
    """
    return "\n".join(
        f".cm-{tone} {{ background: "
        f"{live_wash(pair, PANEL_ALPHA, PANEL_DARK_ALPHA)};"
        f" color: {live_colour(pair)};"
        f" border-left: {NOTICE_EDGE} solid {live_colour(pair)}; }}"
        for tone, pair in NOTICE_TONES.items()
    )


def banner_style():
    """The banner stylesheet, built from the recorded pairs."""
    return BANNER_STYLE.format(
        synth_fill=live_wash(SYNTHETIC_COLOUR, PANEL_ALPHA),
        synth_edge=live_colour(SYNTHETIC_COLOUR),
        synth_word=live_colour(SYNTHETIC_COLOUR),
        synth_width=SYNTHETIC_EDGE,
        notices=notice_rules(),
    )


BANNER_TONES = {
    "synthetic": "cm-synthetic",
    **{tone: f"cm-{tone}" for tone in NOTICE_TONES},
}

SYNTHETIC_LEAD = "SYNTHETIC"

# `like this` -> <code>like this</code>, and nothing else. Streamlit's
# alert blocks rendered markdown, so the one notice that names a command
# showed it as code; a banner is our own HTML, and without this the
# backticks would arrive on the page as two literal characters. Applied
# AFTER escaping, so what goes inside the tag is already inert - this
# turns on two backticks, and cannot turn on anything a notice says.
CODE_SPAN = re.compile(r"`([^`]+)`")


def banner_html(text, tone, lead=None):
    """One banner: a lead word in the hue, then the sentence."""
    shown = CODE_SPAN.sub(r"<code>\1</code>", html.escape(text))
    if lead:
        shown = (
            f'<span class="cm-banner-lead">{html.escape(lead)}</span> {shown}'
        )
    return f'<div class="cm-banner {BANNER_TONES[tone]}">{shown}</div>'


def synthetic_banner_html(label):
    """The synthetic statement, with its own first word as the lead.

    The label already begins with the word, because it is the same
    sentence the workbook and every file name carry; taking it off the
    front and giving it back as the coloured lead keeps one statement
    rather than making a second one up for the page.
    """
    text = label
    if text.startswith(SYNTHETIC_LEAD):
        # The word only. Everything after it, separator included, is the
        # label as it was written, so the banner and the file names and
        # the workbook all still say one sentence rather than three
        # versions of one.
        text = text[len(SYNTHETIC_LEAD):].lstrip()
    return banner_html(text, "synthetic", lead=SYNTHETIC_LEAD)


# The scroll-to-top iframe has to be at least a pixel tall - Streamlit
# refuses a height of zero - so its container is collapsed here instead.
# Nothing is drawn in it: it exists to run one line of script.
SCROLL_KEY = "cm-scroll-top"

# Where every stylesheet on this page is written. A <style> block still
# arrives inside an element container, and four of them left four
# element-sized gaps under the title - so they go in one container, and
# the first rule in the sheet collapses it. A collapsed container still
# applies its CSS: display is about what is drawn, not about what is
# declared.
STYLE_KEY = "cm-styles"

COLLAPSE_STYLE = """<style>
.st-key-{style_key}, .st-key-{scroll_key} {{
    height: 0; min-height: 0; overflow: hidden; margin: 0; padding: 0; }}
</style>"""


def collapse_style():
    """The rule that hides the two containers nothing is drawn in."""
    return COLLAPSE_STYLE.format(style_key=STYLE_KEY, scroll_key=SCROLL_KEY)


# THE CHART LABELS, READABLE IN BOTH PALETTES (Siri, 11/25: "both
# modes readable").
#
# Moving the legend under the picture put it where a light-mode reader
# looks, which is what turned up the problem: measured in the browser,
# Streamlit's Vega theme paints every axis and legend label #808495,
# and on this page's cream that is 3.48:1 - under the 4.5 AA asks and
# under the bar every other word on this page is held to. Dark was
# never in doubt at 14.30:1.
#
# NOT a redesign of the charts. The light shade is Streamlit's own
# gray at its own hue and saturation, darkened the smallest number of
# steps that clears AA with a margin: 4.88:1, and dE 9.5 from the
# colour it replaces, so it still reads as the muted label gray rather
# than as body text promoted onto the axis. The dark end is left
# exactly as Streamlit paints it, because it was already fine.
#
# Vega writes fill as a presentation ATTRIBUTE, not an inline style, so
# an ordinary rule beats it and no !important is needed - checked in
# the browser rather than assumed.
CHART_LABEL_COLOUR = ("#686c7d", "#e6eaf1")   # 4.88:1 / 14.30:1

CHART_STYLE = """<style>
/* Axis and legend labels only. The marks, the gridlines and the title
   are Streamlit's, and stay Streamlit's - a pair here, like every
   other colour this page draws, so the browser picks the shade on
   every repaint and nothing depends on which palette is showing. */
.stVegaLiteChart g[class*="role-axis-label"] text,
.stVegaLiteChart g[class*="role-legend-label"] text {{ fill: {label}; }}
</style>"""


def chart_style():
    """Axis and legend labels, as a pair for the browser to choose."""
    return CHART_STYLE.format(label=live_colour(CHART_LABEL_COLOUR))


EXCEL_BUTTON_KEY = "cm-excel-button"
CSV_BUTTON_KEY = "cm-csv-button"

# CLEAR SPACE BETWEEN A TABLE AND ITS DOWNLOADS (Siri, 11/25, round 3:
# "at least ~16px gap at 1440 and 1920").
#
# The row of buttons, not the table, carries the space: Siri asked for a
# gap between a table and the buttons under it, and all three tables on
# this page - the detail rows, Compare all, and a custom run's rows - are
# followed by this one row, so one rule does all three. A margin on the
# table instead would also push down the caption that sits between the
# two in the Compare-all view, which is not what was asked for.
#
# The value is MEASURED, not guessed, and the measurement is in the
# browser test: Streamlit's own block gap left the buttons close enough
# under the detail table that the row read as part of it.
DOWNLOAD_ROW_KEY = "cm-download-row"
TABLE_GAP = "1.5rem"

DOWNLOAD_STYLE = """<style>
/* The air under a table, above its two buttons. */
.st-key-{row_key} {{ margin-top: {table_gap}; }}
/* The two downloads, told apart by hue rather than by Streamlit's
   primary-blue and secondary-gray. A wash and a 1px edge in the hue, and
   THE LABEL IS LEFT ALONE: text that follows the theme is text that is
   readable in both palettes without anybody choosing a shade for it.
   Reached through the container key Streamlit puts on the element, which
   is the only handle a stylesheet gets on a widget it does not own. */
.st-key-{excel_key} button {{ background: {excel_fill} !important;
                              border: {edge_width} solid {excel_edge} !important;
                              color: inherit !important; }}
.st-key-{csv_key} button {{ background: {csv_fill} !important;
                            border: {edge_width} solid {csv_edge} !important;
                            color: inherit !important; }}
.st-key-{excel_key} button:hover {{ background: {excel_hover} !important; }}
.st-key-{csv_key} button:hover {{ background: {csv_hover} !important; }}
.st-key-{excel_key} button p, .st-key-{csv_key} button p {{
    color: inherit !important; }}
</style>"""


def download_style():
    """The two buttons' colours, built from the recorded pairs."""
    return DOWNLOAD_STYLE.format(
        excel_key=EXCEL_BUTTON_KEY,
        csv_key=CSV_BUTTON_KEY,
        row_key=DOWNLOAD_ROW_KEY,
        table_gap=TABLE_GAP,
        edge_width=BUTTON_EDGE,
        excel_fill=live_wash(EXCEL_COLOUR, EXCEL_FILL_ALPHA),
        excel_edge=live_colour(EXCEL_COLOUR),
        excel_hover=live_wash(EXCEL_COLOUR, EXCEL_FILL_ALPHA + HOVER_BOOST),
        csv_fill=live_wash(CSV_COLOUR, CSV_FILL_ALPHA),
        csv_edge=live_colour(CSV_COLOUR),
        csv_hover=live_wash(CSV_COLOUR, CSV_FILL_ALPHA + HOVER_BOOST),
    )


def _channel(value):
    value = value / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _parts(colour):
    """The three 0-255 parts of a #rrggbb colour."""
    digits = colour.lstrip("#")
    return tuple(int(digits[at:at + 2], 16) for at in (0, 2, 4))


def relative_luminance(colour):
    """WCAG relative luminance of a #rrggbb colour."""
    red, green, blue = _parts(colour)
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


# A translucent wash has no contrast ratio of its own - it only becomes a
# colour once the browser has put it over something. These two do that
# arithmetic in Python so the tests can hold a wash to AA the same way
# they hold a solid colour to it, against each page background in turn.
WASH = re.compile(
    r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)"
)


def is_wash(colour):
    """True for an "rgba(r, g, b, a)" string, False for anything else.

    Anything else really does mean anything: a hex, None, or the per-mode
    dict a tint used to be. A test asks this about every scope tint, and
    it should be able to report "that is not translucent" rather than
    raise on the shape of what it was handed.
    """
    return (
        isinstance(colour, str)
        and WASH.fullmatch(colour.strip()) is not None
    )


def composite_over(wash, background):
    """The #rrggbb a translucent wash actually comes out as on a background.

    Source-over alpha compositing, which is what the browser does when it
    paints an rgba background over the page: result = a*wash + (1-a)*page.
    A plain hex is returned unchanged, so a caller can mix the two.

    Within one step per channel of what a browser really paints, not
    always equal to it: a browser quantises the alpha to 8 bits first, so
    its arithmetic is not quite this arithmetic. Checked rather than
    assumed - the six tints were read back off the real canvas and five
    came out identical to this, the sixth one step off in one channel.
    That is nothing against the margin being measured: the tightest of
    them clears AA at 8.3:1 where 4.5 is the bar.
    """
    found = WASH.fullmatch(wash.strip()) if wash else None
    if found is None:
        return wash
    red, green, blue = (int(found.group(at)) for at in (1, 2, 3))
    alpha = float(found.group(4))
    base = _parts(background)
    mixed = (
        round(alpha * channel + (1 - alpha) * under)
        for channel, under in zip((red, green, blue), base)
    )
    return "#%02x%02x%02x" % tuple(mixed)


def _lab(colour):
    """A colour in CIE L*a*b*, which is laid out the way eyes see.

    sRGB distances lie: #000080 and #0000a0 are 32 apart in RGB and all
    but identical to look at, while two pale tints 32 apart can be
    obviously different colours. Lab is built so that a step of the same
    size looks like a step of the same size, which is what lets the
    question "could a reader mistake these two for each other?" be
    answered with a number instead of a shrug.
    """
    red, green, blue = (_channel(part) for part in _parts(colour))
    # sRGB primaries under D65, the white point every screen claims.
    x = 0.4124 * red + 0.3576 * green + 0.1805 * blue
    y = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    z = 0.0193 * red + 0.1192 * green + 0.9505 * blue

    def bend(value):
        return value ** (1 / 3) if value > 216 / 24389 else (841 / 108) * value + 4 / 29

    fx, fy, fz = bend(x / 0.95047), bend(y / 1.0), bend(z / 1.08883)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def colour_distance(one, other):
    """How far apart two colours look: CIE76 delta-E.

    Roughly: under 2 is a match nobody can see, around 10 is clearly a
    different colour, and over 40 is "nobody would ever confuse these".
    Used to hold the palette to Siri's rule that the no-change blue must
    not read as the teal accent, and that no two meanings share a look.
    """
    return math.dist(_lab(one), _lab(other))


# How far apart two colours have to be before this model will let them
# both carry meaning on the same page. There are two bars, because the
# page asks two different things of a reader.
#
# DISTINCT (40) is "could not be confused, even from memory". It applies
# where a colour is the ONLY thing carrying the difference and the things
# it separates are the same kind of object in the same place: the three
# card verdicts against each other, and every card verdict against the
# teal accent - Siri's own rule, since an accent-coloured control that
# read as "no change" would be a lie about a number. 40 is well past the
# textbook "clearly different" mark.
#
# DIFFERENT (20) is "obviously another colour". It applies where the
# colour is a second signal rather than the only one, or where the two
# things are never the same kind of object: a kind is always printed as
# its own word (Choice, Risk, Sensitivity) and a card always states its
# direction in words and an arrow, so a reader who could not separate a
# kind label from a card colour still loses nothing. Holding those pairs
# to 40 would be over-specification, and it is unreachable besides: the
# sensitivity gray is a NEUTRAL, and a neutral cannot be 40 from a
# mid-chroma teal at any lightness that keeps them both readable.
#
# THE THREE KINDS AGAINST EACH OTHER MOVED TO THIS BAR (Siri, 11/20).
# They were held to 40 while they were measured as the hexes this project
# declares, which passed. Measuring what Streamlit actually paints showed
# the real figures to be 36.0 and 25.8 in light and 19.3 and 27.9 in
# dark - so the 40 the code claimed had never been true of the page, in
# either palette. Siri's call was that 20 is the honest bar for them,
# because a kind is never colour alone: every kind label on this page is
# printed as its own word next to its own colour, in the sidebar, in the
# Kind column and in the key. The gray was then re-hued so the worst pair
# clears 20 with margin rather than scraping it (19.3 -> 26.2).
DISTINCT = 40.0
DIFFERENT = 20.0


# ---------------------------------------------------------------------------
# The theme's own four colours, in both palettes
#
# These are the same hexes .streamlit/config.toml holds, and a test fails
# if the two ever disagree. They are repeated here because every contrast
# check in the suite needs a BACKGROUND to measure against, and a test
# that parsed the TOML to get one would pass happily if somebody deleted
# the section it was reading.
#
# WHY CREAM AND NOT WHITE. Siri's call (11/19): a pure white page is the
# one background that makes a table of figures look like a spreadsheet,
# and a warm cream reads as a document. The dark palette answers it with
# a deep charcoal rather than black, for the same reason in reverse -
# black plus off-white is glare.
#
# WHY TEAL. The page already spends six colours on meaning - better,
# worse, no change, and the three scenario kinds - so the accent on every
# slider, button and focus ring has to be a colour that means nothing. A
# test holds it at arm's length (delta-E) from all six.
# ---------------------------------------------------------------------------

ACCENT_COLOURS = {
    LIGHT_MODE: "#3a7a8c",   # teal, white label at 4.84:1
    DARK_MODE: "#107e9c",    # the same teal, more vivid: label at 4.68:1
}
PAGE_COLOURS = {
    LIGHT_MODE: "#faf7f2",   # warm cream, never pure white
    DARK_MODE: "#191b1e",    # deep charcoal, never pure black
}
SIDEBAR_COLOURS = {
    LIGHT_MODE: "#f1ebe0",   # a step darker, so the panel reads as a panel
    DARK_MODE: "#232629",    # a step lighter, for the same reason
}
TEXT_COLOURS = {
    LIGHT_MODE: "#24262b",   # dark charcoal, 14.17:1 on the cream page
    DARK_MODE: "#edeae4",    # warm off-white, 14.37:1 on the charcoal page
}

# Keyed by the config.toml option each one sets, so a test can walk the
# file and this table together instead of naming four pairs by hand.
THEME_COLOURS = {
    "primaryColor": ACCENT_COLOURS,
    "backgroundColor": PAGE_COLOURS,
    "secondaryBackgroundColor": SIDEBAR_COLOURS,
    "textColor": TEXT_COLOURS,
}


def page_colour(mode=None):
    """The background a card, a chart and a key are drawn on."""
    return shade(PAGE_COLOURS, mode)


def sidebar_colour(mode=None):
    """The background the run picker and its kind captions are drawn on."""
    return shade(SIDEBAR_COLOURS, mode)


def text_colour(mode=None):
    """The body text colour, which is what sits on a scope tint."""
    return shade(TEXT_COLOURS, mode)


def accent_colour(mode=None):
    """The teal on sliders, buttons, toggles and focus outlines."""
    return shade(ACCENT_COLOURS, mode)


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
