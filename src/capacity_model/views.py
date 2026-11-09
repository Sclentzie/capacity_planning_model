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

import math

import pandas as pd

from . import api, export

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

# Which way is better, recorded once.
#
# Cost, overtime, uncovered hours, turnaround and weeks over goal are all
# better lower. Required FTE is NEUTRAL on purpose: a scenario that needs
# more people is not failing, it is being honest about the work, and
# colouring that red would be a verdict the model never made.
BETTER_LOWER = "lower"
NEUTRAL = "neutral"

CARD_MEASURES = (
    ("total_cost", export.MONEY, BETTER_LOWER),
    ("network_required_fte", export.FTE, NEUTRAL),
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


def format_difference(difference, number_format):
    """A difference, signed, or the words for no difference at all."""
    if difference is None:
        return ""
    if abs(difference) < 1e-9:
        return NO_CHANGE
    sign = "+" if difference > 0 else "-"
    return sign + format_value(abs(difference), number_format)


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


def sensitivity_base(key, keys):
    """The run this one varies, when it varies exactly one input of it.

    Derived from the keys, not declared: a run whose key is another run's
    key plus a suffix is a variant of it, and the longest such match is
    its parent. The two the spec calls sensitivities in so many words -
    the 0.5% true error rate and the 2-hour borrow minimum - come out of
    this rule, along with the 50% automation and 100% AI variants.

    NOTE for the architect: there is no sensitivity marker in the scenario
    files, so this is inferred from naming. If the grouping should be
    declared rather than inferred, it wants a field in the scenario
    identity block.
    """
    bases = [
        other for other in keys if other != key and key.startswith(other + "_")
    ]
    return max(bases, key=len) if bases else None


def scenario_menu(results):
    """Every run, grouped the way a reader would look for one.

    The baseline on its own, because it is what everything else is
    measured against. Then the eight a leader would act on, in the order
    the Summary tab puts them. Then the rest, with each variant marked
    for the run it varies - a sensitivity read as a scenario in its own
    right is how "AI pays back" gets quoted from the 50% run.
    """
    scenarios = results["scenarios"]
    finding_of = findings()
    key_of_name = {run["name"]: key for key, run in scenarios.items()}
    keys = set(scenarios)

    def item(key):
        run = scenarios[key]
        base = sensitivity_base(key, keys)
        label = run["name"]
        if base is not None:
            label = f"{run['name']} (sensitivity of {scenarios[base]['name']})"
        return {
            "key": key,
            "name": run["name"],
            "label": label,
            "sensitivity_of": base,
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
        (key for key in scenarios if key not in taken),
        key=lambda key: scenarios[key]["name"],
    )

    groups = []
    if BASELINE_KEY in scenarios:
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
                "direction": direction,
                "format": number_format,
            }
        )
    return cards


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

NETWORK_KIND = "network"
CLUSTER_KIND = "cluster"
SITE_KIND = "site"
NETWORK_LABEL = "Network (every site and the back office)"


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


COST_HALVES = (("Labour", "labour_cost"), ("Machine", "machine_cost"))


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


def plan_display(view):
    """The same table, rounded to the precision the workbook prints.

    A copy for the screen only. The download gets the exact figures,
    because a CSV is something a reader does arithmetic on.
    """
    rounded = view.copy()
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


def csv_bytes(view):
    """The filtered rows as a CSV, at full precision."""
    return view.to_csv(index=False).encode("utf-8")


def csv_name(key, scope_code):
    """SYNTHETIC in the file name, as every generated file here is.

    A downloaded file travels further than the page it came from, and the
    file name is the only label that travels with it.
    """
    return f"capacity_plan_{key}_{scope_code}_SYNTHETIC.csv"
