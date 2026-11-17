"""The build-your-own control panel: what can be changed, and how (7c).

WHAT THIS MODULE IS. A catalogue of every lever the 21 scenario files
pull, described as data: where the input lives in the YAML, what it is
called in plain words, what the baseline has it set to, and a range a
reader can move it over. The dashboard turns each record into a widget;
the tests read the same records. Nothing here runs the model and nothing
here computes a figure.

WHY A CATALOGUE AND NOT A PANEL FULL OF WIDGETS. Three reasons. A test
can check that every lever the scenario files change is reachable from
the panel, by comparing this list against the files themselves rather
than against a screenshot. The baseline value shown beside each control
is read from assumptions.yaml, so it cannot drift from the model's own
default. And a control's path is the path the doorway validates, so a
control that pointed at a key the model does not have would be refused
by the same code that refuses a typo in a scenario file.

WHAT IS NOT HERE. The ranges. A minimum and a maximum are a statement
about what is worth exploring on a screen, not an input of the model, so
they live in this catalogue rather than in assumptions.yaml. Everything
the model actually reads comes from the YAML.
"""

import functools
import json
import re

from . import api, config, scenarios

# ---------------------------------------------------------------------------
# The groups, in the order a reader meets them
# ---------------------------------------------------------------------------

DEMAND = "Demand"
SUPPLY = "Supply"
SERVICE = "Service"
STRUCTURE = "Structure"
AUTOMATION = "AI and automation"
TIMING = "Timing"

GROUP_ORDER = (DEMAND, SUPPLY, SERVICE, STRUCTURE, AUTOMATION, TIMING)

GROUP_NOTES = {
    DEMAND: "How much work arrives.",
    SUPPLY: "Who is on the roster, and how their hours are shaped.",
    SERVICE: "What the operation promises, and when people are away from "
             "the queue.",
    STRUCTURE: "Changes to how the work is organised, rather than to how "
               "much of it there is.",
    AUTOMATION: "Spec 7.1. DEFLECTION is demand-side - the contact never "
                "arrives. AUTOMATION is supply-side - the work arrives and "
                "a machine does it. The two are separate on purpose, and "
                "realization scales all four rates. The button below sets "
                "this group to 7.1's own defaults, which is the programme "
                "the saved AI and automation run reports.",
    TIMING: "When the change starts. The figures are always the whole "
            "52-week plan year, which is why a late change looks mild.",
}

# How a control is shown. The dashboard maps these to widgets; keeping
# them as names means a test can read a control without a browser.
SLIDER = "slider"
NUMBER = "number"
TOGGLE = "toggle"
SELECT = "select"
MULTISELECT = "multiselect"
TIME = "time"

# How a number is written beside its control.
PERCENT = "percent"
PLAIN = "plain"

# Where a control's value goes in the mapping the doorway takes. The
# doorway's own three blocks, named here so a path cannot be mistyped.
ASSUMPTIONS = "overrides"
SITES = "site_overrides"
HEADER = "scenario"


def _control(
    key,
    group,
    label,
    help,
    kind,
    block=ASSUMPTIONS,
    path=(),
    row=None,
    field=(),
    minimum=None,
    maximum=None,
    step=None,
    options=None,
    option_names=None,
    spec=False,
    written=PLAIN,
):
    """One control, as a plain record.

    `path` is where the input sits in its YAML file. A control that
    changes one row of a list - a service goal, a break window - carries
    `row` (the row's own code) and `field` (the path inside that row)
    instead of pointing at a leaf, because that is how the model's merge
    finds a row: by its code, never by its position in the list.

    `spec` marks a control spec 7.1's table gives a default for, so the
    panel can show that default beside the baseline and load the whole
    set at once. It is a flag rather than a number: the default itself is
    read from the YAML (see `catalogue`), never retyped here.
    """
    return {
        "key": key,
        "group": group,
        "label": label,
        "help": help,
        "kind": kind,
        "block": block,
        "path": tuple(path),
        "row": row,
        "field": tuple(field),
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "options": options,
        "option_names": option_names,
        "spec": spec,
        "written": written,
    }


# ---------------------------------------------------------------------------
# Spec 7.1's own defaults
#
# WHY A SCENARIO FILE AND NOT A TABLE OF NUMBERS HERE. The AI and
# automation group has a button that loads spec 7.1's defaults, and 7.1's
# defaults are already written down twice: once in assumptions.yaml, which
# holds the deflection shares at their 7.1 values and the four supply-side
# rates at zero, and once in scenarios/ai_and_automation.yaml, which is
# the run described as "the four 7.1 rates at their defaults".
#
# So the button reads that file and merges it, exactly as a run of it
# would. Retyping 0.40, 0.10 and 0.10 into this module would have made a
# third copy, and the first time one of the three moved, the panel would
# have been loading a default the model no longer had.
# ---------------------------------------------------------------------------

SPEC_FILE = "ai_and_automation.yaml"
SPEC_LABEL = "7.1 default"


def spec_config(assumptions=None):
    """assumptions.yaml with the 7.1 defaults run merged onto it."""
    assumptions = assumptions or config.load_assumptions()
    path = scenarios.scenario_dir(assumptions) / SPEC_FILE
    return scenarios.apply_scenario(scenarios.load_scenario(path), assumptions)


def spec_defaults(controls):
    """Control key -> spec 7.1's default, for the controls 7.1 names.

    Not every control in the AI and automation group is in 7.1's table:
    the proof-of-concept pair is a Section 7 addition from 11/02, and
    reading its value out of the 7.1 run would have reported the baseline
    as a "7.1 default" - a claim the spec does not make.
    """
    return {
        control["key"]: control["spec_default"]
        for control in controls
        if control["spec"]
    }


def work_type_names(assumptions=None):
    """Work-type code -> its plain name, from the table in the YAML.

    So the panel can offer "Digital retrieval" rather than "W8". Same
    rule as the roles (11/16): a reader should not have to learn the
    codes to use the dashboard, and the code stays available for anyone
    tracing a figure back to the spec.
    """
    assumptions = assumptions or config.load_assumptions()
    return {row["code"]: row["name"] for row in assumptions["work_types"]}


def catalogue(assumptions=None, sites=None):
    """Every control, with its baseline value read from the YAML."""
    assumptions = assumptions or config.load_assumptions()
    sites = sites if sites is not None else config.load_sites()
    names = work_type_names(assumptions)

    eligible = sorted(
        (
            code
            for code, row in assumptions["automation"]["work_types"].items()
            if row.get("eligible")
        ),
        key=lambda code: int(code[1:]),
    )

    controls = [
        # -------------------------------------------------------------- demand
        _control(
            "demand_multiplier", DEMAND, "Volume multiplier",
            "Every request type scaled by this at once. The +15% risk run "
            "is 1.15.",
            SLIDER, path=("scenario", "demand_multiplier"),
            minimum=0.5, maximum=2.0, step=0.05,
        ),
        # -------------------------------------------------------------- supply
        _control(
            "attrition_multiplier", SUPPLY, "Attrition multiplier",
            "Scales every role's weekly attrition. The risk run that "
            "doubles attrition is 2.0; the roster is NOT re-tuned to cope, "
            "so the run reports what the change costs rather than what it "
            "would have cost had somebody seen it coming.",
            SLIDER, path=("scenario", "attrition_multiplier"),
            minimum=0.0, maximum=3.0, step=0.05,
        ),
        _control(
            "onsite_shape", SUPPLY, "Onsite roster shape",
            "Flat is the same hours every weekday. Part-time moves a share "
            "of paid hours to Monday and Tuesday. Ideal match shapes the "
            "roster to the arrival pattern exactly - an upper bound nobody "
            "can hire, kept as something to measure against.",
            SELECT, path=("scenario", "onsite_schedule", "shape"),
            options=["flat", "part_time", "ideal_match"],
        ),
        _control(
            "part_time_share", SUPPLY, "Monday part-time share",
            "The share of Onsite Associate paid hours moved to Monday and "
            "Tuesday part-timers. Only used when the shape above is "
            "part-time. The sweep picked 2%: beyond that it costs more "
            "than it covers.",
            SLIDER, path=("scenario", "onsite_schedule", "part_time_share"),
            minimum=0.0, maximum=0.10, step=0.01, written=PERCENT,
        ),
        _control(
            "site_shortfall_site", SUPPLY, "Site losing hours",
            "A site short of people for a spell - sickness, a resignation "
            "nobody has backfilled. None means no shortfall. The risk run "
            "uses N1.",
            SELECT, path=("scenario", "site_shortfall", "site"),
            options=[""] + [site["code"] for site in sites["sites"]],
        ),
        _control(
            "site_shortfall_share", SUPPLY, "Share of that site's hours lost",
            "How much of the site's Onsite Associate hours go missing. The "
            "risk run loses a fifth.",
            SLIDER, path=("scenario", "site_shortfall", "share_of_hours_lost"),
            minimum=0.0, maximum=1.0, step=0.05, written=PERCENT,
        ),
        _control(
            "site_shortfall_weeks", SUPPLY, "Weeks the shortfall lasts",
            "How many weeks it goes on for, starting at the week below.",
            SLIDER, path=("scenario", "site_shortfall", "weeks"),
            minimum=0, maximum=assumptions["calendar"]["plan_weeks"], step=1,
        ),
        _control(
            "minimum_borrow_hours", SUPPLY, "Borrow minimum hours",
            "The smallest useful loan of a person from one site to "
            "another. Lower it and more small gaps can be covered, at the "
            "cost of sending somebody across town for an hour. This one "
            "input lives in sites.yaml rather than assumptions.yaml.",
            SLIDER, block=SITES, path=("travel", "minimum_borrow_hours"),
            minimum=1, maximum=8, step=1,
        ),
        # ------------------------------------------------------------- service
        _control(
            "phone_service_level", SERVICE, "Phone service level",
            "The share of calls to be answered inside the target below. "
            "The goal runs move this and the target together.",
            SLIDER, path=("live_queues", "goals"), row="phones",
            field=("service_level",),
            minimum=0.5, maximum=1.0, step=0.01, written=PERCENT,
        ),
        _control(
            "phone_target_seconds", SERVICE, "Phone target seconds",
            "The seconds a caller should wait at most. 80/30 and 80/60 are "
            "the two goal runs.",
            NUMBER, path=("live_queues", "goals"), row="phones",
            field=("target_seconds",),
            minimum=5, maximum=300, step=5,
        ),
        _control(
            "walk_in_service_level", SERVICE, "Walk-in service level",
            "The share of walk-in requesters to be seen inside the target "
            "below.",
            SLIDER, path=("live_queues", "goals"), row="walk_ins",
            field=("service_level",),
            minimum=0.5, maximum=1.0, step=0.01, written=PERCENT,
        ),
        _control(
            "walk_in_target_seconds", SERVICE, "Walk-in target seconds",
            "The seconds a requester should wait at the counter at most.",
            NUMBER, path=("live_queues", "goals"), row="walk_ins",
            field=("target_seconds",),
            minimum=60, maximum=1800, step=30,
        ),
        _control(
            "lunch_from", SERVICE, "Lunch window opens",
            "The earliest an unpaid lunch can be taken. A wider window "
            "lets the model place lunches where the queue is quiet.",
            TIME, path=("breaks_and_lunch", "reliefs"), row="lunch",
            field=("window", "from"),
        ),
        _control(
            "lunch_to", SERVICE, "Lunch window closes",
            "The latest an unpaid lunch can start.",
            TIME, path=("breaks_and_lunch", "reliefs"), row="lunch",
            field=("window", "to"),
        ),
        _control(
            "afternoon_break_from", SERVICE, "Afternoon break opens",
            "The earliest the paid afternoon break can be taken. Moving it "
            "with the lunch window is the second lunch run.",
            TIME, path=("breaks_and_lunch", "reliefs"),
            row="afternoon_break", field=("window", "from"),
        ),
        _control(
            "afternoon_break_to", SERVICE, "Afternoon break closes",
            "The latest the paid afternoon break can start.",
            TIME, path=("breaks_and_lunch", "reliefs"),
            row="afternoon_break", field=("window", "to"),
        ),
        # ----------------------------------------------------------- structure
        _control(
            "central_phones", STRUCTURE, "Centralized phones",
            "Pool every site's calls into one queue. Erlang C rewards the "
            "bigger pool; the handle time goes up a little because the "
            "person answering is not at the site the caller is asking "
            "about.",
            TOGGLE, path=("scenario", "central_phones", "enabled"),
        ),
        _control(
            "records_hub", STRUCTURE, "Central records hub",
            "Move the non-digital work types to one hub site, with "
            "delivery days back to the others.",
            TOGGLE, path=("scenario", "records_hub", "enabled"),
        ),
        _control(
            "digitization", STRUCTURE, "Coordinated digitization",
            "Run a digitization project: a Records Specialist spends part "
            "of the week converting the archive instead of working the "
            "queue.",
            TOGGLE, path=("scenario", "digitization", "enabled"),
        ),
        _control(
            "digitization_hours", STRUCTURE, "Digitization hours per week",
            "Hours per week the project takes out of the queue. Only used "
            "when coordinated digitization is on.",
            SLIDER, path=("scenario", "digitization", "hours_per_week"),
            minimum=0, maximum=80, step=5,
        ),
        # ---------------------------------------------------------- automation
        _control(
            "deflection_enabled", AUTOMATION, "Deflection on",
            "Demand-side: the requester self-serves or an AI answers, so "
            "the contact never arrives. Off by default, because a run that "
            "buys automation is not the same claim as a run that also "
            "assumes requesters stop calling.",
            TOGGLE, path=("scenario", "deflection", "enabled"),
            spec=True,
        ),
        _control(
            "simple_contact_share", AUTOMATION, "Simple contact share",
            "The share of status calls and emails simple enough to be "
            "deflected at all (7.1).",
            SLIDER, path=("scenario", "deflection", "simple_contact_share"),
            minimum=0.0, maximum=1.0, step=0.05, written=PERCENT,
            spec=True,
        ),
        _control(
            "ai_deflection", AUTOMATION, "AI deflection of simple contacts",
            "The first of the four rates (7.1). Demand-side: an AI answers "
            "the contact.",
            SLIDER, path=("scenario", "deflection", "ai_deflection"),
            minimum=0.0, maximum=1.0, step=0.01, written=PERCENT,
            spec=True,
        ),
        _control(
            "self_service", AUTOMATION, "Self-service of simple contacts",
            "The second of the four rates. Demand-side: the requester "
            "serves themselves. This and AI deflection add together inside "
            "the simple share, and the model refuses a total above 100%.",
            SLIDER, path=("scenario", "deflection", "self_service"),
            minimum=0.0, maximum=1.0, step=0.01, written=PERCENT,
            spec=True,
        ),
        _control(
            "simple_task_share", AUTOMATION, "Simple task share",
            "The share of an eligible work type's items simple enough for "
            "a machine (7.1).",
            SLIDER, path=("automation", "rates", "simple_task_share"),
            minimum=0.0, maximum=1.0, step=0.05, written=PERCENT,
            spec=True,
        ),
        _control(
            "ai_handling", AUTOMATION, "AI handling of simple tasks",
            "The third of the four rates. Supply-side: the work arrives "
            "and an AI does it.",
            SLIDER, path=("automation", "rates", "ai_handling"),
            minimum=0.0, maximum=1.0, step=0.01, written=PERCENT,
            spec=True,
        ),
        _control(
            "automation_displacement", AUTOMATION,
            "Automation displacement of simple tasks",
            "The fourth rate. Supply-side: a rule, not a model, does the "
            "work. This and AI handling add together inside the simple "
            "share.",
            SLIDER, path=("automation", "rates", "automation_displacement"),
            minimum=0.0, maximum=1.0, step=0.01, written=PERCENT,
            spec=True,
        ),
        _control(
            "realization", AUTOMATION, "Realization",
            "How much of all four rates and both uplifts actually lands "
            "(decided 10/27). The 50% sensitivity run halves the program "
            "to ask whether its finding survives.",
            SLIDER, path=("automation", "rates", "realization"),
            minimum=0.0, maximum=1.0, step=0.05, written=PERCENT,
            spec=True,
        ),
        _control(
            "easy_contact_ratio", AUTOMATION,
            "Easy-contact handle time, relative to average",
            "What a deflected contact would have taken, against the "
            "average one. This sets the demand-side mix uplift, which is "
            "calculated rather than typed in: removing the easy work "
            "leaves a harder mix.",
            SLIDER,
            path=("scenario", "deflection", "easy_handle_time_relative"),
            minimum=0.0, maximum=1.0, step=0.05, written=PERCENT,
            spec=True,
        ),
        _control(
            "easy_task_ratio", AUTOMATION,
            "Easy-task handle time, relative to average",
            "The same ratio on the supply side, for the tasks a machine "
            "takes. It sets the supply-side mix uplift the same way.",
            SLIDER, path=("automation", "easy_handle_time_relative"),
            minimum=0.0, maximum=1.0, step=0.05, written=PERCENT,
            spec=True,
        ),
        _control(
            "escalation_uplift", AUTOMATION, "Escalation uplift on contacts",
            "The input that can make deflection COST hours: a contact that "
            "tried AI first and still reaches a person is harder to "
            "handle. The break-even is 4.17% at the defaults (7.2) - above "
            "that, deflection gives the hours back.",
            SLIDER, path=("scenario", "deflection", "escalation_uplift"),
            minimum=0.0, maximum=0.25, step=0.005, written=PERCENT,
            spec=True,
        ),
        _control(
            "extra_uplift", AUTOMATION, "Extra uplift on tasks",
            "The same idea on the supply side: the eligible tasks left for "
            "people are harder than the mix uplift alone accounts for.",
            SLIDER, path=("automation", "extra_uplift"),
            minimum=0.0, maximum=0.25, step=0.005, written=PERCENT,
            spec=True,
        ),
        _control(
            "only_work_types", AUTOMATION, "Automate only these work types",
            "Empty means every eligible work type. The targeted run picks "
            "W5 and W8, the two that pay back after their setup fee. Which "
            "rows the machine is pointed at is a decision, not an "
            "optimisation, so the panel asks rather than choosing.",
            MULTISELECT, path=("automation", "rates", "only_work_types"),
            options=eligible, option_names=names,
            spec=True,
        ),
        _control(
            "poc_enabled", AUTOMATION, "After the proof of concept",
            "Run as though the automation has passed its error-rate gate, "
            "so QA sampling drops to the after-gate multiple.",
            TOGGLE, path=("scenario", "proof_of_concept", "enabled"),
        ),
        _control(
            "poc_true_error_rate", AUTOMATION, "Proof-of-concept true error rate",
            "The error rate the machine really has, as against the one the "
            "gate is set to catch. The 0.5% sensitivity run asks what "
            "happens when the two are the same number.",
            SLIDER, path=("scenario", "proof_of_concept", "true_error_rate"),
            minimum=0.0, maximum=0.05, step=0.001, written=PERCENT,
        ),
        # -------------------------------------------------------------- timing
        _control(
            "starts_week", TIMING, "Change starts in plan week",
            "Week 1 is the start of the plan year. A change that starts in "
            "week 13 is still reported over the whole 52 weeks, because "
            "the question a leader asks is what the year costs.",
            SLIDER, block=HEADER, path=("starts_week",),
            minimum=1, maximum=assumptions["calendar"]["plan_weeks"], step=1,
        ),
    ]

    # The 7.1 defaults are read from the 7.1 run itself, once, rather
    # than restated. Where the baseline already holds 7.1's value - the
    # deflection shares, the easy ratios, realization - the two agree,
    # and a reader seeing them agree learns something true: that input is
    # at its spec default already.
    spec = spec_config(assumptions)
    for control in controls:
        control["baseline"] = baseline_value(control, assumptions, sites)
        control["spec_default"] = (
            baseline_value(control, spec, sites) if control["spec"] else None
        )
    return controls


def _dig(tree, path):
    """Follow a path into a loaded YAML tree."""
    node = tree
    for step in path:
        node = node[step]
    return node


def baseline_value(control, assumptions=None, sites=None):
    """What this input is set to in the YAML, before anybody changes it.

    Read rather than restated, so the number beside a control is the
    model's own default and cannot drift from it.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites if sites is not None else config.load_sites()

    if control["block"] == HEADER:
        return _dig(assumptions["scenario"], control["path"])
    tree = sites if control["block"] == SITES else assumptions
    node = _dig(tree, control["path"])
    if control["row"] is None:
        return node
    # A list keyed by code: find the row this control speaks for, the
    # same way the model's merge finds it.
    for row in node:
        if row.get("code") == control["row"]:
            return _dig(row, control["field"])
    raise KeyError(f"{control['key']}: no row {control['row']!r} to change")


# ---------------------------------------------------------------------------
# Turning a panel full of values into something the doorway will take
# ---------------------------------------------------------------------------


def _set_in(tree, path, value):
    node = tree
    for step in path[:-1]:
        node = node.setdefault(step, {})
    node[path[-1]] = value


def changed(controls, values):
    """The controls whose value is not the baseline's, in catalogue order.

    Compared against the YAML's own value, so a control a reader moved
    and moved back is not reported as a change.
    """
    moved = []
    for control in controls:
        if control["key"] not in values:
            continue
        value = values[control["key"]]
        if value != control["baseline"]:
            moved.append((control, value))
    return moved


def overrides_from_values(controls, values, advanced=None):
    """The mapping the doorway takes, built from what a reader moved.

    ONLY WHAT MOVED. A panel that sent every control on every run would
    declare thirty-odd overrides each time, and the run's own
    "what it changes" line - which the doorway builds from the keys
    declared - would stop meaning anything.
    """
    mapping = {}
    rows = {}

    for control, value in changed(controls, values):
        if control["block"] == HEADER:
            _set_in(mapping, (HEADER,) + control["path"], value)
        elif control["row"] is None:
            _set_in(mapping, (control["block"],) + control["path"], value)
        else:
            # Rows are collected per list and per code, so two controls
            # on the same break window arrive as one row rather than two
            # rows the merge would have to reconcile.
            spot = (control["block"],) + control["path"]
            row = rows.setdefault(spot, {}).setdefault(
                control["row"], {"code": control["row"]}
            )
            _set_in(row, control["field"], value)

    for spot, by_code in rows.items():
        _set_in(mapping, spot, list(by_code.values()))

    for path, value in (advanced or {}).items():
        _set_in(mapping, (ASSUMPTIONS,) + tuple(path.split(".")), value)

    return mapping


def describe_changes(controls, values, advanced=None):
    """What was changed, in plain words, for the line above the results.

    The control's own label and the value as the control writes it, so
    the summary reads like the panel rather than like the YAML.
    """
    lines = []
    for control, value in changed(controls, values):
        lines.append(f"{control['label']}: {write(control, value)} "
                     f"(baseline {write(control, control['baseline'])})")
    for path, value in sorted((advanced or {}).items()):
        lines.append(f"{path}: {value}")
    return lines


# ---------------------------------------------------------------------------
# Break and lunch windows: refused before the run, not during it
#
# WHY THIS IS HERE AND NOT ONLY IN THE MODEL. intervals.relief_windows
# already refuses a window that ends before it starts, and one with no
# interval inside it. Two things were wrong with leaving it at that. The
# refusal arrived twenty-five seconds into a run, as a sentence about
# interval masks; and the model's own test is weaker than the panel's
# needs - a window from 07:00 still has intervals inside it, so the model
# accepts a lunch that starts before the doors open.
#
# So the panel checks the whole window against the opening hours, in
# words, before anything runs. The model keeps its own check: this one is
# a courtesy, not the guard.
# ---------------------------------------------------------------------------


def _clock(text):
    """"HH:MM" as minutes since midnight."""
    hours, minutes = str(text).split(":")
    return int(hours) * 60 + int(minutes)


def window_problems(controls, values, assumptions=None):
    """Break and lunch windows that could not be worked, in plain words.

    One message per window, naming the window the way the YAML names it,
    so a reader who has moved two of them knows which one is wrong.
    """
    assumptions = assumptions or config.load_assumptions()
    hours = assumptions["intraday"]["site_hours"]
    opens, closes = hours["opens"], hours["closes"]
    step = hours["interval_minutes"]
    named = {
        relief["code"]: relief["name"]
        for relief in assumptions["breaks_and_lunch"]["reliefs"]
    }

    # The panel's time controls, gathered back into the windows they
    # describe: {relief code: {"from": "11:00", "to": "14:00"}}.
    windows = {}
    for control in controls:
        if control["kind"] != TIME or control["row"] is None:
            continue
        edge = control["field"][-1]
        value = values.get(control["key"], control["baseline"])
        if hasattr(value, "strftime"):
            value = value.strftime("%H:%M")
        windows.setdefault(control["row"], {})[edge] = value

    problems = []
    for code, window in windows.items():
        name = named.get(code, code)
        start, end = window.get("from"), window.get("to")
        if start is None or end is None:
            continue
        if _clock(end) <= _clock(start):
            problems.append(
                f"{name}: the window opens at {start} and closes at {end}. "
                "A window has to close after it opens, or there is nowhere "
                "to put the break."
            )
            continue
        if _clock(start) < _clock(opens) or _clock(end) > _clock(closes):
            problems.append(
                f"{name}: the window runs {start} to {end}, but the sites "
                f"are open {opens} to {closes}. A break can only be taken "
                "while the site is open."
            )
            continue
        span = _clock(end) - _clock(start)
        if span % step:
            problems.append(
                f"{name}: {start} to {end} is {span} minutes, which is not "
                f"a whole number of the model's {step}-minute intervals."
            )
    return problems


# ---------------------------------------------------------------------------
# The Advanced box's own guards (11/17)
#
# WHAT THEY ARE FOR. The 36 controls above cannot leave their ranges: a
# slider has ends. The Advanced box reaches every other input in
# assumptions.yaml by name, and before this it refused only what the
# MODEL refuses - so automation.uptime = 5.0, a machine up 500% of the
# time, was accepted because no formula divides by it in a way that
# complains.
#
# WHAT THEY ARE NOT. They are not bounds on each input, which would be
# an input-file change and Siri's to make. They are rules about what KIND
# of number an input is, decided 11/17: a share or a rate lies between 0%
# and 100%; hours, counts and costs cannot be negative; and a multiplier
# cannot be negative either. The kind is read off the input's own name,
# and a test checks that every input in the file passes the rule its name
# earns it - which is what catches a name this classification reads
# wrongly.
#
# A MULTIPLIER MAY BE ZERO (Siri, 11/17). The first build refused it, on
# the reasoning that a multiplier of nothing is meaningless. It is not:
# `attrition_multiplier` at zero is "nobody leaves", which is a question
# worth asking, and the Supply slider has offered it since 7c. A text box
# that refused what the slider beside it allows would have been the page
# contradicting itself.
# ---------------------------------------------------------------------------

SHARE_RULE = "share"
NOT_NEGATIVE_RULE = "not negative"
MULTIPLIER_RULE = "multiplier"

# Read in this order: a multiplier first, because `attrition_multiplier`
# carries an attrition word too and it is the multiplier that it is.
MULTIPLIER_WORDS = ("multiplier", "multiple", "factor")
SHARE_WORDS = (
    "share", "rate", "rates", "shrinkage", "attrition", "occupancy",
    "uptime", "realization", "deflection", "uplift",
)
COUNT_WORDS = (
    "hour", "hours", "week", "weeks", "day", "days", "minute", "minutes",
    "second", "seconds", "cost", "fee", "headcount", "count", "item",
    "items", "people", "size", "charts", "passes", "interval", "intervals",
    "staff", "class", "classes", "threshold", "points", "dev", "tolerance",
    "target", "seed", "budget", "fte",
)

# Names the word rules above would read wrongly, so they are named.
# `occupancy_cap` and `overtime_cap` are shares whose word is "cap", which
# is also in `capacity_cap_items_per_week`, a count; `relative_std_dev` is
# not the same kind of thing as `easy_handle_time_relative`.
SHARE_SEGMENTS = (
    "occupancy_cap", "overtime_cap", "paid_shrinkage",
    "easy_handle_time_relative", "human_ramp_threshold", "mape_target",
)

# More than this many times its baseline, or less than one divided by it,
# and the value gets a warning rather than a refusal. A screen judgement,
# like the slider ranges, and kept beside them for the same reason.
TESTED_RANGE_FACTOR = 3


def _words(segment):
    """One key segment as its underscore-separated words."""
    return set(str(segment).split("_"))


def _segment_rule(segment):
    if segment in SHARE_SEGMENTS:
        return SHARE_RULE
    words = _words(segment)
    if words & set(MULTIPLIER_WORDS):
        return MULTIPLIER_RULE
    if words & set(SHARE_WORDS):
        return SHARE_RULE
    if words & set(COUNT_WORDS):
        return NOT_NEGATIVE_RULE
    return None


def input_rule(path):
    """Which kind of number this input is, from its own name.

    The last segment decides it where it can. Where it cannot - the
    leaves of `fte.starting_headcount` are site codes,
    `intraday.day_of_week_share` is keyed by weekday - the walk carries
    on up the path, because in a table of numbers the kind is written on
    the block and not on each row. An input no segment speaks for gets no
    rule, and nothing is claimed about it.
    """
    segments = str(path).split(".")
    for segment in reversed(segments):
        rule = _segment_rule(segment)
        if rule:
            return rule
    return None


def guard(path, value, rule=None):
    """Refuse a value that is not the kind of number the input is.

    Raises ValueError with a sentence a reader can act on. Only numbers
    are checked: a name, a date or a true/false has no range to be
    outside of, and `coerce` has already refused the wrong type.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    rule = rule or input_rule(path)
    if rule == SHARE_RULE and not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{path} is a share or a rate, so it lies between 0% and 100% "
            f"- that is, between 0 and 1. You typed {value:g}."
        )
    if rule == NOT_NEGATIVE_RULE and value < 0:
        raise ValueError(
            f"{path} is an amount - hours, a count or a cost - so it "
            f"cannot be negative. You typed {value:g}."
        )
    if rule == MULTIPLIER_RULE and value < 0:
        raise ValueError(
            f"{path} is a multiplier, so it cannot be negative. Zero is "
            f"allowed - on an attrition multiplier it means nobody "
            f"leaves. You typed {value:g}."
        )
    return value


TESTED_RANGE_NOTE = "outside the range this model was tested in"


def tested_range_warning(path, value, baseline):
    """A warning, not a refusal, for a value far from its baseline.

    A value the rules above allow can still be one nobody has ever run
    this model at. Three times is the line (11/17): far enough that
    something has been typed deliberately, close enough that the ordinary
    sensitivity runs - halved realization, doubled attrition - pass
    without a warning nobody would then read.

    Nothing is said when the baseline is zero. The rule is a ratio, and
    there is no multiple of zero: the four automation rates sit at zero in
    the baseline, so a ratio rule would warn about every AI run there is.
    """
    numbers = (
        isinstance(value, (int, float))
        and isinstance(baseline, (int, float))
        and not isinstance(value, bool)
        and not isinstance(baseline, bool)
    )
    if not numbers or not baseline:
        return None
    ratio = value / baseline
    if 1 / TESTED_RANGE_FACTOR <= ratio <= TESTED_RANGE_FACTOR:
        return None
    return (
        f"{path} = {value:g} is {ratio:.3g} times its baseline of "
        f"{baseline:g}: {TESTED_RANGE_NOTE}. The run is allowed; read its "
        "figures as an extrapolation."
    )


def write(control, value):
    """One control's value, written the way its control shows it."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (list, tuple)):
        if not value:
            return "all eligible"
        # Plain names, with the code kept in brackets: the name is what a
        # reader recognises, and the code is what they need to look the
        # row up in the spec or in the Next Dollar ranking.
        names = control.get("option_names") or {}
        return ", ".join(
            f"{names[item]} ({item})" if item in names else str(item)
            for item in value
        )
    if control["written"] == PERCENT and isinstance(value, (int, float)):
        # %g rather than a fixed precision: the rates run from 0.5% to
        # 100%, and a fixed two decimals would print either "100.00%" or
        # "0.50%" as "1%" depending which way it was rounded.
        return f"{value * 100:g}%"
    return f"{value:g}" if isinstance(value, float) else str(value)


# ---------------------------------------------------------------------------
# The Advanced box: every other input, by name
# ---------------------------------------------------------------------------

# A key line in a YAML file: indentation, a name, a colon.
KEY_LINE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*):(.*)$")
COMMENT_LINE = re.compile(r"^\s*#\s?(.*)$")


@functools.lru_cache(maxsize=1)
def input_descriptions():
    """Dotted key -> the comment assumptions.yaml carries above it.

    The file is commented throughout, in Siri's own words, explaining
    what each input is and why it is set where it is. Those sentences are
    the best plain description of an input that exists in this project,
    so the Advanced search shows them rather than a sentence invented
    here for the occasion.

    It reads the file as text, because a loaded YAML has no comments
    left in it. A heuristic, deliberately: it attaches the comment block
    immediately above a key to that key, and nothing else. Where there
    is no comment there is no description, and the search says so rather
    than guessing.
    """
    path = config.REPO_ROOT / "assumptions.yaml"
    descriptions = {}
    stack = []
    block = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        comment = COMMENT_LINE.match(raw)
        if comment:
            text = comment.group(1).strip()
            # The banner rules between sections describe the section, not
            # the next key, so they are not carried down onto it.
            if text and set(text) != {"="} and set(text) != {"-"}:
                block.append(text)
            continue
        if not raw.strip():
            block = []
            continue
        match = KEY_LINE.match(raw)
        if not match:
            block = []
            continue
        indent, name, _rest = match.groups()
        depth = len(indent)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, name))
        if block:
            dotted = ".".join(part for _depth, part in stack)
            descriptions[dotted] = " ".join(block)
        block = []
    return descriptions


def _leaves(node, path=()):
    """Every scalar input in a loaded tree, by its dotted path."""
    if isinstance(node, dict):
        for name, value in node.items():
            yield from _leaves(value, path + (str(name),))
    elif isinstance(node, (list, tuple)):
        # A list is changed as a whole row keyed by code, which a text
        # box cannot do safely, so lists are not offered in Advanced.
        return
    else:
        yield ".".join(path), node


def searchable_inputs(assumptions=None):
    """Every single-value input in assumptions.yaml, with its description."""
    assumptions = assumptions or config.load_assumptions()
    described = input_descriptions()
    found = []
    for dotted, value in _leaves(assumptions):
        if dotted.split(".")[0] in ("meta", "paths", "random"):
            # Housekeeping rather than a modelling input: file names, the
            # seed, and the document's own metadata.
            continue
        found.append(
            {
                "path": dotted,
                "baseline": value,
                "description": described.get(dotted.split(".")[-1], "")
                if dotted not in described else described[dotted],
            }
        )
    return sorted(found, key=lambda row: row["path"])


def baseline_of_input(path, assumptions=None):
    """One Advanced input's baseline value, by its dotted key.

    So a queued change can be compared with the YAML's own value on a
    later rerun, when whatever was searched for is long gone from the
    screen.
    """
    for row in searchable_inputs(assumptions):
        if row["path"] == path:
            return row["baseline"]
    return None


def search_inputs(query, assumptions=None, limit=25):
    """The inputs whose key or description matches what was typed."""
    wanted = (query or "").strip().lower()
    if not wanted:
        return []
    words = wanted.split()
    found = []
    for row in searchable_inputs(assumptions):
        haystack = f"{row['path']} {row['description']}".lower()
        if all(word in haystack for word in words):
            found.append(row)
    # A key match is what somebody searching a key name wants first.
    found.sort(key=lambda row: (wanted not in row["path"].lower(), row["path"]))
    return found[:limit]


def coerce(text, baseline):
    """What a reader typed, as the type the input already is.

    A text box hands back a string. The model refuses a string where it
    wants a number, which is correct of it but unhelpful here, so the
    value is coerced to the baseline's own type first and anything that
    will not convert is refused with a message that says so.
    """
    if isinstance(baseline, bool):
        lowered = str(text).strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
        raise ValueError(f"{text!r} is not true or false")
    if isinstance(baseline, int) and not isinstance(baseline, bool):
        return int(str(text).strip())
    if isinstance(baseline, float):
        return float(str(text).strip())
    return str(text)


# ---------------------------------------------------------------------------
# Running one
# ---------------------------------------------------------------------------

CUSTOM_KEY = api.CUSTOM_RUN_KEY


def signature(mapping):
    """A stable text form of one set of overrides, for the cache.

    Sorted, so two panels that moved the same controls in a different
    order are one run rather than two.
    """
    return json.dumps(mapping, sort_keys=True, default=str)


def merged_config(mapping, assumptions=None):
    """assumptions.yaml with one panel's overrides merged onto it.

    For the checks that have to read an input as the RUN will see it
    rather than as the file has it. The break windows are checked against
    the opening hours, and the opening hours are themselves an input the
    Advanced box can reach.
    """
    assumptions = assumptions or config.load_assumptions()
    scenario = api.scenario_from_overrides(mapping, assumptions)
    return scenarios.apply_scenario(scenario, assumptions)


@functools.lru_cache(maxsize=32)
def run_signature(text):
    """One custom run, remembered by its overrides.

    THIS IS THE "INSTANT THE SECOND TIME" PART. A run takes about
    twenty-five seconds, and a reader comparing two sets of inputs goes
    back and forth between them; doing the arithmetic again for inputs
    already run would be the slowest possible way to answer a question
    already answered. Bounded at 32 so a long session cannot grow
    without limit.

    Keyed by the signature rather than the mapping because a dict cannot
    be a cache key, and because the signature is sorted - the same
    changes in a different order are the same run.
    """
    return api.run(json.loads(text), key=CUSTOM_KEY)


def forget_runs():
    """Drop the remembered runs (the Reset button)."""
    run_signature.cache_clear()
