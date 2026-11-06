"""The coverage events log (spec 6.7).

Split out of plan.py on 10/26 (spec Section 2). plan.py decides what to do
about a shortfall; this module writes down that it happened.

The audit trail is the whole point. A plan that only ever reported totals
would leave "we were busy in March" as the best anybody could say a year
later. One row per shortfall, carrying what was borrowed, what the drive
cost, what the model thinks caused it and what the requester experienced,
turns that into a question somebody can actually answer: how often did N3
need help, and why.

The fields are spec 6.7's F1 to F24. Two things about them are worth
knowing before reading the code:

    - F13 (cause) is the model's SUGGESTION and F15, F23 and F24 belong to
      a person. That split is in the spec and it is the honest one: a model
      can see that Wednesday lost 8% to meetings, but not that two people
      were at the same funeral.
    - F14 (planned or unplanned) is DERIVED from the cause rather than
      typed beside it, so the two can never disagree. The distinction earns
      its place because planned time can be moved and unplanned cannot.

This module reads plan.py and nothing reads this one, which keeps the
Section 2 chain one-way.
"""

import numpy as np
import pandas as pd

from . import config, plan, supply

# Re-bound from plan.py rather than re-declared, so there is exactly one
# definition of each and no chance of the two drifting apart.
ALL_WEEK = plan.ALL_WEEK
NETWORK = plan.NETWORK
WEEKDAYS = supply.WEEKDAYS


# ---------------------------------------------------------------------------
# Coverage events (spec 6.7)
#
# The audit trail. One row every time a site and role come up short, which
# is what turns "we were busy in March" into something a leader can act on:
# how often, where, which role, what it cost, and why.
#
# F13 (cause) is the model's SUGGESTION and F15, F23 and F24 belong to a
# person - that split is in the spec, and it is the honest one, because a
# model can see that Wednesday lost 8% to meetings but not that two people
# were at the same funeral. F14 (planned or unplanned) is DERIVED from the
# cause rather than typed, so the two can never disagree; the distinction
# earns its place because planned time can be moved and unplanned cannot.
# ---------------------------------------------------------------------------

ALL_TYPES = ("E1", "E2", "E3", "E4")

# (field, columns, event types it applies to, grain it applies to)
# "live" = the counter, which has a service level and no inventory;
# "queue" = a back-office queue, which has inventory and no service level.
MODEL_FIELDS = (
    ("F1", ("event_id",), ALL_TYPES, None),
    ("F2", ("event_type",), ALL_TYPES, None),
    ("F3", ("week_index", "week_start"), ALL_TYPES, None),
    ("F4", ("day_of_week",), ALL_TYPES, None),
    ("F5", ("cluster",), ALL_TYPES, None),
    ("F6", ("short_site",), ALL_TYPES, None),
    ("F7", ("covering_site",), ("E1",), None),
    ("F8", ("role", "skill"), ALL_TYPES, None),
    ("F9", ("fte_covered",), ALL_TYPES, None),
    ("F10", ("hours_short_before_cover",), ALL_TYPES, None),
    ("F11", ("hours_covered",), ALL_TYPES, None),
    ("F12", ("hours_uncovered",), ALL_TYPES, None),
    ("F13", ("cause",), ALL_TYPES, None),
    ("F14", ("planned_or_unplanned",), ALL_TYPES, None),
    ("F16", ("service_level_phones_before", "service_level_phones_after",
             "service_level_walk_ins_before", "service_level_walk_ins_after"),
     ALL_TYPES, "live"),
    ("F17", ("days_of_inventory_before", "days_of_inventory_after"),
     ALL_TYPES, "queue"),
    ("F18", ("requests_delayed",), ALL_TYPES, None),
    ("F19", ("travel_hours",), ("E1",), None),
    ("F20", ("travel_miles", "travel_cost"), ("E1",), None),
    ("F21", ("overtime_cost",), ("E2",), None),
    ("F22", ("action_taken",), ALL_TYPES, None),
)

PERSON_FIELDS = (
    ("F15", ("cause_notes",)),
    ("F23", ("approved_by",)),
    ("F24", ("follow_up",)),
)

EVENT_COLUMNS = (
    [column for _f, columns, _t, _g in MODEL_FIELDS for column in columns]
    + [column for _f, columns in PERSON_FIELDS for column in columns]
    + ["grain"]
)


def field_applies(field, event_type, grain):
    """Whether a spec 6.7 field is expected on this row.

    Spec 6.7 already scopes two fields by event type - F7 is "E1 only" and
    F21 is "E2 only" - so a blank there is correct, not missing. The same
    logic covers the two consequence fields: a live queue has a service
    level and no inventory, and a back-office queue the other way round.
    """
    for name, _columns, types, needs_grain in MODEL_FIELDS:
        if name != field:
            continue
        if event_type not in types:
            return False
        return needs_grain is None or needs_grain == grain
    return any(name == field for name, _columns in PERSON_FIELDS)


class CauseFinder:
    """The model's suggested cause for a shortfall (F13).

    Three candidates, in the order a planner would actually check them:

        1. a volume spike - the work was unusually heavy, so blaming
           whoever was out would be unfair and would point the fix the
           wrong way;
        2. a vacancy - attrition has taken a whole person off the roster,
           which is a hiring problem and not an absence problem;
        3. otherwise, the shrinkage line furthest ABOVE ITS OWN NORMAL
           LEVEL that day - its share that day against its annual average.

    Rule 3 was the biggest line that day until 10/26, and that rule always
    answered "vacation": at 10% it is the largest line in every week of the
    year, so the field carried no information. Scoring each line against
    its own average asks the question a planner actually asks - what was
    UNUSUAL about this day - and so surfaces winter sickness, a
    meeting-heavy Wednesday, or the summer vacation peak.

    The vacancy test needs a WHOLE person gone, not a fraction. Headcount
    in this model is continuous - attrition takes a fraction of a team
    every week - so a looser test would call everything a vacancy from
    about March onwards and the field would stop carrying information.

    A person confirms it (spec 6.7), which is why the model only has to be
    reasonable here, not right.
    """

    def __init__(self, assumptions, prepared, required_by_grain):
        self.settings = assumptions["coverage_events"]
        self.threshold = float(self.settings["volume_spike_threshold"])
        self.vacancy_fte = float(self.settings["vacancy_fte_threshold"])
        self.lines = prepared["ledger"]["shrinkage_lines"]
        self.weeks = list(prepared["ledger"]["weeks"])
        self.week_index = {week: i for i, week in enumerate(self.weeks)}
        self.mean_required = {
            key: float(np.mean(values)) for key, values in required_by_grain.items()
        }
        ledger = prepared["ledger"]["ledger"]
        opening = ledger[ledger["week_index"] == self.weeks[0]]
        self.opening_headcount = {
            (row.role, row.scope): row.starting_headcount for row in opening.itertuples()
        }
        self.headcount = ledger.set_index(["week_index", "role", "scope"])[
            "ending_headcount"
        ]

    def most_above_normal(self, role, scope, week, day):
        """The shrinkage line furthest above its own average (spec 6.7, 10/26).

        Scored as the line's share that day divided by its annual average,
        so a 1% line that lands entirely on Wednesdays beats a 10% line
        spread evenly - which is the point. Comparing raw sizes only ever
        named vacation.

        The average is taken from the role and scope's OWN plan-year
        series, so a site whose sick draw ran high all year is not
        permanently blamed for it: the yardstick moves with the site.

        Only lines the F13 list names are candidates. Several lines are
        flat and even, so on an unremarkable day they all sit at exactly
        1.0 and the winner is a tie - broken here by the line's SIZE,
        because among equally unusual lines the one costing more hours is
        the better explanation, and then by code so the answer never
        depends on floating-point noise in the last bit.
        """
        index = self.week_index[week]
        lines = self.lines[role][scope]
        ranked = []
        for code, values in sorted(lines.items()):
            if code not in self.settings["causes"]:
                continue  # not on the spec 6.7 F13 list
            normal = float(values.mean())
            if normal <= 0:
                continue
            if day == ALL_WEEK:
                share = float(values[index].mean())
            else:
                share = float(values[index, WEEKDAYS.index(day)])
            ranked.append((round(share / normal, 9), round(normal, 9), code))
        if not ranked:
            return "other"
        return max(ranked)[2]

    def cause(self, role, scope, week, day, required_hours):
        average = self.mean_required.get((role, scope, day if day == ALL_WEEK else "day"))
        if average and required_hours > average * self.threshold:
            return "volume_spike"
        opened = self.opening_headcount.get((role, scope), 0.0)
        now = float(self.headcount.get((week, role, scope), opened))
        if opened - now >= self.vacancy_fte:
            return "vacancy"
        return self.most_above_normal(role, scope, week, day)

    def tag(self, cause):
        """F14, derived from the cause so the two can never disagree."""
        return self.settings["causes"][cause]


def _sample_text(assumptions):
    """Deterministic sample values for the person-entered fields (6.7)."""
    text = assumptions["coverage_events"]["sample_text"]
    rng = np.random.default_rng(assumptions["coverage_events"]["sample_seed"])
    return text, rng


def coverage_events(back_office, onsite, prepared, assumptions=None, sites=None):
    """The spec 6.7 event log: E1 borrows, E2 overtime, E3 gaps, E4 floor breaches.

    E5 (automation outage) cannot occur in the baseline, because automation
    is off until Session 5 - so the column exists in the spec and no row
    claims it here.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    _members, of_site = plan.cluster_lookup(sites)
    actions = assumptions["coverage_events"]["actions"]
    floor = int(assumptions["supply"].get("records_specialist_floor_per_site", 2))

    weeks = prepared["ledger"]["weeks"]
    mondays = {week: prepared["ledger"]["mondays"][i].isoformat()
               for i, week in enumerate(weeks)}

    # The yardstick for "unusually heavy", per role and scope.
    cost = back_office["cost"]
    daily = onsite["daily"]
    required_by_grain = {}
    for (role, scope), group in cost.groupby(["role", "scope"]):
        required_by_grain[(role, scope, ALL_WEEK)] = group["required_hours"].to_numpy()
    for (role, scope), group in daily.groupby(["role", "site"]):
        required_by_grain[(role, scope, "day")] = group["required_hours"].to_numpy()
    finder = CauseFinder(assumptions, prepared, required_by_grain)

    # Consequence lookups, so a borrow row can report what the day or week
    # actually came to after every form of cover was applied.
    live = daily.set_index(["week_index", "site", "day_of_week"])
    queue = cost.set_index(["week_index", "role", "scope"])

    def consequence(role, scope, week, day, grain):
        if grain == "live":
            row = live.loc[(week, scope, day)]
            return {
                "service_level_phones_before": float(row["service_level_phones_before"]),
                "service_level_phones_after": float(row["service_level_phones_after"]),
                "service_level_walk_ins_before": float(row["service_level_walk_ins_before"]),
                "service_level_walk_ins_after": float(row["service_level_walk_ins_after"]),
                "requests_delayed": float(row["requests_delayed"]),
            }
        row = queue.loc[(week, role, scope)]
        return {
            "days_of_inventory_before": float(row["days_of_inventory_before"]),
            "days_of_inventory_after": float(row["days_of_inventory_after"]),
            "requests_delayed": float(row["requests_delayed"]),
        }

    rows = []

    def add(event_type, role, scope, week, day, grain, short, covered, uncovered, **extra):
        cause = extra.pop("cause", None) or finder.cause(
            role, scope, week, day, extra.pop("required_hours", 0.0)
        )
        per_day = plan.productive_hours_per_person_day(role, assumptions)
        basis = per_day if day != ALL_WEEK else per_day * len(WEEKDAYS)
        record = {
            "event_type": event_type,
            "week_index": week,
            "week_start": mondays[week],
            "day_of_week": day,
            "cluster": of_site.get(scope, "network"),
            "short_site": scope if scope != NETWORK else "network",
            "role": role,
            "skill": role,
            "fte_covered": covered / basis if basis > 0 else 0.0,
            "hours_short_before_cover": short,
            "hours_covered": covered,
            "hours_uncovered": uncovered,
            "cause": cause,
            "planned_or_unplanned": finder.tag(cause),
            "action_taken": actions[event_type],
            "grain": grain,
        }
        record.update(consequence(role, scope, week, day, grain))
        record.update(extra)
        rows.append(record)

    # --- E1: every borrow, at either grain ------------------------------
    for frame, grain in ((back_office["borrows"], "queue"), (onsite["borrows"], "live")):
        for borrow in frame.itertuples():
            role, site, week, day = borrow.role, borrow.short_site, borrow.week_index, borrow.day_of_week
            if grain == "live":
                row = live.loc[(week, site, day)]
                short, uncovered = float(row["shortfall_hours"]), float(row["uncovered_hours"])
                required = float(row["required_hours"])
            else:
                row = queue.loc[(week, role, site)]
                short = max(0.0, float(row["required_hours"]) - float(row["available_hours"]))
                uncovered, required = float(row["uncovered_hours"]), float(row["required_hours"])
            add(
                "E1", role, site, week, day, grain,
                short, borrow.hours, uncovered,
                required_hours=required,
                covering_site=borrow.covering_site,
                travel_hours=borrow.travel_hours,
                travel_miles=borrow.travel_miles,
                travel_cost=borrow.travel_cost,
            )

    # --- E2 and E3: overtime used, and what it could not reach ----------
    for row in daily.itertuples():
        if row.overtime_hours > 1e-9:
            add("E2", row.role, row.site, row.week_index, row.day_of_week, "live",
                row.shortfall_after_borrow_hours, row.overtime_hours, row.uncovered_hours,
                required_hours=row.required_hours, overtime_cost=row.overtime_cost)
        if row.uncovered_hours > 1e-9:
            add("E3", row.role, row.site, row.week_index, row.day_of_week, "live",
                row.shortfall_hours, row.covered_hours - row.available_hours,
                row.uncovered_hours, required_hours=row.required_hours)

    for row in cost.itertuples():
        scope = row.scope
        day = ALL_WEEK
        if row.overtime_hours > 1e-9:
            add("E2", row.role, scope, row.week_index, day, "queue",
                row.shortfall_hours, row.overtime_hours, row.uncovered_hours,
                required_hours=row.required_hours, overtime_cost=row.overtime_cost)
        if row.uncovered_hours > 1e-9:
            add("E3", row.role, scope, row.week_index, day, "queue",
                row.required_hours - row.available_hours,
                row.borrowed_hours + row.overtime_hours, row.uncovered_hours,
                required_hours=row.required_hours)

    # --- E4: a site below its Records Specialist floor ------------------
    #
    # Spec 6.7, decided 10/26: counted only when the ROUNDED headcount
    # falls below the floor, because a fraction of a person leaving is not
    # a person leaving. Headcount is continuous here - a site that opens at
    # exactly 2.00 is at 1.99 the week attrition starts - and logging that
    # as a breach every week was 52 events describing nothing.
    #
    # Each real breach opens a one-person backfill requisition at once,
    # outside the minimum class size (spec 6.5, decided 10/26). The hiring
    # mechanics arrive in task 4e, so the event carries the flag and
    # nothing is lost in the meantime.
    breach = assumptions["coverage_events"]["floor_breach"]
    ledger = prepared["ledger"]["ledger"]
    specialists = ledger[ledger["role"] == "records_specialist"].copy()
    specialists["rounded_headcount"] = (
        specialists["ending_headcount"].round()
        if breach.get("round_headcount", True)
        else specialists["ending_headcount"]
    )
    for row in specialists[specialists["rounded_headcount"] < floor].itertuples():
        below = floor - float(row.rounded_headcount)
        per_week = plan.productive_hours_per_person_day("records_specialist", assumptions) * len(WEEKDAYS)
        add("E4", "records_specialist", row.scope, row.week_index, ALL_WEEK, "queue",
            below * per_week, 0.0, below * per_week, cause="vacancy",
            follow_up=breach["follow_up"])

    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(columns=["event_id"] + list(EVENT_COLUMNS))

    # Every spec 6.7 column, present whether or not this run produced an
    # event that fills it. Found 10/28: a scenario with no cluster borrows
    # emits no row carrying `covering_site`, so pandas never created the
    # column and the sort below raised a KeyError - a crash that depended
    # on whether borrowing happened to occur, which is the worst possible
    # trigger for one. The log's shape is a contract, not a by-product of
    # which events fired.
    for column in EVENT_COLUMNS:
        if column not in events.columns:
            events[column] = pd.NA

    # A stable order, so the event IDs mean the same thing on every run.
    events["_day"] = [
        -1 if day == ALL_WEEK else WEEKDAYS.index(day) for day in events["day_of_week"]
    ]
    events = events.sort_values(
        ["week_index", "_day", "event_type", "role", "short_site", "covering_site"],
        na_position="first",
    ).drop(columns="_day").reset_index(drop=True)
    events["event_id"] = [f"EV{n:05d}" for n in range(1, len(events) + 1)]

    # --- the person-entered fields (F15, F23, F24) ----------------------
    text, rng = _sample_text(assumptions)
    notes = text["notes"]
    events["cause_notes"] = [
        rng.choice(notes[tag]) for tag in events["planned_or_unplanned"]
    ]
    # F24 (spec 6.7, decided 10/26): the model fills it on a floor breach -
    # "backfill requisition opened" - because 6.5 opens that requisition
    # automatically and nobody should have to notice the E4 rows to find
    # out. Every other row gets a sample value a person would have typed.
    drawn = rng.choice(text["follow_up"], size=len(events))
    if "follow_up" in events.columns:
        events["follow_up"] = events["follow_up"].fillna(pd.Series(drawn, index=events.index))
    else:
        events["follow_up"] = drawn
    # Escalations go to WFM; a borrow or some overtime is the site manager's
    # call. Sample values, but not arbitrary ones.
    events["approved_by"] = np.where(
        events["event_type"].isin(["E3", "E4"]),
        text["approved_by"][-1],
        text["approved_by"][0],
    )

    # reindex, not select: a column belonging only to an event type that
    # never occurred - E4 in a year where no site loses a whole person -
    # still has to exist, so the file's shape does not depend on the data.
    ordered = ["event_id"] + [c for c in EVENT_COLUMNS if c != "event_id"]
    return events.reindex(columns=ordered)
