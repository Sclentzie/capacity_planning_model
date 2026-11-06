"""The hiring plan (spec 6.5).

Split out of plan.py on 10/26 (spec Section 2). plan.py measures the gap;
this module decides who to hire for it and WHEN TO START, which is the part
that cannot be done late.

A Tier 2 Specialist is eight weeks of requisition, seven of onboarding and
four of ramp: nearly five months between signing a requisition and getting
a productive hour out of it. A plan that notices the shortfall in the week
it lands has already missed, which is why this module works backwards from
the shortfall instead of forwards from today.

The judgement the whole module turns on (decided 10/26): the trigger reads
the gap after cluster coverage and BEFORE overtime. Overtime is there to
bridge a dip and to carry a new class through its ramp, so a trigger that
waited for overtime to run out would not fire until the cap was exhausted -
months after the requisition needed signing.

This module reads plan.py and nothing reads this one, which keeps the
Section 2 chain one-way.
"""

import math

import numpy as np
import pandas as pd

from . import config, plan, supply


# ---------------------------------------------------------------------------
# The hiring plan (spec 6.5)
#
# This is the output a leader actually acts on. Everything before it says
# how bad the gap is; this says what to do about it and WHEN TO START,
# which is the part that cannot be done late. A Tier 2 Specialist is
# eight weeks of requisition, seven of onboarding and four of ramp: nearly
# five months between signing a requisition and getting a productive hour.
# A plan that spots the shortfall in the week it lands has already missed.
#
# The trigger reads the gap after cluster coverage and BEFORE overtime
# (decided 10/26). That ordering is the whole judgement: overtime is there
# to bridge a dip and to carry a new class through its ramp, so a trigger
# that waited for overtime to run out would not fire until the cap was
# exhausted - months after the requisition needed signing.
# ---------------------------------------------------------------------------


def fte_basis(role, assumptions=None):
    """Spec 6.1's denominator: 40 x (1 - shrinkage), productive hours per FTE."""
    assumptions = assumptions or config.load_assumptions()
    fte = assumptions["fte"]
    return fte["paid_hours_per_week"] * (1.0 - fte["paid_shrinkage"][role])


def weekly_gap_fte(back_office, onsite, assumptions=None, sites=None):
    """The gap in FTE per role, scope and week (spec 6.1, 6.5).

    After cluster coverage, before overtime - which is what 6.5 now asks
    for. Returned as two non-negative columns rather than one signed one,
    because the two are not opposites at a live queue.

    For the BACK OFFICE a week is a week: hours are fungible inside it, so
    short and surplus are the two sides of one number.

    At the COUNTER they are not. A Monday that ran 5 hours short is not
    repaid by a slack Friday - the walk-in has already left - so the
    shortfall is the sum of the days' own shortfalls. The surplus is the
    week's net, because "should we stop hiring here" is genuinely a
    question about the week's total resourcing, not about Monday.

    That asymmetry is deliberate, and it is also self-correcting: a site
    whose only problem is a heavy Monday accumulates a fraction of an FTE
    and never triggers hiring, which is right. Its problem is the roster
    shape, and another body would sit idle on Friday.
    """
    assumptions = assumptions or config.load_assumptions()
    rows = []

    for row in back_office["cost"].itertuples():
        basis = fte_basis(row.role, assumptions)
        rows.append(
            {
                "week_index": row.week_index,
                "role": row.role,
                "scope": row.scope,
                "required_hours": row.required_hours,
                "covered_hours": row.covered_hours,
                "short_fte": row.shortfall_hours / basis,
                "surplus_fte": max(0.0, row.covered_hours - row.required_hours) / basis,
            }
        )

    basis = fte_basis("onsite_associate", assumptions)
    live = onsite["daily"].groupby(["week_index", "site"], as_index=False).agg(
        required_hours=("required_hours", "sum"),
        covered_hours=("covered_hours", "sum"),
        available_hours=("available_hours", "sum"),
        borrowed_hours=("borrowed_hours", "sum"),
        daily_shortfall=("shortfall_after_borrow_hours", "sum"),
    )
    for row in live.itertuples():
        before_overtime = row.available_hours + row.borrowed_hours
        rows.append(
            {
                "week_index": row.week_index,
                "role": "onsite_associate",
                "scope": row.site,
                "required_hours": row.required_hours,
                "covered_hours": before_overtime,
                "short_fte": row.daily_shortfall / basis,
                "surplus_fte": max(0.0, before_overtime - row.required_hours) / basis,
            }
        )

    return pd.DataFrame(rows).sort_values(["role", "scope", "week_index"]).reset_index(
        drop=True
    )


def runs_over(weeks, values, threshold, minimum_length):
    """Stretches of consecutive weeks above a threshold (spec 6.5).

    Returns [(first week, last week, the worst value in it), ...]. The
    "consecutive" part matters: one bad week is a scheduling problem and
    three in a row is a staffing problem, and only the second one is worth
    a requisition.
    """
    found, start, worst = [], None, 0.0
    previous = None
    for week, value in zip(weeks, values):
        over = value > threshold
        broken = previous is not None and week != previous + 1
        if start is not None and (not over or broken):
            if week - start >= minimum_length or (previous - start + 1) >= minimum_length:
                found.append((start, previous, worst))
            start, worst = None, 0.0
        if over:
            if start is None:
                start, worst = week, value
            worst = max(worst, value)
        previous = week
    if start is not None and (previous - start + 1) >= minimum_length:
        found.append((start, previous, worst))
    return found


def class_size(role, worst_short_fte, assumptions=None):
    """How many to hire for a shortfall of `worst_short_fte` (spec 6.5).

        size = ceil(worst shortfall / share still here at full ramp)

    Grossed up because a class of 4 is not 4 people by the time it reaches
    full speed. Tier 2 waits 8 weeks of requisition, 7 of onboarding and 4
    of ramp; at 1.2% monthly attrition that is a few percent of the class
    gone before a single productive hour. Hiring the gap exactly means
    hiring short, and then hiring again - at another five months' notice.

    Minimum class of 2, because a class of one has no cover: if that
    person leaves in training the whole requisition starts over.
    """
    assumptions = assumptions or config.load_assumptions()
    hiring = assumptions["hiring"]
    settings = supply.role_settings(role, assumptions)

    weeks_to_full = supply.onboarding_weeks(settings) + supply.weeks_to_full_speed(
        settings, hiring["gross_up_to_ramp"]
    )
    # The SCENARIO's attrition rate, not the baseline's: an attrition-
    # doubles scenario loses twice as many of a class before it reaches
    # full speed, and sizing against the baseline rate would hire short
    # exactly where hiring short hurts most.
    attrition = supply.weekly_attrition(
        supply.scenario_monthly_attrition(role, assumptions)
    )
    survivors = (1.0 - attrition) ** weeks_to_full

    wanted = math.ceil(worst_short_fte / survivors - 1e-9)
    return max(int(wanted), int(hiring["minimum_class"])), survivors, weeks_to_full


def schedule_class(role, first_short_week, assumptions=None):
    """When the requisition opens, the class starts, and it first produces.

    Spec 6.5's hire-by date, plus what happens when it is already in the
    past (decided 10/26): the class still goes in at the earliest week the
    plan can place it, flagged, and overtime carries the gap until it ramps.
    That is the honest answer - the hours cannot be conjured backwards, and
    a plan that silently moved the shortfall would be lying about it.
    """
    assumptions = assumptions or config.load_assumptions()
    settings = supply.role_settings(role, assumptions)
    earliest = assumptions["calendar"]["history_weeks"]

    hire_by = supply.hire_by_week(role, first_short_week, assumptions)
    requisition = max(hire_by, earliest)
    start = requisition + int(settings["requisition_weeks"])
    return {
        "hire_by_week": hire_by,
        "requisition_week": requisition,
        "start_week": start,
        "first_productive_week": start + supply.onboarding_weeks(settings),
        "flag": "hire-by date passed" if hire_by < earliest else "",
    }


def floor_breach_runs(prepared, assumptions=None):
    """Weeks where a site's rounded Records Specialist headcount is under 2.

    Grouped into consecutive runs, because 6.5's "opened at once" means
    immediately - not once per week. A 52-week breach is one empty seat,
    not fifty-two requisitions.
    """
    assumptions = assumptions or config.load_assumptions()
    floor = int(assumptions["supply"].get("records_specialist_floor_per_site", 2))
    breach = assumptions["coverage_events"]["floor_breach"]

    ledger = prepared["ledger"]["ledger"]
    mine = ledger[ledger["role"] == "records_specialist"].copy()
    mine["rounded"] = (
        mine["ending_headcount"].round()
        if breach.get("round_headcount", True)
        else mine["ending_headcount"]
    )

    runs = []
    for scope, group in mine.groupby("scope"):
        group = group.sort_values("week_index")
        under = group["rounded"] < floor
        weeks = group["week_index"].to_numpy()[under.to_numpy()]
        for first, last, _worst in runs_over(weeks, np.ones(len(weeks)), 0.5, 1):
            short = floor - float(
                group[group["week_index"] == first]["rounded"].iloc[0]
            )
            runs.append((scope, int(first), int(last), short))
    return sorted(runs)


def find_candidates(gap, prepared, assumptions=None, placed=()):
    """Every requisition the rules say to open now, earliest gap first.

    Two kinds, and they are not the same decision. A FLOOR BACKFILL is one
    person for an empty seat, opened at once, outside the minimum class
    size - a site with one Records Specialist is a compliance problem, not
    a capacity one. A HIRING CLASS answers a sustained capacity gap and
    waits for the three-week rule to be sure it is real.
    """
    assumptions = assumptions or config.load_assumptions()
    hiring = assumptions["hiring"]
    earliest = assumptions["calendar"]["history_weeks"]
    placed = set(placed)
    candidates = []

    for scope, first, last, short in floor_breach_runs(prepared, assumptions):
        key = ("floor", "records_specialist", scope, first)
        if key in placed:
            continue
        schedule = schedule_class("records_specialist", first, assumptions)
        candidates.append(
            {
                "key": key,
                "kind": "floor backfill",
                "role": "records_specialist",
                "scope": scope,
                "class_size": int(
                    assumptions["coverage_events"]["floor_breach"][
                        "backfill_requisition_people"
                    ]
                ),
                "first_short_week": first,
                "last_short_week": last,
                "weeks_short": last - first + 1,
                "worst_short_fte": short,
                "gross_up_survivors": 1.0,
                "weeks_to_full_ramp": 0,
                "why": (
                    f"Records Specialist headcount at {scope} rounds below the "
                    f"floor of 2 from plan week {first - earliest + 1}"
                ),
                **schedule,
            }
        )

    for (role, scope), group in gap.groupby(["role", "scope"]):
        group = group.sort_values("week_index")
        runs = runs_over(
            group["week_index"].to_numpy(),
            group["short_fte"].to_numpy(),
            float(hiring["trigger_fte"]),
            int(hiring["trigger_weeks"]),
        )
        for first, last, worst in runs:
            key = ("class", role, scope, int(first))
            if key in placed:
                continue
            size, survivors, weeks_to_full = class_size(role, worst, assumptions)
            schedule = schedule_class(role, int(first), assumptions)
            candidates.append(
                {
                    "key": key,
                    "kind": "hiring class",
                    "role": role,
                    "scope": scope,
                    "class_size": size,
                    "first_short_week": int(first),
                    "last_short_week": int(last),
                    "weeks_short": int(last - first + 1),
                    "worst_short_fte": worst,
                    "gross_up_survivors": survivors,
                    "weeks_to_full_ramp": weeks_to_full,
                    "why": (
                        f"short by more than {hiring['trigger_fte']} FTE for "
                        f"{last - first + 1} weeks from plan week "
                        f"{first - earliest + 1}; worst {worst:.2f} FTE"
                    ),
                    **schedule,
                }
            )

    # Earliest gap first, because an early class changes every later gap -
    # which is exactly why the ledger is rebuilt after each one.
    return sorted(
        candidates,
        key=lambda c: (c["first_short_week"], c["hire_by_week"], c["role"], c["scope"]),
    )


def surplus_flags(gap, assumptions=None):
    """Where the plan recommends PAUSING hiring (spec 6.5).

    A plan that only ever finds gaps is a plan that only ever recommends
    spending. Over by more than 2 FTE for six weeks or more is worth
    saying out loud, because the alternative to hiring there is not doing
    nothing - it is moving the requisition somewhere it is needed.
    """
    assumptions = assumptions or config.load_assumptions()
    hiring = assumptions["hiring"]
    earliest = assumptions["calendar"]["history_weeks"]

    rows = []
    for (role, scope), group in gap.groupby(["role", "scope"]):
        group = group.sort_values("week_index")
        for first, last, worst in runs_over(
            group["week_index"].to_numpy(),
            group["surplus_fte"].to_numpy(),
            float(hiring["surplus_fte"]),
            int(hiring["surplus_weeks"]),
        ):
            rows.append(
                {
                    "kind": "pause hiring",
                    "role": role,
                    "scope": scope,
                    "class_size": 0,
                    "first_week": int(first),
                    "last_week": int(last),
                    "weeks": int(last - first + 1),
                    "worst_surplus_fte": worst,
                    "why": (
                        f"over by more than {hiring['surplus_fte']} FTE for "
                        f"{last - first + 1} weeks from plan week "
                        f"{first - earliest + 1}; peak {worst:.2f} FTE over"
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["kind", "role", "scope", "class_size", "first_week", "last_week",
                 "weeks", "worst_surplus_fte", "why"],
    )


def build_hiring_plan(assumptions=None, sites=None, work=None, demand=None):
    """Place classes until no trigger fires (spec 6.5).

    One class at a time, earliest gap first, rebuilding the ledger after
    each - because a class placed in week 18 changes every gap after it,
    and sizing them all against the untouched baseline would hire the same
    shortfall several times over. It is slower and it is the only way the
    arithmetic stays honest.

    The loop ends when no candidate is left. If it hits the safety stop,
    that is reported rather than swallowed.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    demand = demand or plan.demand_side(assumptions, sites, work)

    cohorts = supply.opening_cohorts(assumptions, sites)
    limit = int(assumptions["hiring"]["max_classes"])
    placed, records = set(), []
    prepared = gap = None
    exhausted = False

    for _round in range(limit + 1):
        prepared = plan.prepare(assumptions, sites, cohorts=cohorts, demand=demand)
        back_office = plan.build_plan(assumptions, sites, prepared=prepared)
        onsite = plan.build_onsite(assumptions, sites, prepared=prepared)
        gap = weekly_gap_fte(back_office, onsite, assumptions, sites)

        candidates = find_candidates(gap, prepared, assumptions, placed)
        if not candidates:
            break
        if len(records) >= limit:
            exhausted = True
            break

        chosen = candidates[0]
        placed.add(chosen["key"])
        cohorts = supply.add_hiring_class(
            cohorts, chosen["role"], chosen["scope"],
            chosen["start_week"], chosen["class_size"],
        )
        records.append({k: v for k, v in chosen.items() if k != "key"})

    first_plan_week = assumptions["calendar"]["history_weeks"]
    classes = pd.DataFrame(
        records,
        columns=[
            "kind", "role", "scope", "class_size", "hire_by_week",
            "requisition_week", "start_week", "first_productive_week",
            "first_short_week", "last_short_week", "weeks_short",
            "worst_short_fte", "gross_up_survivors", "weeks_to_full_ramp",
            "why", "flag",
        ],
    )
    for column in ("hire_by_week", "requisition_week", "start_week",
                   "first_productive_week", "first_short_week"):
        if len(classes):
            classes["plan_" + column.replace("_week", "")] = (
                classes[column] - first_plan_week + 1
            )

    return {
        "classes": classes,
        "surplus": surplus_flags(gap, assumptions),
        "cohorts": cohorts,
        "gap": gap,
        "rounds": len(records),
        "exhausted": exhausted,
    }
