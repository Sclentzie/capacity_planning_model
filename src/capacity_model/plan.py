"""Capacity Plan: the gap, cluster coverage, overtime and cost (spec 6.1-6.4).

The plan compares required and available hours, fills what it can, and
reports what the shortfall did to the requester. The coverage events log
lives in coverage.py and the hiring plan in hiring.py (split 10/26, spec
Section 2); both read this module and neither is read by it.

    Spec 6.2 order:
        1. cluster borrowing   - same role, same cluster, travel paid
        2. overtime            - up to 10% of paid hours, at 1.5x
        3. uncovered gap       - what is left, and what it costs

**Two grains, for one reason** (spec 6.2, decided 10/26). Live work at the
counter and on the phones is checked BY DAY, because a walk-in cannot wait
until Thursday. Back-office queues - Tier 1, Tier 2, QA and the Records
Specialists - are checked BY WEEK, because a Wednesday shortfall can wait
for Thursday and still clear inside the turnaround goal. The queue absorbs
the within-week timing; the counter cannot.

That split is the whole reason this module has two builders:

    build_plan     back-office roles, weekly, consequence = days of inventory
    build_onsite   Onsite Associates, daily, consequence = service level

The WFM idea that makes any of this worth modelling: A SHORTFALL DOES NOT
VANISH. Hours that nobody worked become work that nobody did. In a
back-office queue that work is still sitting there on Monday, so the gap
feeds the backlog projection in 4.3 and comes out as turnaround. At a live
queue the work does not wait at all - the requester does - so the gap feeds
Erlang C back at the staff actually on the floor and comes out as service
level. A plan that stopped at "we were 40 hours short" would be describing
the staffing and not the consequence.

Overtime is treated as fully productive, which is deliberately generous:
real overtime is tired time. The cap is what keeps that assumption honest.

Run it:
    .venv\\Scripts\\python -m capacity_model.plan
"""

import math

import numpy as np
import pandas as pd

from . import (
    automation, backlog, config, digitization, erlang, hub, intervals, supply,
    work_volume,
)

MINUTES_PER_HOUR = 60
NETWORK = "NETWORK"
WEEKDAYS = supply.WEEKDAYS

# Weekly-grain events cover the whole week, not one day of it.
ALL_WEEK = "all week"


def remote_roles(assumptions=None):
    """The network-wide pools (spec 4.4): everything not tied to a site."""
    assumptions = assumptions or config.load_assumptions()
    per_site = set(assumptions["supply"]["per_site_roles"])
    return [role for role in assumptions["supply"]["roles"] if role not in per_site]


def back_office_roles(assumptions=None):
    """The roles with a backlog queue, checked weekly (spec 6.2, 10/26).

    Everyone except the Onsite Associates: the remote pools plus the
    Records Specialists, who are per site and so can borrow within their
    cluster even though their work queues like back-office work.
    """
    assumptions = assumptions or config.load_assumptions()
    live = set(assumptions["coverage_events"]["daily_check_roles"])
    return [role for role in assumptions["supply"]["roles"] if role not in live]


def daily_split(required_week, day_shares):
    """Weekly hours spread across Monday to Friday (spec 3.4).

    The same day-of-week shares the arrivals use, because the work arrives
    on those days: Monday carries 23% of the week and Friday 17%, so a team
    that is level-staffed is short at the start of the week and slack at
    the end even when the weekly totals balance.
    """
    return {day: required_week * share for day, share in day_shares.items()}


def allocate_overtime(shortfall_by_day, weekly_cap):
    """Spread a week's overtime budget across the days that need it (6.2).

    For LIVE work only, which spec 6.2 checks by day because a walk-in
    cannot wait. Back-office queues are checked by week (decided 10/26) and
    take their overtime in one weekly lump.

    Proportionally to each day's shortfall rather than first-come: a
    manager authorising overtime on Monday does not know yet that Thursday
    will be worse, but a weekly plan does, and spending the whole budget on
    the first short day would be an artefact of iteration order rather than
    a decision anybody made.

    Overtime never exceeds the day's own shortfall - there is nothing to
    gain from paying 1.5x for hours with no work behind them.
    """
    total = sum(shortfall_by_day.values())
    if total <= 0 or weekly_cap <= 0:
        return {day: 0.0 for day in shortfall_by_day}
    share = min(1.0, weekly_cap / total)
    return {day: hours * share for day, hours in shortfall_by_day.items()}


# ---------------------------------------------------------------------------
# Cluster coverage (spec 6.2 step 1, 6.7; defaults decided 10/26)
#
# A cluster is a set of sites close enough to help each other on short
# notice. Borrowing is temporary help, never a transfer - the person goes
# home at the end of the day - and it is the cheapest of the three coverage
# options, which is why 6.2 tries it first: the hours already exist and are
# already paid for.
#
# But it is not free, and the model would abuse it if it were. Somebody
# drives. The round trip comes off the LENDING site's hours, because the
# lender is the one losing the time, and the mileage is a real cost on
# somebody's expense line. Leaving both out would make borrowing look free
# and the plan would borrow constantly instead of hiring.
#
# Two rules keep it realistic:
#   - never across clusters, because the drive is the point of a cluster;
#   - never under four hours (a half day), because nobody drives half an
#     hour each way to cover one hour of work.
# ---------------------------------------------------------------------------


# Column types are declared rather than inferred, because a plan with no
# borrows at all is a legitimate outcome - and an untyped empty frame would
# quietly turn every hours column into an object column downstream.
BORROW_COLUMNS = {
    "week_index": "int64",
    "day_of_week": "object",
    "grain": "object",
    "role": "object",
    "cluster": "object",
    "short_site": "object",
    "covering_site": "object",
    "hours": "float64",
    "person_days": "int64",
    "travel_hours": "float64",
    "travel_miles": "float64",
    "travel_cost": "float64",
}


def borrow_frame(rows):
    """A borrows table with stable columns and dtypes, empty or not."""
    frame = pd.DataFrame(rows, columns=list(BORROW_COLUMNS))
    return frame.astype(BORROW_COLUMNS)


def travel_pairs(sites=None):
    """One-way minutes and miles between sites, both directions (spec 6.7).

    The YAML holds each pair once; this returns it keyed both ways so the
    lookup does not care which site is lending.
    """
    sites = sites or config.load_sites()
    of_site = cluster_lookup(sites)[1]
    pairs = {}
    for entry in sites["travel"]["pairs"]:
        a, b = entry["sites"]
        if of_site[a] != of_site[b]:
            raise ValueError(f"travel pair {a}-{b} crosses clusters (spec 6.2)")
        leg = {"minutes": float(entry["minutes"]), "miles": float(entry["miles"])}
        pairs[(a, b)] = leg
        pairs[(b, a)] = dict(leg)
    return pairs


def cluster_lookup(sites=None):
    """(members by cluster, cluster of each site) from sites.yaml."""
    sites = sites or config.load_sites()
    members = {c["code"]: list(c["sites"]) for c in sites["clusters"]}
    of_site = {site: cluster for cluster, group in members.items() for site in group}
    return members, of_site


def productive_hours_per_person_day(role, assumptions=None):
    """What one person actually delivers in a day (spec 5.1).

    Paid hours less the role's shrinkage. This is the unit a borrow is
    counted in: a visiting Records Specialist covers 5.76 productive hours
    in a day, not 8, so a 12-hour borrow is three person-days of travel and
    not one and a half.
    """
    assumptions = assumptions or config.load_assumptions()
    fte = assumptions["fte"]
    per_day = fte["paid_hours_per_week"] / assumptions["calendar"]["business_days_per_week"]
    return per_day * (1.0 - fte["paid_shrinkage"][role])


def borrow_offer(
    shortfall, surplus, one_way_minutes, hours_per_person_day, minimum_hours
):
    """The biggest borrow that fits in the lender's surplus once travel is paid.

    Returns (hours borrowed, person-days, travel hours).

    The constraint that matters:

        hours borrowed + travel hours <= the lender's surplus

    so a borrow never pushes the LENDER into a shortfall. Travel is charged
    per person-day because each covering person drives there and back, so
    the cost is a step function, not a smooth one - and that is exactly why
    the answer has to be searched over the number of trips rather than
    solved in one line. Fewer trips mean less travel but cap the hours one
    visitor can deliver; more trips deliver more hours and burn more of the
    surplus getting there.

    A borrow below the minimum returns nothing at all, which is the point
    of the minimum: it stops the model buying coverage no supervisor would
    authorise.
    """
    if shortfall < minimum_hours or surplus <= 0 or hours_per_person_day <= 0:
        return 0.0, 0, 0.0

    round_trip = 2.0 * one_way_minutes / MINUTES_PER_HOUR
    ceiling = min(shortfall, surplus)
    most_days = int(math.ceil(ceiling / hours_per_person_day - 1e-9))

    best = (0.0, 0, 0.0)
    for days in range(1, most_days + 1):
        travel = days * round_trip
        hours = min(shortfall, surplus - travel, days * hours_per_person_day)
        if hours < minimum_hours - 1e-9:
            continue
        # Skip a trip count that delivers no more than one fewer trip could:
        # that option was already scored, and with less travel.
        if days > 1 and hours <= (days - 1) * hours_per_person_day + 1e-9:
            continue
        if hours > best[0] + 1e-12:
            best = (hours, days, travel)
    return best


def allocate_borrowing(shortfalls, surpluses, role, assumptions=None, sites=None):
    """Spec 6.2 step 1: fill short sites from their own cluster.

    `shortfalls` and `surpluses` are {site: hours} for one role at one
    grain (a day for the counter, a week for a back-office queue).

    Worst-short site first, and for each, the NEAREST lender first - a
    coverage plan a supervisor would actually make. Nearest first is also
    the cheapest, because travel is the only thing borrowing costs.

    Returns a list of borrow records; the caller nets them off its own
    hours. Nothing is mutated, so a scenario can re-run the same inputs.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    members, of_site = cluster_lookup(sites)
    pairs = travel_pairs(sites)

    travel = sites["travel"]
    minimum = float(travel["minimum_borrow_hours"])
    rate = float(travel["mileage_rate"])
    per_day = productive_hours_per_person_day(role, assumptions)

    spare = {site: float(hours) for site, hours in surpluses.items()}
    borrows = []

    # Worst first, with the site code breaking ties so the result is stable.
    short_sites = sorted(
        ((site, float(hours)) for site, hours in shortfalls.items() if hours > 0),
        key=lambda item: (-item[1], item[0]),
    )

    for site, need in short_sites:
        cluster = of_site.get(site)
        if cluster is None:
            continue  # a network pool has no cluster and nobody to borrow from
        remaining = need
        lenders = sorted(
            (other for other in members[cluster] if other != site and spare.get(other, 0.0) > 0),
            key=lambda other: (pairs[(site, other)]["minutes"], other),
        )
        for lender in lenders:
            if remaining < minimum - 1e-9:
                break
            leg = pairs[(site, lender)]
            hours, days, travel_hours = borrow_offer(
                remaining, spare[lender], leg["minutes"], per_day, minimum
            )
            if hours <= 0:
                continue
            miles = days * 2.0 * leg["miles"]
            borrows.append(
                {
                    "role": role,
                    "short_site": site,
                    "covering_site": lender,
                    "cluster": cluster,
                    "hours": hours,
                    "person_days": days,
                    "travel_hours": travel_hours,
                    "travel_miles": miles,
                    "travel_cost": miles * rate,
                }
            )
            spare[lender] -= hours + travel_hours
            remaining -= hours
    return borrows


# ---------------------------------------------------------------------------
# Shared inputs
#
# Both builders need the same four things, and two of them are expensive, so
# they are built once. The ORDER matters: Records Specialist help reduces
# what the counter needs, and the help that is actually used reduces what
# the Records Specialists have left for their own queue. Each side has to
# see the other's answer.
# ---------------------------------------------------------------------------


def demand_side(assumptions=None, sites=None, work=None):
    """Everything that does not depend on who is on the payroll.

    Split out because the hiring loop rebuilds the supply side once per
    class it places, and these three are the expensive parts: work volume
    takes 3 seconds and the interval build runs Erlang C over 28,000
    site-day-intervals. Neither changes when a hiring class is added - the
    work arrives whether or not anybody is there to do it, which is rather
    the point of the whole model.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if work is None:
        work = work_volume.build(assumptions, sites)["work"]

    plan_weeks, mondays = supply.plan_window(assumptions)
    built_hours = backlog.required_hours(assumptions, sites, work)
    required = built_hours["hours"]
    queues = backlog.build_queues(assumptions, sites, required)
    queues = queues[queues["week_index"].isin(plan_weeks)]

    plan_work = work[work["week_index"].isin(plan_weeks)]
    onsite = intervals.onsite_intervals(assumptions, sites, plan_work)

    return {
        "work": work,
        "weeks": list(plan_weeks),
        "mondays": mondays,
        "required": required,
        # Carried so 4.4's requester view can put the automated items back
        # into the arrival rate without rebuilding the automation plan.
        # None when automation is off, which is what keeps the baseline's
        # turnaround file identical.
        "automated": built_hours.get("automation"),
        # The proof-of-concept gate series (spec 7, 11/02), carried so the
        # report prints the very weeks the hours were priced from. None in
        # every scenario that did not switch the gate on.
        "proof_of_concept": built_hours.get("proof_of_concept"),
        "queues": queues,
        "onsite_hours": onsite["hours"],
        "raw_intervals": onsite["intervals"],
    }


def prepare(assumptions=None, sites=None, work=None, cohorts=None, demand=None):
    """Required hours, queues, the ledger, and the helped onsite schedule.

    `cohorts` lets a caller plan a different roster - which is exactly what
    the hiring loop does, one class at a time. `demand` lets it reuse the
    supply-independent half instead of recomputing it on every pass.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    demand = demand or demand_side(assumptions, sites, work)

    built = supply.build_ledger(assumptions, sites, cohorts=cohorts)
    required = demand["required"]
    spare = supply.records_specialist_spare_hours(built["daily"], required, assumptions)
    helped = supply.apply_records_specialist_help(
        demand["raw_intervals"], spare, assumptions
    )

    return {
        **demand,
        "ledger": built,
        "onsite_intervals": helped,
        "spare": spare,
        "cohorts": cohorts,
    }


def _help_hours_by_week(helped):
    """Hours Records Specialists actually spent at the counter, per week and site."""
    return helped.groupby(["week_index", "site"])["help_hours"].sum()


# ---------------------------------------------------------------------------
# The back office: weekly (spec 6.2, decided 10/26)
# ---------------------------------------------------------------------------


def queue_key(row):
    """What makes one backlog queue distinct from another (spec 4.3).

    Role, scope, work type and medium are not enough once automation is on
    (5.5): the human line and the exception or sign-off line for the same
    work type sit on the SAME owning role, and keying without the kind
    collapsed them - the later row silently replaced the earlier one and a
    whole queue vanished from the projection. So the WORK KIND is in the
    key, and that is what separates them.

    The HANDLE TIME IS NOT, and must never be. Found 10/28: this key is
    what the rolling backlog is stored under, so every field in it has to
    be a property of the QUEUE and not of the week. The handle time is a
    property of the week - 7.1's mix uplift is priced from the share
    actually removed, which moves with the ramp and with uptime - so
    including it meant a queue got a new key whenever its handle time
    changed, the state lookup missed, and the queue silently restarted at
    its target backlog with its real position thrown away.

    It was invisible in the baseline, where the handle time never moves,
    and it broke 196 of 3,519 week-to-week rolls in the full AI run and
    820 in the half-realized one - which is why that run looked worse than
    doing nothing. A test now asserts the roll holds in every scenario.

    Nothing is lost by dropping it: (week, role, scope, work type, medium,
    work kind) is already unique in every run, and a test asserts that too.
    """
    return (
        row.role,
        row.scope,
        row.work_type,
        row.media,
        getattr(row, "work_kind", "human"),
    )


def build_plan(assumptions=None, sites=None, work=None, unconstrained=False, prepared=None):
    """Gap, coverage, backlog consequence and cost for the back-office roles.

    Walks the plan year one week at a time, because the backlog rolls: this
    week's shortfall is next week's opening position, and next week's
    required hours include a slice of it. That coupling is the whole reason
    the projection cannot be vectorised across weeks.

    Within each week the work happens in three passes, and it has to, because
    coverage is a cluster-level decision: every scope's position has to be
    known before any borrowing can be arranged, and the borrowing has to be
    settled before anyone buys overtime.

        1. what each scope needs, and what it has
        2. cluster borrowing between the scopes of a per-site role
        3. overtime, the uncovered remainder, and the queue consequence

    `unconstrained=True` gives every queue exactly the hours it asked for,
    which is the FULLY STAFFED counterfactual. It is what makes the
    shortfall's effect measurable: turnaround moves with holidays and
    demand as well as with staffing, so the honest question is not "how bad
    was turnaround" but "how much worse was it than it needed to be".
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    prepared = prepared or prepare(assumptions, sites, work)

    settings = assumptions["backlog"]
    business_days = assumptions["calendar"]["business_days_per_week"]
    catch_up = settings["catch_up_weeks"]
    paid_hours = assumptions["fte"]["paid_hours_per_week"]
    day_shares = intervals.day_of_week_shares(assumptions)
    role_settings = assumptions["supply"]["roles"]
    per_site = set(assumptions["supply"]["per_site_roles"])
    roles = set(back_office_roles(assumptions))

    queues = prepared["queues"]
    queues = queues[queues["role"].isin(roles)]

    ledger = prepared["ledger"]["ledger"].set_index(["week_index", "role", "scope"])
    daily = prepared["ledger"]["daily"]
    available = daily[daily["role"].isin(roles)].set_index(
        ["week_index", "role", "scope", "day_of_week"]
    )["productive_hours"]

    # Hours the Records Specialists gave to the counter are gone from their
    # own queue (spec 4.5). Only spare hours are ever offered, so this can
    # never push them below their own required work - but it has to come off
    # the total, or the same hour would be spent twice.
    help_hours = _help_hours_by_week(prepared["onsite_intervals"])

    weeks = sorted(queues["week_index"].unique())
    labelled_kinds = "work_kind" in queues.columns

    # Opening backlog: every queue starts at its target (spec 4.3).
    first = queues[queues["week_index"] == weeks[0]]
    state = {queue_key(row): row.target_backlog for row in first.itertuples()}

    gap_rows, queue_rows, cost_rows, borrow_rows = [], [], [], []

    for week in weeks:
        this_week = queues[queues["week_index"] == week]

        # --- pass 1: what each scope needs, and what it has ---------------
        positions = {}
        for (role, scope), group in this_week.groupby(["role", "scope"]):
            needs = {}
            for row in group.itertuples():
                key = queue_key(row)
                opening = state.get(key, row.target_backlog)
                needs[key] = {
                    "row": row,
                    "opening": opening,
                    "required_hours": float(
                        backlog.required_hours_for_week(
                            row.arrivals, opening, row.handle_time_minutes,
                            row.goal_business_days, catch_up, business_days,
                        )
                    ),
                }
            required_week = sum(item["required_hours"] for item in needs.values())

            by_day = daily_split(required_week, day_shares)
            available_by_day = {
                day: float(available.get((week, role, scope, day), 0.0)) for day in by_day
            }
            lent_to_counter = (
                float(help_hours.get((week, scope), 0.0))
                if role == "records_specialist"
                else 0.0
            )
            available_week = sum(available_by_day.values()) - lent_to_counter

            positions[(role, scope)] = {
                "needs": needs,
                "required": required_week,
                "by_day": by_day,
                "available_by_day": available_by_day,
                "available": available_week,
                "help_to_counter": lent_to_counter,
                "group": group,
            }

        # --- pass 2: cluster borrowing (spec 6.2 step 1) ------------------
        #
        # Only for per-site roles. A network pool has no second site to
        # borrow from: when you already are every site, step 1 is a no-op
        # and the order falls straight through to overtime.
        borrowed_in = {key: 0.0 for key in positions}
        lent_out = {key: 0.0 for key in positions}
        for role in sorted({r for r, _s in positions} & per_site):
            scopes = [s for r, s in positions if r == role]
            shortfalls = {
                scope: max(0.0, positions[(role, scope)]["required"]
                           - positions[(role, scope)]["available"])
                for scope in scopes
            }
            surpluses = {
                scope: max(0.0, positions[(role, scope)]["available"]
                           - positions[(role, scope)]["required"])
                for scope in scopes
            }
            for borrow in allocate_borrowing(shortfalls, surpluses, role, assumptions, sites):
                borrow = {**borrow, "week_index": week, "day_of_week": ALL_WEEK,
                          "grain": "week"}
                borrow_rows.append(borrow)
                borrowed_in[(role, borrow["short_site"])] += borrow["hours"]
                lent_out[(role, borrow["covering_site"])] += (
                    borrow["hours"] + borrow["travel_hours"]
                )

        # --- pass 3: overtime, the remainder, and the consequence ---------
        for (role, scope), position in sorted(positions.items()):
            required_week = position["required"]
            available_week = position["available"] - lent_out[(role, scope)]
            covered_week = available_week + borrowed_in[(role, scope)]

            headcount = float(ledger.loc[(week, role, scope), "starting_headcount"])
            cap = role_settings[role]["overtime_cap"] * headcount * paid_hours
            shortfall_week = max(0.0, required_week - covered_week)
            overtime_week = min(shortfall_week, cap)
            uncovered_week = shortfall_week - overtime_week

            # Hours applied to the queues are capped at what they asked for.
            # "Required" in 4.3 means enough to hold the backlog AT TARGET,
            # not enough to empty it: a healthy back-office queue is not a
            # queue of zero. Without the cap a week of slack would drive
            # every backlog to nothing and turnaround would collapse below
            # the goal, which is not what a team with spare hours does -
            # they stop early, or the hours go somewhere else.
            effective = (
                required_week
                if unconstrained
                else min(covered_week + overtime_week, required_week)
            )
            # The same arithmetic with step 1 and step 2 switched off, which
            # is what "before cover" in the event log means.
            before_cover = required_week if unconstrained else min(available_week, required_week)

            for day in position["by_day"]:
                gap_rows.append(
                    {
                        "week_index": week,
                        "week_start": position["group"]["week_start"].iloc[0],
                        "role": role,
                        "scope": scope,
                        "day_of_week": day,
                        "required_hours": position["by_day"][day],
                        "available_hours": position["available_by_day"][day],
                        # Reported, not acted on: the queue absorbs this.
                        "daily_mismatch_hours": (
                            position["by_day"][day] - position["available_by_day"][day]
                        ),
                        "week_shortfall_hours": shortfall_week,
                        "week_borrowed_hours": borrowed_in[(role, scope)],
                        "week_lent_hours": lent_out[(role, scope)],
                        "week_overtime_hours": overtime_week,
                        "week_uncovered_hours": uncovered_week,
                    }
                )

            # --- run the backlog forward (spec 4.3) -----------------------
            #
            # Effective hours are shared across the role's queues in
            # proportion to what each one asked for. No priority order is
            # stated for the remote pipeline in 4.5 - that rule is about
            # the onsite team - so every queue falls behind at the same
            # rate rather than one being starved to protect another.
            inventory_before, inventory_after, weight, delayed = 0.0, 0.0, 0.0, 0.0
            for key, item in position["needs"].items():
                row = item["row"]
                share = (
                    item["required_hours"] / required_week if required_week > 0 else 0.0
                )
                hours = effective * share
                outcome = backlog.project_week(
                    row.arrivals, item["opening"], hours, row.handle_time_minutes,
                    row.goal_business_days, business_days,
                )
                # The counterfactual: the same week without borrowing or
                # overtime. Scored, never stored - the state that rolls
                # forward is always the one that actually happened.
                plain = backlog.project_week(
                    row.arrivals, item["opening"], before_cover * share,
                    row.handle_time_minutes, row.goal_business_days, business_days,
                )
                state[key] = float(outcome["ending_backlog"])
                weight += row.arrivals
                inventory_after += float(outcome["days_of_inventory"]) * row.arrivals
                inventory_before += float(plain["days_of_inventory"]) * row.arrivals
                delayed += max(
                    0.0, float(outcome["ending_backlog"]) - float(row.target_backlog)
                )
                queue_rows.append(
                    {
                        "week_index": week,
                        "week_start": row.week_start,
                        "role": role,
                        "scope": scope,
                        "stage": row.stage,
                        "work_type": row.work_type,
                        "media": row.media,
                        "arrivals": row.arrivals,
                        "goal_business_days": row.goal_business_days,
                        "opening_backlog": item["opening"],
                        "target_backlog": row.target_backlog,
                        "required_hours": item["required_hours"],
                        "allocated_hours": hours,
                        "completed": float(outcome["completed"]),
                        "ending_backlog": float(outcome["ending_backlog"]),
                        "days_of_inventory": float(outcome["days_of_inventory"]),
                        "goal_met": bool(outcome["goal_met"]),
                    }
                )
                # Carried through only when automation is on, so 4.4 can
                # leave the loop lines out of the turnaround weighting
                # (decided 10/27). A baseline file that read "human" on
                # every row would be a column of noise.
                if labelled_kinds:
                    queue_rows[-1]["work_kind"] = row.work_kind

            # --- cost (spec 6.4) ------------------------------------------
            hourly = role_settings[role]["hourly_cost"]
            overtime_rate = role_settings[role]["overtime_pay_multiple"]
            cost_rows.append(
                {
                    "week_index": week,
                    "role": role,
                    "scope": scope,
                    "paid_headcount": headcount,
                    "overtime_cap_hours": cap,
                    "shortfall_hours": shortfall_week,
                    "regular_cost": headcount * paid_hours * hourly,
                    "overtime_hours": overtime_week,
                    "overtime_cost": overtime_week * hourly * overtime_rate,
                    "available_hours": available_week,
                    "borrowed_hours": borrowed_in[(role, scope)],
                    "lent_hours": lent_out[(role, scope)],
                    "covered_hours": covered_week,
                    "required_hours": required_week,
                    "uncovered_hours": uncovered_week,
                    "help_to_counter_hours": position["help_to_counter"],
                    "days_of_inventory_before": (
                        inventory_before / weight if weight > 0 else 0.0
                    ),
                    "days_of_inventory_after": (
                        inventory_after / weight if weight > 0 else 0.0
                    ),
                    "requests_delayed": delayed,
                    "short": uncovered_week > 1e-9,
                }
            )

    costs = pd.DataFrame(cost_rows)
    costs["total_cost"] = costs["regular_cost"] + costs["overtime_cost"]
    borrows = borrow_frame(borrow_rows)
    return {
        "gap": pd.DataFrame(gap_rows),
        "queues": pd.DataFrame(queue_rows),
        "cost": costs,
        "borrows": borrows,
    }


# ---------------------------------------------------------------------------
# The counter: daily (spec 6.2, 6.3)
#
# The requirement is what spec 5.1 settled on 10/26: the scheduled live-work
# hours the Onsite Associates are left with AFTER Records Specialist help,
# plus any email that would not fit into idle time. Required coverage in
# each interval is unchanged by the help - Erlang C still says the same
# number of people must be on the floor - so what the help changes is who
# pays for them.
#
# The consequence is spec 6.3: service level recomputed with Erlang C at the
# staff actually available. Fractional staff is floored, because a counter
# cannot be covered by seven tenths of a person, and flooring is the honest
# direction to round.
# ---------------------------------------------------------------------------


def onsite_requirement(prepared, assumptions=None):
    """Required and available hours for the counter, by week, site and day."""
    assumptions = assumptions or config.load_assumptions()
    live = (
        prepared["onsite_intervals"]
        .groupby(["week_index", "site", "day_of_week"], as_index=False)
        .agg(live_hours=("scheduled_hours", "sum"))
    )
    email = prepared["onsite_hours"][
        ["week_index", "week_start", "site", "day_of_week", "email_paid_hours"]
    ]
    table = live.merge(email, on=["week_index", "site", "day_of_week"], how="left")
    table["required_hours"] = table["live_hours"] + table["email_paid_hours"].fillna(0.0)

    daily = prepared["ledger"]["daily"]
    mine = daily[daily["role"] == "onsite_associate"].rename(columns={"scope": "site"})
    table = table.merge(
        mine[["week_index", "site", "day_of_week", "productive_hours"]],
        on=["week_index", "site", "day_of_week"],
        how="left",
    ).rename(columns={"productive_hours": "available_hours"})
    table["available_hours"] = table["available_hours"].fillna(0.0)
    return table


def _interval_bundle(helped, assumptions):
    """Per site-day arrays Erlang C needs to re-score a short day (spec 6.3).

    TWO queues, because a scenario can split them. Blended (the baseline,
    4.1) answers both channels from one pooled team, so the phone columns
    are the pooled ones. Centralized phones (spec 7, 10/27) answers calls
    from a network queue and walk-ins from the counter, so they are two
    different queues with two different handle times and the service level
    has to be read off whichever one each channel actually sits in.
    """
    goals = {g["code"]: g for g in assumptions["live_queues"]["goals"]}
    phones = next(g for g in goals.values() if g["code"] == "phones")
    walk_ins = next(g for g in goals.values() if g["code"] == "walk_ins")

    bundle = {}
    columns = [
        "required_staff", "traffic_pooled", "pooled_handle_seconds",
        "walk_in_arrivals", "call_arrivals",
    ]
    central = "central_required_staff" in helped.columns
    for key, group in helped.groupby(["week_index", "site", "day_of_week"]):
        entry = {name: group[name].to_numpy() for name in columns}
        # The counter's queue is what the site staffs either way.
        entry["counter_staff"] = group["required_counter"].to_numpy()
        entry["counter_traffic"] = group["traffic_counter"].to_numpy()
        if central and group["central_required_staff"].notna().all():
            entry["phone_staff"] = group["central_required_staff"].to_numpy()
            entry["phone_traffic"] = group["central_traffic"].to_numpy()
            entry["phone_handle"] = group["central_handle_seconds"].to_numpy()
            entry["counter_handle"] = np.full(
                len(group), float(group["pooled_handle_seconds"].iloc[0])
            )
            entry["central"] = True
        else:
            # Blended: one pooled team answers both.
            entry["phone_staff"] = entry["required_staff"]
            entry["phone_traffic"] = entry["traffic_pooled"]
            entry["phone_handle"] = entry["pooled_handle_seconds"]
            entry["counter_staff"] = entry["required_staff"]
            entry["counter_traffic"] = entry["traffic_pooled"]
            entry["counter_handle"] = entry["pooled_handle_seconds"]
            entry["central"] = False
        bundle[key] = entry
    return bundle, phones, walk_ins


def _blended_service_level(staffed, offered, handle, target_seconds):
    """Erlang C at a FRACTIONAL staffing, by blending the two whole numbers.

    Erlang C only answers for whole people, and rightly so - half a person
    cannot take a call. But a day that was 99% covered is not a day that
    was one person short in every interval, and flooring would say it was,
    turning a rounding remainder into a service-level collapse.

    So 2.4 people means the interval ran at 2 for 60% of the half-hour and
    at 3 for the other 40%, and its service level is the blend. At a whole
    number the weight is zero and this is exactly Erlang C again, which is
    the property that matters: a fully covered day reproduces the service
    levels the schedule was designed to deliver, so every miss the plan
    reports is a real miss.
    """
    low = np.floor(staffed + 1e-9).astype(int)
    weight = np.clip(staffed - low, 0.0, 1.0)
    out = np.empty(len(low))
    for i, (n, w, a, h) in enumerate(zip(low, weight, offered, handle)):
        if h <= 0:
            out[i] = 1.0
            continue
        lower = erlang.service_level(n, a, target_seconds, h)
        upper = erlang.service_level(n + 1, a, target_seconds, h) if w > 0 else lower
        out[i] = (1.0 - w) * lower + w * upper
    return out


def service_levels(day_intervals, ratio, phones, walk_ins, phone_ratio=None):
    """Erlang C at the staff actually on the floor (spec 6.3).

    `ratio` is the share of the day's required hours that were actually
    covered, and the staff in each interval scale with it. Scaling rather
    than taking the shortfall out of the quiet hours is the pessimistic and
    honest reading: a team that is 10% short is 10% short at 9am too, when
    it can least afford to be.

    `phone_ratio` is the same thing for the phone queue when it is a
    separate team. With centralized phones (spec 7) the queue is staffed
    from a NETWORK pool, so it degrades with the network's coverage rather
    than with the site's - a site that is short does not make the central
    queue short on its own. Left out, the phones share the counter's
    ratio, which is the blended baseline.

    Weighted by each channel's own arrivals, because the service level a
    requester experiences is the one in the interval they turned up in.
    """
    walk_in_arrivals = day_intervals["walk_in_arrivals"]
    call_arrivals = day_intervals["call_arrivals"]

    counter_ratio = min(1.0, max(0.0, ratio))
    phones_ratio = counter_ratio if phone_ratio is None else min(1.0, max(0.0, phone_ratio))

    phone_sl = _blended_service_level(
        day_intervals["phone_staff"] * phones_ratio,
        day_intervals["phone_traffic"],
        day_intervals["phone_handle"],
        phones["target_seconds"],
    )
    counter_sl = _blended_service_level(
        day_intervals["counter_staff"] * counter_ratio,
        day_intervals["counter_traffic"],
        day_intervals["counter_handle"],
        walk_ins["target_seconds"],
    )

    calls_total = float(call_arrivals.sum())
    walk_ins_total = float(walk_in_arrivals.sum())
    return {
        "phones": float((phone_sl * call_arrivals).sum() / calls_total) if calls_total > 0 else 1.0,
        "walk_ins": (
            float((counter_sl * walk_in_arrivals).sum() / walk_ins_total)
            if walk_ins_total > 0
            else 1.0
        ),
        "calls": calls_total,
        "walk_in_contacts": walk_ins_total,
        "requests_delayed": (
            float((call_arrivals * (1.0 - phone_sl)).sum())
            + float((walk_in_arrivals * (1.0 - counter_sl)).sum())
        ),
    }


def build_onsite(assumptions=None, sites=None, work=None, prepared=None):
    """Daily gap, cluster coverage, overtime and service level for the counter.

    The day is the unit of decision here, but the overtime CAP is weekly -
    10% of the role's paid hours - so the week is still the unit of budget.
    That is why the days of a week are solved together: the budget has to be
    shared out across the days that need it, and a Monday decision made in
    ignorance of Thursday would be an artefact of iteration order.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    prepared = prepared or prepare(assumptions, sites, work)

    requirement = onsite_requirement(prepared, assumptions)
    bundle, phones, walk_ins = _interval_bundle(prepared["onsite_intervals"], assumptions)

    ledger = prepared["ledger"]["ledger"].set_index(["week_index", "role", "scope"])
    role_settings = assumptions["supply"]["roles"]["onsite_associate"]
    paid_hours = assumptions["fte"]["paid_hours_per_week"]
    _members, of_site = cluster_lookup(sites)

    position = requirement.set_index(["week_index", "site", "day_of_week"])
    site_codes = sorted(requirement["site"].unique())
    weeks = sorted(requirement["week_index"].unique())

    rows, borrow_rows = [], []

    for week in weeks:
        # --- pass 1 and 2: the day's position, then borrowing -------------
        day_state, borrowed_in, lent_out = {}, {}, {}
        for day in WEEKDAYS:
            needs, has = {}, {}
            for site in site_codes:
                key = (week, site, day)
                if key not in position.index:
                    continue
                cell = position.loc[key]
                needs[site] = float(cell["required_hours"])
                has[site] = float(cell["available_hours"])
            day_state[day] = {"required": needs, "available": has}

            shortfalls = {s: max(0.0, needs[s] - has[s]) for s in needs}
            surpluses = {s: max(0.0, has[s] - needs[s]) for s in needs}
            borrowed_in[day] = {s: 0.0 for s in needs}
            lent_out[day] = {s: 0.0 for s in needs}
            for borrow in allocate_borrowing(
                shortfalls, surpluses, "onsite_associate", assumptions, sites
            ):
                borrow = {**borrow, "week_index": week, "day_of_week": day,
                          "grain": "day"}
                borrow_rows.append(borrow)
                borrowed_in[day][borrow["short_site"]] += borrow["hours"]
                lent_out[day][borrow["covering_site"]] += (
                    borrow["hours"] + borrow["travel_hours"]
                )

        # --- the network's own coverage, for a shared phone queue -------
        #
        # A central phone team is staffed from the network pool (spec 7),
        # so it degrades with the network's coverage and not with any one
        # site's. Computed per day across all six sites before the
        # per-site loop, because every site needs the same figure.
        network_ratio = {}
        for day in WEEKDAYS:
            need = sum(
                day_state[day]["required"].get(site, 0.0) for site in site_codes
            )
            have = sum(
                day_state[day]["available"].get(site, 0.0) for site in site_codes
            )
            network_ratio[day] = min(1.0, have / need) if need > 0 else 1.0

        # --- pass 3: overtime, within the week's cap, then the outcome ----
        for site in site_codes:
            headcount = float(ledger.loc[(week, "onsite_associate", site), "starting_headcount"])
            cap = role_settings["overtime_cap"] * headcount * paid_hours
            after_borrow = {}
            for day in WEEKDAYS:
                if site not in day_state[day]["required"]:
                    continue
                need = day_state[day]["required"][site]
                have = (
                    day_state[day]["available"][site]
                    - lent_out[day][site]
                    + borrowed_in[day][site]
                )
                after_borrow[day] = max(0.0, need - have)
            overtime = allocate_overtime(after_borrow, cap)

            for day, shortfall_after_borrow in after_borrow.items():
                need = day_state[day]["required"][site]
                raw_available = day_state[day]["available"][site] - lent_out[day][site]
                covered = raw_available + borrowed_in[day][site] + overtime[day]
                uncovered = max(0.0, need - covered)

                ratio_before = min(1.0, raw_available / need) if need > 0 else 1.0
                ratio_after = min(1.0, covered / need) if need > 0 else 1.0
                day_bundle = bundle[(week, site, day)]
                shared = network_ratio[day] if day_bundle["central"] else None
                before = service_levels(
                    day_bundle, ratio_before, phones, walk_ins, phone_ratio=shared
                )
                after = service_levels(
                    day_bundle, ratio_after, phones, walk_ins, phone_ratio=shared
                )

                rows.append(
                    {
                        "week_index": week,
                        "role": "onsite_associate",
                        "cluster": of_site[site],
                        "site": site,
                        "day_of_week": day,
                        "paid_headcount": headcount,
                        "required_hours": need,
                        "available_hours": raw_available,
                        "lent_hours": lent_out[day][site],
                        "borrowed_hours": borrowed_in[day][site],
                        "shortfall_hours": max(0.0, need - day_state[day]["available"][site]
                                               + lent_out[day][site]),
                        "shortfall_after_borrow_hours": shortfall_after_borrow,
                        "overtime_cap_hours": cap,
                        "overtime_hours": overtime[day],
                        "covered_hours": covered,
                        "uncovered_hours": uncovered,
                        "service_level_phones_before": before["phones"],
                        "service_level_phones_after": after["phones"],
                        "service_level_walk_ins_before": before["walk_ins"],
                        "service_level_walk_ins_after": after["walk_ins"],
                        "requests_delayed": after["requests_delayed"],
                        "calls": after["calls"],
                        "walk_ins": after["walk_in_contacts"],
                        "short": uncovered > 1e-9,
                    }
                )

    daily = pd.DataFrame(rows)
    hourly = role_settings["hourly_cost"]
    daily["overtime_cost"] = daily["overtime_hours"] * hourly * role_settings["overtime_pay_multiple"]
    borrows = borrow_frame(borrow_rows)
    return {"daily": daily, "borrows": borrows}


# The scope name the digitization project's added hours are booked
# under. A scope rather than a role, so the hours carry the Records
# Specialist rate while never being mistaken for the Records Specialists
# who answer the retrieval queue.
DIGITIZATION_SCOPE = "digitization_project"


def plan_cost(prepared, back_office, onsite, assumptions=None):
    """Regular and overtime cost by role, scope and week (spec 6.4).

    Regular cost pays EVERYBODY on the payroll, including the people still
    in compliance training who produce nothing. That is not an accounting
    quirk: it is precisely why hiring late is expensive, and a cost model
    that only paid productive people would hide the whole argument for
    planning ahead.
    """
    assumptions = assumptions or config.load_assumptions()
    paid_hours = assumptions["fte"]["paid_hours_per_week"]
    settings = assumptions["supply"]["roles"]

    ledger = prepared["ledger"]["ledger"]
    overtime = {}
    for row in back_office["cost"].itertuples():
        overtime[(row.week_index, row.role, row.scope)] = (
            row.overtime_hours, row.overtime_cost
        )
    live = onsite["daily"].groupby(["week_index", "site"], as_index=False).agg(
        overtime_hours=("overtime_hours", "sum"),
        overtime_cost=("overtime_cost", "sum"),
    )
    for row in live.itertuples():
        overtime[(row.week_index, "onsite_associate", row.site)] = (
            row.overtime_hours, row.overtime_cost
        )

    rows = []
    for row in ledger.itertuples():
        hourly = settings[row.role]["hourly_cost"]
        hours, cost = overtime.get((row.week_index, row.role, row.scope), (0.0, 0.0))
        rows.append(
            {
                "week_index": row.week_index,
                "role": row.role,
                "scope": row.scope,
                "paid_headcount": row.starting_headcount,
                "regular_cost": row.starting_headcount * paid_hours * hourly,
                "overtime_hours": hours,
                "overtime_cost": cost,
            }
        )
    table = pd.DataFrame(rows)
    table["total_cost"] = table["regular_cost"] + table["overtime_cost"]

    # Coordinated digitization's added paid hours (spec 7, decided 10/28).
    #
    # Added as their own rows rather than folded into the Records
    # Specialists', because they are not the Records Specialists' hours -
    # they are a project team bought at the same rate. Keeping them
    # separate is what lets the scenario report its own cost and lets a
    # reader see that the retrieval queue was never raided to pay for it.
    #
    # Only when the scenario is on, so the baseline cost file does not
    # change at all.
    if digitization.enabled(assumptions):
        block = digitization.settings(assumptions)
        role = block.get("project_role", "records_specialist")
        hours = float(block.get("hours_per_week", 0.0))
        cost = digitization.weekly_cost(assumptions)
        start = digitization.from_week(assumptions)
        plan_weeks, _mondays = supply.plan_window(assumptions)
        project = pd.DataFrame(
            [
                {
                    "week_index": week,
                    "role": role,
                    "scope": DIGITIZATION_SCOPE,
                    "paid_headcount": hours / paid_hours,
                    "regular_cost": cost,
                    "overtime_hours": 0.0,
                    "overtime_cost": 0.0,
                    "total_cost": cost,
                }
                for week in plan_weeks
                if week >= start
            ]
        )
        table = pd.concat([table, project], ignore_index=True)
    return table


# ---------------------------------------------------------------------------
# Next Dollar: headcount or automation (spec 6.6)
#
# The question a leader actually has to answer. The next dollar buys either
# an hour of somebody's time or an item through a machine - which one, and
# how soon does it start helping?
#
# Lead time is half the answer and the half that usually decides it.
# Fourteen weeks from requisition to a productive Tier 1 hire is a long
# time to wait for relief a machine could start delivering in four. A
# comparison of unit costs alone would make automation look like a slow,
# cheap option when it is often a fast one.
#
# The other half is that automated work comes back (spec 5.5). An item run
# through the machine still costs an exception some of the time and a QA
# look some of the time, each at the loaded rate of whoever is paid for it
# - the owning role for an exception, the QA team for a QA look, and a QA
# hour costs more. That is why the automated figure is not simply the
# per-item price.
#
# Fixed costs decide it more often than the per-item fee does (decided
# 10/27). Setup is $10,000 a work type spread over a year, and the shared
# $500-a-week platform fee is split by automated volume, so a row doing
# twelve items a week carries a crushing fixed cost per item and a row
# doing two thousand barely notices it. Both figures are reported - before
# fixed costs, which is the per-item price on its own, and after, which is
# what an item really costs - and the ranking uses the second.
# ---------------------------------------------------------------------------


def productive_hour_cost(role, assumptions=None):
    """What an hour of ACTUAL WORK costs, not an hour of payroll (spec 6.6).

        hourly cost / (1 - shrinkage)

    A Tier 1 hour is $28 on the payroll but only 72% of it reaches the
    queue, so the work costs $38.89 an hour. Comparing a per-item price
    against the payroll rate instead of this one is the most common way an
    automation case gets overstated.
    """
    assumptions = assumptions or config.load_assumptions()
    fte = assumptions["fte"]
    hourly = assumptions["supply"]["roles"][role]["hourly_cost"]
    return hourly / (1.0 - fte["paid_shrinkage"][role])


def human_cost_per_item(role, handle_time_minutes, assumptions=None):
    """Spec 6.6: hourly / (1 - shrinkage) x handle time / 60.

    A general helper: the cost of working one item of THIS length. 6.6 asks
    it for the EASY-item length (decided 10/28), because the items a
    machine takes are the simple ones and the saving is what they would
    have cost by hand - 3 minutes on a W1 intake item, $1.94, not the
    6-minute average's $3.89.

    Kept general rather than folding the easy ratio in here, because the
    caller is the only one who knows whether it is pricing an item the
    machine took or an item a person still works.
    """
    return (
        productive_hour_cost(role, assumptions)
        * handle_time_minutes
        / MINUTES_PER_HOUR
    )


def automated_cost_per_item(
    work_type, role, handle_time_minutes, assumptions=None, sign_off=False
):
    """Spec 6.6, before fixed costs, broken into its parts.

        cost per item
        + exception rate x exception hours x owning role's hour cost
        + QA sample rate x review time    x QA TEAM's hour cost

    The spec's worked example, intake, revised 10/28: $0.50 + 10% x 4.5
    min at the Tier 1 rate ($0.29) + 5% x 2 min at the QA team's $44.44
    hour ($0.07) = $0.87. The 4.5 minutes is 1.5 x the EASY-item time of 3
    minutes, not 1.5 x the 6-minute average, which used to give $1.16.

    Each term is priced at the rate of whoever actually does the work
    (decided 10/27). Exceptions and sign-off go back to the role that owns
    the work type, so they carry that role's loaded rate; the QA sample
    goes to the QA team per 5.5, so it carries theirs - and a QA hour costs
    more than a Tier 1 hour, which is why this is not a rounding question.
    Pricing the QA term at the owning role's rate would have quietly
    flattered automation on every work type the QA team does not own, and
    would have disagreed with the hours the plan actually bills.

    Sign-off is returned as its own number rather than folded in, because
    6.6's formula does not mention it and its worked example has no
    sign-off row. For W7, an R8 release or a notarisation it is the largest
    term of the three, so leaving it invisible would be worse than leaving
    it out of the total.
    """
    assumptions = assumptions or config.load_assumptions()
    loop = automation.settings(assumptions)["human_in_the_loop"]
    row = automation.work_type_settings(work_type, assumptions)
    cost = automation.settings(assumptions)["cost"]
    hour = productive_hour_cost(role, assumptions)

    per_item = (
        cost["ai_per_item"] if row["kind"] == "ai" else cost["rule_per_item"]
    )
    # The easy-item time, not the average (5.5, 6.6, decided 10/28): an
    # exception on an item the machine took is 1.5x a SIMPLE job.
    easy_minutes = automation.easy_item_minutes(handle_time_minutes, assumptions)
    exception_minutes = (
        float(loop["exception_handle_time_multiple"]) * easy_minutes
    )
    exception = (
        float(row.get("exception_rate", 0.0))
        * exception_minutes
        / MINUTES_PER_HOUR
        * hour
    )
    # The QA team's rate, not the owning role's: they do the work (5.5),
    # and their hour costs more.
    qa = (
        float(row.get("qa_sample_rate", 0.0))
        * float(loop["qa_review_minutes"])
        / MINUTES_PER_HOUR
        * productive_hour_cost(automation.QA_ROLE, assumptions)
    )
    sign_off_cost = 0.0
    if sign_off:
        sign_off_cost = (
            float(loop["sign_off_rate"])
            * float(loop["sign_off_handle_time_multiple"])
            * easy_minutes
            / MINUTES_PER_HOUR
            * hour
        )
    return {
        "item_cost": float(per_item),
        "exception_cost": exception,
        "qa_cost": qa,
        "sign_off_cost": sign_off_cost,
        "total": float(per_item) + exception + qa,
    }


def human_lead_weeks(role, assumptions=None):
    """Requisition to a productive hour (spec 6.6).

    Requisition lead time, then onboarding, then ramp to 90% - the same
    threshold 6.5 uses for the hire-by date, so the two outputs cannot
    quote different numbers. Tier 1 comes to 6 + 5 + 3 = 14 weeks, the
    figure named in 6.6.
    """
    assumptions = assumptions or config.load_assumptions()
    threshold = assumptions["next_dollar"]["human_ramp_threshold"]
    settings = supply.role_settings(role, assumptions)
    return (
        int(settings["requisition_weeks"])
        + supply.onboarding_weeks(settings)
        + supply.weeks_to_full_speed(settings, threshold)
    )


def automation_lead_weeks(assumptions=None):
    """Go-live to full capacity (spec 5.4, 6.6).

    The ramp only - 25/50/75/100% is four weeks - because unlike a hire
    there is no requisition queue and no compliance training. This is the
    number that makes automation the FAST option as often as the cheap one.
    """
    return len(automation.settings(assumptions)["ramp"])


def fixed_cost_per_item(automated, week_index=None, assumptions=None):
    """Setup share plus platform-fee share, per automated item (spec 5.4, 6.6).

    `week_index` prices one week at that week's own automated volume;
    omitted, it prices the whole plan year, which is the volume-weighted
    average of the weekly figures and the right basis for a plan-level
    ranking.

    Returns {work type: dollars per item}. A work type the machine never
    touched is absent rather than infinite: there is no per-item cost when
    there are no items, and a NaN would poison the ranking.
    """
    assumptions = assumptions or config.load_assumptions()
    if automated is None or automated.empty:
        return {}
    if week_index is None:
        plan_weeks, _mondays = supply.plan_window(assumptions)
        rows = automated[automated["week_index"].isin(plan_weeks)]
    else:
        rows = automated[automated["week_index"] == week_index]

    out = {}
    for code, group in rows.groupby("work_type"):
        items = group["automated_items"].sum()
        if items <= 0:
            continue
        spend = group["setup_cost"].sum() + group["platform_cost"].sum()
        out[code] = spend / items
    return out


def weekly_cost_comparison(
    work_type, week_index, assumptions=None, sites=None, required=None
):
    """One week's automation cost against doing the same items by hand (6.6).

    Decided 10/27: both sides must cover the SAME items and the SAME work,
    or the comparison is rhetoric. So the automation side carries its
    human-in-the-loop labour - the exception and QA hours, priced at a
    human's productive-hour cost - and its share of the fixed costs,
    alongside the per-item fee.

    Leaving the loop labour off would be comparing a machine that never
    needs a second look against people who do. Leaving the platform share
    off would hand a leader a number that gets worse the moment they ask
    the finance team about it.

    Sign-off is reported and kept OUT of the total, the same way 6.6's
    formula keeps it out: the spec's figure is the three terms it names.
    For W7, an R8 release or a notarisation it is the largest human term,
    so it is shown rather than buried.

    Every hour here is priced at the rate of whoever is paid for it, which
    since 10/27 is also what 6.6's per-item formula does - so this figure
    and next_dollar()'s reconcile exactly, and a test says so.

    The by-hand side values the automated items at the EASY-item handle
    time (decided 10/28), the same as 6.6, because those are the items the
    machine actually took.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if required is None:
        required = backlog.required_hours(assumptions, sites)
    automated = required.get("automation")
    if automated is None or automated.empty:
        return None

    mine = automated[
        (automated["work_type"] == work_type)
        & (automated["week_index"] == week_index)
    ]
    if mine.empty or mine["automated_items"].iloc[0] <= 0:
        return None
    row = mine.iloc[0]
    share = float(row["automated_fraction"])
    if share >= 1.0:
        raise ValueError(
            f"{work_type} week {week_index}: the machine took every item, so "
            "there are no human rows left to read the role and media mix from"
        )

    # The automated share is uniform within a work type and week, so the
    # automated items split across role, media and handle time in exactly
    # the same proportions as the human ones. Scaling the human rows up is
    # therefore the same attribution, done from the side that still has
    # rows in it.
    human = required["human"]
    human = human[
        (human["work_type"] == work_type) & (human["week_index"] == week_index)
    ]
    scale = share / (1.0 - share)
    by_hand = 0.0
    attributed = 0.0
    # Columns by name, not through itertuples: "items" collides with a
    # DataFrame method, so pandas renames that field and a positional
    # fallback silently reads required_hours instead - which is the same
    # number divided by the handle time, and so wrong by a factor nobody
    # would notice on inspection.
    for role, handle_time, items in zip(
        human["role"], human["handle_time_minutes"], human["items"]
    ):
        automated_here = items * scale
        attributed += automated_here
        # Priced at the EASY-item time (6.6, decided 10/28): these are the
        # items the machine took, and it took the simple ones. Pricing them
        # at the average credited automation with a saving on work it was
        # never going to touch, which is the same error 6.6's per-item
        # formula was making - and both are fixed together so the two
        # figures still reconcile.
        #
        # Measured against the ORIGINAL handle time, so the mix uplift
        # this row carries is divided back out first: the easy item is
        # half of 6.00, never half of the 6.26 a human row reads after the
        # uplift (decided 10/28).
        by_hand += automated_here * human_cost_per_item(
            role,
            automation.easy_item_minutes(
                handle_time,
                assumptions,
                code=work_type,
                week_index=week_index,
                removed_share=share,
                uplifted=True,
            ),
            assumptions,
        )

    loop = required["human_in_the_loop"]
    loop = loop[
        (loop["work_type"] == work_type) & (loop["week_index"] == week_index)
    ]
    labour = {kind: {"hours": 0.0, "cost": 0.0} for kind in automation.RETURN_KINDS}
    for entry in loop.itertuples():
        hour = productive_hour_cost(entry.role, assumptions)
        labour[entry.work_kind]["hours"] += entry.required_hours
        labour[entry.work_kind]["cost"] += entry.required_hours * hour

    automation_total = (
        float(row["item_cost"])
        + float(row["setup_cost"])
        + float(row["platform_cost"])
        + labour["exception"]["cost"]
        + labour["qa_sample"]["cost"]
    )
    items = float(row["automated_items"])
    return {
        "work_type": work_type,
        "week_index": week_index,
        "automated_items": items,
        "attributed_items": attributed,
        "item_cost": float(row["item_cost"]),
        "setup_cost": float(row["setup_cost"]),
        "platform_cost": float(row["platform_cost"]),
        "exception_hours": labour["exception"]["hours"],
        "exception_cost": labour["exception"]["cost"],
        "qa_hours": labour["qa_sample"]["hours"],
        "qa_cost": labour["qa_sample"]["cost"],
        # Reported, deliberately outside the total (spec 6.6).
        "sign_off_hours": labour["sign_off"]["hours"],
        "sign_off_cost": labour["sign_off"]["cost"],
        "automation_total": automation_total,
        "by_hand_total": by_hand,
        "saving": by_hand - automation_total,
        "automation_cost_per_item": automation_total / items if items > 0 else 0.0,
        "by_hand_cost_per_item": by_hand / items if items > 0 else 0.0,
    }


def next_dollar(
    assumptions=None, sites=None, work=None, required=None, week_index=None,
    by_role_media=False,
):
    """Human against automated cost per item, by work type (spec 6.6).

    One row per work type, role and medium that actually occurs, so the
    handle time is the real one: W5 is 5 minutes at Tier 1 and 12 at Tier
    2, and W8's retrieval time comes from the media table. A single figure
    per work type would be an average of jobs nobody does.

    Ranked by savings per dollar AFTER fixed costs (decided 10/27), which
    is the number a leader can act on: a work type that saves $1.60 on a
    $2.27 item returns more per dollar spent than one that saves $4 on a
    $10 item, and the next dollar should go to the first. Ranking on the
    before-fixed figure would put a low-volume row near the top on the
    strength of a fee it cannot actually achieve.

        savings per dollar = (human cost - automated cost) / automated cost

    with both sides after fixed costs.

    `week_index` prices the fixed costs at one week's volume; omitted, the
    whole plan year, which is the volume-weighted average.

    A work type the machine never touched - W11 and W24 are pinned to 0%
    by spec Section 1 - is reported with no after-fixed figures at all and
    sorts last. There is no cost per item when there are no items, and
    calling it zero would put the two rows automation is forbidden from
    touching at the top of the ranking.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if required is None:
        required = backlog.required_hours(assumptions, sites, work)
    hours = required["human"] if "human" in required else required["hours"]
    automated = required.get("automation")

    plan_weeks, _mondays = supply.plan_window(assumptions)
    in_plan = hours[hours["week_index"].isin(plan_weeks)]
    eligible = set(automation.eligible_codes(assumptions))

    # Fixed costs spread over the items the machine actually did, which is
    # the only denominator that means anything: a $500 platform fee on a
    # thousand items is cheap and on ten items is not.
    fixed = fixed_cost_per_item(automated, week_index, assumptions)

    # The ORIGINAL handle time, with 7.1's mix uplift divided back out
    # (decided 10/28). Two reasons, and the second is the one that shows:
    #
    #   the easy item is what the machine TOOK, so it is half of 6.00 and
    #     not half of the 6.26 a human row carries after the uplift;
    #
    #   the uplift varies by week with the ramp, so grouping on the
    #     uplifted figure split every work type into one row per ramp step
    #     - five rows where the model used to report one, all saying the
    #     same thing about the same job.
    #
    # Undone per row, where the week is still known, using exactly the
    # multiple that was applied.
    shares = automation.automated_fraction_by_week(automated)
    originals = [
        automation.easy_item_minutes(
            handle_time,
            assumptions,
            code=code,
            week_index=week,
            removed_share=shares.get((week, code), 0.0),
            uplifted=True,
        )
        for code, week, handle_time in zip(
            in_plan["work_type"], in_plan["week_index"],
            in_plan["handle_time_minutes"],
        )
    ]
    in_plan = in_plan.assign(easy_item_minutes=originals)

    rows = []
    for (code, role, media, easy_minutes), group in in_plan.groupby(
        ["work_type", "role", "media", "easy_item_minutes"]
    ):
        if code not in eligible:
            continue
        row = automation.work_type_settings(code, assumptions)
        sign_off = row.get("sign_off", "none") != "none"
        # The by-hand side prices the items the MACHINE WOULD TAKE, and
        # those are the easy ones (6.6, decided 10/28). Valuing them at the
        # average handle time credited automation with a saving on work it
        # was never going to touch: $3.89 an item instead of $1.94.
        ratio = automation.settings(assumptions)["easy_handle_time_relative"]
        handle_time = easy_minutes / ratio if ratio > 0 else easy_minutes
        human = human_cost_per_item(role, easy_minutes, assumptions)
        machine = automated_cost_per_item(
            code, role, handle_time, assumptions, sign_off=sign_off
        )
        after_fixed = machine["total"] + fixed.get(code, float("nan"))
        rows.append(
            {
                "work_type": code,
                "role": role,
                "media": media,
                "handle_time_minutes": handle_time,
                "easy_item_minutes": easy_minutes,
                "plan_year_items": group["items"].sum(),
                "kind": row["kind"],
                "target_share": automation.row_target_share(code, assumptions),
                "human_cost_per_item": human,
                "automated_cost_per_item": machine["total"],
                "automated_item_cost": machine["item_cost"],
                "automated_exception_cost": machine["exception_cost"],
                "automated_qa_cost": machine["qa_cost"],
                "sign_off_cost_per_item": machine["sign_off_cost"],
                # A row the machine never touched has NO measurable cost
                # per item, and saying zero would be the worst possible
                # answer: W11 and W24 are pinned to 0% by spec Section 1,
                # so a zero fixed cost made them look 60 times more
                # profitable than anything real and put them top of the
                # ranking. Not measurable is not the same as free.
                "automated": code in fixed,
                "fixed_cost_per_item": fixed.get(code, float("nan")),
                "automated_cost_per_item_after_fixed": after_fixed,
                "savings_per_item_before_fixed": human - machine["total"],
                "savings_per_item_after_fixed": human - after_fixed,
                "savings_per_dollar": _savings_per_dollar(human, after_fixed),
                "human_lead_weeks": human_lead_weeks(role, assumptions),
                "automation_lead_weeks": automation_lead_weeks(assumptions),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if not by_role_media:
        table = _next_dollar_by_work_type(table, fixed)
    # Ties broken by work type so the order never depends on grouping
    # order, and the rows with no automated volume sort last rather than
    # first - they are reported, not recommended.
    return table.sort_values(
        ["savings_per_dollar", "work_type", "role", "media"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _next_dollar_by_work_type(detail, fixed):
    """One row per work type, volume-weighted (spec 6.6, decided 10/28).

    A work type can be worked at more than one handle time - W5 is 5
    minutes at Tier 1 and 12 at Tier 2, and W8's retrieval time comes from
    the media table - so the per-item figures are averaged by VOLUME, not
    straight. A straight average would weight a rare job the same as a
    common one and describe a mix nobody works.

    Which is also how the "AI targeted" list is picked, so the table a
    leader reads is now the table the decision was made on.

    The fixed cost per item is already a per-work-type figure (setup plus
    its share of one network platform fee, over that work type's automated
    items), so it is carried across rather than averaged, and the
    after-fixed figures are rebuilt from the weighted variable costs. The
    role and medium shown are the highest-volume ones, with
    `combinations` saying how many were folded in - a reader who needs the
    split can call this with by_role_media=True.
    """
    weighted = [
        "handle_time_minutes", "easy_item_minutes",
        "human_cost_per_item", "automated_cost_per_item",
        "automated_item_cost", "automated_exception_cost",
        "automated_qa_cost", "sign_off_cost_per_item",
    ]
    out = []
    for code, group in detail.groupby("work_type", sort=False):
        items = group["plan_year_items"].to_numpy()
        total_items = items.sum()
        # A work type with no plan-year volume cannot be volume-weighted;
        # fall back to a straight mean so the row still reports.
        weights = items if total_items > 0 else np.ones_like(items)
        # Ties broken by role and medium, not by row order: W5's Tier 1
        # and Tier 2 volumes are an exact 50/50 split, and letting
        # idxmax pick would make the reported role - and the lead time
        # taken from it - depend on how the groupby happened to sort.
        biggest = group.sort_values(
            ["plan_year_items", "role", "media"], ascending=[False, True, True]
        ).iloc[0]
        record = {
            "work_type": code,
            "role": biggest["role"],
            "media": biggest["media"],
            "combinations": int(len(group)),
            "plan_year_items": float(total_items),
            "kind": biggest["kind"],
            "target_share": biggest["target_share"],
        }
        for column in weighted:
            record[column] = float(np.average(group[column], weights=weights))
        record["automated"] = bool(biggest["automated"])
        record["fixed_cost_per_item"] = float(biggest["fixed_cost_per_item"])
        human = record["human_cost_per_item"]
        machine = record["automated_cost_per_item"]
        after_fixed = machine + record["fixed_cost_per_item"]
        record["automated_cost_per_item_after_fixed"] = after_fixed
        record["savings_per_item_before_fixed"] = human - machine
        record["savings_per_item_after_fixed"] = human - after_fixed
        record["savings_per_dollar"] = _savings_per_dollar(human, after_fixed)
        record["human_lead_weeks"] = int(biggest["human_lead_weeks"])
        record["automation_lead_weeks"] = int(biggest["automation_lead_weeks"])
        out.append(record)
    return pd.DataFrame(out)


def _savings_per_dollar(human, automated):
    """(human - automated) / automated, after fixed costs (spec 6.6, 10/27).

    A cost of zero means there is nothing to divide by and nothing to buy,
    so it returns nothing rather than infinity - an infinite return would
    sort straight to the top of the ranking on the strength of a missing
    number.
    """
    if automated <= 0:
        return float("nan")
    return (human - automated) / automated


def write_next_dollar(table, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    out_dir = config.resolve_path(assumptions["paths"]["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / assumptions["paths"]["next_dollar_csv"]
    table.to_csv(path, index=False, float_format="%.6f")
    return path


def weekly_turnaround(projected, assumptions=None, sites=None, work=None, prepared=None):
    """Turnaround by stage each week, from the projected inventory (4.4).

    Every queue on the turnaround path is projected - the remote pools and
    the Records Specialists both - so this is the whole picture, not a
    partial one.

    TWO VIEWS, decided 10/28, and the difference only exists when
    automation is on:

      REQUESTER VIEW (the headline). Every request counts once and an
        automated item counts as completed by the machine. So the backlog
        is the one people are still working, and the arrival rate is every
        request that came in - including the ones a machine finished.

      HUMAN-QUEUE VIEW (secondary, for planners). The same backlog over
        only the arrivals a person handled. This is what the model reported
        until 10/28, and on its own it is misleading: taking 8% of the
        items out of the denominator inflates days of inventory by about
        5% whatever the service actually did. The AI run looked worse in 18
        weeks of the year on this measure and in 4 on the requester's -
        the 4 being the go-live ramp, which is real, because queue targets
        drop the moment the machine starts and the catch-up takes four
        weeks to follow them down.

      It is still worth reporting, because a planner staffing the human
        queues needs days of inventory against the work those queues
        actually receive. It is the wrong number to put in front of a
        leader asking whether requests are going out on time.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    prepared = prepared or prepare(assumptions, sites, work)

    columns = [
        "week_index", "role", "scope", "stage", "work_type", "media",
        "arrivals", "days_of_inventory", "goal_business_days",
    ]
    roles = set(back_office_roles(assumptions))
    held = prepared["queues"]
    held = held[~held["role"].isin(roles)]

    # work_kind rides along only when automation is on (5.5). Defaulted
    # here rather than demanded, because the held frame - the steps no
    # back-office role owns - never carries loop work either way.
    def with_kind(frame):
        out = frame[columns].copy()
        out["work_kind"] = (
            frame["work_kind"].to_numpy()
            if "work_kind" in frame.columns
            else backlog.HUMAN_WORK_KIND
        )
        # The inventory behind each days-of-inventory figure, so the
        # requester view can re-divide it. The projection carries an
        # ending backlog; the held steps - the ones no back-office role
        # owns - sit at their opening position all year.
        out["inventory"] = (
            frame["ending_backlog"].to_numpy()
            if "ending_backlog" in frame.columns
            else frame["starting_backlog"].to_numpy()
        )
        return out

    combined = pd.concat(
        [with_kind(projected), with_kind(held)], ignore_index=True
    )

    # The central records hub (spec 7, item 3, 11/02): a chart pulled at
    # the hub for another site has to travel, so the delivery days go on
    # the DAYS OF INVENTORY of every hub-retrieved step - not on the goal.
    # The 5-day promise does not move because the records moved, which is
    # the point: a goal that followed the operating model could never be
    # failed by it.
    delivery_days = 0.0
    if hub.enabled(assumptions):
        combined, delivery_days = hub.add_delivery_to_turnaround(
            combined, hub.hub_site(assumptions, sites), assumptions, sites
        )

    shares = automation.automated_fraction_by_week(prepared.get("automated"))
    business_days = assumptions["calendar"]["business_days_per_week"]

    rows = []
    for week, group in combined.groupby("week_index"):
        by_stage, _by_step, total = backlog.turnaround(group, assumptions)
        record = {"week_index": week, "turnaround_days": total}
        for stage_row in by_stage.itertuples():
            record[stage_row.stage] = stage_row.days_of_inventory
        rows.append(record)
    table = pd.DataFrame(rows).sort_values("week_index").reset_index(drop=True)

    # Shown on its own line, so a reader can see how much of the hub's
    # turnaround is the van rather than the queue. Only when the hub is
    # on, so the baseline's turnaround file is unchanged.
    if delivery_days:
        no_delivery = combined.copy()
        no_delivery["days_of_inventory"] = (
            no_delivery["days_of_inventory"] - no_delivery["delivery_days"]
        )
        without = []
        for week, group in no_delivery.groupby("week_index"):
            _by_stage, _by_step, total = backlog.turnaround(group, assumptions)
            without.append({"week_index": week, "turnaround_days_before_delivery": total})
        table = table.merge(pd.DataFrame(without), on="week_index", how="left")
        table["delivery_days_in_turnaround"] = (
            table["turnaround_days"] - table["turnaround_days_before_delivery"]
        )

    if not shares:
        # No automation, so the two views are the same number and a second
        # column would be noise. The baseline's turnaround file does not
        # change at all.
        return table

    # The requester view: the same backlog over the FULL arrival rate.
    # Human arrivals are (1 - share) of everything that came in, so
    # dividing by (1 - share) puts the automated items back without
    # needing the item counts again.
    requester = combined.copy()
    scale = np.array(
        [
            1.0 - shares.get((week, code), 0.0)
            for week, code in zip(requester["week_index"], requester["work_type"])
        ]
    )
    full_arrivals = np.where(scale > 0, requester["arrivals"] / scale, requester["arrivals"])
    daily = full_arrivals / business_days
    requester["arrivals"] = full_arrivals
    requester["days_of_inventory"] = np.where(
        daily > 0, requester["inventory"] / np.where(daily > 0, daily, 1.0), 0.0
    )

    headline = []
    for week, group in requester.groupby("week_index"):
        by_stage, _by_step, total = backlog.turnaround(group, assumptions)
        record = {"week_index": week, "turnaround_days": total}
        for stage_row in by_stage.itertuples():
            record[stage_row.stage] = stage_row.days_of_inventory
        headline.append(record)
    headline = pd.DataFrame(headline).sort_values("week_index").reset_index(drop=True)

    # The headline columns keep their names: every reader of this table
    # wants the requester's view, and the planner's is the one that needs
    # a qualifier.
    stages = [s for s in backlog.TURNAROUND_STAGES if s in table.columns]
    out = headline.rename(
        columns={"turnaround_days": "turnaround_days", **{s: s for s in stages}}
    )
    out["turnaround_days_human_queue"] = table["turnaround_days"].to_numpy()
    for stage in stages:
        out[f"{stage}_human_queue"] = table[stage].to_numpy()
    return out


def write_plan(result, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    paths = assumptions["paths"]
    out_dir = config.resolve_path(paths["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for key, name in (
        ("gap", "plan_gap_csv"),
        ("queues", "plan_queues_csv"),
        ("cost", "plan_cost_csv"),
        ("turnaround", "plan_turnaround_csv"),
        ("onsite", "plan_onsite_csv"),
        ("events", "coverage_events_csv"),
        ("hiring", "hiring_plan_csv"),
    ):
        if key not in result:
            continue
        path = out_dir / paths[name]
        result[key].to_csv(path, index=False, float_format="%.6f")
        written[key] = path
    return written


def build_all(assumptions=None, sites=None, work=None, hiring=True, demand=None):
    """Everything Session 4 covers, from one set of shared inputs.

    Built TWICE when hiring is on: once on the opening roster, which is the
    counterfactual the hiring plan is argued against, and once on the
    roster the hiring plan produces. Without the first, "hiring saved 1,900
    hours of overtime" would be a claim with nothing behind it.

    `demand` lets a caller running many variants reuse the supply-
    independent half - work volume and the Erlang interval solve, which
    together are most of the cost. Only safe when the variants leave the
    demand side alone, which a roster sweep does by construction; pass
    nothing and it is rebuilt per run.
    """
    # Imported here, not at the top: coverage.py and hiring.py both read
    # this module, so a module-level import would make the dependency
    # circular. build_all is the one place the chain runs the other way,
    # and it is an orchestrator rather than a step in the chain.
    from . import coverage, hiring as hiring_plan

    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    demand = demand or demand_side(assumptions, sites, work)

    def run(cohorts):
        prepared = prepare(assumptions, sites, cohorts=cohorts, demand=demand)
        back_office = build_plan(assumptions, sites, prepared=prepared)
        onsite = build_onsite(assumptions, sites, prepared=prepared)
        result = dict(back_office)
        result["prepared"] = prepared
        result["onsite"] = onsite["daily"]
        result["onsite_borrows"] = onsite["borrows"]
        result["events"] = coverage.coverage_events(
            back_office, onsite, prepared, assumptions, sites
        )
        result["turnaround"] = weekly_turnaround(
            back_office["queues"], assumptions, sites, prepared=prepared
        )
        result["role_cost"] = plan_cost(prepared, back_office, onsite, assumptions)
        result["gap_fte"] = hiring_plan.weekly_gap_fte(
            back_office, onsite, assumptions, sites
        )
        return result

    opening = run(None)
    if not hiring:
        opening["hiring"] = pd.DataFrame()
        opening["surplus"] = pd.DataFrame()
        return opening

    plan_hiring = hiring_plan.build_hiring_plan(assumptions, sites, demand=demand)
    result = run(plan_hiring["cohorts"])
    result["hiring"] = plan_hiring["classes"]
    result["surplus"] = plan_hiring["surplus"]
    result["hiring_rounds"] = plan_hiring["rounds"]
    result["hiring_exhausted"] = plan_hiring["exhausted"]
    result["before_hiring"] = opening
    return result


def _quarters(frame, assumptions):
    cal = assumptions["calendar"]
    per_quarter = cal["plan_weeks"] // 4
    return ((frame["week_index"] - cal["history_weeks"]) // per_quarter + 1).clip(upper=4)


def _plan_week(week_index, assumptions):
    return int(week_index) - assumptions["calendar"]["history_weeks"] + 1


def main():
    assumptions, sites, _skills = config.load_all()
    work = work_volume.build(assumptions, sites)["work"]

    result = build_all(assumptions, sites, work)
    before = result["before_hiring"]
    written = write_plan(result, assumptions)
    for key, path in written.items():
        print(f"  {key:<11} -> {path}")

    # --- the hiring plan (spec 6.5) ------------------------------------
    classes = result["hiring"]
    print(
        f"\nHIRING PLAN (spec 6.5): {len(classes)} classes placed over "
        f"{result['hiring_rounds']} rounds"
        + ("  [SAFETY STOP HIT]" if result["hiring_exhausted"] else "")
    )
    if len(classes):
        print(
            f"  {'role':<19}{'site':>8}{'size':>6}{'hire-by':>9}{'req':>6}"
            f"{'start':>7}{'first prod':>12}{'short from':>12}{'wks':>5}"
            f"{'worst FTE':>11}  flag"
        )
        for row in classes.itertuples():
            print(
                f"  {row.role:<19}{row.scope:>8}{row.class_size:>6}"
                f"{_plan_week(row.hire_by_week, assumptions):>9}"
                f"{_plan_week(row.requisition_week, assumptions):>6}"
                f"{_plan_week(row.start_week, assumptions):>7}"
                f"{_plan_week(row.first_productive_week, assumptions):>12}"
                f"{_plan_week(row.first_short_week, assumptions):>12}"
                f"{row.weeks_short:>5}{row.worst_short_fte:>11.2f}  {row.flag}"
            )
        print(f"  all weeks are PLAN weeks, 1 to {assumptions['calendar']['plan_weeks']}")

    surplus = result["surplus"]
    print(f"\nSurplus flags - pause hiring (spec 6.5): {len(surplus)}")
    for row in surplus.itertuples():
        print(
            f"  {row.role:<19}{row.scope:>8}  plan weeks "
            f"{_plan_week(row.first_week, assumptions)}-"
            f"{_plan_week(row.last_week, assumptions)} "
            f"({row.weeks} weeks), peak {row.worst_surplus_fte:.2f} FTE over"
        )

    # --- before and after, by role ------------------------------------
    print("\nOvertime, uncovered hours and cost by role, before and after hiring:")
    print(
        f"  {'role':<19}{'OT h before':>13}{'OT h after':>12}"
        f"{'unc before':>12}{'unc after':>11}{'$ before':>13}{'$ after':>13}"
        f"{'change':>11}"
    )

    def by_role(built):
        cost = built["role_cost"].groupby("role")[
            ["overtime_hours", "total_cost"]
        ].sum()
        uncovered = built["cost"].groupby("role")["uncovered_hours"].sum()
        live = built["onsite"].groupby("role")["uncovered_hours"].sum()
        cost["uncovered_hours"] = pd.concat([uncovered, live]).reindex(cost.index).fillna(0.0)
        return cost

    old, new = by_role(before), by_role(result)
    for role in new.index:
        change = new.loc[role, "total_cost"] - old.loc[role, "total_cost"]
        print(
            f"  {role:<19}{old.loc[role, 'overtime_hours']:>13.1f}"
            f"{new.loc[role, 'overtime_hours']:>12.1f}"
            f"{old.loc[role, 'uncovered_hours']:>12.1f}"
            f"{new.loc[role, 'uncovered_hours']:>11.1f}"
            f"${old.loc[role, 'total_cost']:>12,.0f}"
            f"${new.loc[role, 'total_cost']:>12,.0f}{change:>+11,.0f}"
        )
    print(
        f"  {'ALL':<19}{old['overtime_hours'].sum():>13.1f}"
        f"{new['overtime_hours'].sum():>12.1f}"
        f"{old['uncovered_hours'].sum():>12.1f}"
        f"{new['uncovered_hours'].sum():>11.1f}"
        f"${old['total_cost'].sum():>12,.0f}${new['total_cost'].sum():>12,.0f}"
        f"{new['total_cost'].sum() - old['total_cost'].sum():>+11,.0f}"
    )

    # --- coverage events and turnaround -------------------------------
    print("\nCoverage events (spec 6.7), before and after hiring:")
    print(f"  {'type':<6}{'before':>9}{'after':>8}")
    codes = sorted(set(before["events"]["event_type"]) | set(result["events"]["event_type"]))
    old_counts = before["events"]["event_type"].value_counts()
    new_counts = result["events"]["event_type"].value_counts()
    for code in codes:
        print(f"  {code:<6}{old_counts.get(code, 0):>9}{new_counts.get(code, 0):>8}")
    print(f"  {'ALL':<6}{len(before['events']):>9}{len(result['events']):>8}")

    goal = assumptions["backlog"]["overall_turnaround_goal_days"]
    for label, built in (("before hiring", before), ("after hiring", result)):
        turnaround = built["turnaround"]
        worst = turnaround.loc[turnaround["turnaround_days"].idxmax()]
        print(
            f"\nWorst turnaround {label} (goal {goal} days): plan week "
            f"{_plan_week(worst['week_index'], assumptions)} at "
            f"{worst['turnaround_days']:.2f} days"
        )
        print("  " + "  ".join(
            f"{stage} {worst[stage]:.2f}" for stage in backlog.TURNAROUND_STAGES
        ))


if __name__ == "__main__":
    main()
