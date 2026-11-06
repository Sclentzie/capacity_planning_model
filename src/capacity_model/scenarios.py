"""The scenario engine and the Capacity Scenarios output (spec Section 7).

A scenario is assumptions.yaml with a few keys replaced. That is the whole
design, and it is worth being stubborn about: ONE chain, several sets of
inputs. A scenario that got its own code path would eventually disagree
with the baseline about something, and nobody would know which was right.

    scenarios/volume_plus_15.yaml  ->  deep merge  ->  the same plan.build_all

Two rules from spec 7, decided 10/27:

  - Every scenario starts from the BASELINE ROSTER. Starting headcount is
    tuned once, on the baseline, and held. A scenario that re-tuned its own
    roster would open balanced and report no gap at all - it would answer
    "what if we had already hired for this", which is not the question.
  - A scenario's changes take effect from its STARTS WEEK, and the weeks
    before it match the baseline. The history is never touched.

The loader REFUSES an override key that does not exist in the baseline. A
typo in a scenario file would otherwise do nothing at all, silently, and
the run would come back looking exactly like the baseline - which is the
single most dangerous failure mode this module has, because the output
looks perfectly reasonable.

Run it:
    .venv\\Scripts\\python -m capacity_model.scenarios
"""

import copy
import math

import numpy as np
import pandas as pd
import yaml

from . import (
    automation, backlog, config, digitization, hiring, hub, intervals, plan,
    proof_of_concept, supply, work_volume,
)

# The keys a scenario file always sets, because they identify it rather
# than change the model. The diff test treats them separately from the
# overrides a scenario actually declares.
IDENTITY_KEYS = ("name", "starts_week", "what_it_changes")


def scenario_dir(assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    return config.resolve_path(assumptions["paths"]["scenarios_dir"])


def list_scenarios(assumptions=None):
    """Every scenario file, baseline first and the rest alphabetically.

    Baseline first because every comparison is read against it, and a
    table whose reference column is in the middle is a table people
    misread.
    """
    paths = sorted(scenario_dir(assumptions).glob("*.yaml"))
    baseline = [p for p in paths if p.stem == "baseline"]
    return baseline + [p for p in paths if p.stem != "baseline"]


def load_scenario(path):
    """Read one scenario file: its identity and its overrides.

    The file is data, not code: it names inputs and values, and nothing in
    it can reach the model except through the merge.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    unknown = set(raw) - {"scenario", "overrides", "site_overrides"}
    if unknown:
        raise ValueError(
            f"{path.name}: unexpected top-level keys {sorted(unknown)}; a "
            "scenario file holds 'scenario', 'overrides' and 'site_overrides'"
        )
    header = raw.get("scenario") or {}
    missing = set(IDENTITY_KEYS) - set(header)
    if missing:
        raise ValueError(f"{path.name}: the scenario block needs {sorted(missing)}")

    return {
        "key": path.stem,
        "path": path,
        "name": header["name"],
        "starts_week": int(header["starts_week"]),
        "what_it_changes": header["what_it_changes"],
        "overrides": raw.get("overrides") or {},
        # Some inputs live in sites.yaml rather than assumptions.yaml - the
        # travel table and the borrow minimum among them - and a scenario
        # has to be able to reach those too. Same strict merge, so a typo
        # fails the same way.
        "site_overrides": raw.get("site_overrides") or {},
    }


def is_coded_list(value):
    """A list whose every entry is a dict carrying a `code`.

    These are tables, not lists: the request types, the work types, the
    live-queue goals, the shrinkage lines. The order means nothing and the
    code is the key, so a scenario can name one row of one of them. A plain
    list - the automation ramp, a set of weights - has no key and is
    replaced whole.
    """
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(entry, dict) and "code" in entry for entry in value)
    )


def merge_overrides(baseline, overrides, path=()):
    """Deep-merge `overrides` onto `baseline`, refusing unknown keys.

    Refusing is the point. A scenario file is the one place in this model
    where a typo produces no error and no effect: `atrition_multiplier:
    2.0` would merge in happily, change nothing, and hand back a run that
    looks exactly like the baseline. Nobody would catch that by reading the
    output, so it has to fail here.

    A dict is merged key by key. A list of CODED entries - the live-queue
    goals, the work types, the shrinkage lines - is merged row by row on
    the code, so a scenario that moves the phone answer target names that
    one field and nothing else. Any other list, and any scalar, replaces
    the baseline value whole: a plain list has no key to merge on, so a
    scenario that wants a different ramp states the whole ramp.

    Merging coded lists by code rather than wholesale is not a nicety. The
    alternative is restating every row to change one number, which is how
    a scenario file acquires a copy-paste error that nothing can detect -
    a walk-in goal silently reverted while the file claims to be about
    phones.
    """
    merged = copy.deepcopy(baseline)
    for key, value in overrides.items():
        where = path + (key,)
        if key not in baseline:
            trail = ".".join(where)
            raise KeyError(
                f"override {trail!r} is not a baseline input; a scenario can "
                "only change inputs that already exist (check the spelling)"
            )
        if isinstance(value, dict):
            if not isinstance(baseline[key], dict):
                trail = ".".join(where)
                raise TypeError(
                    f"override {trail!r} is a block, but the baseline input is "
                    f"a {type(baseline[key]).__name__}"
                )
            merged[key] = merge_overrides(baseline[key], value, where)
        elif is_coded_list(value) and is_coded_list(baseline[key]):
            merged[key] = _merge_coded_list(baseline[key], value, where)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_coded_list(baseline, overrides, path):
    """Merge a coded table row by row, refusing a code that is not there."""
    rows = [copy.deepcopy(entry) for entry in baseline]
    index = {entry["code"]: position for position, entry in enumerate(rows)}
    for entry in overrides:
        code = entry["code"]
        if code not in index:
            trail = ".".join(path + (str(code),))
            raise KeyError(
                f"override {trail!r} is not a baseline input; a scenario can "
                "only change inputs that already exist (check the spelling)"
            )
        rows[index[code]] = merge_overrides(
            rows[index[code]], entry, path + (str(code),)
        )
    return rows


def apply_scenario(scenario, baseline=None):
    """The merged config for one scenario, with its identity written in."""
    baseline = baseline or config.load_assumptions()
    merged = merge_overrides(baseline, scenario["overrides"])
    merged["scenario"]["name"] = scenario["name"]
    merged["scenario"]["starts_week"] = scenario["starts_week"]
    merged["scenario"]["what_it_changes"] = scenario["what_it_changes"]
    return merged


def apply_site_overrides(scenario, sites=None):
    """The merged sites.yaml for one scenario (spec 7).

    Separate from apply_scenario because the two files are separate, and
    keeping them apart means the diff test can say which of them a
    scenario touched.
    """
    sites = sites or config.load_sites()
    return merge_overrides(sites, scenario.get("site_overrides") or {})


def changed_keys(baseline, merged, path=()):
    """Every leaf path where two configs differ, as dotted strings.

    Used by the diff test in spec Section 9: a scenario must change only
    the inputs its own file names. Anything else means a merge is reaching
    further than the file says it does.
    """
    found = []
    for key in sorted(set(baseline) | set(merged)):
        where = path + (str(key),)
        if key not in baseline or key not in merged:
            found.append(".".join(where))
            continue
        left, right = baseline[key], merged[key]
        if isinstance(left, dict) and isinstance(right, dict):
            found.extend(changed_keys(left, right, where))
        elif is_coded_list(left) and is_coded_list(right):
            # Reported by code, the same way it is merged, so the diff test
            # names live_queues.goals.phones.target_seconds rather than
            # saying the whole goals table moved.
            by_code = {entry["code"]: entry for entry in left}
            for entry in right:
                code = entry["code"]
                if code not in by_code:
                    found.append(".".join(where + (str(code),)))
                    continue
                found.extend(
                    changed_keys(by_code[code], entry, where + (str(code),))
                )
        elif left != right:
            found.append(".".join(where))
    return found


def declared_keys(scenario):
    """The dotted leaf paths a scenario file declares, identity included."""
    found = []

    def walk(node, path):
        for key, value in node.items():
            where = path + (str(key),)
            if isinstance(value, dict):
                walk(value, where)
            elif is_coded_list(value):
                for entry in value:
                    row = dict(entry)
                    code = row.pop("code")
                    walk(row, where + (str(code),))
            else:
                found.append(".".join(where))

    walk(scenario["overrides"], ())
    found.extend(f"scenario.{key}" for key in IDENTITY_KEYS)
    return sorted(set(found))


def declared_site_keys(scenario):
    """The dotted leaf paths a scenario declares in sites.yaml."""
    found = []

    def walk(node, path):
        for key, value in node.items():
            where = path + (str(key),)
            if isinstance(value, dict):
                walk(value, where)
            elif is_coded_list(value):
                for entry in value:
                    row = dict(entry)
                    walk(row, where + (str(row.pop("code")),))
            else:
                found.append(".".join(where))

    walk(scenario.get("site_overrides") or {}, ())
    return sorted(set(found))


# ---------------------------------------------------------------------------
# Running them
# ---------------------------------------------------------------------------


def machine_spend(built):
    """The automation's OWN plan-year invoice: per item, setup, platform.

    Spec 6.4's cost table is LABOUR - regular pay plus overtime - so the
    comparison table's total_cost is what the people cost and not what
    the programme cost. For scenarios that only move people around, the
    two are the same thing. For an automation scenario they are not, and
    reporting the labour line alone would credit the machine with its
    saving and bill it for nothing.

    Found 11/02 and worth saying plainly: on these volumes the invoice is
    the same order as the saving, so which number is quoted decides
    whether the programme looks like a gain or a loss. Both are reported
    here, and the all-in figure is their sum.

    Restricted to plan weeks, because the machine does not exist in the
    history and a setup fee spread over a year is a plan-year cost.
    """
    table = built["prepared"]["automated"]
    if table is None or len(table) == 0:
        return {
            "automated_items": 0.0,
            "item_cost": 0.0,
            "setup_cost": 0.0,
            "platform_cost": 0.0,
            "total": 0.0,
        }
    plan_weeks, _mondays = supply.plan_window(built["assumptions"])
    live = table[table["week_index"].isin(plan_weeks)]
    return {
        "automated_items": float(live["automated_items"].sum()),
        "item_cost": float(live["item_cost"].sum()),
        "setup_cost": float(live["setup_cost"].sum()),
        "platform_cost": float(live["platform_cost"].sum()),
        "total": float(live["total_cost"].sum()),
    }


def all_in_cost(built):
    """Labour plus the machine's own invoice, for one run."""
    labour = float(built["role_cost"]["total_cost"].sum())
    machine = machine_spend(built)["total"]
    return {"labour": labour, "machine": machine, "all_in": labour + machine}


def run_scenario(scenario, baseline=None, sites=None, demand=None):
    """One scenario, through exactly the chain the baseline uses.

    The roster is NOT re-tuned (spec 7, decided 10/27): the merged config
    keeps the baseline's starting headcount, so the gap, the overtime and
    the hiring trigger all report what the change costs rather than what it
    would have cost had somebody seen it coming.
    """
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()
    assumptions = apply_scenario(scenario, baseline)
    merged_sites = apply_site_overrides(scenario, sites)

    # A shared demand side is only safe while the scenario leaves the
    # demand inputs alone. A sites.yaml override could move a size factor,
    # so a scenario that touches that file rebuilds its own.
    if scenario.get("site_overrides"):
        demand = None
    work = None if demand is not None else work_volume.build(assumptions, merged_sites)["work"]
    built = plan.build_all(assumptions, merged_sites, work, demand=demand)
    built["assumptions"] = assumptions
    built["sites"] = merged_sites
    built["scenario"] = scenario
    return built


def summarise(built, assumptions=None, sites=None):
    """The one row per scenario that the comparison table is made of.

    Every figure is a plan-year figure, and every one is after coverage and
    overtime except the peak gap, which is deliberately before overtime -
    that is the number the hiring trigger reads (spec 6.5, 10/26), and the
    number that says how far short the ROSTER was rather than how hard it
    was worked.
    """
    assumptions = assumptions or built["assumptions"]
    sites = sites or config.load_sites()
    cal = assumptions["calendar"]
    plan_weeks = cal["plan_weeks"]
    goal = assumptions["backlog"]["overall_turnaround_goal_days"]

    cost = built["cost"]
    onsite = built["onsite"]
    role_cost = built["role_cost"]
    gap = built["gap_fte"]

    # Required FTE: each role's own hours over its own productive-hour
    # basis, averaged across the plan year. Dividing total hours by one
    # shrinkage figure would misprice the counter, which carries 21.75%
    # against the back office's 28%.
    #
    # Reported as TWO figures, named apart. A single "required FTE" column
    # is the network total, and reading it as the counter's is an easy
    # mistake to make - a scenario that only touches the Onsite Associates
    # moves 25 of the 108 and the column barely twitches.
    onsite_fte = onsite["required_hours"].sum() / hiring.fte_basis(
        "onsite_associate", assumptions
    ) / plan_weeks
    back_office_fte = 0.0
    for role, group in cost.groupby("role"):
        back_office_fte += group["required_hours"].sum() / hiring.fte_basis(
            role, assumptions
        )
    back_office_fte /= plan_weeks

    # Weeks short: role-scope-weeks that ended with work nobody did. The
    # counter is counted at the week grain here even though it is checked
    # daily, so the column means the same thing in every row.
    short_weeks = int((cost["uncovered_hours"] > 1e-9).sum())
    live_short = onsite[onsite["uncovered_hours"] > 1e-9]
    short_weeks += int(len(live_short.groupby(["week_index", "site"])))

    uncovered = cost["uncovered_hours"].sum() + onsite["uncovered_hours"].sum()

    # THREE cost columns, not one (spec 6.4, decided 11/03).
    #
    # Spec 6.4's cost is LABOUR: regular pay plus overtime. The machine's
    # own invoice - per item, setup and the shared platform fee (5.4) -
    # is a real cheque the operation writes and it was nowhere in this
    # table, so every automation scenario's headline was its saving with
    # none of its cost.
    #
    # Measured 11/02 and the reason this changed: on these volumes the
    # invoice is the same ORDER as the labour it saves. AI targeted saves
    # $57,360 of labour and costs $43,280 to run; on top of +15% volume
    # it saves $9,788 and costs $44,465, which turns a saving into a
    # loss. One number could not be both, so there are three and the
    # all-in one is the headline.
    #
    # Nothing changes for a scenario with no machine: its machine cost is
    # zero and all-in IS labour, which is why the baseline pins and every
    # people-only scenario are untouched by this.
    labour_cost = role_cost["total_cost"].sum()
    machine_cost = machine_spend(built)["total"]
    total_cost = labour_cost + machine_cost

    work = built["prepared"]["work"]
    plan_week_set = set(built["prepared"]["weeks"])
    requests = work[
        (work["work_type"] == "W1") & (work["week_index"].isin(plan_week_set))
    ]["work_items"].sum()

    # Phone service level, weighted by the calls that actually arrived -
    # the level a requester experienced, not the average of the days.
    calls = onsite["calls"].sum()
    phone_sl = (
        float((onsite["service_level_phones_after"] * onsite["calls"]).sum() / calls)
        if calls > 0
        else float("nan")
    )
    # Monday against Friday, because that contrast IS the Monday-scheduling
    # question: a flat roster meets a 23% Monday and a 17% Friday with the
    # same hours, so the service level a requester gets depends on which
    # day they turn up.
    by_day = {}
    for day, group in onsite.groupby("day_of_week"):
        calls_that_day = group["calls"].sum()
        by_day[day] = (
            float(
                (group["service_level_phones_after"] * group["calls"]).sum()
                / calls_that_day
            )
            if calls_that_day > 0
            else float("nan")
        )

    walk_in_sl = (
        float((onsite["service_level_walk_ins_after"] * onsite["walk_ins"]).sum()
              / onsite["walk_ins"].sum())
        if onsite["walk_ins"].sum() > 0
        else float("nan")
    )

    # Two turnaround figures, because there are two different questions
    # and they have different answers (see the module note below).
    turnaround = built["turnaround"]
    opening = turnaround.loc[turnaround["week_index"].idxmin(), "turnaround_days"]

    # The planner's view of the same weeks (4.4, decided 10/28): the same
    # backlog over only the arrivals a PERSON handled. Present only when
    # automation is on, because with no machine the two views are one
    # number. Reported alongside rather than instead, so a staffing
    # question and a service question can each read the right column.
    planner = {}
    if "turnaround_days_human_queue" in turnaround.columns:
        human = turnaround["turnaround_days_human_queue"]
        planner = {
            "mean_human_queue_turnaround_days": human.mean(),
            "worst_week_human_queue_turnaround_days": human.max(),
            "weeks_over_goal_human_queue": int((human > goal).sum()),
        }

    return {
        "scenario": built["scenario"]["name"],
        "starts_week": built["scenario"]["starts_week"],
        "what_it_changes": built["scenario"]["what_it_changes"],
        "plan_year_requests": requests,
        "onsite_required_fte": onsite_fte,
        "back_office_required_fte": back_office_fte,
        "network_required_fte": onsite_fte + back_office_fte,
        "peak_gap_fte": gap["short_fte"].max(),
        "weeks_short": short_weeks,
        "overtime_hours": role_cost["overtime_hours"].sum(),
        "uncovered_hours": uncovered,
        # All-in is the headline (spec 6.4, 11/03); the two halves sit
        # beside it so a reader can always see which is which.
        "total_cost": total_cost,
        "labour_cost": labour_cost,
        "machine_cost": machine_cost,
        "cost_per_request": total_cost / requests if requests > 0 else float("nan"),
        "hiring_classes": int(len(built["hiring"])),
        "hired_people": int(built["hiring"]["class_size"].sum()) if len(built["hiring"]) else 0,
        "phone_service_level": phone_sl,
        "monday_phone_service_level": by_day.get("monday", float("nan")),
        "friday_phone_service_level": by_day.get("friday", float("nan")),
        "walk_in_service_level": walk_in_sl,
        # Named for what they measure. The opening week is the yardstick -
        # every queue is PLACED at its target backlog in plan week 1 (spec
        # 4.3), so its days of inventory are its goal by construction and
        # the total is the turnaround the goals describe. The 52-week mean
        # is the projection: each week's queue carries what the week before
        # actually left.
        "opening_week_turnaround_days": opening,
        "mean_projected_turnaround_days": turnaround["turnaround_days"].mean(),
        "worst_week_turnaround_days": turnaround["turnaround_days"].max(),
        "weeks_over_turnaround_goal": int(
            (turnaround["turnaround_days"] > goal).sum()
        ),
        **planner,
    }




# ---------------------------------------------------------------------------
# Two turnaround figures, and why they differ
#
# The comparison table reports both, because they answer different
# questions and a single column would be read as answering whichever one
# the reader had in mind.
#
#   opening_week_turnaround_days   the turnaround the GOALS describe
#   mean_projected_turnaround_days what the plan delivers, averaged
#
# Both are the REQUESTER's view since 10/28: every request counts once and
# an automated item counts as completed by the machine. When automation is
# on, three more columns carry the planner's view of the same weeks -
# `*_human_queue` - which measures the same backlog over only the arrivals
# a person handled. See plan.weekly_turnaround for why both exist.
#
# Plan week 1 opens every queue AT its target backlog (spec 4.3,
# start_at_target). Days of inventory at target IS the goal, by
# definition - so week 1's total is the weighted sum of the five stage
# goals and nothing else. It is a yardstick, not an outcome, and it is the
# 5.23 days Session 3 recorded.
#
# From week 2 the queue carries what the week before actually left, and
# days of inventory divides that by THIS week's arrival rate. So the
# figure moves with the arrival stream as much as with the staffing: a
# week whose volume jumps shows a lower days-of-inventory on the same
# backlog, and a quiet week shows a higher one. Over the plan year that
# correlation is -0.67, and rising weeks average 4.50 days against 4.83 in
# falling ones.
#
# Which is why the mean sitting below the 5-day goal is not the plan
# beating its goal through excellence. The numbers a leader should read
# are the worst week and the count of weeks over the goal, both of which
# the table carries.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phone service goal (spec 7)
#
# Session 3's finding is the reason this scenario exists: at these volumes
# it is the ANSWER TARGET, not headcount or demand, that sets onsite
# staffing. Blended occupancy sits near 30% and the 85% cap never binds,
# which means the counter is staffed to be available rather than to be
# busy - so the service goal is the real cost driver, and it is the one
# input a leader changes by deciding to.
# ---------------------------------------------------------------------------

PHONE_GOAL_SECONDS = (20, 30, 60)


def phone_goal_variant(seconds, baseline=None):
    """The baseline with the phone answer target moved (spec 7)."""
    baseline = baseline or config.load_assumptions()
    return merge_overrides(
        baseline,
        {"live_queues": {"goals": [{"code": "phones", "target_seconds": seconds}]}},
    )


def onsite_shape(assumptions, sites=None, work=None):
    """What the counter's schedule looks like under one set of inputs.

    TWO FTE figures, and they are different questions (reconciled 10/27):

      floor_fte   bodies the schedule puts on the counter floor, before
                  anybody helps. This is what the service goal buys, and
                  what Erlang C actually demands.
      required_fte what the ONSITE ASSOCIATES themselves owe: the floor
                  less the Records Specialists' share of it, plus any
                  email that would not fit into idle time. Spec 5.1's
                  definition, decided 10/26, and the one every comparison
                  table uses.

    On the baseline the gap between them is 1.1085 FTE, which is the 1,804
    hours of Records Specialist help - nothing else. Reporting only the
    first would overstate the counter's requirement by a whole person;
    reporting only the second would hide what a service goal costs in
    bodies on the floor.
    """
    sites = sites or config.load_sites()
    if work is None:
        work = work_volume.build(assumptions, sites)["work"]
    plan_weeks, _mondays = supply.plan_window(assumptions)
    built = intervals.onsite_intervals(
        assumptions, sites, work[work["week_index"].isin(plan_weeks)]
    )

    # Records Specialist help, so this agrees with summarise() to the cent.
    # The ledger is cheap; the interval solve above is the expensive part.
    required = backlog.required_hours(assumptions, sites, work)["hours"]
    ledger = supply.build_ledger(assumptions, sites)
    spare = supply.records_specialist_spare_hours(ledger["daily"], required, assumptions)
    helped = supply.apply_records_specialist_help(built["intervals"], spare, assumptions)
    spillover = built["hours"]["email_paid_hours"].sum()

    table = built["intervals"]
    interval_hours = assumptions["live_queues"]["interval_seconds"] / 3600.0

    # Blended occupancy pooled across every interval, not averaged across
    # them: traffic over staff is a ratio of totals, and averaging ratios
    # would weight a quiet 8am half-hour the same as a busy 11am one.
    staffed = table["required_staff"].sum()
    occupancy = float(table["traffic_pooled"].sum() / staffed) if staffed else 0.0

    # Hours bought purely to cover people on break and at lunch: the gap
    # between who is scheduled and who Erlang C needs on the queue.
    cover = (table["scheduled_staff"] - table["required_staff"]) * interval_hours

    peaks = table.groupby(["week_index", "site", "day_of_week"])["scheduled_staff"].max()
    basis = hiring.fte_basis("onsite_associate", assumptions) * len(plan_weeks)
    owed = helped["scheduled_hours"].sum() + spillover
    return {
        # Spec 5.1's definition: after Records Specialist help, plus email
        # spillover. The same figure the comparison table reports.
        "required_hours": owed,
        "required_fte": owed / basis,
        # And the floor it comes out of, named apart so neither can be
        # read as the other.
        "floor_hours": table["scheduled_hours"].sum(),
        "floor_fte": table["scheduled_hours"].sum() / basis,
        "records_specialist_help_hours": helped["help_hours"].sum(),
        "email_spillover_hours": spillover,
        "blended_occupancy": occupancy,
        "break_cover_hours": float(cover.sum()),
        "mean_daily_peak_staff": float(peaks.mean()),
        "worst_daily_peak_staff": float(peaks.max()),
        "mean_required_staff": float(table["required_staff"].mean()),
        "walk_in_service_level_min": float(table["walk_in_service_level"].min())
        if "walk_in_service_level" in table
        else float("nan"),
        "intervals": table,
    }


def phone_goal_report(baseline=None, sites=None, work=None):
    """Onsite FTE and occupancy at each phone goal, and what 10 seconds buys.

    The marginal column is the one that answers the question, and the
    answer at these volumes is "almost nothing, almost linearly": about a
    quarter of an FTE per ten seconds across the whole 20-to-60-second
    range, for the entire six-site network.

    That is the Session 3 occupancy finding showing up as money. Blended
    occupancy is near 32%, so the counter is staffed to be AVAILABLE
    rather than to be busy - and relaxing the answer target does not
    change how many people have to be present to answer at all. A leader
    cannot buy their way out of onsite staffing by letting the phone ring
    longer; tripling the permitted wait buys about one FTE.
    """
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()
    if work is None:
        work = work_volume.build(baseline, sites)["work"]

    rows = []
    previous = None
    for seconds in PHONE_GOAL_SECONDS:
        shape = onsite_shape(phone_goal_variant(seconds, baseline), sites, work)
        row = {
            "phone_goal": f"80% in {seconds}s",
            "target_seconds": seconds,
            "onsite_required_fte": shape["required_fte"],
            "onsite_required_hours": shape["required_hours"],
            "blended_occupancy": shape["blended_occupancy"],
            "mean_daily_peak_staff": shape["mean_daily_peak_staff"],
            "break_cover_hours": shape["break_cover_hours"],
            "fte_saved_vs_baseline": 0.0,
            "fte_saved_per_10_seconds": float("nan"),
        }
        if previous is not None:
            row["fte_saved_vs_baseline"] = rows[0]["onsite_required_fte"] - row[
                "onsite_required_fte"
            ]
            step = seconds - previous["target_seconds"]
            row["fte_saved_per_10_seconds"] = (
                (previous["onsite_required_fte"] - row["onsite_required_fte"])
                / step
                * 10
            )
        rows.append(row)
        previous = row
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Wider lunch window (spec 7)
#
# One 30-minute unpaid lunch spread across a 3-hour window puts 16.7% of
# the team away on average, and that relief has to fit inside six
# half-hours of a midday that is BUSY - requesters chase records on their
# own lunch break, which is why both interval curves bump at noon.
#
# Half an hour more window is 30/210 = 14.3% away instead of 16.7%. Nobody
# takes a longer lunch and nobody takes fewer; the same relief is spread
# thinner. Lunch is unpaid, so whatever that saves is free - which is why
# it is worth checking before anybody signs a requisition.
#
# WHAT THE PLAIN RUN SHOWS, and it is not what the scenario expected.
# The six half-hours from 11:00 to 13:30 do thin, and the staffing in them
# falls with it. But the window now reaches 14:00, where the AFTERNOON
# BREAK window already starts, so lunch relief stacks on break relief in
# that one half-hour. The single overlapping half-hour costs about what
# the six thinned ones save, so network onsite FTE moves the wrong way.
#
# That run found the away cap too: it was enforced per relief, so the
# combined share at 14:00 went past the 25% the input exists to impose.
# Fixed 10/27 (spec 3.4): the cap now limits everyone away in a half-hour
# across all reliefs, and intervals.relief_shares places them in day order
# into a shared headroom.
#
# Decided 10/27: keep the plain run as a recorded finding, and add a
# second that moves the afternoon break to 14:30 so the two windows stop
# overlapping. That is the run that tells a leader whether the idea pays.
# ---------------------------------------------------------------------------


def wider_lunch_variant(closes, baseline=None, break_opens=None):
    """The baseline with the lunch window's end moved (spec 7).

    `break_opens` also moves the afternoon break's start, which is the
    second run added on 10/27: without it the two windows overlap at 14:00
    and the widening cancels itself out.
    """
    baseline = baseline or config.load_assumptions()
    reliefs = [{"code": "lunch", "window": {"to": closes}}]
    if break_opens is not None:
        reliefs.append({"code": "afternoon_break", "window": {"from": break_opens}})
    return merge_overrides(baseline, {"breaks_and_lunch": {"reliefs": reliefs}})


# The three runs the lunch report lines up: as it is, widened, and
# widened with the afternoon break moved out of the way.
LUNCH_VARIANTS = (
    ("Baseline", "14:00", "14:00"),
    ("Wider lunch", "14:30", "14:00"),
    ("Wider lunch, break moved", "14:30", "14:30"),
)

# The half-hours both windows cover, so the two runs can be compared like
# for like. Comparing "the lunch window" between them would compare six
# intervals against seven and call the difference an effect.
SHARED_LUNCH_INTERVALS = ("11:00", "11:30", "12:00", "12:30", "13:00", "13:30")
BREAK_OVERLAP_INTERVAL = "14:00"


def lunch_window_report(baseline=None, sites=None, work=None):
    """Peak staff, break cover and the away share, at each lunch window.

    Reported on the SHARED half-hours as well as on each window as a
    whole, because the two windows are not the same size: widening the
    window adds a half-hour, so "the mean away share in the lunch window"
    changes partly because the window changed. The shared columns are the
    like-for-like comparison and they are the ones that answer whether
    spreading lunch thinner actually thins it.

    The overlap column is the finding. 14:00 belongs to the afternoon
    break window too, so under the wider lunch window it carries both
    reliefs at once.
    """
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()
    if work is None:
        work = work_volume.build(baseline, sites)["work"]

    rows = []
    for label, closes, break_opens in LUNCH_VARIANTS:
        assumptions = wider_lunch_variant(closes, baseline, break_opens)
        shape = onsite_shape(assumptions, sites, work)
        table = shape["intervals"]
        inside = np.array(intervals.relief_windows(assumptions)["lunch"])
        in_window = inside[table["interval"].to_numpy()]
        lunch_rows = table[in_window]
        shared = table[table["interval_start"].isin(SHARED_LUNCH_INTERVALS)]
        overlap = table[table["interval_start"] == BREAK_OVERLAP_INTERVAL]

        peak_in_window = lunch_rows.groupby(
            ["week_index", "site", "day_of_week"]
        )["scheduled_staff"].max()
        day_peak_interval = table.loc[
            table.groupby(["week_index", "site", "day_of_week"])[
                "scheduled_staff"
            ].idxmax(),
            "interval",
        ]
        rows.append(
            {
                "run": label,
                "lunch_window": f"11:00 to {closes}",
                "afternoon_break_window": f"{break_opens} to 16:00",
                "window_half_hours": int(inside.sum()),
                "nominal_away_share": 30.0 / (30.0 * inside.sum()),
                "mean_away_share_in_window": float(lunch_rows["away_share"].mean()),
                "shared_mean_away_share": float(shared["away_share"].mean()),
                "shared_mean_scheduled_staff": float(shared["scheduled_staff"].mean()),
                "shared_peak_scheduled_staff": float(
                    shared.groupby(["week_index", "site", "day_of_week"])[
                        "scheduled_staff"
                    ].max().mean()
                ),
                "overlap_away_share": float(overlap["away_share"].mean()),
                "overlap_scheduled_staff": float(overlap["scheduled_staff"].mean()),
                "max_away_share_any_interval": float(table["away_share"].max()),
                "lunch_window_peak_staff": float(peak_in_window.mean()),
                "onsite_required_fte": shape["required_fte"],
                "mean_daily_peak_staff": shape["mean_daily_peak_staff"],
                "break_cover_hours": shape["break_cover_hours"],
                "day_peak_in_lunch_window_share": float(
                    inside[day_peak_interval.to_numpy()].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The part-time share sweep (spec 7, decided 10/27)
#
# At 10% the part-time run overshoots: Monday lands on 25.5% of paid hours
# against 23% of demand, and every point of the share has to come off all
# five days, so Wednesday and Thursday fall below theirs. It fixes one hole
# by digging two smaller ones and comes out worse than the baseline.
#
# The share is an input, so the obvious question is where the curve turns.
# The arithmetic says Monday's share of paid hours is 20% + 0.55 x share
# and every other day loses 20% x share, so Monday reaches its 23% of
# demand at a share of 5.45% - and by then Tuesday, Wednesday and Thursday
# are all already below theirs. The instrument has two destination days
# and takes from five, so it can never match the pattern; the sweep says
# how close it gets and at what price.
# ---------------------------------------------------------------------------

PART_TIME_SWEEP = tuple(round(0.01 * step, 2) for step in range(11))


def part_time_variant(share, baseline=None):
    """The part-time scenario at one share, as a scenario record.

    A share of 0.0 is the baseline exactly: the weights come out as ones -
    (1 - 0) / 5 x 5 - and multiplying the ledger's hours by one is a no-op,
    not an approximation. The sweep's first row is therefore a check on the
    whole mechanism, and the tests treat it as one.
    """
    return {
        "key": f"part_time_{int(round(share * 100)):02d}",
        "path": None,
        "name": f"Part-time {share:.0%}",
        "starts_week": 1,
        "what_it_changes": (
            f"{share:.0%} of Onsite Associate paid hours to Monday/Tuesday part-timers"
        ),
        "overrides": {
            "scenario": {
                "onsite_schedule": {"shape": "part_time", "part_time_share": share}
            }
        },
    }


def part_time_sweep(baseline=None, sites=None, shares=PART_TIME_SWEEP):
    """Uncovered hours, overtime and cost against the part-time share.

    One full plan run per step, sharing the demand side across them -
    reshaping the roster cannot touch the work that arrives, which is the
    property that makes the sharing safe and is itself tested.
    """
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()
    demand = plan.demand_side(baseline, sites)

    reference = None
    rows = []
    for share in shares:
        scenario = part_time_variant(share, baseline)
        built = run_scenario(scenario, baseline, sites, demand=demand)
        summary = summarise(built, built["assumptions"], sites)
        weights = supply.day_weights("onsite_associate", built["assumptions"]) / len(
            supply.WEEKDAYS
        )
        if reference is None:
            reference = summary
        rows.append(
            {
                "part_time_share": share,
                "monday_share_of_paid_hours": float(weights[0]),
                "wednesday_share_of_paid_hours": float(weights[2]),
                "uncovered_hours": summary["uncovered_hours"],
                "overtime_hours": summary["overtime_hours"],
                "monday_phone_service_level": summary["monday_phone_service_level"],
                "onsite_required_fte": summary["onsite_required_fte"],
                "total_cost": summary["total_cost"],
                "cost_change_vs_baseline": summary["total_cost"]
                - reference["total_cost"],
                "uncovered_change_vs_baseline": summary["uncovered_hours"]
                - reference["uncovered_hours"],
            }
        )
    return pd.DataFrame(rows)


def write_sweep(table, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    out_dir = config.resolve_path(assumptions["paths"]["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / assumptions["paths"]["monday_sweep_csv"]
    table.to_csv(path, index=False, float_format="%.6f")
    return path


# ---------------------------------------------------------------------------
# Coordinated digitization (spec 7, decided 10/28)
#
# The report answers one question in two different ways, because the two
# answers are months apart and a leader needs both.
#
#   THE LABOUR-VALUE BREAK-EVEN. Hours saved, priced at the rate of
#     whoever would have worked them, against the project's added cost.
#     This is the economics of the project itself and the number the spec
#     asks for.
#
#   THE PLAN'S OWN COST DELTA. What the scenario actually bills, which is
#     dominated by the roster: with the baseline headcount held (spec 7,
#     10/27), hours saved turn into less overtime and a smaller gap, not
#     into a smaller payroll. A project can be clearly worth doing on the
#     first measure and invisible on the second for a year.
#
# Reporting only the first would be a business case with no plan behind
# it; only the second would bury a real saving under a fixed roster.
# ---------------------------------------------------------------------------


DIGITIZATION_WORK_TYPES = ("W8", "W11", "W12")


def digitization_report(baseline=None, sites=None):
    """Week-by-week: the non-digital share, hours saved, and break-even."""
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()

    scenario = load_scenario(
        scenario_dir(baseline) / "coordinated_digitization.yaml"
    )
    merged = apply_scenario(scenario, baseline)
    plan_weeks, _mondays = supply.plan_window(baseline)
    first = min(plan_weeks)

    # The conversion walk, and the share it leaves behind.
    walk = digitization.converted_by_week(merged, sites)
    base_share = digitization.non_digital_share(baseline)

    # Required hours on both sides, by week and work type. Built straight
    # from work volume rather than from a full plan run: the hours a work
    # type NEEDS do not depend on who is rostered, and this is the figure
    # the project changes.
    def hours_by_week(assumptions):
        work = work_volume.build(assumptions, sites)["work"]
        table = backlog.required_hours(assumptions, sites, work)["hours"]
        live = table[table["week_index"].isin(plan_weeks)]
        return live, work[work["week_index"].isin(plan_weeks)]

    base_hours, base_work = hours_by_week(baseline)
    run_hours, run_work = hours_by_week(merged)

    rates = {
        role: float(settings["hourly_cost"])
        for role, settings in baseline["supply"]["roles"].items()
    }

    def by_week(frame, code):
        mine = frame[frame["work_type"] == code]
        return mine.groupby("week_index")["required_hours"].sum()

    def value_by_week(frame, code):
        mine = frame[frame["work_type"] == code].copy()
        mine["value"] = mine["required_hours"] * mine["role"].map(rates)
        return mine.groupby("week_index")["value"].sum()

    weekly_cost = digitization.weekly_cost(merged)
    start = digitization.from_week(merged)

    rows = []
    cumulative_saving = 0.0
    cumulative_cost = 0.0
    for week in plan_weeks:
        site_rows = walk[walk["week_index"] == week]
        converted = float(site_rows["converted_this_week"].sum())
        cumulative_converted = float(site_rows["converted_cumulative"].sum())
        archive = float(site_rows["archive_charts"].sum())
        # The network's requested non-digital share: each site's remaining
        # onsite share weighted by that site's share of the archive, which
        # is its size factor - the same weight its request volume carries.
        remaining = float(
            (
                (1.0 - site_rows["requests_removed_share"])
                * site_rows["archive_charts"]
            ).sum()
            / archive
        ) if archive > 0 else 1.0

        record = {
            "week_index": week,
            "plan_week": week - first + 1,
            "converted_this_week": converted,
            "converted_cumulative": cumulative_converted,
            "converted_share_of_archive": (
                cumulative_converted / archive if archive > 0 else 0.0
            ),
            "non_digital_share": base_share * remaining,
            "non_digital_share_baseline": base_share,
            # True once the project has run out of hot charts still on
            # paper and is converting the cold tail instead.
            "not_yet_requested_cap_binds": bool(site_rows["cap_binds"].any()),
        }

        saving = 0.0
        for code in DIGITIZATION_WORK_TYPES:
            base_h = float(by_week(base_hours, code).get(week, 0.0))
            run_h = float(by_week(run_hours, code).get(week, 0.0))
            record[f"{code.lower()}_hours_baseline"] = base_h
            record[f"{code.lower()}_hours"] = run_h
            record[f"{code.lower()}_hours_saved"] = base_h - run_h
            base_v = float(value_by_week(base_hours, code).get(week, 0.0))
            run_v = float(value_by_week(run_hours, code).get(week, 0.0))
            saving += base_v - run_v

        added = weekly_cost if week >= start else 0.0
        cumulative_saving += saving
        cumulative_cost += added
        record["hours_saved_net"] = sum(
            record[f"{code.lower()}_hours_saved"]
            for code in DIGITIZATION_WORK_TYPES
        )
        record["labour_saving"] = saving
        record["added_cost"] = added
        record["net_this_week"] = saving - added
        record["cumulative_saving"] = cumulative_saving
        record["cumulative_added_cost"] = cumulative_cost
        record["cumulative_net_cost"] = cumulative_cost - cumulative_saving
        rows.append(record)

    table = pd.DataFrame(rows)

    # Break-even: the first plan week where the cumulative saving has
    # covered the cumulative cost.
    #
    # Only weeks the project has actually spent something in count. The
    # weeks before the starts week have a cumulative net of exactly zero -
    # nothing spent, nothing saved - and reading those as "paid back in
    # week 1" is the kind of answer that looks like good news.
    spending = table[table["cumulative_added_cost"] > 0]
    paid_back = spending[spending["cumulative_net_cost"] <= 0]
    inside = int(paid_back["plan_week"].iloc[0]) if len(paid_back) else None

    # If it does not pay back inside the plan year, project when it would
    # AT THE SAME RATE - which means holding the conversion rate, not the
    # saving. The weekly saving GROWS: every week adds another 160 charts
    # that stop being retrieved, so the saving is proportional to the
    # records converted so far.
    #
    # That makes the arithmetic a quadratic rather than a division:
    #
    #     weekly saving after w live weeks  = g x w
    #     cumulative saving                 = g x w^2 / 2
    #     cumulative cost                   = C x w
    #     break-even                        = w where the two meet = 2C/g
    #
    # and the week the project first pays for ITSELF, before clearing its
    # debt, is C/g - half as long.
    #
    # Dividing the outstanding balance by the last week's net instead
    # would assume the saving stops growing the moment the plan year ends,
    # which is both wrong and much too pessimistic.
    #
    # The straight line holds only while there are hot charts still on
    # paper, where every chart converted carries the same high request
    # probability. The report says how long that is and whether the
    # break-even it projects falls inside it - because at the calibrated
    # archive (11/02) it no longer does, and a projection that has walked
    # outside its own assumption has to say so.
    projection = None
    if inside is None:
        live = table[table["cumulative_added_cost"] > 0]
        weeks_live = np.arange(1, len(live) + 1, dtype=float)
        savings = live["labour_saving"].to_numpy(dtype=float)
        # Slope through the origin: a project that has converted nothing
        # saves nothing, so the line is not free to pick an intercept.
        slope = float((weeks_live * savings).sum() / (weeks_live ** 2).sum())
        cost = float(weekly_cost)
        hot_weeks = digitization.live_weeks_of_hot_charts(merged, sites)
        if slope > 0:
            pays_for_itself = cost / slope
            break_even = 2.0 * cost / slope
            projection = {
                "saving_growth_per_live_week": slope,
                "weekly_cost": cost,
                "live_weeks_to_weekly_break_even": pays_for_itself,
                "live_weeks_to_cumulative_break_even": break_even,
                "years_to_cumulative_break_even": break_even / 52.0,
                "live_weeks_of_hot_charts_on_paper": hot_weeks,
                "still_inside_the_band_at_break_even": break_even < hot_weeks,
                "outstanding_at_plan_year_end": float(
                    table["cumulative_net_cost"].iloc[-1]
                ),
                "basis": (
                    "the conversion rate held constant, so the weekly "
                    "saving keeps growing in proportion to the records "
                    "converted; a straight line fitted through the origin "
                    "on the plan year's live weeks"
                ),
            }

    # ------------------------------------------------------------------
    # The per-chart form of the answer (added 11/02).
    #
    # Everything above is the project's own arithmetic. This is the test a
    # leader can carry out of the room: converting a chart costs a known
    # amount and saves nothing at all UNLESS that chart gets pulled again.
    # So there is a request probability at which the project breaks even,
    # and it can be compared against what the archive actually offers.
    #
    # The value of one avoided paper pull is taken from the model's own
    # hours and rates rather than restated - W11 and W12 saved, W8 added
    # back - so it cannot drift away from the weekly table above.
    # ------------------------------------------------------------------
    def items_of(frame, code):
        return float(frame[frame["work_type"] == code]["work_items"].sum())

    avoided_requests = items_of(base_work, "W11") - items_of(run_work, "W11")
    total_saving = float(table["labour_saving"].sum())
    converted_total = float(table["converted_this_week"].sum())
    per_chart = None
    if avoided_requests > 0 and converted_total > 0:
        value_per_request = total_saving / avoided_requests
        cost_per_chart = weekly_cost / digitization.items_per_week(merged)
        probability = cost_per_chart / value_per_request
        hot_archive, hot_requests = digitization.concentration(merged)
        archive_total = float(
            walk.groupby("week_index")["archive_charts"].sum().iloc[0]
        )
        lam_hot = (
            hot_requests
            * digitization.reactive_rate(merged)
            / (hot_archive * archive_total)
        )
        in_a_year = 1.0 - np.exp(-lam_hot * 52.0)
        per_chart = {
            "value_of_one_avoided_paper_pull": value_per_request,
            "cost_per_chart_converted": cost_per_chart,
            "break_even_request_probability": probability,
            "hot_chart_request_probability_in_a_year": in_a_year,
            "targeting_multiple_needed": (
                probability / in_a_year if in_a_year > 0 else float("inf")
            ),
            "weeks_for_a_hot_chart_to_reach_it": (
                -np.log(1.0 - probability) / lam_hot
                if 0.0 < probability < 1.0 and lam_hot > 0
                else float("inf")
            ),
            "avoided_paper_pulls_in_the_plan_year": avoided_requests,
            "charts_converted_in_the_plan_year": converted_total,
        }

    # The calibration, carried with the report so the archive is never
    # quoted without the history it was solved against (spec 7, 11/02).
    hot_archive, hot_requests = digitization.concentration(merged)
    calibration = {
        "archive_charts": float(
            walk.groupby("week_index")["archive_charts"].sum().iloc[0]
        ),
        "concentration_archive_share": hot_archive,
        "concentration_request_share": hot_requests,
        "hot_density": hot_requests / hot_archive,
        "mean_density": digitization.mean_density(merged),
        "reactive_conversions_per_week": digitization.reactive_rate(merged),
        "history_weeks": int(baseline["calendar"]["history_weeks"]),
        "history_drift_points": digitization.history_drift_points(merged),
        "history_drift_budget_points": digitization.drift_budget_points(merged),
    }
    calibration["holds"] = (
        calibration["history_drift_points"]
        < calibration["history_drift_budget_points"]
    )

    return {
        "weekly": table,
        "walk": walk,
        "break_even_plan_week": inside,
        "projection": projection,
        "per_chart": per_chart,
        "calibration": calibration,
        "weekly_added_cost": weekly_cost,
        "items_per_week": digitization.items_per_week(merged),
        "not_yet_requested_cap_ever_binds": bool(walk["cap_binds"].any()),
    }


# ---------------------------------------------------------------------------
# Central records hub (spec 7, decided 11/02)
#
# Reported in two halves, because the scenario asks a question the
# as-run numbers cannot answer on their own.
#
#   AS RUN keeps the baseline roster (spec 7, 10/27), which is what the
#     task asked for and what a transition really looks like in its first
#     year: the same people, one address. It shows the SERVICE cost of
#     pooling - delivery days, counter help, overtime, uncovered hours -
#     against a payroll that has not changed.
#
#   SIZED TO WORK re-sizes both teams from their own required hours. That
#     is where the hub's saving lives, because the saving is a structural
#     one: six per-site teams each round up to a whole person and each
#     carry a floor of 2, and one hub rounds up once and is floored once.
#
# Reporting only the first would say the hub is pointless; reporting only
# the second would say it is free. It is neither.
# ---------------------------------------------------------------------------

HUB_ROLES = ("records_specialist", "onsite_associate")


def hub_report(baseline=None, sites=None):
    """The hub against the baseline: team, delivery, help, and cost."""
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()

    scenario = load_scenario(scenario_dir(baseline) / "central_records_hub.yaml")
    merged = apply_scenario(scenario, baseline)

    chosen, runner_up, ranking = hub.choose_hub(merged, sites)

    # The choice is made on size factors, because it has to be made before
    # the work table exists (see hub.volume_by_site). Check it against the
    # volume the model really built, and say so either way - two
    # definitions of "volume" that are never compared is how a model ends
    # up quietly contradicting itself.
    work = work_volume.build(baseline, sites)["work"]
    realised = hub.volume_by_site_from_work(work, merged)
    realised_ranking = hub.hub_ranking(merged, sites, shares=realised)
    realised_hub = str(realised_ranking["site"].iloc[0])

    runs = {
        "baseline": run_scenario(
            load_scenario(scenario_dir(baseline) / "baseline.yaml"), baseline, sites
        ),
        "hub": run_scenario(scenario, baseline, sites),
    }

    rates = {
        role: float(cfg["hourly_cost"])
        for role, cfg in baseline["supply"]["roles"].items()
    }

    # --- Records Specialist hours and cost, network and by site --------
    specialist = {}
    for name, built in runs.items():
        rows = built["cost"]
        mine = rows[rows["role"] == "records_specialist"]
        specialist[name] = {
            "required_hours": float(mine["required_hours"].sum()),
            "paid_hours": float(mine["paid_hours"].sum())
            if "paid_hours" in mine.columns
            else float("nan"),
            "by_site": mine.groupby("scope")["required_hours"].sum(),
            "overtime_hours": float(mine["overtime_hours"].sum()),
            "uncovered_hours": float(mine["uncovered_hours"].sum()),
        }

    # --- counter help: lost, and what it costs Onsite Associates -------
    help_by_site, onsite_by_site, onsite_total = {}, {}, {}
    for name, built in runs.items():
        helped = built["prepared"]["onsite_intervals"]
        help_by_site[name] = helped.groupby("site")["help_hours"].sum()
        onsite = built["onsite"]
        onsite_by_site[name] = onsite.groupby("site")["required_hours"].sum()
        onsite_total[name] = float(onsite["required_hours"].sum())

    help_lost = help_by_site["baseline"] - help_by_site["hub"]
    onsite_added = onsite_by_site["hub"] - onsite_by_site["baseline"]
    counter_help = {
        "applied_hours_baseline": float(help_by_site["baseline"].sum()),
        "applied_hours_hub": float(help_by_site["hub"].sum()),
        "hours_lost_net": float(help_lost.sum()),
        "hours_lost_at_non_hub_sites": float(
            help_lost[help_lost.index != chosen].sum()
        ),
        "by_site_help_baseline": help_by_site["baseline"],
        "by_site_help_hub": help_by_site["hub"],
        "by_site_onsite_hours_added": onsite_added,
        "onsite_hours_added_net": onsite_total["hub"] - onsite_total["baseline"],
        "onsite_cost_added_net": (
            onsite_total["hub"] - onsite_total["baseline"]
        ) * rates["onsite_associate"],
    }

    # --- turnaround, requester view, delivery shown apart --------------
    goal = baseline["backlog"]["overall_turnaround_goal_days"]
    turnaround = {}
    for name, built in runs.items():
        table = built["turnaround"]
        days = table["turnaround_days"]
        entry = {
            "mean": float(days.mean()),
            "worst": float(days.max()),
            "weeks_over_goal": int((days > goal).sum()),
        }
        if "delivery_days_in_turnaround" in table.columns:
            before = table["turnaround_days_before_delivery"]
            entry.update(
                {
                    "mean_before_delivery": float(before.mean()),
                    "worst_before_delivery": float(before.max()),
                    "weeks_over_goal_before_delivery": int((before > goal).sum()),
                    "mean_delivery_days": float(
                        table["delivery_days_in_turnaround"].mean()
                    ),
                }
            )
        turnaround[name] = entry

    # --- all-in cost ---------------------------------------------------
    cost = {
        name: float(built["role_cost"]["total_cost"].sum())
        for name, built in runs.items()
    }
    cost["delta"] = cost["hub"] - cost["baseline"]

    # --- sized to work: where the structural saving actually is --------
    #
    # Two things make six teams cost more than one for the same hours, and
    # they are worth separating because only one of them is the 2-per-site
    # floor the spec talks about:
    #
    #   the FLOOR, where a site's own work needs less than 2 people;
    #   the ROUNDING, where each site rounds its own fraction up to a
    #     whole person and six fractions round up six times.
    sized = {}
    for name, config_used in (("baseline", baseline), ("hub", merged)):
        table = supply.tune_starting_headcount(config_used, sites)
        entry = {}
        for role in HUB_ROLES:
            mine = table[table["role"] == role]
            floor_cost = float(
                (mine["floor"] - mine["required_fte"]).clip(lower=0.0).sum()
            )
            entry[role] = {
                "scopes": int(len(mine)),
                "required_fte": float(mine["required_fte"].sum()),
                "headcount": int(mine["headcount"].sum()),
                "slack_fte": float(mine["rounding_slack_fte"].sum()),
                "slack_from_the_floor_fte": floor_cost,
                "slack_from_rounding_fte": float(
                    mine["rounding_slack_fte"].sum() - floor_cost
                ),
                "by_scope": mine.set_index("scope")[
                    ["required_fte", "floor", "headcount"]
                ],
            }
        sized[name] = entry

    paid_hours_a_year = (
        baseline["fte"]["paid_hours_per_week"] * baseline["calendar"]["plan_weeks"]
    )
    heads = {
        role: sized["hub"][role]["headcount"] - sized["baseline"][role]["headcount"]
        for role in HUB_ROLES
    }
    sized_saving = -sum(
        heads[role] * rates[role] * paid_hours_a_year for role in HUB_ROLES
    )

    return {
        "hub": chosen,
        "runner_up": runner_up,
        "ranking": ranking,
        "weighted_delivery_days": hub.weighted_delivery_days(chosen, merged, sites),
        "delivery_days_by_site": hub.delivery_days_by_site(chosen, merged, sites),
        "choice_check": {
            "hub_by_size_factor": chosen,
            "hub_by_realised_volume": realised_hub,
            "agree": chosen == realised_hub,
            "realised_ranking": realised_ranking,
        },
        "records_specialist": specialist,
        "counter_help": counter_help,
        "turnaround": turnaround,
        "cost": cost,
        "sized_to_work": sized,
        "sized_headcount_change": heads,
        "sized_annual_saving": sized_saving,
        "paid_hours_a_year": paid_hours_a_year,
        "archive_move_cost_modelled": False,
        "not_modelled": (
            "the one-time cost of physically moving the paper archive to "
            "the hub: boxes, transport, re-indexing, and running the "
            "retrieval service at two addresses during the move. Every "
            "figure here is an upper bound on the hub's case."
        ),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# After proof of concept (spec 7, added 11/02)
#
# The scenario is one line of spec 5.5 - the QA sample rate - and the
# report is built to say so rather than to flatter it.
#
# Of the three things automation sends back to people, only the QA sample
# is about TRUST. Exceptions are a property of the work and the legal
# sign-off is a property of the law, so neither moves because a machine
# measured well. That leaves a review sample as the entire prize, and
# whether it is worth having depends on something the spec paragraph does
# not mention: how many items the machine took. A sample is a fraction of
# a fraction.
#
# The gate itself is the more interesting half. It is not a date, it is a
# measurement on a random sample, and the report prints the sample size
# every week because that is the number which decides everything:
#
#     a week passes only on 0 errors whenever 1/n is above the target
#     P(a week passes) = (1 - true rate) ^ n
#     P(the gate clears) = P(a week passes) ^ 12
#
# which has the perverse consequence the report names out loud: a SMALLER
# sample passes more easily. Looking less is the cheapest way to clear a
# gate written on an observed rate.
# ---------------------------------------------------------------------------


POC_WORK_KINDS = ("qa_sample", "sign_off", "exception")


def _loop_hours_by_kind(built):
    """Human-in-the-loop hours from a run, split the way 5.5 splits them.

    Taken from the plan's OWN required-hours table rather than recomputed,
    so the figures reported are the hours the plan was actually built on.
    An empty dict when automation is off, which is the baseline.
    """
    required = built["prepared"]["required"]
    if "work_kind" not in required.columns:
        return {}
    out = {}
    for kind in POC_WORK_KINDS:
        mine = required[required["work_kind"] == kind]
        out[kind] = {
            "hours": float(mine["required_hours"].sum()),
            "items": float(mine["items"].sum()),
        }
    return out


def week_pass_probability(true_rate, sampled, target):
    """P(a week of this sample size reads at or under the target), exactly.

    Whenever one error in n would read above the target - which is every
    sample of fewer than 1/target items, so fewer than 200 at a 0.5%
    target - a week passes only on ZERO errors, and the probability is
    simply (1 - true rate) ^ n. The general binomial tail is written out
    anyway so the figure stays right if the inputs move.

    Here rather than in proof_of_concept.py because it prices nothing in
    the plan: it is the arithmetic that explains the run, and a reader
    should be able to check it on paper against one seed's outcome.
    """
    n = int(sampled)
    if n <= 0:
        return 0.0
    allowed = int(math.floor(target * n + 1e-12))
    return float(
        sum(
            math.comb(n, errors)
            * true_rate**errors
            * (1.0 - true_rate) ** (n - errors)
            for errors in range(min(allowed, n) + 1)
        )
    )


POC_RUNS = (
    ("baseline", "baseline.yaml"),
    ("ai_targeted", "ai_targeted.yaml"),
    ("proof_of_concept", "after_proof_of_concept.yaml"),
    ("true_rate_on_target", "after_proof_of_concept_true_05.yaml"),
)


def proof_of_concept_report(baseline=None, sites=None):
    """What PROVING the automation frees, against installing it (spec 7).

    Reported against AI TARGETED, not against the baseline, because that
    is what the scenario is built on (Siri 11/02): the baseline has no
    automation, so it has no QA sample to lower, and comparing with it
    would measure the machine rather than the proof of it.
    """
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()

    runs, merged = {}, {}
    for key, filename in POC_RUNS:
        scenario = load_scenario(scenario_dir(baseline) / filename)
        merged[key] = apply_scenario(scenario, baseline)
        runs[key] = run_scenario(scenario, baseline, sites)

    goal = baseline["backlog"]["overall_turnaround_goal_days"]
    qa_hour = plan.productive_hour_cost(automation.QA_ROLE, baseline)

    gates = {}
    for key in ("proof_of_concept", "true_rate_on_target"):
        used = merged[key]
        series = runs[key]["prepared"]["proof_of_concept"]
        live = series[series["sampled_items"] > 0]
        freed = proof_of_concept.hours_freed(series, used)
        automated_plan = runs[key]["prepared"]["automated"]
        gate = proof_of_concept.gate_week(automated_plan, used)
        truth = proof_of_concept.true_error_rate(used)
        target = proof_of_concept.target_error_rate(used)
        hold = proof_of_concept.weeks_to_hold(used)

        # The gate's own arithmetic at the typical sample this run saw,
        # so the outcome can be read against the statistics rather than
        # taken on the strength of one seed.
        typical = int(round(live["sampled_items"].mean())) if len(live) else 0
        per_week = week_pass_probability(truth, typical, target)
        gates[key] = {
            "true_error_rate": truth,
            "target_error_rate": target,
            "weeks_to_hold": hold,
            "gate_week_absolute": gate,
            "gate_plan_week": proof_of_concept.gate_plan_week(automated_plan, used),
            "cleared": gate is not None,
            "live_weeks": int(len(live)),
            "weeks_over_target": int((~live["at_or_under_target"]).sum()),
            "longest_clean_run": int(live["consecutive_weeks"].max())
            if len(live)
            else 0,
            "sampled_items_total": int(live["sampled_items"].sum()),
            "sampled_items_mean": float(live["sampled_items"].mean())
            if len(live)
            else 0.0,
            "error_weeks": [
                (int(row.week_index), int(row.sampled_items), int(row.errors_found))
                for row in live.itertuples()
                if row.errors_found
            ],
            "qa_items_not_sampled": freed["items_not_sampled"],
            "qa_hours_freed": freed["hours"],
            "qa_cost_freed": freed["hours"] * qa_hour,
            "typical_sample": typical,
            "p_week_passes": per_week,
            "p_gate_in_twelve": per_week**hold,
            "series": series,
        }

    # --- the hours, from the plan's own table ---------------------------
    loop = {key: _loop_hours_by_kind(built) for key, built in runs.items()}

    # Sign-off is the CHECK, not a saving (spec Section 9). Reported as
    # the rate actually in force and the hours actually billed, so
    # "sign-offs stayed at 100%" is a measurement rather than a claim.
    sign_off = {
        "legal_requirement_changed": proof_of_concept.legal_requirement_changed(
            merged["proof_of_concept"]
        ),
        "rate_in_force": sorted(
            set(
                proof_of_concept.sign_off_rate_by_week(
                    runs["proof_of_concept"]["prepared"]["automated"],
                    merged["proof_of_concept"],
                ).values()
            )
        ),
        "hours_by_run": {
            key: entry.get("sign_off", {}).get("hours", 0.0)
            for key, entry in loop.items()
        },
        # Which automated rows carry a legal sign-off at all. Empty on the
        # targeted list, and that is the honest headline: the two rows
        # that pay are not rows a statute speaks to, so there is no
        # sign-off here to argue about in the first place.
        "automated_rows_needing_sign_off": sorted(
            code
            for code in automation.targeted_work_types(merged["proof_of_concept"])
            if automation.work_type_settings(code, merged["proof_of_concept"]).get(
                "sign_off", "none"
            )
            != "none"
        ),
    }

    # Labour and the machine's invoice, reported apart and together. The
    # gate moves only the labour line - it buys a smaller review sample,
    # not a cheaper platform - so the all-in figures for the two gated
    # runs differ from AI targeted by the QA saving and nothing else.
    cost = {key: all_in_cost(built) for key, built in runs.items()}
    turnaround = {}
    for key, built in runs.items():
        days = built["turnaround"]["turnaround_days"]
        turnaround[key] = {
            "mean": float(days.mean()),
            "worst": float(days.max()),
            "weeks_over_goal": int((days > goal).sum()),
        }

    return {
        "gates": gates,
        "loop_hours": loop,
        "sign_off": sign_off,
        "cost": {key: entry["labour"] for key, entry in cost.items()},
        "all_in_cost": cost,
        "machine_spend": {key: machine_spend(built) for key, built in runs.items()},
        "cost_vs_ai_targeted": {
            key: cost[key]["labour"] - cost["ai_targeted"]["labour"] for key in cost
        },
        "all_in_vs_ai_targeted": {
            key: cost[key]["all_in"] - cost["ai_targeted"]["all_in"] for key in cost
        },
        "turnaround": turnaround,
        "qa_productive_hour_cost": qa_hour,
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Combined (spec 7, decided 11/02)
#
# Volume +15% from week 1 and AI targeted from week 13: a growth scenario
# and a relief scenario running at different dates on purpose, because
# the demand arrives a quarter before the machine does and that quarter
# is where the backlog and the hiring trigger live.
#
# The report is built around ONE ratio, the question a leader actually
# asks - how much of the rise does the automation absorb:
#
#     absorbed = (volume-only FTE - combined FTE)
#                / (volume-only FTE - baseline FTE)
#
# and around one warning. "AI targeted" is a list of work types picked
# from the Next Dollar ranking AT BASELINE VOLUME. Setup is fixed at
# $7,692 a row a year while the saving scales with items, so a row that
# loses a cent an item at baseline volume can pay at +15%. The report
# re-prices the ranking on the combined volume and names the rows that
# cross over - as a FINDING, not as a change: the scenario file keeps the
# list Siri decided it builds on.
# ---------------------------------------------------------------------------


COMBINED_RUNS = (
    ("baseline", "baseline.yaml"),
    ("volume_only", "volume_plus_15.yaml"),
    ("ai_targeted_only", "ai_targeted.yaml"),
    ("combined", "combined.yaml"),
    ("combined_ai_100", "combined_ai_100.yaml"),
)

COMBINED_ABSORB_COLUMNS = (
    "onsite_required_fte",
    "back_office_required_fte",
    "network_required_fte",
    "overtime_hours",
    "uncovered_hours",
    "total_cost",
)


def combined_variant(baseline=None):
    """+15% volume with the FULL program but deflection off.

    Not a registered scenario, because it answers a question about the
    other two rather than one of its own: "AI 100%" differs from "AI
    targeted" in BOTH breadth (19 rows against 2) and deflection (on
    against off), and a single number for the gap would not say which did
    the work. This run holds deflection off so the two can be separated.
    """
    baseline = baseline or config.load_assumptions()
    scenario = load_scenario(scenario_dir(baseline) / "combined_ai_100.yaml")
    variant = copy.deepcopy(scenario)
    variant["key"] = "combined_ai_100_no_deflection"
    variant["name"] = "Combined, AI 100%, deflection off"
    variant["what_it_changes"] = (
        "Requests x 1.15 from week 1 plus the full automation program, "
        "deflection off, to separate breadth from deflection"
    )
    variant["overrides"]["scenario"]["deflection"]["enabled"] = False
    return variant


def combined_repick_variant(codes, baseline=None):
    """Combined with the targeted list re-picked at the HIGHER volume.

    Also not a registered scenario, and deliberately so: which rows the
    machine is pointed at is Siri's decision (spec 7, "AI targeted"), and
    the registered Combined run keeps the list she decided it builds on.
    This variant exists to put a number on what that list costs, because
    "the list would be longer at +15%" is a caveat and "it is worth
    another forty thousand" is an answer.

    `codes` comes from the re-priced ranking rather than being written in,
    so if the volumes or the setup fee ever move, this follows them.
    """
    baseline = baseline or config.load_assumptions()
    variant = copy.deepcopy(load_scenario(scenario_dir(baseline) / "combined.yaml"))
    variant["key"] = "combined_repicked"
    variant["name"] = "Combined, list re-picked at +15%"
    variant["what_it_changes"] = (
        "Requests x 1.15 from week 1 plus AI on every row that pays after "
        "fixed costs AT THE HIGHER volume: " + ", ".join(sorted(codes))
    )
    variant["overrides"]["automation"]["rates"]["only_work_types"] = sorted(codes)
    return variant


def combined_report(baseline=None, sites=None):
    """Volume +15% with and without the automation it can justify (spec 7)."""
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()

    runs = {}
    for key, filename in COMBINED_RUNS:
        runs[key] = run_scenario(
            load_scenario(scenario_dir(baseline) / filename), baseline, sites
        )
    runs["combined_ai_100_no_deflection"] = run_scenario(
        combined_variant(baseline), baseline, sites
    )

    # --- would the targeted list be longer at +15%? ---------------------
    #
    # Priced on the combined volume with every row switched on, which is
    # the only run where a row the machine did not touch has an automated
    # cost per item to rank at all. Exactly how the original list was
    # picked (spec 6.6, 10/28).
    #
    # Through next_dollar's OWN path, letting it rebuild the hours rather
    # than handing it the run's table. Found 11/02: a run carries the
    # COMBINED hours (human plus 5.5's exception, QA and sign-off rows),
    # and feeding those in as the human side inflates every item count and
    # dilutes every handle time - W8 read 175,312 items at 3.50 an item
    # instead of 156,532 at 3.90, and the answer came out backwards. The
    # rebuild costs a few seconds and means this is the ranking a leader
    # reads.
    #
    # The mechanism is worth naming because it is the asymmetry that
    # decides the list: SETUP IS FIXED and the saving SCALES with items.
    # A setup fee spread over 15% more items is 13% less per item, so
    # every row's after-fixed saving improves by about twenty cents and a
    # row sitting just under the line can cross it.
    rank_base = apply_scenario(
        load_scenario(scenario_dir(baseline) / "ai_and_automation.yaml"), baseline
    )
    at_baseline_volume = plan.next_dollar(rank_base, sites)
    ranking = plan.next_dollar(runs["combined_ai_100"]["assumptions"], sites)

    picked = set(automation.targeted_work_types(runs["combined"]["assumptions"]))
    pays = set(ranking.loc[ranking["savings_per_item_after_fixed"] > 0, "work_type"])
    paid_before = set(
        at_baseline_volume.loc[
            at_baseline_volume["savings_per_item_after_fixed"] > 0, "work_type"
        ]
    )
    repick = {
        "list_in_the_scenario": sorted(picked),
        "pays_at_baseline_volume": sorted(paid_before),
        "would_pay_at_plus_15": sorted(pays),
        "newly_positive": sorted(pays - picked),
        "ranking": ranking[
            [
                "work_type",
                "plan_year_items",
                "human_cost_per_item",
                "fixed_cost_per_item",
                "automated_cost_per_item_after_fixed",
                "savings_per_item_after_fixed",
            ]
        ].sort_values("savings_per_item_after_fixed", ascending=False),
        "ranking_at_baseline_volume": at_baseline_volume[
            ["work_type", "plan_year_items", "savings_per_item_after_fixed"]
        ].sort_values("savings_per_item_after_fixed", ascending=False),
    }
    if pays != picked:
        runs["combined_repicked"] = run_scenario(
            combined_repick_variant(pays, baseline), baseline, sites
        )

    table = pd.DataFrame(
        [
            summarise(built, built["assumptions"], built["sites"])
            for built in runs.values()
        ],
        index=list(runs),
    )

    # --- how much of the rise the automation absorbs --------------------
    #
    # Reported on required FTE as well as on cost, because the two answer
    # different questions: FTE is the size of the hole the volume dug,
    # and cost mixes in overtime, hiring and coverage, which are the
    # plan's RESPONSE to the hole rather than the hole.
    rise = {}
    for column in COMBINED_ABSORB_COLUMNS:
        base = float(table.loc["baseline", column])
        volume = float(table.loc["volume_only", column])
        headroom = volume - base
        rise[column] = {
            "baseline": base,
            "volume_only": volume,
            "rise_from_volume": headroom,
            "absorbed": {},
        }
        for key in [k for k in runs if k not in ("baseline", "volume_only",
                                                "ai_targeted_only")]:
            value = float(table.loc[key, column])
            rise[column]["absorbed"][key] = {
                "value": value,
                "absorbed": volume - value,
                "share_of_rise": (volume - value) / headroom
                if headroom
                else float("nan"),
            }

    # Breadth against deflection, each on its own line, because the two
    # are different kinds of decision: breadth is supply-side and costs a
    # setup fee per row, deflection is demand-side and costs nothing per
    # row but assumes requesters self-serve.
    #
    # Each reported on LABOUR and on ALL-IN, and the pair is the finding
    # (11/03). Breadth saves $51,217 of labour and buys seventeen more
    # setups to do it, so all-in it COSTS about $101,000: going wide is a
    # worse decision than it looks on the labour line alone. Deflection
    # saves $27,511 of labour and adds nothing to the invoice - the
    # contact never arrives, so there is no item to pay a fee on - so it
    # is the same number either way. The cheaper lever is the one with no
    # machine attached to it.
    def lever(wider, narrower):
        return {
            part: float(
                table.loc[narrower, column] - table.loc[wider, column]
            )
            for part, column in (
                ("labour", "labour_cost"),
                ("machine", "machine_cost"),
                ("all_in", "total_cost"),
            )
        }

    split = {
        "breadth_19_rows_vs_2": lever("combined_ai_100_no_deflection", "combined"),
        "deflection": lever("combined_ai_100", "combined_ai_100_no_deflection"),
    }

    # --- the all-in cost, which reverses the ordering -------------------
    #
    # The comparison table's total_cost is LABOUR (spec 6.4). Measured
    # 11/02, the machine's own invoice on these volumes is the same order
    # as the labour it saves, so the two lines have to be read together:
    # the targeted machine saves about ten thousand of labour on top of
    # +15% and costs about forty-four thousand to run.
    #
    # And the reason the labour line is so thin is the held roster (spec
    # 7, 10/27). Regular pay is headcount x paid hours, so required hours
    # falling is only cash where it cancels an overtime hour or a hiring
    # class. At +15% the plan is already hiring 23 people and the machine
    # removes well under one FTE of requirement, so it cancels neither.
    cost = {key: all_in_cost(built) for key, built in runs.items()}
    machine = {key: machine_spend(built) for key, built in runs.items()}
    against_volume = {
        key: {
            "labour": entry["labour"] - cost["volume_only"]["labour"],
            "machine": entry["machine"] - cost["volume_only"]["machine"],
            "all_in": entry["all_in"] - cost["volume_only"]["all_in"],
        }
        for key, entry in cost.items()
    }
    against_baseline = {
        key: {
            "labour": entry["labour"] - cost["baseline"]["labour"],
            "machine": entry["machine"] - cost["baseline"]["machine"],
            "all_in": entry["all_in"] - cost["baseline"]["all_in"],
        }
        for key, entry in cost.items()
    }

    return {
        "comparison": table,
        "rise": rise,
        "breadth_against_deflection": split,
        "repick": repick,
        "all_in_cost": cost,
        "machine_spend": machine,
        "against_volume_only": against_volume,
        "against_baseline": against_baseline,
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Centralized phones (spec 7, decided 10/27)
#
# Reported as separate lines rather than one net number, because the three
# effects pull in different directions and a leader needs to see which one
# dominates:
#
#   pooling gain      six phone-only queues against one, same goal and
#                     same handle time. Erlang C's spare capacity stops
#                     being duplicated six times.
#   cap gives back    the 85% occupancy cap, which Session 3 found bites
#                     hardest in large queues.
#   uplift costs      +5% handle time for the local knowledge lost.
#
# And then the effect a pooling argument on its own would miss entirely:
# the counters lose their blend. A site taking a handful of walk-ins an
# hour still needs somebody standing at the counter, and today that
# somebody answers the phone between requesters.
# ---------------------------------------------------------------------------


def central_phones_report(baseline=None, sites=None, work=None):
    """Phone and counter staffing, central against per-site (spec 7).

    Everything is in STAFF-INTERVALS over the plan year from the changeover
    week onward - the natural unit, since Erlang C answers per half-hour -
    with hours and FTE beside them.
    """
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()
    if work is None:
        work = work_volume.build(baseline, sites)["work"]

    scenario = load_scenario(scenario_dir(baseline) / "central_phones.yaml")
    merged = apply_scenario(scenario, baseline)
    from_week = intervals.central_phones_from_week(merged)
    plan_weeks, _mondays = supply.plan_window(merged)
    weeks = [week for week in plan_weeks if week >= from_week]
    interval_hours = merged["live_queues"]["interval_seconds"] / 3600.0
    basis = hiring.fte_basis("onsite_associate", merged)

    central = intervals.central_phone_queue(merged, sites, work)
    central = central[central["week_index"].isin(weeks)]

    # The counter, walk-ins alone, and the blended team it replaces.
    shaped = intervals.onsite_intervals(merged, sites, work)["intervals"]
    shaped = shaped[shaped["week_index"].isin(weeks)]
    flat = intervals.onsite_intervals(baseline, sites, work)["intervals"]
    flat = flat[flat["week_index"].isin(weeks)]

    def fte(staff_intervals):
        return staff_intervals * interval_hours / basis / len(weeks)

    network = {
        "weeks": len(weeks),
        "phones_six_queues": central["required_separate"].sum(),
        "phones_central_service_level_only": central["required_no_cap"].sum(),
        "phones_central_capped": central["required_capped"].sum(),
        "phones_central_final": central["required_staff"].sum(),
        "counter_blended_baseline": flat["required_staff"].sum(),
        "counter_walk_ins_only": shaped["required_counter"].sum(),
    }
    network["pooling_gain"] = (
        network["phones_six_queues"] - network["phones_central_service_level_only"]
    )
    network["cap_gives_back"] = (
        network["phones_central_capped"]
        - network["phones_central_service_level_only"]
    )
    network["uplift_costs"] = (
        network["phones_central_final"] - network["phones_central_capped"]
    )
    network["phones_net_gain"] = (
        network["phones_six_queues"] - network["phones_central_final"]
    )
    # The counter loses its blend: what it needs on walk-ins alone against
    # what the blended team needed for both channels.
    network["counter_extra"] = (
        network["counter_walk_ins_only"] + network["phones_central_final"]
    ) - network["counter_blended_baseline"]
    for key in list(network):
        if key != "weeks":
            network[key + "_fte"] = fte(network[key])

    per_site = []
    for site, group in shaped.groupby("site"):
        was = flat[flat["site"] == site]
        share = float(group["central_call_share"].mean())
        per_site.append(
            {
                "site": site,
                "call_share": share,
                "blended_staff_intervals": was["required_staff"].sum(),
                "counter_only_staff_intervals": group["required_counter"].sum(),
                "central_share_staff_intervals": group["central_share_staff"].sum(),
                "total_staff_intervals": group["required_staff"].sum(),
                "change_vs_blended": group["required_staff"].sum()
                - was["required_staff"].sum(),
                "blended_fte": fte(was["required_staff"].sum()),
                "central_fte": fte(group["required_staff"].sum()),
            }
        )
    return {"network": network, "per_site": pd.DataFrame(per_site)}


# ---------------------------------------------------------------------------
# THE FTE GLOSSARY (reconciled 10/27)
#
# Four different FTE figures got quoted across tasks 5b to 5e and three of
# them were right. They differ on three axes and every table now says
# which it is using.
#
#   floor FTE
#       Bodies the schedule puts on the counter floor, before anybody
#       helps. What Erlang C demands and what a service goal buys.
#       Baseline, plan weeks 1-52: 25.14.
#
#   onsite required FTE (spec 5.1, decided 10/26)
#       What the ONSITE ASSOCIATES owe: the floor less the Records
#       Specialists' share of it, plus any email that would not fit into
#       idle time. The definition every comparison table uses.
#       Baseline, plan weeks 1-52: 24.03.
#
#   network required FTE
#       All five roles. Onsite plus back office.
#       Baseline, plan weeks 1-52: 107.94.
#
# The three axes, because they are what made the figures look like they
# disagreed:
#
#   WEEKS. 24.03 is the 52-week plan year; the centralized-phones table's
#       22.00 is plan weeks 13-52 only, because that is when the scenario
#       runs. Same definition, 40 weeks instead of 52.
#   FLOOR vs NET OF HELP. 25.14 against 24.03 is exactly the 1,804 hours
#       the Records Specialists give the counter - 1.11 FTE.
#   SHRINKAGE. Every figure divides by its own role's productive hours:
#       40 x (1 - 21.75%) = 31.30 for the counter, 40 x (1 - 28%) = 28.80
#       for the back office. One shrinkage figure across the network would
#       misprice the counter by 9%.
#
# And the 2.66 against 2.3: the 2.66 is the centralized-phones saving over
# weeks 13-52 measured on the counter alone; the 2.3 is the whole plan
# year's onsite required FTE, which dilutes 40 weeks of saving across 52.
# ---------------------------------------------------------------------------

FTE_GLOSSARY = (
    (
        "floor FTE",
        "bodies on the counter floor, before Records Specialist help",
        "the scenario's own weeks",
    ),
    (
        "onsite required FTE",
        "what the Onsite Associates owe: floor less help, plus email "
        "spillover (spec 5.1)",
        "plan weeks 1-52",
    ),
    (
        "network required FTE",
        "all five roles, each over its own productive-hour basis",
        "plan weeks 1-52",
    ),
)


def print_fte_glossary():
    """So no table in the output can be read against the wrong definition."""
    print("\nFTE definitions used above (reconciled 10/27):")
    for name, meaning, weeks in FTE_GLOSSARY:
        print(f"  {name:<22}{weeks:<22}{meaning}")
    print(
        "  Each divides by its own role's productive hours: 40 x (1 - 21.75%)"
        " = 31.30 onsite,"
    )
    print("  40 x (1 - 28%) = 28.80 back office.")


def run(paths=None, baseline=None, sites=None):
    """Run the baseline and every scenario, and line them up (spec 7)."""
    baseline = baseline or config.load_assumptions()
    sites = sites or config.load_sites()
    paths = paths if paths is not None else list_scenarios(baseline)

    rows, runs = [], {}
    for path in paths:
        scenario = load_scenario(path)
        built = run_scenario(scenario, baseline, sites)
        runs[scenario["key"]] = built
        rows.append(summarise(built, built["assumptions"], built["sites"]))
    return {"comparison": pd.DataFrame(rows), "runs": runs}


def write_comparison(table, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    out_dir = config.resolve_path(assumptions["paths"]["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / assumptions["paths"]["scenario_comparison_csv"]
    table.to_csv(path, index=False, float_format="%.6f")
    return path


def main():
    baseline, sites, _skills = config.load_all()
    paths = list_scenarios(baseline)
    print(f"Capacity Scenarios (spec 7): {len(paths)} runs, all SYNTHETIC")
    for path in paths:
        print(f"  {path.name}")

    result = run(paths, baseline, sites)
    table = result["comparison"]
    written = write_comparison(table, baseline)
    print(f"\n  comparison -> {written}")

    reference = table.iloc[0]
    print(f"\n{'':<24}" + "".join(f"{row.scenario:>22}" for row in table.itertuples()))
    rows = (
        ("plan-year requests", "plan_year_requests", "{:>22,.0f}"),
        ("onsite FTE", "onsite_required_fte", "{:>22.1f}"),
        ("back-office FTE", "back_office_required_fte", "{:>22.1f}"),
        ("network FTE", "network_required_fte", "{:>22.1f}"),
        ("peak gap FTE", "peak_gap_fte", "{:>22.2f}"),
        ("weeks short", "weeks_short", "{:>22d}"),
        ("overtime hours", "overtime_hours", "{:>22,.0f}"),
        ("uncovered hours", "uncovered_hours", "{:>22,.1f}"),
        # All-in first, because it is the headline (spec 6.4, 11/03), with
        # its two halves indented under it so nobody has to work out which
        # of three cost numbers they are looking at.
        ("ALL-IN cost", "total_cost", "{:>22,.0f}"),
        ("  labour", "labour_cost", "{:>22,.0f}"),
        ("  machine", "machine_cost", "{:>22,.0f}"),
        ("all-in per request", "cost_per_request", "{:>22.2f}"),
        ("hiring classes", "hiring_classes", "{:>22d}"),
        ("people hired", "hired_people", "{:>22d}"),
        ("phone service level", "phone_service_level", "{:>21.1%} "),
        ("  Monday", "monday_phone_service_level", "{:>21.1%} "),
        ("  Friday", "friday_phone_service_level", "{:>21.1%} "),
        ("walk-in service level", "walk_in_service_level", "{:>21.1%} "),
        ("turnaround, opening wk", "opening_week_turnaround_days", "{:>22.2f}"),
        ("turnaround, 52-wk mean", "mean_projected_turnaround_days", "{:>22.2f}"),
        ("turnaround, worst week", "worst_week_turnaround_days", "{:>22.2f}"),
        ("weeks over the goal", "weeks_over_turnaround_goal", "{:>22d}"),
    )
    for label, column, fmt in rows:
        line = f"  {label:<22}"
        for value in table[column]:
            line += fmt.format(value)
        print(line)

    print_fte_glossary()

    print("\nChange against the baseline:")
    for row in table.itertuples():
        if row.scenario == reference["scenario"]:
            continue
        # The all-in change, with the machine's share named: a scenario
        # that saves labour and spends more than it saved should not be
        # able to print a negative number here (spec 6.4, 11/03).
        print(
            f"  {row.scenario:<22}"
            f"all-in {row.total_cost - reference['total_cost']:+,.0f} "
            f"({row.total_cost / reference['total_cost'] - 1:+.1%}), "
            f"of which machine {row.machine_cost - reference['machine_cost']:+,.0f}, "
            f"overtime {row.overtime_hours - reference['overtime_hours']:+,.0f} h, "
            f"uncovered {row.uncovered_hours - reference['uncovered_hours']:+,.1f} h, "
            f"hires {row.hired_people - reference['hired_people']:+d}"
        )


if __name__ == "__main__":
    main()
