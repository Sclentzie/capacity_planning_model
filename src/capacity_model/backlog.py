"""Required hours by role for the back-office work types (spec 4.3, 4.4).

Session 3, task 3. This is the step that turns "how many items" into "how
many hours, and whose". The backlog inventory model of 4.3 builds on top of
it and is not here yet: these are STRAIGHT workload hours, with no
automation, no deflection and no backlog catch-up.

Two ideas do the work:

  ROUTING BY SKILL (spec 4.4). Most work types name their owner outright in
  the Section 1 table - intake is Tier 1, quality review is the QA team,
  onsite retrieval is a Records Specialist. Three rows instead say "by
  request type", and those follow the REQUEST to whichever tier holds its
  skill: a patient asking for their own records is Tier 1 work, a subpoena
  is Tier 2. The model never assigns work to a tier without the skill, and
  skills.yaml is the matrix it checks against.

  HANDLE TIME IS NOT ALWAYS ONE NUMBER. Three rows refuse to be a single
  figure, and each refusal is the interesting part:
    W5  5 minutes for Tier 1, 12 for Tier 2 - validating a subpoena is a
        different job from validating a patient's own request
    W11 25 / 35 / 45 minutes for paper / microfiche / archive room - the
        media mix is what actually decides onsite retrieval hours
    W22 half of whatever W14 costs, derived rather than typed, so rework
        cannot drift away from the review it reworks

    Formula (spec 4.4):
        Hours per role = items routed to that role
                         x that step's handle time for the request type / 60

Onsite Associates are deliberately absent. Their live work (W19, W20) is
staffed by Erlang C in intervals.py, and their email and portal work (W21)
is absorbed into idle live-queue time there too - so routing any of the
three here would pay for the same work twice.

Remote roles are ONE network pool, not six site teams: a Tier 1 Specialist
works the queue wherever the request came from. Site is kept as a reporting
field so the plan can still show where demand arises, but hours roll up to
the network before they become FTE.

Run it:
    .venv\\Scripts\\python -m capacity_model.backlog
"""

import numpy as np
import pandas as pd

from . import automation, config, hub, proof_of_concept, work_volume

MINUTES_PER_HOUR = 60
BY_REQUEST_TYPE = "by_request_type"


def request_type_tiers(skills=None):
    """Which tier may work each request type (spec 4.4, skills.yaml).

    This is the skill matrix 4.4 says the model must never route around.
    """
    skills = skills or config.load_skills()
    tiers = {}
    for tier, codes in skills["request_type_skills"].items():
        for code in codes:
            if code in tiers:
                raise ValueError(f"{code} is claimed by both {tiers[code]} and {tier}")
            tiers[code] = tier
    return tiers


def role_for(work_type, request_type, tiers):
    """The role that works one item (spec Section 1 owner column, 4.4).

    A named owner wins outright. "By request type" follows the request to
    the tier that holds its skill.
    """
    owner = work_type["owner"]
    if owner != BY_REQUEST_TYPE:
        return owner
    if request_type not in tiers:
        raise ValueError(f"{request_type} is not in the skill matrix")
    return tiers[request_type]


def retrieval_times(assumptions=None):
    """Minutes to fetch a chart, by medium (spec Section 1 record media).

    Revised 10/22: this table is the ONLY source of retrieval times. W8, W9
    and W11 all read from it, so the work-type table and the media table
    can never disagree about how long it takes to fetch a chart - which
    they briefly did, at 12 minutes for digital in both places.
    """
    assumptions = assumptions or config.load_assumptions()
    times = {}
    for medium in assumptions["record_media"]:
        if "retrieval_time_minutes" not in medium:
            raise ValueError(f"record media {medium['code']} has no retrieval time")
        times[medium["code"]] = float(medium["retrieval_time_minutes"])
    return times


def handle_time_minutes(work_type, role, media, by_code, media_times):
    """Minutes per item for one work type, role and medium (spec Section 1).

    Order matters: a work type that names a medium takes its time from the
    media table and may not carry one of its own, which is what makes the
    media table the single source.
    """
    if "media" in work_type:
        if "handle_time_minutes" in work_type or "handle_time_minutes_by_media" in work_type:
            raise ValueError(
                f"{work_type['code']} names a medium, so its time must come from "
                "record_media only (spec Section 1, revised 10/22)"
            )
        if media not in media_times:
            raise ValueError(f"{work_type['code']}: no retrieval time for media {media!r}")
        return media_times[media]

    if "handle_time_minutes_by_tier" in work_type:
        times = work_type["handle_time_minutes_by_tier"]
        if role not in times:
            raise ValueError(f"{work_type['code']} has no handle time for {role}")
        return float(times[role])

    if "handle_time_fraction_of" in work_type:
        link = work_type["handle_time_fraction_of"]
        parent = by_code[link["work_type"]]
        return float(link["fraction"]) * handle_time_minutes(
            parent, role, media, by_code, media_times
        )

    if "handle_time_minutes" not in work_type:
        raise ValueError(f"{work_type['code']} has no handle time")
    return float(work_type["handle_time_minutes"])


def required_hours(assumptions=None, sites=None, work=None):
    """Weekly required hours per site, role and work type (spec 4.4, 5.5).

    Returns the hours table plus the items this module deliberately did not
    route, so the caller can prove every work item was accounted for exactly
    once rather than taking it on trust.

    When automation is on (spec 5.4) the items split here, because this is
    where items become hours and so the only place the split can be made
    without counting an hour twice. Each eligible work type keeps the share
    the machine could not take, and the automated share comes BACK as
    human-in-the-loop work (spec 5.5): exceptions and legal sign-off to the
    role that owns the work, the QA sample to the QA team.

    Automation is OFF in the baseline, and this function then takes exactly
    the path it took before 5.4 existed - same columns, same rows, same
    numbers. That is deliberate: the baseline has no automation, so it has
    no business carrying an automation column.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if work is None:
        work = work_volume.build(assumptions, sites)["work"]

    by_code = {w["code"]: w for w in assumptions["work_types"]}
    media_times = retrieval_times(assumptions)
    tiers = request_type_tiers()
    handled_elsewhere = onsite_work_types(assumptions)

    rows = work[~work["work_type"].isin(handled_elsewhere)].copy()
    skipped = work[work["work_type"].isin(handled_elsewhere)]

    rows["role"] = [
        role_for(by_code[code], request_type, tiers)
        for code, request_type in zip(rows["work_type"], rows["request_type"])
    ]
    rows["media"] = rows["media"].fillna("")
    rows["handle_time_minutes"] = [
        handle_time_minutes(by_code[code], role, media or None, by_code, media_times)
        for code, role, media in zip(rows["work_type"], rows["role"], rows["media"])
    ]

    # The central records hub (spec 7, 11/02). The site column is what
    # becomes a staffing scope, so re-attributing it HERE is what moves
    # the work to the hub - and it is the right place rather than the
    # work-volume table, because that table's site is where the REQUEST
    # came from and the onsite interval build still needs six counters.
    #
    # Item counts are untouched: a chart retrieved at the hub is the same
    # chart (the Section 9 hub check).
    if hub.enabled(assumptions):
        rows = hub.move_to_hub(rows, hub.hub_site(assumptions, sites), assumptions)
    # --- the automation split (spec 5.4, 5.5) --------------------------
    automated_plan = automation.build(assumptions, sites, work)
    shares = automation.automated_fraction_by_week(automated_plan)

    # "After proof of concept" (spec 7, 11/02). The gate is read off the
    # automation plan - the QA sample is drawn from the items the machine
    # actually took - so it has to be computed HERE, after the plan exists
    # and before 5.5's return minutes are priced. Empty unless the
    # scenario switched it on, and then every rate below is the 5.5 rate.
    poc_series = None
    qa_multiple, sign_off_rate = {}, {}
    if shares and proof_of_concept.enabled(assumptions):
        poc_series = proof_of_concept.observed_series(automated_plan, assumptions)
        qa_multiple = proof_of_concept.qa_multiple_by_week(automated_plan, assumptions)
        sign_off_rate = proof_of_concept.sign_off_rate_by_week(
            automated_plan, assumptions
        )

    loop_rows = []
    if shares:
        rows["automated_items"] = [
            items * shares.get((week, code), 0.0)
            for items, week, code in zip(
                rows["work_items"], rows["week_index"], rows["work_type"]
            )
        ]
        # What the machine could not take stays with people (spec 5.4), so a
        # short outage shows up as human workload and never as lost work.
        rows["work_items"] = rows["work_items"] - rows["automated_items"]
        # Built from the BASE handle time, before the uplift below: 5.5's
        # multiples price what comes back OFF the machine, and 7.1's
        # uplift prices what people were left with. Each is counted once.
        loop_rows = _human_in_the_loop_rows(
            rows, assumptions, qa_multiple, sign_off_rate
        )
        rows = rows.drop(columns="automated_items")

    # Spec 7.1's supply-side uplift: the machine took the simple items, so
    # the ones left for people are harder. Only automation-eligible work
    # types, only from the go-live week, and exactly 1.0 in the baseline.
    # Priced from the share ACTUALLY removed that week, not the nominal
    # target: the ramp and uptime mean the two differ, and the mix uplift
    # is only honest if it follows what really left.
    rows["handle_time_minutes"] = rows["handle_time_minutes"] * [
        automation.supply_handle_time_multiple(
            code, week, assumptions, removed_share=shares.get((week, code), 0.0)
        )
        for code, week in zip(rows["work_type"], rows["week_index"])
    ]

    rows["required_hours"] = (
        rows["work_items"] * rows["handle_time_minutes"] / MINUTES_PER_HOUR
    )

    # Request types collapse here because the handle time no longer varies
    # within a (role, work type, media) group - W5's split is already
    # carried by the role column. The full per-request-type detail stays in
    # the work-volume file for anything that needs it.
    table = (
        rows.groupby(
            [
                "week_index",
                "week_start",
                "period",
                "site",
                "role",
                "stage",
                "work_type",
                "media",
                "handle_time_minutes",
            ],
            as_index=False,
        )
        .agg(items=("work_items", "sum"), required_hours=("required_hours", "sum"))
        .sort_values(["week_index", "site", "role", "work_type", "media"])
        .reset_index(drop=True)
    )
    if not shares:
        return {
            "hours": table,
            "human": table,
            "human_in_the_loop": pd.DataFrame(columns=table.columns),
            "automation": automated_plan,
            "proof_of_concept": poc_series,
            "routed_elsewhere": skipped,
        }

    # The work_kind column appears only when automation is on. A baseline
    # file that carried a column reading "human" on every row would be
    # noise, and the whole point of 4f is that the baseline file does not
    # change at all.
    table["work_kind"] = "human"
    loop = _group_hours(pd.DataFrame(loop_rows))
    combined = (
        pd.concat([table, loop], ignore_index=True)
        .sort_values(["week_index", "site", "role", "work_type", "media", "work_kind"])
        .reset_index(drop=True)
    )
    return {
        "hours": combined,
        "human": table,
        "human_in_the_loop": loop,
        "automation": automated_plan,
        # The gate series, so the report prints exactly the weeks the
        # hours above were priced from rather than a second computation
        # that could quietly disagree. None unless the scenario is on.
        "proof_of_concept": poc_series,
        "routed_elsewhere": skipped,
    }


HOURS_KEYS = [
    "week_index", "week_start", "period", "site", "role", "stage",
    "work_type", "media", "handle_time_minutes", "work_kind",
]


def _group_hours(rows):
    """Collapse to the hours table's grain, keeping work_kind apart.

    Each kind of returned work keeps its own row because each has its own
    handle time, and 4.3 builds one queue per (role, work type, handle
    time). A two-minute QA look and a nine-minute exception are not the
    same queue and averaging them would flatter both.
    """
    if rows.empty:
        return pd.DataFrame(columns=HOURS_KEYS + ["items", "required_hours"])
    return (
        rows.groupby(HOURS_KEYS, as_index=False)
        .agg(items=("items", "sum"), required_hours=("required_hours", "sum"))
    )


def _human_in_the_loop_rows(
    rows, assumptions, qa_multiple_by_week=None, sign_off_rate_by_week=None
):
    """Automated work coming back to people (spec 5.5).

    Three destinations, and they are not the same place:

      - exceptions go to the tier that OWNS the work type, at 1.5x the
        human handle time, because an item the machine could not finish is
        the hard end of the distribution;
      - the legal sign-off goes to the same owning role at half the handle
        time - a person reads the machine's prep and decides;
      - the QA sample goes to the QA TEAM. Spec Section 9 is explicit that
        every QA item lands on the QA team and never on whoever did the
        work, and for automated items that means never on the machine's own
        owner either.

    Sign-off is scoped per row, not per work type: W16's R8 releases need a
    person because a court order does, while the same work type's routine
    releases do not.

    The two week maps are "After proof of concept" (spec 7, 11/02): the QA
    sample rate falls once the gate is met, and the legal sign-off rate
    moves only if an input says the legal requirement itself changed. Both
    are keyed by WEEK because the gate is met in a particular week, and
    both default to the 5.5 rates, so every other scenario is unchanged.
    """
    qa_multiple_by_week = qa_multiple_by_week or {}
    sign_off_rate_by_week = sign_off_rate_by_week or {}
    emitted = []
    for row in rows.itertuples():
        automated = row.automated_items
        if automated <= 0:
            continue
        sign_off = automation.needs_sign_off(
            row.work_type, row.request_type, assumptions
        )
        minutes = automation.return_minutes(
            row.work_type,
            row.handle_time_minutes,
            assumptions,
            sign_off=sign_off,
            qa_multiple=qa_multiple_by_week.get(row.week_index, 1.0),
            sign_off_rate=sign_off_rate_by_week.get(row.week_index),
        )
        for kind, per_item in minutes.items():
            if per_item <= 0:
                continue
            role = automation.QA_ROLE if kind == "qa_sample" else row.role
            emitted.append(
                {
                    "week_index": row.week_index,
                    "week_start": row.week_start,
                    "period": row.period,
                    "site": row.site,
                    "role": role,
                    "stage": row.stage,
                    "work_type": row.work_type,
                    "media": row.media,
                    "handle_time_minutes": per_item,
                    "work_kind": kind,
                    "items": automated,
                    "required_hours": automated * per_item / MINUTES_PER_HOUR,
                }
            )
    return emitted


def onsite_work_types(assumptions=None):
    """The work types intervals.py already costed (spec 4.1, 4.2).

    W19 and W20 are Erlang C queues; W21 is absorbed into their idle time.
    Routing any of them here would buy the same hour twice.
    """
    assumptions = assumptions or config.load_assumptions()
    live = set(assumptions["breaks_and_lunch"]["relief_placement"]["load_curves"].values())
    support = {assumptions["onsite_team"]["support_work"]["work_type"]}
    return live | support


def fte_by_role(table, assumptions=None):
    """Hours turned into FTE on the spec 5.1 basis.

    Remote roles pool across the network, so hours are summed over sites
    before the division - six sites each needing a third of a person is two
    people in one queue, not six.
    """
    assumptions = assumptions or config.load_assumptions()
    paid = assumptions["fte"]["paid_hours_per_week"]
    shrinkage = assumptions["fte"]["paid_shrinkage"]

    summary = table.groupby("role", as_index=False)["required_hours"].sum()
    summary["shrinkage"] = summary["role"].map(shrinkage)
    if summary["shrinkage"].isna().any():
        missing = summary.loc[summary["shrinkage"].isna(), "role"].tolist()
        raise ValueError(f"no shrinkage in assumptions.yaml for {missing}")
    summary["productive_hours_per_fte"] = paid * (1 - summary["shrinkage"])
    summary["fte"] = summary["required_hours"] / summary["productive_hours_per_fte"]
    return summary.sort_values("required_hours", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The backlog (inventory) model, weekly (spec 4.3, 4.4)
#
# The WFM idea people find counter-intuitive: a healthy back-office queue is
# NOT empty. Work that can wait should wait a little, because a team sized to
# clear every item the moment it lands is a team idle between arrivals.
# Little's Law says how much is allowed to sit there:
#
#     target backlog = daily arrivals x the goal in days
#
# A queue at target is on plan. A queue above target is behind, and the hours
# needed to catch up are spread over the catch-up period rather than demanded
# in one week - which is what stops the plan recommending a hiring spike to
# fix a two-week blip.
#
# Limitation (spec 4.3, repeated in the README): a weekly grain cannot see
# within-day aging, so a 1-day goal is checked as days of inventory rather
# than as a true response time.
# ---------------------------------------------------------------------------


def queue_scope(role, site, assumptions):
    """Which queue an item joins (spec 4.4).

    Remote roles are ONE network queue - a Tier 1 Specialist works the
    pipeline wherever the request came from, so six site queues would be a
    fiction that costs real people. Records Specialists are per site,
    because they walk to a filing cabinet in their own building.
    """
    per_site = assumptions["backlog"]["per_site_roles"]
    return site if role in per_site else assumptions["backlog"]["network_scope"]


def target_backlog(arrivals, goal_business_days, business_days_per_week):
    """Little's Law: the healthy amount of work in the queue (spec 4.3).

    Daily arrivals times the goal in days. Not zero - zero would mean the
    team had nothing to pick up the moment it finished an item.
    """
    if business_days_per_week <= 0:
        raise ValueError("business days per week must be positive")
    return np.asarray(arrivals, dtype=float) / business_days_per_week * goal_business_days


def required_hours_for_week(
    arrivals, starting_backlog, handle_time_minutes, goal_business_days,
    catch_up_weeks, business_days_per_week,
):
    """Hours to work this week's arrivals plus a slice of any excess (4.3).

        required = (arrivals + max(0, starting - target) / catch-up weeks)
                   x handle time / 60

    Only the EXCESS over target is chased, and only a quarter of it at the
    default four-week catch-up. A queue sitting exactly on target asks for
    nothing beyond its arrivals, which is the whole point of having a target.
    """
    if catch_up_weeks <= 0:
        raise ValueError("catch-up weeks must be positive")
    target = target_backlog(arrivals, goal_business_days, business_days_per_week)
    excess = np.maximum(0.0, np.asarray(starting_backlog, dtype=float) - target)
    return (np.asarray(arrivals, dtype=float) + excess / catch_up_weeks) * (
        handle_time_minutes / MINUTES_PER_HOUR
    )


def project_week(
    arrivals, starting_backlog, available_hours, handle_time_minutes,
    goal_business_days, business_days_per_week,
):
    """What actually happens in a week, given the hours on hand (spec 4.3).

    Available hours are an INPUT here, not something this module decides:
    supply arrives in Session 4. Until then the projection answers "if the
    plan gives this queue these hours, where does it end up?".

        capacity  = hours x 60 / handle time
        completed = min(starting + arrivals, capacity)
        ending    = starting + arrivals - completed
        days of inventory = ending / (arrivals / 5)

    Capacity is capped by the work available, so a quiet week does not bank
    phantom output, and the backlog can never go negative.
    """
    if handle_time_minutes <= 0:
        raise ValueError("handle time must be positive")

    arrivals = np.asarray(arrivals, dtype=float)
    starting = np.asarray(starting_backlog, dtype=float)
    capacity = np.asarray(available_hours, dtype=float) * MINUTES_PER_HOUR / handle_time_minutes

    completed = np.minimum(starting + arrivals, capacity)
    ending = starting + arrivals - completed

    daily = arrivals / business_days_per_week
    # A week with no arrivals has no meaningful "days" of inventory: with
    # nothing coming in, any remaining items would sit forever. Reported as
    # zero when the queue is also empty, and as infinite when it is not.
    with np.errstate(divide="ignore", invalid="ignore"):
        days_of_inventory = np.where(
            daily > 0, np.divide(ending, np.where(daily > 0, daily, 1.0)),
            np.where(ending > 0, np.inf, 0.0),
        )

    return {
        "capacity_items": capacity,
        "completed": completed,
        "ending_backlog": ending,
        "days_of_inventory": days_of_inventory,
        "goal_met": days_of_inventory <= goal_business_days + 1e-12,
    }


def roll_forward(
    arrivals, available_hours, handle_time_minutes, goal_business_days,
    catch_up_weeks, starting_backlog, business_days_per_week=5,
):
    """Roll one queue forward week by week (spec 4.3).

    The starting backlog rolls: each week opens with what the last one left
    behind, which is what makes this an inventory model rather than a series
    of unconnected weeks.

    Returns a tidy table, one row per week, in the same columns as the hand
    roll in docs/validation/backlog_hand_roll.xlsx.
    """
    arrivals = np.asarray(arrivals, dtype=float)
    available_hours = np.asarray(available_hours, dtype=float)
    if len(arrivals) != len(available_hours):
        raise ValueError("arrivals and available hours must cover the same weeks")

    rows = []
    opening = float(starting_backlog)
    for week, (week_arrivals, hours) in enumerate(zip(arrivals, available_hours), start=1):
        target = float(
            target_backlog(week_arrivals, goal_business_days, business_days_per_week)
        )
        required = float(
            required_hours_for_week(
                week_arrivals, opening, handle_time_minutes, goal_business_days,
                catch_up_weeks, business_days_per_week,
            )
        )
        outcome = project_week(
            week_arrivals, opening, hours, handle_time_minutes,
            goal_business_days, business_days_per_week,
        )
        rows.append(
            {
                "week": week,
                "arrivals": float(week_arrivals),
                "available_hours": float(hours),
                "starting_backlog": opening,
                "target_backlog": target,
                "required_hours": required,
                "capacity_items": float(outcome["capacity_items"]),
                "completed": float(outcome["completed"]),
                "ending_backlog": float(outcome["ending_backlog"]),
                "days_of_inventory": float(outcome["days_of_inventory"]),
                "goal_met": bool(outcome["goal_met"]),
            }
        )
        opening = rows[-1]["ending_backlog"]

    return pd.DataFrame(rows)


def goal_days(assumptions=None):
    """The backlog goal in business days for each work type."""
    assumptions = assumptions or config.load_assumptions()
    goals = {}
    for work_type in assumptions["work_types"]:
        if "goal_business_days" in work_type:
            goals[work_type["code"]] = float(work_type["goal_business_days"])
    return goals


def build_queues(assumptions=None, sites=None, hours_table=None):
    """One backlog queue per role, work type and scope (spec 4.3, 4.4).

    The baseline opens every queue at its target backlog, so the plan starts
    healthy rather than with an invented hole or pile - and so the turnaround
    it reports is the one the goals describe, not an artefact of where the
    model happened to begin.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if hours_table is None:
        hours_table = required_hours(assumptions, sites)["hours"]

    settings = assumptions["backlog"]
    business_days = assumptions["calendar"]["business_days_per_week"]
    goals = goal_days(assumptions)

    rows = hours_table.copy()
    missing = set(rows["work_type"]) - set(goals)
    if missing:
        raise ValueError(f"no goal_business_days for {sorted(missing)}")

    rows["scope"] = [
        queue_scope(role, site, assumptions)
        for role, site in zip(rows["role"], rows["site"])
    ]
    # work_kind only exists when automation is on (5.5). Grouping by it
    # keeps the loop work as its own queue - it has its own handle time and
    # its own goal - while labelling it, so 4.4 can leave it out of the
    # turnaround weighting (decided 10/27).
    keys = ["week_index", "week_start", "period", "scope", "role", "stage",
            "work_type", "media", "handle_time_minutes"]
    if "work_kind" in rows.columns:
        keys.append("work_kind")
    queues = (
        rows.groupby(keys, as_index=False)
        .agg(arrivals=("items", "sum"), workload_hours=("required_hours", "sum"))
    )
    queues["goal_business_days"] = queues["work_type"].map(goals)
    queues["target_backlog"] = target_backlog(
        queues["arrivals"], queues["goal_business_days"], business_days
    )

    # Baseline: every queue opens at target (spec 4.3). With supply absent
    # until Session 4 there is nothing yet to carry a backlog forward from,
    # so each week is planned from a healthy opening position.
    queues["starting_backlog"] = (
        queues["target_backlog"] if settings.get("start_at_target", True) else 0.0
    )
    queues["required_hours"] = required_hours_for_week(
        queues["arrivals"],
        queues["starting_backlog"],
        queues["handle_time_minutes"],
        queues["goal_business_days"],
        settings["catch_up_weeks"],
        business_days,
    )
    queues["days_of_inventory"] = np.where(
        queues["arrivals"] > 0,
        queues["starting_backlog"] / (queues["arrivals"] / business_days),
        0.0,
    )
    return queues


TURNAROUND_STAGES = ("intake", "validation", "retrieval", "review", "release")


HUMAN_WORK_KIND = "human"


def on_the_turnaround_path(queues):
    """The queue lines a requester actually waits through (spec 4.4, 10/27).

    Human-in-the-loop work (5.5) carries hours and cost to its role, but a
    QA look at an automated intake item is part of THAT item's journey -
    not a second wait on top of it. Counting it as another queue line with
    its own days of inventory let 8% of the volume in at a fifth of the
    inventory and flattered the stage average.

    Each item counts once. With automation off there is no work_kind column
    and nothing is filtered, so the baseline is untouched.
    """
    if "work_kind" not in queues.columns:
        return queues
    return queues[queues["work_kind"] == HUMAN_WORK_KIND]


def turnaround(queues, assumptions=None):
    """Turnaround as the sum of five stages (spec 4.4, decided 10/22).

    Each stage is the VOLUME-WEIGHTED AVERAGE days of inventory of its work
    types, and the five stage figures are then added.

    Averaging within a stage and adding across them is the whole trick, and
    getting it backwards would be badly wrong in both directions. Steps
    inside a stage are alternatives or run side by side - a chart is
    retrieved digitally OR from a filing cabinet, not both - so adding them
    would invent delay that no single request ever experiences. Stages are
    genuinely sequential: nothing is released before it is reviewed.

    Weighting by volume is what makes the average honest. Onsite retrieval
    takes 3 days but touches only about 12% of charts, so it lifts the
    retrieval stage to roughly 2.1 rather than dragging it to 3.

    Returns (per-stage table, per-step detail, total days).
    """
    assumptions = assumptions or config.load_assumptions()
    in_path = {w["code"] for w in assumptions["work_types"] if w.get("in_turnaround")}

    # Each item counts once (4.4, decided 10/27).
    queues = on_the_turnaround_path(queues)
    steps = queues[queues["work_type"].isin(in_path)].copy()
    unexpected = set(steps["stage"]) - set(TURNAROUND_STAGES)
    if unexpected:
        raise ValueError(f"work types on the request path sit outside the stages: {unexpected}")

    by_step = (
        steps.groupby(["stage", "work_type"], as_index=False)
        .agg(
            arrivals=("arrivals", "sum"),
            days_of_inventory=("days_of_inventory", "mean"),
            goal_business_days=("goal_business_days", "max"),
        )
        .sort_values(["stage", "work_type"])
    )

    rows = []
    for stage in TURNAROUND_STAGES:
        in_stage = by_step[by_step["stage"] == stage]
        if in_stage.empty:
            raise ValueError(f"no work types found for the {stage} stage")
        weights = in_stage["arrivals"].to_numpy()
        if weights.sum() <= 0:
            raise ValueError(f"the {stage} stage has no volume to weight by")
        rows.append(
            {
                "stage": stage,
                "arrivals": float(weights.sum()),
                "days_of_inventory": float(
                    np.average(in_stage["days_of_inventory"], weights=weights)
                ),
                "goal_business_days": float(
                    np.average(in_stage["goal_business_days"], weights=weights)
                ),
            }
        )

    by_stage = pd.DataFrame(rows)
    return by_stage, by_step, float(by_stage["days_of_inventory"].sum())


def write_required_hours(result, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    paths = assumptions["paths"]
    out_dir = config.resolve_path(paths["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / paths["required_hours_csv"]
    result["hours"].to_csv(path, index=False, float_format="%.6f")
    return path


def main():
    assumptions, sites, _skills = config.load_all()
    result = required_hours(assumptions, sites)
    path = write_required_hours(result, assumptions)

    table = result["hours"]
    first_plan_week = assumptions["calendar"]["history_weeks"]
    week = table[table.week_index == first_plan_week]
    monday = week["week_start"].iloc[0]

    print(f"Wrote {len(table):,} rows to {path}")
    print(f"  {len(result['routed_elsewhere']):,} rows left to intervals.py "
          f"({', '.join(sorted(onsite_work_types(assumptions)))})")

    print(f"\nFirst plan week ({monday}): required hours by role, network")
    summary = fte_by_role(week, assumptions)
    headcount = {
        role: sum(scopes.values())
        for role, scopes in assumptions["fte"]["starting_headcount"].items()
    }
    print(f"  {'role':<20}{'hours':>9}{'h/FTE':>8}{'FTE':>8}{'spec 5.1':>10}{'gap':>8}")
    for row in summary.itertuples():
        have = headcount.get(row.role)
        gap = "" if have is None else f"{have - row.fte:+8.1f}"
        print(
            f"  {row.role:<20}{row.required_hours:9.1f}"
            f"{row.productive_hours_per_fte:8.1f}{row.fte:8.2f}"
            f"{'' if have is None else have:>10}{gap}"
        )
    print(f"  {'TOTAL':<20}{summary.required_hours.sum():9.1f}"
          f"{'':8}{summary.fte.sum():8.2f}")

    print(f"\nBy stage, first plan week:")
    for stage, hours in week.groupby("stage")["required_hours"].sum().sort_values(
        ascending=False
    ).items():
        print(f"  {stage:<12}{hours:9.1f} h")

    # --- the backlog queues (spec 4.3, 4.4) -------------------------------
    queues = build_queues(assumptions, sites, table)
    paths = assumptions["paths"]
    queue_path = config.resolve_path(
        paths["synthetic_data_dir"], paths["backlog_queues_csv"]
    )
    queues.to_csv(queue_path, index=False, float_format="%.6f")
    print(f"\nWrote {len(queues):,} backlog-queue rows to {queue_path}")
    print(
        f"  {queues['scope'].nunique()} scopes: "
        f"{', '.join(sorted(queues['scope'].unique()))}"
    )

    plan_queues = queues[queues.week_index == first_plan_week]
    by_stage, by_step, total = turnaround(plan_queues, assumptions)
    goal = assumptions["backlog"]["overall_turnaround_goal_days"]

    print("\nBaseline turnaround, first plan week (spec 4.4):")
    print(f"  {'stage':<12}{'days':>7}{'share':>8}   steps (goal d, volume)")
    for row in by_stage.itertuples():
        detail = by_step[by_step.stage == row.stage]
        parts = ", ".join(
            f"{s.work_type} {s.goal_business_days:g}d x{s.arrivals / row.arrivals:.0%}"
            for s in detail.itertuples()
        )
        print(
            f"  {row.stage:<12}{row.days_of_inventory:7.3f}"
            f"{row.days_of_inventory / total:8.1%}   {parts}"
        )
    print(
        f"  {'TOTAL':<12}{total:7.3f}           against a {goal}-business-day goal"
        f"  ({total - goal:+.3f})"
    )
    print(
        "  stages are added; steps inside a stage are volume-weighted averages,\n"
        "  because a chart is retrieved digitally OR from a cabinet, not both"
    )


if __name__ == "__main__":
    main()
