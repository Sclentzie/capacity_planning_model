"""Automated throughput and human-in-the-loop work (spec 5.4, 5.5).

Automation is a SUPPLY line, not a discount on demand. That distinction is
spec 7.1's, decided 10/21, and it is the one that keeps the arithmetic
honest: deflection means the contact never arrives, automation means the
work still arrives and a machine does it. So a machine is planned exactly
like a hiring class - it has a capacity, a go-live date, a ramp and a cost -
and it belongs in the Capacity Ledger beside the people.

    Spec 5.4:
        automated items = min(eligible items x target share,
                              capacity cap x uptime x ramp)

The $500 weekly platform fee is ONE fee for the whole network (decided
10/27), split across the automated work types by each one's share of that
week's automated items. Setup is per work type and spread over 52 weeks
from go-live. Both matter to 6.6, because a per-item comparison that left
the fixed costs out would make a row doing twelve items a week look as
cheap as one doing two thousand.

Items the machine cannot take that week stay with people, so an outage
shows up as human workload and never as lost work. That is the whole reason
the formula is a minimum of two things rather than a single percentage.

The part that catches planners out is 5.5: AUTOMATED WORK COMES BACK.
Exceptions return to the tier that owns the work at one and a half times
the human handle time, the QA sample goes to the QA team, and a legally
required sign-off sends every single item to a person. An 8% target share
does not free 8% of the hours, and a plan that assumed it did would promise
savings it cannot deliver.

    Spec 5.5:
        human-in-the-loop hours
            = automated items x (exception rate x exception time
                                 + QA sample rate x review time
                                 + sign-off rate x sign-off time) / 60

This is a different thing from the handle-time uplift in 7.1. The uplift
says the work LEFT for people is harder; human-in-the-loop says automated
work comes BACK to people. Each is counted once, and this module has
nothing to do with the uplift.

THE BASELINE KEEPS EVERY RATE AT ZERO (spec 7.1; turning them up is a
Session 5 scenario), so `enabled()` is false and every function here
returns nothing. That is deliberate and it is tested: the baseline results
must be bit-identical to the plan before this module existed.

This module depends only on config, so backlog.py and supply.py can both
read it without a circular import.
"""

import numpy as np
import pandas as pd

from . import config

MINUTES_PER_HOUR = 60

# The three ways automated work comes back to a person (spec 5.5), and who
# takes each one. The QA sample is the odd one out: it is the only kind that
# leaves the owning role, because spec Section 9 insists every QA item lands
# on the QA team and never on the tier that did the work - which for
# automated items means never back on the machine's own owner.
RETURN_KINDS = ("exception", "qa_sample", "sign_off")
QA_ROLE = "qa_specialist"


def settings(assumptions=None):
    """The automation block (spec 5.4, 5.5)."""
    assumptions = assumptions or config.load_assumptions()
    return assumptions["automation"]


def rates(assumptions=None):
    """The four supply-side rates from spec 7.1."""
    return settings(assumptions)["rates"]


def target_share(assumptions=None):
    """Share of eligible items the machine is aimed at (spec 5.4).

        simple task share x (AI handling + automation displacement)
        x realization

    At spec 7.1's defaults that is 40% x 20% x 100% = 8%, the figure named
    in the 5.4 input table. At the baseline's zeroes it is 0%.

    Spec 7.1: rates in the same group add together within the simple share,
    and a total above 100% is refused rather than quietly clipped - a
    scenario that asks for 60% AI plus 60% automation has a typo in it, and
    silently capping at 100% would hide the typo and the plan would be
    argued from a number nobody chose.
    """
    rate = rates(assumptions)
    group = float(rate["ai_handling"]) + float(rate["automation_displacement"])
    if group > 1.0 + 1e-12:
        raise ValueError(
            f"AI handling {rate['ai_handling']:.0%} plus automation displacement "
            f"{rate['automation_displacement']:.0%} is {group:.0%}, over 100% "
            "(spec 7.1)"
        )
    simple = (
        1.0
        if rate.get("apply_to_all_volume", False)
        else float(rate["simple_task_share"])
    )
    return simple * group * float(rate["realization"])


def enabled(assumptions=None):
    """Whether any automation runs at all.

    The baseline answer is False, and every caller checks it before doing
    anything, so the baseline plan takes exactly the path it took before
    this module existed. Cheaper than trusting that multiplying by zero is
    always exactly zero - and much easier to prove.
    """
    assumptions = assumptions or config.load_assumptions()
    if target_share(assumptions) > 0:
        return True
    # A row can be aimed independently of the network rates, so a scenario
    # that automates one work type only still counts as switched on.
    return any(
        float(row.get("share_override", 0.0)) > 0
        for row in settings(assumptions)["work_types"].values()
        if row.get("eligible", False)
    )


def work_type_settings(code, assumptions=None):
    """One work type's automation row (spec Section 1, last column)."""
    rows = settings(assumptions)["work_types"]
    if code not in rows:
        raise ValueError(f"no automation settings for work type {code!r}")
    return rows[code]


def eligible_codes(assumptions=None):
    """Work types a supply-side machine can take (spec Section 1).

    W19, W20 and W21 are absent: they are DEMAND-side (a self-service
    kiosk, call and email deflection), so the contact never arrives and
    there is nothing for a machine on the supply side to do. Counting them
    here would take the same saving twice.
    """
    rows = settings(assumptions)["work_types"]
    return sorted(code for code, row in rows.items() if row.get("eligible", False))


def targeted_work_types(assumptions=None):
    """The eligible rows the machine is pointed at (spec 7, added 10/28).

    Empty means all of them. A non-empty list is the "AI targeted" run:
    automate only what pays back after fixed costs, because setup is
    $7,692 a work type a year whether the row does two thousand items a
    week or a hundred.
    """
    assumptions = assumptions or config.load_assumptions()
    return list(rates(assumptions).get("only_work_types") or [])


def row_target_share(code, assumptions=None):
    """The share aimed at one work type (spec Section 1, 5.4, 7).

    Usually the network target share, but three things can pin it to zero:

      - spec Section 1 pins two rows whatever the network rates say: W11
        onsite retrieval, because somebody physically walks to the archive
        room, and W24 the fulfiller self-check. An override of 0 is a real
        value, not a missing one.
      - a row that is not eligible at all.
      - a targeted run (10/28) naming the rows the machine is pointed at,
        which leaves every other eligible row untouched.
    """
    row = work_type_settings(code, assumptions)
    if not row.get("eligible", False):
        return 0.0
    targeted = targeted_work_types(assumptions)
    if targeted and code not in targeted:
        return 0.0
    if "share_override" in row:
        return float(row["share_override"])
    return target_share(assumptions)


def ramp_fraction(weeks_since_go_live, assumptions=None):
    """How much of the capacity is usable in its nth week (spec 5.4).

    25 / 50 / 75 / 100%, holding at full speed afterwards. Before go-live
    it is zero - which is the point of having a go-live date at all, and
    the reason 6.6 can compare automation's lead time against a hire's.
    """
    ramp = settings(assumptions)["ramp"]
    if weeks_since_go_live < 0:
        return 0.0
    return float(ramp[min(int(weeks_since_go_live), len(ramp) - 1)])


def go_live_week_index(assumptions=None):
    """The go-live week as an absolute week index.

    The YAML holds it as a PLAN week, because "week 13" is how a leader
    says it; everything downstream counts in absolute week indexes.
    """
    assumptions = assumptions or config.load_assumptions()
    return (
        assumptions["calendar"]["history_weeks"]
        + int(settings(assumptions)["go_live_week"])
        - 1
    )


def available_fraction(week_index, assumptions=None):
    """Uptime x ramp: the share of the cap the machine can actually use."""
    assumptions = assumptions or config.load_assumptions()
    ramp = ramp_fraction(week_index - go_live_week_index(assumptions), assumptions)
    return float(settings(assumptions)["uptime"]) * ramp


def capacity_cap(code, eligible_items, assumptions=None):
    """Items per week the platform can take (spec 5.4).

    Network-wide per work type, because an automation platform is not
    installed per site. `null` in the YAML means "sized to the target
    share", so only uptime and the ramp hold it back - the honest default,
    because a cap nobody has measured is a guess dressed as a constraint.
    """
    stated = settings(assumptions)["capacity_cap_items_per_week"]
    if stated is None:
        return eligible_items * row_target_share(code, assumptions)
    if isinstance(stated, dict):
        if code not in stated:
            return eligible_items * row_target_share(code, assumptions)
        return float(stated[code])
    return float(stated)


def automated_items(code, eligible_items, week_index, assumptions=None):
    """Spec 5.4's formula, for one work type in one week, network-wide.

        min(eligible items x target share, cap x uptime x ramp)

    Never more than the items that exist, which the minimum already
    guarantees while the target share is a share. A scenario that set a cap
    above the volume would still only automate what arrived.
    """
    assumptions = assumptions or config.load_assumptions()
    aimed = eligible_items * row_target_share(code, assumptions)
    capacity = capacity_cap(code, eligible_items, assumptions) * available_fraction(
        week_index, assumptions
    )
    return max(0.0, min(aimed, capacity, eligible_items))


def build(assumptions=None, sites=None, work=None):
    """The automated supply line, week by week (spec 5.4).

    One row per week and eligible work type, network-wide, because the
    machine is network-wide. Site-level splitting happens where the hours
    are built (backlog.py), in proportion to each site's own items, so no
    site is credited with capacity it did not have.

    Returns an empty frame when automation is off, which is the baseline.
    """
    assumptions = assumptions or config.load_assumptions()
    columns = [
        "week_index", "week_start", "period", "work_type", "kind",
        "eligible_items", "target_share", "ramp", "uptime", "capacity_cap",
        "capacity_items", "automated_items", "human_items",
        "automated_fraction", "item_cost", "setup_cost", "platform_cost",
        "total_cost",
    ]
    if not enabled(assumptions):
        return pd.DataFrame(columns=columns)

    if work is None:
        from . import work_volume

        sites = sites or config.load_sites()
        work = work_volume.build(assumptions, sites)["work"]

    codes = set(eligible_codes(assumptions))
    mine = work[work["work_type"].isin(codes)]
    totals = mine.groupby(
        ["week_index", "week_start", "period", "work_type"], as_index=False
    ).agg(eligible_items=("work_items", "sum"))

    cost = settings(assumptions)["cost"]
    per_item = {"ai": float(cost["ai_per_item"]), "rule": float(cost["rule_per_item"])}
    setup_weekly = float(cost["setup_per_work_type"]) / float(cost["setup_spread_weeks"])

    rows = []
    for row in totals.itertuples():
        code = row.work_type
        row_settings = work_type_settings(code, assumptions)
        share = row_target_share(code, assumptions)
        weeks_since = row.week_index - go_live_week_index(assumptions)
        ramp = ramp_fraction(weeks_since, assumptions)
        uptime = float(settings(assumptions)["uptime"])
        cap = capacity_cap(code, row.eligible_items, assumptions)
        automated = automated_items(code, row.eligible_items, row.week_index, assumptions)
        kind = row_settings["kind"]
        live = automated > 0
        rows.append(
            {
                "week_index": row.week_index,
                "week_start": row.week_start,
                "period": row.period,
                "work_type": code,
                "kind": kind,
                "eligible_items": row.eligible_items,
                "target_share": share,
                "ramp": ramp,
                "uptime": uptime,
                "capacity_cap": cap,
                "capacity_items": cap * uptime * ramp,
                "automated_items": automated,
                "human_items": row.eligible_items - automated,
                "automated_fraction": (
                    automated / row.eligible_items if row.eligible_items > 0 else 0.0
                ),
                "item_cost": automated * per_item[kind],
                # Setup is spread over a year from go-live (spec 5.4) and
                # charged only while the row is actually running: a platform
                # nobody switched on has not been bought.
                "setup_cost": setup_weekly if live else 0.0,
                "platform_cost": 0.0,
            }
        )

    table = pd.DataFrame(rows, columns=[c for c in columns if c != "total_cost"])
    if table.empty:
        return pd.DataFrame(columns=columns)

    # ONE shared platform fee for the whole network (spec 5.4, decided
    # 10/27), split across the automated work types by each one's share of
    # that week's automated items. Not one fee per work type: there is one
    # platform, and charging it per row would multiply a single invoice by
    # however many work types happened to be switched on.
    #
    # Pro rata by items rather than evenly, because evenly would load the
    # same $500 onto a row doing 12 items a week as onto one doing 2,000,
    # and make the small row look uneconomic for a reason that has nothing
    # to do with it. A row that automated nothing carries none of it.
    platform = float(cost["platform_per_week"])
    for _week, group in table.groupby("week_index"):
        live = group["automated_items"].sum()
        if live <= 0:
            continue
        table.loc[group.index, "platform_cost"] = (
            platform * group["automated_items"] / live
        )

    table["total_cost"] = (
        table["item_cost"] + table["setup_cost"] + table["platform_cost"]
    )
    return table.sort_values(["week_index", "work_type"]).reset_index(drop=True)


def automated_fraction_by_week(table):
    """{(week, work type): share automated}, for splitting site-level items.

    The machine does not care which site an item came from, so the share is
    uniform within a work type and a week. Applying one fraction to every
    site's rows is what makes human items plus automated items add back to
    the total exactly, which is the conservation check in Section 9.
    """
    if table.empty:
        return {}
    return {
        (row.week_index, row.work_type): row.automated_fraction
        for row in table.itertuples()
    }


def easy_item_minutes(
    handle_time_minutes, assumptions=None, code=None, week_index=None,
    removed_share=None, uplifted=False,
):
    """What an item the machine took was worth by hand (spec 7.1, 10/28).

        easy-item handle time = ORIGINAL handle time x easy ratio

    One helper for the whole model, because three different places need
    the same number and they must not disagree: 5.5's exception and
    sign-off times, 6.6's by-hand cost, and the weekly comparison.

    The WFM point is a single sentence. A machine does not take a random
    sample of the work; it takes the simple end. Everything downstream of
    "which items left" has to be priced on those items rather than on the
    average, or the model flatters automation in one place and punishes it
    in another - which is exactly what it did until 10/28, charging a
    9-minute exception on a 3-minute item while valuing that item at a
    6-minute saving.

    ORIGINAL, not uplifted (decided 10/28). The handle time on a human
    required-hours row carries 7.1's mix uplift, so a W1 intake row reads
    6.26 minutes rather than 6.00 - and half of 6.26 is not what an easy
    intake item was ever worth. The mix uplift describes what is LEFT
    after the machine took its share; the easy item is what it TOOK. Using
    the uplifted figure let the uplift feed back into the valuation of the
    very items whose removal caused it.

    `uplifted=True` says the handle time passed in carries that uplift and
    must have it divided out. The caller names the row (`code`, and either
    `week_index` or `removed_share`) so the exact multiple can be
    reconstructed - the same multiple that was applied, not an estimate.
    """
    assumptions = assumptions or config.load_assumptions()
    minutes = float(handle_time_minutes)
    if uplifted:
        if code is None or week_index is None:
            raise ValueError(
                "uplifted=True needs the work type and week, to know which "
                "multiple to divide out"
            )
        multiple = supply_handle_time_multiple(
            code, week_index, assumptions, removed_share=removed_share
        )
        if multiple <= 0:
            raise ValueError(f"{code}: a handle-time multiple of {multiple} cannot be undone")
        minutes = minutes / multiple
    return minutes * float(settings(assumptions)["easy_handle_time_relative"])


def return_minutes(
    code,
    handle_time_minutes,
    assumptions=None,
    sign_off=True,
    qa_multiple=1.0,
    sign_off_rate=None,
):
    """Minutes per AUTOMATED item that come back to a person (spec 5.5).

    Returned broken out by kind, because the three go to different places:
    exceptions and sign-off to the role that owns the work, the QA sample
    to the QA team. A single total would have to be billed to somebody, and
    whichever one it was would be wrong.

    `sign_off=False` is for the part of a work type where no legal
    requirement applies - W16's non-R8 releases, for instance - where the
    machine's output is simply the answer.

    Priced off the EASY-item handle time since 10/28, not the average one.

    `qa_multiple` and `sign_off_rate` are what "After proof of concept"
    (spec 7, 11/02) moves. Both default to leaving 5.5 exactly as it is,
    so every other scenario takes the path it took before that scenario
    existed. They are passed in per WEEK rather than read from the
    assumptions here, because the gate is met in a particular week and
    this function has no idea which week it is pricing - see
    proof_of_concept.py for the forward pass that decides.
    """
    assumptions = assumptions or config.load_assumptions()
    loop = settings(assumptions)["human_in_the_loop"]
    row = work_type_settings(code, assumptions)
    if sign_off_rate is None:
        sign_off_rate = float(loop["sign_off_rate"])

    # Both multiples are against the EASY-item time (5.5, decided 10/28):
    # the machine took the simple items, so what comes back off it is a
    # multiple of a simple job. The QA review is a flat two minutes and so
    # is not scaled at all.
    easy = easy_item_minutes(handle_time_minutes, assumptions)

    exception_minutes = (
        float(row.get("exception_rate", 0.0))
        * float(loop["exception_handle_time_multiple"])
        * easy
    )
    qa_minutes = (
        float(row.get("qa_sample_rate", 0.0))
        * float(qa_multiple)
        * float(loop["qa_review_minutes"])
    )
    sign_off_minutes = 0.0
    if sign_off and row.get("sign_off", "none") != "none":
        sign_off_minutes = (
            float(sign_off_rate)
            * float(loop["sign_off_handle_time_multiple"])
            * easy
        )
    return {
        "exception": exception_minutes,
        "qa_sample": qa_minutes,
        "sign_off": sign_off_minutes,
    }


def human_in_the_loop_hours(
    code,
    automated,
    handle_time_minutes,
    assumptions=None,
    sign_off=True,
    qa_multiple=1.0,
    sign_off_rate=None,
):
    """Spec 5.5's formula, as hours, broken out by kind."""
    minutes = return_minutes(
        code,
        handle_time_minutes,
        assumptions,
        sign_off,
        qa_multiple=qa_multiple,
        sign_off_rate=sign_off_rate,
    )
    return {
        kind: automated * value / MINUTES_PER_HOUR for kind, value in minutes.items()
    }


def needs_sign_off(code, request_type, assumptions=None):
    """Whether a legal requirement says a person must decide (spec Section 1).

    Three shapes, because the spec has three. W7 is sign-off throughout -
    a legal and sensitive-records review is the decision. W16 and W17 are
    sign-off only for the subpoena and court-order path, so the scope is a
    list of request types. Everything else is none.
    """
    row = work_type_settings(code, assumptions)
    scope = row.get("sign_off", "none")
    if scope == "none":
        return False
    if scope == "all":
        return True
    if scope == "request_types":
        return request_type in set(row.get("sign_off_request_types", []))
    raise ValueError(f"work type {code}: unknown sign_off scope {scope!r}")


# ---------------------------------------------------------------------------
# Deflection, the demand side (spec 7.1, decided 10/21)
#
# Deflection means the contact NEVER ARRIVES. Automation means it arrives
# and a machine does it. Keeping the two apart is the whole reason 7.1 is
# written as two groups of rates, and the reason this module holds both.
#
# The planning lesson is in 7.2 and it is counter-intuitive enough to be
# worth stating twice: deflecting SIMPLE contacts removes the easy ones,
# so what is left takes longer. At the 7.1 defaults 1,000 calls at 5
# minutes (5,000 minutes of workload) becomes 920 calls at 5.5 minutes -
# 5,060 minutes. Workload RISES 1.2% while 80 calls disappear.
#
# Break-even is at an 8.7% uplift: above that, deflection costs hours.
# ---------------------------------------------------------------------------


def deflection_settings(assumptions=None):
    """The deflection block (spec 7.1)."""
    assumptions = assumptions or config.load_assumptions()
    return assumptions["scenario"]["deflection"]


def deflection_enabled(assumptions=None):
    """Whether any contact is deflected. False in the baseline.

    The baseline keeps the 7.1 default rates visible in the YAML and
    switches the block off, rather than zeroing the rates - so a reader
    sees what the scenario will do, and a scenario file flips one boolean.
    """
    return deflected_share(assumptions) > 0


def deflected_share(assumptions=None):
    """Share of contacts that never arrive (spec 7.1).

        simple contact share x (AI deflection + self-service) x realization

    At the 7.1 defaults that is 40% x 20% x 100% = 8%, which is the 80 of
    1,000 calls in the 7.2 worked example.

    Refused above 100% within the group, the same way the supply side is:
    a scenario asking to deflect 60% by AI and 60% by self-service has a
    typo, and clipping it would hide the typo.
    """
    assumptions = assumptions or config.load_assumptions()
    settings = deflection_settings(assumptions)
    if not settings["enabled"]:
        return 0.0
    group = float(settings["ai_deflection"]) + float(settings["self_service"])
    if group > 1.0 + 1e-12:
        raise ValueError(
            f"AI deflection {settings['ai_deflection']:.0%} plus self-service "
            f"{settings['self_service']:.0%} is {group:.0%}, over 100% (spec 7.1)"
        )
    simple = (
        1.0
        if settings.get("apply_to_all_volume", False)
        else float(settings["simple_contact_share"])
    )
    # One realization for both sides (spec 7.1), kept in one place so the
    # two halves of a scenario cannot drift apart.
    return simple * group * float(rates(assumptions)["realization"])


def effective_uplift(stated, assumptions=None):
    """A stated handle-time uplift, scaled by realization (spec 7.1, 10/27).

        effective uplift = stated uplift x realization

    Because the uplift is a CONSEQUENCE of the mix changing, not a promise
    anybody made: half the simple work removed shifts the mix half as
    much. Scaling the rates while leaving the uplifts at full strength
    would have made a half-delivered programme worse than a whole one,
    which is an artefact of the arithmetic rather than a finding.

    At 100% realization this returns the stated uplift unchanged, so spec
    7.2's worked example is untouched.
    """
    assumptions = assumptions or config.load_assumptions()
    return float(stated) * float(rates(assumptions)["realization"])


def mix_uplift(removed_share, easy_ratio):
    """How much more the AVERAGE remaining item costs (spec 7.1, 10/28).

        mix uplift = (1 - removed share x easy ratio)
                     / (1 - removed share) - 1

    Calculated, never chosen. Take a share of the items away and those
    items were cheap - worth `easy_ratio` of the average - so the ones
    left average more minutes each. The numerator is the workload that
    remains and the denominator the count that remains, so the ratio is
    the new average over the old.

    It cannot raise total WORKLOAD, however large it looks. Workload after
    is (1 - removed share x easy ratio) of what it was, which is at most 1
    and only reaches 1 when the removed items were free. That bound is
    `mix_uplift_ceiling`, and the flat +10% this model used until 10/28
    was above it - which implied the deflected calls had taken less than
    zero minutes.
    """
    removed_share = float(removed_share)
    if removed_share <= 0.0:
        return 0.0
    if removed_share >= 1.0:
        raise ValueError(
            f"cannot remove {removed_share:.0%} of the items and price what is left"
        )
    return (1.0 - removed_share * float(easy_ratio)) / (1.0 - removed_share) - 1.0


def mix_uplift_ceiling(removed_share):
    """The most the mix alone can ever lift the average (spec 7.2, 10/28).

        1 / (1 - removed share) - 1

    Reached when the removed items were free. At that ceiling total
    workload is exactly FLAT - the same minutes spread over fewer items -
    so any uplift above it is not a mix effect at all.
    """
    removed_share = float(removed_share)
    if removed_share <= 0.0:
        return 0.0
    if removed_share >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - removed_share) - 1.0


def deflection_mix_uplift(assumptions=None):
    """The demand-side mix uplift at the configured rates (spec 7.1)."""
    assumptions = assumptions or config.load_assumptions()
    settings = deflection_settings(assumptions)
    return mix_uplift(
        deflected_share(assumptions), settings["easy_handle_time_relative"]
    )


def deflection_handle_time_multiple(code, week_index, assumptions=None):
    """What a contact left after deflection costs (spec 7.1, revised 10/28).

        (1 + mix uplift) x (1 + effective escalation)

    Two separate things, because they come from different places. The MIX
    uplift is arithmetic: the cheap contacts went, so the average rose.
    The ESCALATION uplift is a judgement about the requester: somebody who
    tried a portal first and rang anyway is further into their problem.

    Returns exactly 1.0 outside the deflected work types and before the
    starts week, so the baseline multiplies by one and nothing moves.
    """
    assumptions = assumptions or config.load_assumptions()
    if not deflection_enabled(assumptions):
        return 1.0
    if code not in set(deflection_work_types(assumptions)):
        return 1.0
    if week_index < deflection_from_week(assumptions):
        return 1.0
    escalation = effective_uplift(
        deflection_settings(assumptions)["escalation_uplift"], assumptions
    )
    return (1.0 + deflection_mix_uplift(assumptions)) * (1.0 + escalation)


def deflection_break_even_uplift(assumptions=None):
    """The ESCALATION at which deflection stops saving hours (spec 7.2, 10/28).

        workload after = (1 - removed share x easy ratio) x (1 + escalation)
        break-even when that equals 1

    At the 7.1 defaults 8% of contacts go and they were worth half an
    average one, so workload falls to 96% and break-even escalation is
    1/0.96 - 1 = 4.17%. In 7.2's units: 200 minutes saved over the 4,800
    that remain.

    Note what this is NOT. Until 10/28 the model compared a flat uplift
    against 1/(1 - removed share) - 1, which is the ceiling on the MIX
    effect rather than a break-even at all - at that ceiling workload is
    exactly flat. The mix can never cost hours; only escalation can.
    """
    assumptions = assumptions or config.load_assumptions()
    share = deflected_share(assumptions)
    if share <= 0.0:
        return 0.0
    easy = float(deflection_settings(assumptions)["easy_handle_time_relative"])
    remaining = 1.0 - share * easy
    if remaining <= 0.0:
        return float("inf")
    return 1.0 / remaining - 1.0


def deflection_work_types(assumptions=None):
    """Which contact types deflection reaches (spec 7.1).

    Status calls and emails. NOT walk-ins: somebody standing at the
    counter has already arrived, and the self-service kiosk is its own
    input in Section 1.
    """
    return list(deflection_settings(assumptions)["work_types"])


def deflection_from_week(assumptions=None):
    """The week deflection begins - the scenario's own starts week."""
    assumptions = assumptions or config.load_assumptions()
    return config.scenario_starts_week(assumptions)


def supply_handle_time_multiple(
    code, week_index, assumptions=None, removed_share=None
):
    """What a task left for people costs (spec 7.1, revised 10/28).

        (1 + mix uplift) x (1 + effective extra uplift)

    The same split as the demand side. A machine takes the SIMPLE items,
    so the ones left average more minutes each - calculated, from the
    share actually removed that week rather than from the nominal target,
    because the ramp and uptime mean the two differ. Anything beyond the
    mix is the `extra_uplift` input, and its default is zero.

    Counted once. The human-in-the-loop multiples in 5.5 are measured
    against the BASE handle time, because they price what comes back off
    the machine rather than what people were left with.
    """
    assumptions = assumptions or config.load_assumptions()
    if not enabled(assumptions):
        return 1.0
    if week_index < go_live_week_index(assumptions):
        return 1.0
    row = work_type_settings(code, assumptions)
    if not row.get("eligible", False):
        return 1.0
    if removed_share is None:
        removed_share = row_target_share(code, assumptions)
    if removed_share <= 0:
        return 1.0

    block = settings(assumptions)
    mix = mix_uplift(removed_share, block["easy_handle_time_relative"])
    extra = effective_uplift(block["extra_uplift"], assumptions)
    return (1.0 + mix) * (1.0 + extra)


def write_outputs(table, assumptions=None):
    """Write the automated supply line, if there is one.

    Nothing is written when automation is off, so the baseline's output
    directory is exactly what it was before this module existed.
    """
    assumptions = assumptions or config.load_assumptions()
    if table.empty:
        return None
    out_dir = config.resolve_path(assumptions["paths"]["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / assumptions["paths"]["automation_csv"]
    table.to_csv(path, index=False, float_format="%.6f")
    return path


def main():
    assumptions, sites, _skills = config.load_all()
    if not enabled(assumptions):
        print("Automation is OFF in the baseline (spec 7.1: every rate 0%).")
        print(f"  target share of eligible items: {target_share(assumptions):.1%}")
        print(f"  eligible work types:            {len(eligible_codes(assumptions))}")
        print("Turning it on is a Session 5 scenario.")
        return

    table = build(assumptions, sites)
    path = write_outputs(table, assumptions)
    print(f"  automation -> {path}")
    print(f"\nTarget share of eligible items: {target_share(assumptions):.1%}")
    totals = table.groupby("work_type").agg(
        automated=("automated_items", "sum"),
        human=("human_items", "sum"),
        cost=("total_cost", "sum"),
    )
    print(f"  {'work type':<11}{'automated':>12}{'human':>12}{'cost':>12}")
    for code, row in totals.iterrows():
        print(f"  {code:<11}{row.automated:>12.0f}{row.human:>12.0f}${row.cost:>11,.0f}")


if __name__ == "__main__":
    main()
