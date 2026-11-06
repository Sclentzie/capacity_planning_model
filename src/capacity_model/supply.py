"""Capacity Ledger: people and productive hours, week by week (spec 5.1-5.3).

Session 4, task 4a: the ledger mechanics only. No coverage, no overtime and
no hiring decisions - those are 6.2 and 6.5, in later tasks. Starting
headcount is the spec 5.1 default and is deliberately UNTUNED; the 10/26
rule that sets it from required FTE is task 4c.

The WFM idea this module exists for: HEADCOUNT IS NOT CAPACITY. A name on
the roster becomes productive hours only after four separate deductions,
and every one of them is a different kind of problem:

    attrition     people leave, continuously, whatever the plan says
    onboarding    a new hire cannot touch records until compliance training
                  and system access are done - weeks at 0%, not reduced
    ramp          then weeks at partial speed before full productivity
    shrinkage     and the people who are here are not all on the floor

The gap between the roster and the hours is where most capacity plans go
wrong. Hiring late costs more weeks than training alone suggests, which is
what the hire-by calculation in 6.5 exists to show.

    Formula (spec 5.3):
        weekly attrition = 1 - (1 - monthly attrition) ^ (12 / 52)
        next headcount   = this headcount x (1 - weekly attrition) + hires
        productive hours = (sum over cohorts of headcount x ramp %)
                           x paid hours x (1 - shrinkage)

Headcount stays FRACTIONAL inside the maths (spec 5.3): these are expected
values, and rounding 0.4 of a leaver to zero every week would quietly
rebuild the headcount the attrition rate is removing.

Every hiring class is its own cohort at its own site, so the ramp applies to
the right people - and so new hires attrit too, which is the detail that
makes a class of 10 deliver fewer than 10 FTE a year later.

Run it:
    .venv\\Scripts\\python -m capacity_model.supply
"""

import copy
import math

import numpy as np
import pandas as pd

from . import backlog, config, generate_data, hub, intervals, work_volume

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")
MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52


# ---------------------------------------------------------------------------
# Attrition, onboarding and ramp
# ---------------------------------------------------------------------------


def weekly_attrition(monthly_rate):
    """Spec 5.3: weekly attrition = 1 - (1 - monthly) ^ (12 / 52).

    Compounded, not divided. A 2.5% monthly rate is not 2.5 x 12 / 52 a
    week, because each week's leavers come out of a roster the previous
    week already thinned.
    """
    if not 0 <= monthly_rate < 1:
        raise ValueError(f"monthly attrition must be in [0, 1): {monthly_rate}")
    return 1.0 - (1.0 - monthly_rate) ** (MONTHS_PER_YEAR / WEEKS_PER_YEAR)


def scenario_monthly_attrition(role, assumptions=None):
    """A role's monthly attrition with any scenario multiplier (spec 7).

    The multiplier applies to the MONTHLY rate, before the weekly
    conversion, because the two are not interchangeable: 2.5% a month is
    0.5826% a week, and doubling the monthly rate to 5% gives 1.1767% a
    week rather than twice 0.5826% (1.1651%). Doubling the weekly figure
    instead would quietly model a gentler scenario than the one asked for.
    """
    assumptions = assumptions or config.load_assumptions()
    monthly = role_settings(role, assumptions)["monthly_attrition"]
    return monthly * config.scenario_multiplier("attrition", assumptions)


def attrition_profile(role, weeks, assumptions=None):
    """Weekly attrition across `weeks`: (uniform rate, per-week rates).

    Returns a single rate when it does not change over the horizon, which
    is the normal case - the baseline, and any scenario starting in plan
    week 1 - and lets the cohort recursion keep its exact closed form
    `size x (1 - rate) ** age`. That matters for more than speed: the
    closed form and a repeated product do not agree in the last bits of a
    float, so switching to the product unconditionally would shift every
    baseline number by a hair and break the regression pins for no reason.

    A scenario that changes attrition PART WAY through the plan gets the
    per-week array instead, and build_ledger walks the survival product.
    """
    assumptions = assumptions or config.load_assumptions()
    base = weekly_attrition(role_settings(role, assumptions)["monthly_attrition"])
    scenario_rate = weekly_attrition(scenario_monthly_attrition(role, assumptions))
    if scenario_rate == base:
        return base, None

    starts = config.scenario_starts_week(assumptions)
    weeks = list(weeks)
    if weeks and min(weeks) >= starts:
        return scenario_rate, None  # the whole horizon is inside the scenario
    rates = np.array(
        [scenario_rate if week >= starts else base for week in weeks], dtype=float
    )
    return None, rates


def day_weights(role, assumptions=None):
    """Relative paid hours by weekday, averaging 1.0 (spec 7, 10/27).

    The scenario lever behind "Monday scheduling". It changes WHEN a role's
    paid hours are worked and never how many: the weights sum to the number
    of business days, so weekly paid hours come out identical to the
    baseline in every shape. The demand side never sees it at all.

    flat
        Ones. The baseline, and the reason 420 hours a year go uncovered:
        Monday carries 23% of the week against a roster that works the
        same hours every day.

    ideal_match
        The day-of-week pattern itself, scaled to average 1.0 - so Monday
        gets 1.15 of an average day and Friday 0.85. An UPPER BOUND rather
        than a roster: matching a 23/21/20/19/17 pattern exactly needs
        fractional people on every day, which no rota can deliver. It
        exists to say how much of the gap is schedulable at all.

    part_time
        The realistic version. A share of paid hours leaves the full-time
        roster, cut EVENLY across all five days - that is what part-timing
        a post costs, you cannot take the reduction only from Friday - and
        comes back on the part-time days in proportion to how far each sits
        above an even share.

        At the 10% default that is Monday 25.5% of the week and Tuesday
        20.5%, against the baseline's flat 20%. Note Monday OVERSHOOTS its
        23% of demand: 10% is a blunt instrument, every point of it has to
        leave all five days, and the two effects do not cancel neatly. The
        share is an input precisely so that can be tuned.
    """
    assumptions = assumptions or config.load_assumptions()
    settings = config.scenario(assumptions)["onsite_schedule"]
    count = len(WEEKDAYS)

    shape = settings["shape"]
    if shape == "flat" or role not in settings["roles"]:
        return np.ones(count)

    shares = intervals.day_of_week_shares(assumptions)
    missing = set(WEEKDAYS) - set(shares)
    if missing:
        raise ValueError(f"no day-of-week share for {sorted(missing)}")
    pattern = np.array([float(shares[day]) for day in WEEKDAYS])

    if shape == "ideal_match":
        return pattern * count

    if shape == "part_time":
        moved = float(settings["part_time_share"])
        if not 0.0 <= moved < 1.0:
            raise ValueError(f"part_time_share must be in [0, 1): {moved}")
        # Full-timers give up their share across every day, not just the
        # quiet ones: a post cut to four fifths is cut on all five days.
        weights = np.full(count, (1.0 - moved) / count)

        days = list(settings["part_time_days"])
        unknown = set(days) - set(WEEKDAYS)
        if unknown:
            raise ValueError(f"part_time_days names unknown days: {sorted(unknown)}")
        even = 1.0 / count
        excess = np.array([max(0.0, float(shares[day]) - even) for day in days])
        if excess.sum() <= 0:
            raise ValueError(
                f"none of {days} carries more than an even share of the week, "
                "so there is nothing for part-timers to move hours toward"
            )
        split = excess / excess.sum()
        for day, portion in zip(days, split):
            weights[WEEKDAYS.index(day)] += moved * portion
        return weights * count

    raise ValueError(f"unknown onsite schedule shape: {shape!r}")


def shortfall_weeks(assumptions=None):
    """The absolute week indexes a site shortfall covers (spec 7).

    Empty unless a scenario names a site. The window runs from the
    scenario's own starts week, so "week 20 for 4 weeks" is plan weeks 20
    to 23 and nothing else.
    """
    assumptions = assumptions or config.load_assumptions()
    settings = config.scenario(assumptions)["site_shortfall"]
    if not settings["site"] or int(settings["weeks"]) <= 0:
        return []
    first = config.scenario_starts_week(assumptions)
    return list(range(first, first + int(settings["weeks"])))


def shortfall_factor(role, scope, week, assumptions=None):
    """What share of a role's hours survive at one site in one week (spec 7).

    1.0 everywhere except the named site, the named roles and the weeks in
    the window - which is why the baseline multiplies by exactly one and is
    untouched.

    Losing 20% of five Onsite Associates is modelled as 20% of the site's
    hours rather than as one named person, because headcount is continuous
    everywhere else in this model (5.3) and a fifth of a five-person team
    IS one person. The difference would only matter if the model tracked
    individuals, and spec 6.7 is explicit that it does not.
    """
    assumptions = assumptions or config.load_assumptions()
    settings = config.scenario(assumptions)["site_shortfall"]
    if scope != settings["site"] or role not in settings["roles"]:
        return 1.0
    if week not in shortfall_weeks(assumptions):
        return 1.0
    lost = float(settings["share_of_hours_lost"])
    if not 0.0 <= lost < 1.0:
        raise ValueError(f"share_of_hours_lost must be in [0, 1): {lost}")
    return 1.0 - lost


def paid_hours_by_day(role, assumptions=None):
    """Paid hours a full-speed head works on each weekday (spec 5.1, 7).

    Returned as a dict so a reader - or a test - can see the reshaping as
    hours rather than as weights, and check the week still adds to 40.
    """
    assumptions = assumptions or config.load_assumptions()
    cal = assumptions["calendar"]
    per_day = assumptions["fte"]["paid_hours_per_week"] / cal["business_days_per_week"]
    weights = day_weights(role, assumptions)
    return {day: float(per_day * weight) for day, weight in zip(WEEKDAYS, weights)}


def survival_fraction(start_week, at_week, switch):
    """Share of a cohort still present, with one attrition rate change.

    The weeks that attrit are (start_week, at_week]: a cohort does not
    attrit in its own starting week (spec 5.3). Those weeks split at the
    scenario's starts week, and each side carries its own rate, so the
    answer is two powers rather than a loop - and the side before the
    change is exactly what the baseline would have computed.
    """
    if at_week <= start_week:
        return 1.0
    last_old_week = switch["week"] - 1
    before = max(0, min(at_week, last_old_week) - start_week)
    after = max(0, at_week - max(start_week, last_old_week))
    return (1.0 - switch["before"]) ** before * (1.0 - switch["after"]) ** after


def role_settings(role, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    roles = assumptions["supply"]["roles"]
    if role not in roles:
        raise ValueError(f"no supply settings for {role!r}; have {sorted(roles)}")
    return roles[role]


def onboarding_weeks(settings):
    """Weeks at 0% before any productive work (spec 5.1).

    Compliance and training are kept apart because they are relieved by
    different things: a compliance queue is cleared by the client's access
    process, a training queue by trainer capacity.
    """
    return int(settings["compliance_weeks"]) + int(settings["training_weeks"])


def weeks_to_full_speed(settings, threshold=0.90):
    """Ramp weeks until a cohort first reaches `threshold` (spec 6.5).

    6.5 measures the hire-by date against 90% ramp rather than 100%,
    because the last ten points arrive slowly and a plan that waits for
    them hires too late. Tier 2's ramp never passes through 90% exactly -
    it steps 40/60/80/100 - so this takes the first week at or above it.
    """
    for week, fraction in enumerate(settings["ramp"], start=1):
        if fraction >= threshold:
            return week
    return len(settings["ramp"])


def ramp_fraction(settings, weeks_since_start):
    """How productive a cohort is in its nth week (spec 5.1).

    Week 0 is the cohort's first week on the payroll. Onboarding weeks
    return 0.0 - they are paid and unproductive, which is the point - and
    the ramp holds at its last value once it reaches full speed.
    """
    if weeks_since_start < 0:
        return 0.0
    onboarding = onboarding_weeks(settings)
    if weeks_since_start < onboarding:
        return 0.0
    ramp = settings["ramp"]
    return float(ramp[min(weeks_since_start - onboarding, len(ramp) - 1)])


def hire_by_week(role, first_short_week, assumptions=None):
    """When the requisition must open (spec 6.5).

        hire-by = first short week
                  - weeks to reach 90% ramp
                  - onboarding weeks
                  - requisition lead time

    The spec's worked examples: Onsite Associates short from week 30 give
    30 - 3 - 3 - 6 = week 18; Tier 2 short from week 40 give
    40 - 4 - 7 - 8 = week 21. Tier 2 needs nearly five months of warning
    for a shortfall, which is the whole argument for planning ahead.

    A negative answer is returned as-is rather than clamped: "you needed to
    start three weeks before the plan began" is the useful answer, and 6.5
    says the plan should state it plainly.
    """
    settings = role_settings(role, assumptions)
    return (
        int(first_short_week)
        - weeks_to_full_speed(settings)
        - onboarding_weeks(settings)
        - int(settings["requisition_weeks"])
    )


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------


def scopes_for_role(role, sites, assumptions=None):
    """Where a role's people sit: one per site, or one network pool.

    The central records hub (spec 7, 11/02) enters the model HERE and
    almost nowhere else. It makes Records Specialists a one-site role
    instead of a six-site one, and the floor, the opening roster, the
    per-site shrinkage draw and the cluster borrow then all follow
    without any of them having to know that a hub exists.
    """
    assumptions = assumptions or config.load_assumptions()
    supply = assumptions["supply"]
    if role in supply["per_site_roles"]:
        if role == "records_specialist" and hub.enabled(assumptions):
            return hub.records_specialist_scopes(assumptions, sites)
        return [site["code"] for site in sites["sites"]]
    return [supply["network_scope"]]


def opening_cohorts(assumptions=None, sites=None, first_week=None):
    """The people already on the roster when the plan opens (spec 5.1).

    Per-site roles split their network total evenly across the six sites -
    12 Onsite Associates is 2 per site - and the rest form one pool.

    The opening cohort starts one week BEFORE the plan, so that week 1
    shows it as an opening balance that attrits, rather than as 12 people
    hired on the first Monday.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if first_week is None:
        first_week = assumptions["calendar"]["history_weeks"]

    headcount = _as_scope_map(assumptions, sites)
    cohorts = []
    for role in assumptions["supply"]["roles"]:
        scopes = scopes_for_role(role, sites, assumptions)
        stated = headcount[role]
        missing = set(scopes) - set(stated)
        if missing:
            raise ValueError(f"no starting headcount for {role} at {sorted(missing)}")
        for scope in scopes:
            cohorts.append(
                {
                    "role": role,
                    "scope": scope,
                    "start_week": first_week - 1,
                    "size": float(stated[scope]),
                    "opening": True,
                }
            )
    return cohorts


def add_hiring_class(cohorts, role, scope, start_week, size):
    """Add a class that STARTS in `start_week` (spec 5.3).

    start_week is the first week on the payroll, not the week the
    requisition opens - hire_by_week() converts between the two. The class
    is unproductive through onboarding and then ramps, and it attrits from
    its second week like everybody else.

    Returns a new list; the input is left alone so a scenario can branch
    without disturbing the baseline.
    """
    if size <= 0:
        raise ValueError(f"a hiring class needs people in it: {size}")
    return list(cohorts) + [
        {
            "role": role,
            "scope": scope,
            "start_week": int(start_week),
            "size": float(size),
            "opening": False,
        }
    ]


# ---------------------------------------------------------------------------
# Shrinkage (spec 5.2)
# ---------------------------------------------------------------------------


def plan_window(assumptions=None):
    """The plan's week indexes and their Mondays (spec 2).

    The reference period for everything seasonal. Shrinkage patterns are
    normalised across THIS window and never across a shorter slice,
    because "averages its annual percentage" has to mean the year.
    """
    assumptions = assumptions or config.load_assumptions()
    cal = assumptions["calendar"]
    mondays = generate_data.week_start_dates(
        cal["history_start"], cal["history_weeks"] + cal["plan_weeks"]
    )
    weeks = list(range(cal["history_weeks"], cal["history_weeks"] + cal["plan_weeks"]))
    return weeks, [mondays[w] for w in weeks]


def plan_week_factors(assumptions=None, weeks=None):
    """Week factors for `weeks`, normalised over the whole plan year.

    Slicing after normalising is the point. Normalising a 13-week window
    on its own would force that window to average the annual rate, which
    silently deletes the seasonality - a first quarter with heavier sick
    time and lighter vacation would come out looking exactly average.
    """
    assumptions = assumptions or config.load_assumptions()
    plan_weeks, mondays = plan_window(assumptions)
    table = week_factor_table(assumptions, mondays)

    if weeks is None:
        return table
    positions = []
    for week in weeks:
        if week not in plan_weeks:
            raise ValueError(f"week {week} is outside the plan window {plan_weeks[0]}-{plan_weeks[-1]}")
        positions.append(plan_weeks.index(week))
    return {name: values[positions] for name, values in table.items()}


def week_factor_table(assumptions=None, mondays=None):
    """One week-factor series per named pattern, shared across roles.

    Shared on purpose: a bad flu week hits the whole operation, so every
    role should see the same sickness week rather than its own private
    draw. The series is rescaled to average 1.0, which is what keeps each
    line averaging its annual percentage however the pattern is shaped.

    Callers pass the FULL plan year; plan_week_factors() slices afterwards.
    """
    assumptions = assumptions or config.load_assumptions()
    shrinkage = assumptions["shrinkage"]
    normalise = shrinkage.get("normalize_to_annual_rate", True)
    rng = np.random.default_rng(shrinkage["seed"])

    table = {}
    for name in sorted(shrinkage["week_patterns"]):
        pattern = shrinkage["week_patterns"][name]
        factors = np.full(len(mondays), float(pattern.get("default", 1.0)))
        for period in pattern.get("periods", []):
            start = (period["from"]["month"], period["from"]["day"])
            end = (period["to"]["month"], period["to"]["day"])
            inside = np.array(
                [
                    generate_data._within_window((m.month, m.day), start, end)
                    for m in mondays
                ]
            )
            factors[inside] = float(period["weight"])

        noise = pattern.get("noise")
        if noise:
            drawn = rng.normal(0.0, float(noise["relative_std_dev"]), len(mondays))
            factors = np.clip(factors * (1.0 + drawn), 0.0, None)

        if normalise:
            mean = factors.mean()
            if mean <= 0:
                raise ValueError(f"week pattern {name!r} averages zero")
            factors = factors / mean
        table[name] = factors
    return table


def day_factors(pattern):
    """Weights across Monday to Friday, rescaled to average 1.0.

    Rescaling is what lets a line sit on one day without changing the
    week: meetings at 1% land as 5% every Wednesday and nothing on the
    other four days, and the weekly average is still 1%.
    """
    weights = np.array([float(pattern[day]) for day in WEEKDAYS])
    if (weights < 0).any():
        raise ValueError("day weights cannot be negative")
    if weights.sum() <= 0:
        raise ValueError("a day pattern needs at least one day with weight")
    return weights / weights.mean()


def plan_positions(mondays, assumptions=None):
    """Where each Monday sits in the plan year.

    Needed because every shrinkage series is normalised over the WHOLE plan
    year and only then sliced. Normalising a 13-week request on its own
    would force that window to average the annual rate and quietly delete
    the seasonality - the bug that made the first headcount tuning come out
    97% covered.
    """
    _plan_weeks, plan_mondays = plan_window(assumptions)
    index = {monday: i for i, monday in enumerate(plan_mondays)}
    missing = [m for m in mondays if m not in index]
    if missing:
        raise ValueError(
            f"{missing[0]} is outside the plan year "
            f"{plan_mondays[0]} to {plan_mondays[-1]}"
        )
    return [index[monday] for monday in mondays]


def site_draw_index(assumptions=None, sites=None):
    """A stable number for each site, so its random draw is reproducible.

    Position in the sorted site list, not Python's hash(): string hashing
    is randomised per process, so a hash-seeded draw would differ between
    runs and break the fixed-seed promise in CLAUDE.md.
    """
    sites = sites or config.load_sites()
    codes = sorted(site["code"] for site in sites["sites"])
    return {code: position for position, code in enumerate(codes)}


def site_day_factors(line, scope_index, line_index, weeks, assumptions=None):
    """Per-site, per-day noise for an unplanned line (spec 5.2, 10/26).

    Returns a (weeks, 5) multiplier averaging exactly 1.0, so the line still
    averages its annual percentage at every site.

    Why a site needs its own draw: sick time is the one line nobody
    schedules. If every site loses the same share of Tuesday, then on a
    heavy Tuesday every site is short at once and cluster coverage has
    nothing to work with - a site can only cover a neighbour on a day it is
    better off than them. The shared week-level noise stays, because a bad
    flu week really does hit the whole operation; this sits on top of it.

    The seed is (network seed, which line, which site), so each site's
    series is fixed and independent, and adding a seventh site never
    changes the first six.
    """
    assumptions = assumptions or config.load_assumptions()
    seed = assumptions["shrinkage"]["seed"]
    spread = float(line.get("per_site_relative_std_dev", 0.0))
    rng = np.random.default_rng([seed, line_index, scope_index])
    drawn = rng.normal(0.0, spread, (weeks, len(WEEKDAYS)))
    factors = np.clip(1.0 + drawn, 0.0, None)
    mean = factors.mean()
    if mean <= 0:
        raise ValueError(f"site draw for line {line['code']!r} averages zero")
    return factors / mean


def shrinkage_by_line(
    role, mondays, assumptions=None, factors=None, scope=None, sites=None
):
    """Daily shrinkage for one role and scope, per line (spec 5.2).

    Returns {line code: array of (weeks, 5 days)}. A line that does not
    apply to the role is absent rather than zero, so "which lines does this
    role carry" stays answerable from the result.

    `scope` matters for one reason (decided 10/26): the sick line is drawn
    separately for each site. Pass a site code and that site gets its own
    series; pass the network pool, or nothing, and the shared one is used -
    one remote team, one sick pattern.

    Every line's multiplier averages exactly 1.0 over the plan year, which
    is what keeps the line at its annual percentage. For a line without a
    site draw that is automatic: the mean of an outer product is the
    product of the means, and both halves already average 1.0. So the
    network pool's numbers are untouched by this change.
    """
    assumptions = assumptions or config.load_assumptions()
    shrinkage = assumptions["shrinkage"]
    factors = factors if factors is not None else week_factor_table(assumptions, mondays)
    all_roles = set(assumptions["supply"]["roles"])

    site_index = site_draw_index(assumptions, sites)
    drawn_for_site = scope in site_index
    if drawn_for_site:
        plan_weeks, plan_mondays = plan_window(assumptions)
        positions = plan_positions(mondays, assumptions)
        year_factors = week_factor_table(assumptions, plan_mondays)

    lines = {}
    for line_index, line in enumerate(shrinkage["lines"]):
        applies = set(line.get("roles", all_roles))
        unknown = applies - all_roles
        if unknown:
            raise ValueError(f"shrinkage line {line['code']} names unknown roles: {unknown}")
        if role not in applies:
            continue
        day = day_factors(shrinkage["day_patterns"][line["day_pattern"]])

        if drawn_for_site and line.get("per_site_draw"):
            # Built over the whole plan year, normalised, THEN sliced.
            draw = site_day_factors(
                line, site_index[scope], line_index, len(plan_weeks), assumptions
            )
            year = year_factors[line["week_pattern"]][:, None] * day[None, :] * draw
            multiplier = year[positions] / year.mean()
        else:
            multiplier = factors[line["week_pattern"]][:, None] * day[None, :]

        lines[line["code"]] = float(line["rate"]) * multiplier
    return lines


def daily_shrinkage(
    role, mondays, assumptions=None, factors=None, scope=None, sites=None
):
    """Total share of paid time away, per week and weekday (spec 5.2)."""
    lines = shrinkage_by_line(role, mondays, assumptions, factors, scope, sites)
    if not lines:
        raise ValueError(f"no shrinkage lines apply to {role!r}")
    total = np.sum(list(lines.values()), axis=0)
    if (total >= 1.0).any():
        worst = np.unravel_index(np.argmax(total), total.shape)
        raise ValueError(
            f"{role}: shrinkage reaches {total.max():.1%} on week {worst[0]} "
            f"{WEEKDAYS[worst[1]]}; the whole team cannot be away"
        )
    return total


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def build_ledger(assumptions=None, sites=None, cohorts=None, weeks=None):
    """The Capacity Ledger: headcount and productive hours, week by week.

    Returns a dict with the weekly ledger, the daily hours behind it, and
    the shrinkage detail by line.

    The ledger reconciles by construction (spec 9): ending headcount is
    starting plus hires minus attrition, every week, for every role and
    scope. Hires do not attrit in the week they start, which is what the
    5.3 recursion says and what stops a class losing people before it has
    arrived.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    cal = assumptions["calendar"]

    if weeks is None:
        first = cal["history_weeks"]
        weeks = list(range(first, first + cal["plan_weeks"]))
    weeks = list(weeks)
    first_week = weeks[0]

    if cohorts is None:
        cohorts = opening_cohorts(assumptions, sites, first_week)

    mondays = generate_data.week_start_dates(
        cal["history_start"], cal["history_weeks"] + cal["plan_weeks"]
    )
    week_mondays = [mondays[w] for w in weeks]
    # Normalised over the plan year, then sliced - never the other way round.
    factors = plan_week_factors(assumptions, weeks)

    paid_hours = assumptions["fte"]["paid_hours_per_week"]
    business_days = cal["business_days_per_week"]
    hours_per_day = paid_hours / business_days

    ledger_rows, daily_rows = [], []
    groups = sorted({(c["role"], c["scope"]) for c in cohorts})

    # Per (role, scope), not per role: the sick line is drawn separately for
    # each site (spec 5.2, decided 10/26), so two sites on the same role no
    # longer lose the same share of the same Tuesday.
    shrink = {
        (role, scope): daily_shrinkage(
            role, week_mondays, assumptions, factors, scope, sites
        )
        for role, scope in groups
    }

    for role, scope in groups:
        settings = role_settings(role, assumptions)
        uniform, by_week = attrition_profile(role, weeks, assumptions)
        attrition_rate = uniform
        switch = None
        if uniform is None:
            # A scenario changes attrition part way through the plan. The
            # rate is still constant on each side of the change, so the
            # survival stays a closed form - one power for the weeks before
            # and one for the weeks after - rather than a product walked
            # week by week.
            #
            # Exactness is the reason, not speed. A repeated product does
            # not agree with a power in the last bits of a float, so the
            # weeks BEFORE the scenario starts would come out a hair away
            # from the baseline, and "weeks before the starts week match
            # the baseline" would be true only to a tolerance.
            switch = {
                "week": config.scenario_starts_week(assumptions),
                "before": weekly_attrition(
                    role_settings(role, assumptions)["monthly_attrition"]
                ),
                "after": weekly_attrition(
                    scenario_monthly_attrition(role, assumptions)
                ),
            }
            attrition_rate = switch["before"]
        role_day_weights = day_weights(role, assumptions)
        mine = [c for c in cohorts if c["role"] == role and c["scope"] == scope]

        for index, week in enumerate(weeks):
            # Headcount at the END of each week, per cohort. A cohort does
            # not attrit in its own starting week (spec 5.3).
            def present(cohort, at_week):
                age = at_week - cohort["start_week"]
                if age < 0:
                    return 0.0
                if switch is None:
                    return cohort["size"] * (1.0 - attrition_rate) ** age
                return cohort["size"] * survival_fraction(
                    cohort["start_week"], at_week, switch
                )

            ending = sum(present(c, week) for c in mine)
            starting = sum(present(c, week - 1) for c in mine)
            hires = sum(c["size"] for c in mine if c["start_week"] == week)
            week_rate = attrition_rate if by_week is None else float(by_week[index])
            attrition = starting * week_rate

            ramped = sum(
                present(c, week)
                * (1.0 if c["opening"] else ramp_fraction(settings, week - c["start_week"]))
                for c in mine
            )

            week_shrink = shrink[(role, scope)][index]
            # Monday scheduling (spec 7, 10/27): the weights average 1.0, so
            # this moves paid hours between days and never changes the
            # week's total. Flat in the baseline, where every weight is
            # exactly 1.0 and the multiply is a no-op.
            #
            # A site shortfall (spec 7) then removes a share of what is
            # left, at one site for a few weeks. Also exactly 1.0 in the
            # baseline, so neither multiply moves a single bit of it.
            lost = shortfall_factor(role, scope, week, assumptions)
            hours = (
                ramped * hours_per_day * (1.0 - week_shrink) * role_day_weights * lost
            )

            ledger_rows.append(
                {
                    "week_index": week,
                    "week_start": week_mondays[index].isoformat(),
                    "role": role,
                    "scope": scope,
                    "starting_headcount": starting,
                    "hires": hires,
                    "attrition": attrition,
                    "ending_headcount": ending,
                    "ramped_fte": ramped,
                    "shrinkage": float(week_shrink.mean()),
                    "productive_hours": float(hours.sum()),
                }
            )
            for day_index, day in enumerate(WEEKDAYS):
                daily_rows.append(
                    {
                        "week_index": week,
                        "week_start": week_mondays[index].isoformat(),
                        "role": role,
                        "scope": scope,
                        "day_of_week": day,
                        "ramped_fte": ramped,
                        "shrinkage": float(week_shrink[day_index]),
                        "productive_hours": float(hours[day_index]),
                    }
                )

    return {
        "ledger": pd.DataFrame(ledger_rows),
        "daily": pd.DataFrame(daily_rows),
        # Nested role -> scope -> line, because a line is no longer the
        # same series at every site.
        "shrinkage_lines": {
            role: {
                scope: shrinkage_by_line(
                    role, week_mondays, assumptions, factors, scope, sites
                )
                for other, scope in groups
                if other == role
            }
            for role, _scope in groups
        },
        "weeks": weeks,
        "mondays": week_mondays,
    }


# ---------------------------------------------------------------------------
# Records Specialists help the counter (spec 4.5, decided 10/26)
#
# A Records Specialist who has pulled every chart on the list is standing in
# the same building as a queue. Spec 4.5 sends their spare hours to walk-ins
# and phones - but only HALF of them, because they are not reliably free at
# the moment the counter needs them. A chart pull takes as long as it takes,
# so the help is spread evenly across the day rather than promised for a
# particular half-hour.
#
# The crucial property, and the one the tests pin hardest: this changes WHO
# provides coverage, never HOW MUCH is required. Erlang C still says the
# same number of people must be on the floor in every interval; some of them
# are now wearing a different job title.
# ---------------------------------------------------------------------------


def records_specialist_spare_hours(
    ledger_daily, required_hours, assumptions=None
):
    """Spare hours per site and day, after their own work is done.

    Spec 4.5 puts onsite retrieval first - a request is waiting on it - then
    digitization. Only what is left over can go to the counter, so this is
    productive hours minus the role's own required work.

    Their required work is weekly, so it is spread across the week by the
    same day-of-week shares the arrivals use (spec 3.4). A Monday is busier
    than a Friday on both sides of the subtraction.
    """
    assumptions = assumptions or config.load_assumptions()
    settings = assumptions["onsite_team"]["records_specialist_help"]
    own_types = set(settings["own_work_types"])

    own_work = required_hours[required_hours["work_type"].isin(own_types)]
    missing = own_types - set(required_hours["work_type"])
    if missing:
        raise ValueError(f"no required hours found for {sorted(missing)}")
    weekly_need = own_work.groupby(["week_index", "site"])["required_hours"].sum()

    day_share = intervals.day_of_week_shares(assumptions)
    rows = ledger_daily[ledger_daily["role"] == "records_specialist"].copy()
    if rows.empty:
        raise ValueError("the ledger has no Records Specialist rows")

    rows = rows.rename(columns={"scope": "site"})
    rows["own_work_hours"] = [
        weekly_need.get((week, site), 0.0) * day_share[day]
        for week, site, day in zip(rows["week_index"], rows["site"], rows["day_of_week"])
    ]
    rows["spare_hours"] = (rows["productive_hours"] - rows["own_work_hours"]).clip(lower=0.0)
    rows["help_hours_offered"] = rows["spare_hours"] * float(settings["share_of_spare"])
    return rows[
        [
            "week_index", "week_start", "site", "day_of_week",
            "productive_hours", "own_work_hours", "spare_hours", "help_hours_offered",
        ]
    ]


def apply_records_specialist_help(onsite, spare, assumptions=None):
    """Hand half the spare hours to the blended queue (spec 4.5).

    Spread evenly across the day's intervals, because a Records Specialist
    cannot promise to be free at 11:00 specifically. Offered help becomes
    staff on the floor:

        help staff per interval = help hours / (intervals x interval hours)

    What the counter can actually use is capped by what it needed in the
    first place - there is no credit for offering help to an interval that
    was already covered - so applied help never exceeds the scheduled staff
    it replaces, and the Onsite Associate requirement never goes below zero.

    Help never crosses a site: they are standing in one building.
    """
    assumptions = assumptions or config.load_assumptions()
    settings = assumptions["onsite_team"]["records_specialist_help"]
    if not settings.get("enabled", True):
        table = onsite.copy()
        table["help_staff"] = 0.0
        table["scheduled_staff_before_help"] = table["scheduled_staff"]
        return table

    interval_hours = assumptions["live_queues"]["interval_seconds"] / 3600.0
    per_day = assumptions["intraday"]["site_hours"]["intervals_per_day"]

    offered = spare.set_index(["week_index", "site", "day_of_week"])["help_hours_offered"]

    table = onsite.copy()
    keys = list(zip(table["week_index"], table["site"], table["day_of_week"]))
    hours = np.array([offered.get(key, 0.0) for key in keys], dtype=float)

    # Evenly across the day's intervals, converted from hours to bodies.
    table["help_staff_offered"] = hours / (per_day * interval_hours)
    table["scheduled_staff_before_help"] = table["scheduled_staff"]
    table["help_staff"] = np.minimum(
        table["help_staff_offered"], table["scheduled_staff_before_help"]
    )
    table["scheduled_staff"] = table["scheduled_staff_before_help"] - table["help_staff"]
    table["scheduled_hours"] = table["scheduled_staff"] * interval_hours
    table["help_hours"] = table["help_staff"] * interval_hours

    # Coverage is unchanged by construction; assert it rather than trust it,
    # because this is the one property the whole change rests on.
    covered = table["scheduled_staff"] + table["help_staff"]
    if not np.allclose(covered, table["scheduled_staff_before_help"], rtol=1e-9):
        raise ValueError("records specialist help changed the total coverage")
    return table


# ---------------------------------------------------------------------------
# Tuning the opening roster (spec 5.1, decided 10/26)
#
# Starting headcount is not a guess any more: it is the average required FTE
# over the first 13 plan weeks, rounded up, floored at 2 Records Specialists
# a site. The point is to open the plan BALANCED, so that every gap the
# Capacity Plan later reports is caused by something real - attrition,
# growth, a season - rather than by an arbitrary opening number nobody can
# defend.
#
# Per-person hours come from the ledger rather than from 40 x (1 - 28%),
# because the first 13 weeks of the plan are January to March: sick time
# runs about a quarter higher and vacation is at its lightest, so a person
# delivers slightly more than the annual average. Using the flat figure
# would hire a fraction of a person too many on every team.
#
# There is one ordering constraint and it matters. Onsite Associate demand
# depends on how much help the Records Specialists can give, which depends
# on how many Records Specialists there are. So the back office is tuned
# first, the ledger is rebuilt on those numbers, and only then is the
# counter tuned against the help it will actually receive.
# ---------------------------------------------------------------------------

TUNING_WEEKS = 13


def productive_hours_per_person(role, weeks, assumptions=None, sites=None):
    """What one person on the opening roster delivers over `weeks` (5.1).

    Run through the ledger itself rather than calculated beside it, so it
    picks up everything the ledger does to a head: this quarter's own
    shrinkage AND attrition. Both matter and they pull opposite ways.

    Seasonality: the first 13 plan weeks are January to March, where
    vacation is at its lightest and sick time about a quarter heavier. The
    two do not cancel - a back-office person delivers about 29.2 hours a
    week here against 28.8 across the year.

    Attrition: a roster of N people is not N people for 13 weeks. Leaving
    it out would size every team to its opening Monday and leave it short
    by the end of the quarter, which is the error this whole tuning rule
    exists to avoid.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    one_person = [
        {
            "role": role,
            "scope": "ONE",
            "start_week": weeks[0] - 1,
            "size": 1.0,
            "opening": True,
        }
    ]
    ledger = build_ledger(assumptions, sites, cohorts=one_person, weeks=list(weeks))
    return float(ledger["ledger"]["productive_hours"].sum())


def tune_starting_headcount(assumptions=None, sites=None, weeks=TUNING_WEEKS):
    """Opening headcount per role and scope, from required hours (spec 5.1).

    Returns a tidy table with the arithmetic shown, so the number can be
    argued with rather than merely accepted.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    cal = assumptions["calendar"]

    first = cal["history_weeks"]
    window = list(range(first, first + weeks))
    all_mondays = generate_data.week_start_dates(
        cal["history_start"], cal["history_weeks"] + cal["plan_weeks"]
    )
    mondays = [all_mondays[w] for w in window]
    factors = plan_week_factors(assumptions, window)

    per_person = {
        role: productive_hours_per_person(role, window, assumptions, sites)
        for role in assumptions["supply"]["roles"]
    }

    # --- back office: required hours straight from the backlog at target ---
    work = work_volume.build(assumptions, sites)["work"]
    required = backlog.required_hours(assumptions, sites, work)["hours"]
    queues = backlog.build_queues(assumptions, sites, required)
    in_window = queues[queues["week_index"].isin(window)]
    back_office = in_window.groupby(["role", "scope"])["required_hours"].sum()

    rows = []
    for (role, scope), hours in back_office.items():
        rows.append({"role": role, "scope": scope, "required_hours": float(hours)})

    # --- the counter: rebuild the ledger on the tuned back office first ----
    floors = _records_specialist_floor(assumptions)
    interim = {}
    for row in rows:
        headcount = math.ceil(row["required_hours"] / per_person[row["role"]])
        if row["role"] == "records_specialist":
            headcount = max(headcount, floors)
        interim.setdefault(row["role"], {})[row["scope"]] = headcount

    tuned_assumptions = copy.deepcopy(assumptions)
    tuned_assumptions["fte"]["starting_headcount"] = {
        **_as_scope_map(assumptions, sites),
        **interim,
    }
    ledger = build_ledger(tuned_assumptions, sites, weeks=window)

    onsite = intervals.onsite_intervals(
        assumptions, sites, work[work["week_index"].isin(window)]
    )
    spare = records_specialist_spare_hours(ledger["daily"], required, tuned_assumptions)
    helped = apply_records_specialist_help(onsite["intervals"], spare, tuned_assumptions)

    # Spec 5.1, decided 10/26: the counter's requirement is the scheduled
    # live-work hours it is left with after the help, plus any email work
    # that would not fit into idle time.
    counter = helped.groupby("site")["scheduled_hours"].sum()
    spillover = (
        onsite["hours"][onsite["hours"]["week_index"].isin(window)]
        .groupby("site")["email_paid_hours"]
        .sum()
    )
    for site, hours in counter.items():
        rows.append(
            {
                "role": "onsite_associate",
                "scope": site,
                "required_hours": float(hours + spillover.get(site, 0.0)),
            }
        )

    table = pd.DataFrame(rows)
    table["productive_hours_per_person"] = table["role"].map(per_person)
    table["required_fte"] = table["required_hours"] / table["productive_hours_per_person"]
    table["floor"] = np.where(table["role"] == "records_specialist", floors, 0)
    table["headcount"] = np.maximum(
        np.ceil(table["required_fte"] - 1e-9), table["floor"]
    ).astype(int)
    table["rounding_slack_fte"] = table["headcount"] - table["required_fte"]
    return table.sort_values(["role", "scope"]).reset_index(drop=True)


def _records_specialist_floor(assumptions):
    """2 per site (spec Section 1), so a site keeps its records cover."""
    return int(assumptions["supply"].get("records_specialist_floor_per_site", 2))


def _as_scope_map(assumptions, sites):
    """The current headcount as {role: {scope: count}}, whichever form it is in."""
    stated = assumptions["fte"]["starting_headcount"]
    mapped = {}
    for role, value in stated.items():
        if isinstance(value, dict):
            mapped[role] = dict(value)
            continue
        scopes = scopes_for_role(role, sites, assumptions)
        mapped[role] = {scope: value / len(scopes) for scope in scopes}
    # The records hub re-bases Records Specialists rather than deleting
    # the five sites' worth (spec 7, 11/02). See hub.collapse_headcount
    # _to_hub for the bug this prevents.
    return hub.collapse_headcount_to_hub(mapped, assumptions, sites)


def gap_by_quarter(assumptions=None, sites=None, quarters=4):
    """Available minus required, in FTE, by role and quarter (spec 6.1).

        Gap FTE = (available hours - required hours) / (40 x (1 - shrinkage))

    No hiring and no coverage yet, so this is the untouched consequence of
    opening balanced: attrition eats into the roster week by week, demand
    grows at 6% a year, and the seasons move both sides.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    cal = assumptions["calendar"]

    built = build_ledger(assumptions, sites)
    ledger = built["ledger"]

    work = work_volume.build(assumptions, sites)["work"]
    required = backlog.required_hours(assumptions, sites, work)["hours"]
    queues = backlog.build_queues(assumptions, sites, required)
    need = queues.groupby(["week_index", "role", "scope"])["required_hours"].sum()

    onsite = intervals.onsite_intervals(assumptions, sites, work)
    spare = records_specialist_spare_hours(built["daily"], required, assumptions)
    helped = apply_records_specialist_help(onsite["intervals"], spare, assumptions)
    counter = helped.groupby(["week_index", "site"])["scheduled_hours"].sum()
    spillover = onsite["hours"].groupby(["week_index", "site"])["email_paid_hours"].sum()

    paid = assumptions["fte"]["paid_hours_per_week"]
    rows = []
    for row in ledger.itertuples():
        if row.role == "onsite_associate":
            key = (row.week_index, row.scope)
            hours = float(counter.get(key, 0.0) + spillover.get(key, 0.0))
        else:
            hours = float(need.get((row.week_index, row.role, row.scope), 0.0))
        fte_hours = paid * (1.0 - row.shrinkage)
        rows.append(
            {
                "week_index": row.week_index,
                "role": row.role,
                "scope": row.scope,
                "available_hours": row.productive_hours,
                "required_hours": hours,
                "gap_fte": (row.productive_hours - hours) / fte_hours,
                "coverage": row.productive_hours / hours if hours > 0 else np.nan,
            }
        )

    weekly = pd.DataFrame(rows)
    first = cal["history_weeks"]
    per_quarter = cal["plan_weeks"] // quarters
    weekly["quarter"] = (weekly["week_index"] - first) // per_quarter + 1
    weekly["quarter"] = weekly["quarter"].clip(upper=quarters)
    return weekly


def write_ledger(result, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    paths = assumptions["paths"]
    out_dir = config.resolve_path(paths["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = out_dir / paths["capacity_ledger_csv"]
    result["ledger"].to_csv(ledger_path, index=False, float_format="%.6f")
    daily_path = out_dir / paths["ledger_daily_csv"]
    result["daily"].to_csv(daily_path, index=False, float_format="%.6f")
    return ledger_path, daily_path


def main():
    assumptions, sites, _skills = config.load_all()
    result = build_ledger(assumptions, sites)
    ledger_path, daily_path = write_ledger(result, assumptions)

    ledger = result["ledger"]
    first_week = result["weeks"][0]
    monday = ledger["week_start"].iloc[0]

    print(f"Wrote {len(ledger):,} ledger rows to {ledger_path}")
    print(f"Wrote {len(result['daily']):,} daily rows to {daily_path}")
    print("  starting headcount is the spec 5.1 default and is UNTUNED (task 4c)")

    # --- productive vs required, first plan week ---------------------------
    required = backlog.required_hours(assumptions, sites)["hours"]
    required_week = (
        required[required.week_index == first_week].groupby("role")["required_hours"].sum()
    )
    supplied = ledger[ledger.week_index == first_week].groupby("role")[
        "productive_hours"
    ].sum()
    headcount = ledger[ledger.week_index == first_week].groupby("role")[
        "starting_headcount"
    ].sum()

    print(f"\nFirst plan week ({monday}): productive vs required hours (UNTUNED preview)")
    print(f"  {'role':<20}{'HC':>6}{'productive':>12}{'required':>10}{'gap h':>9}{'cover':>8}")
    for role in sorted(supplied.index):
        need = required_week.get(role, float("nan"))
        have = supplied[role]
        if role == "onsite_associate":
            print(
                f"  {role:<20}{headcount[role]:6.1f}{have:12.1f}"
                f"{'see 4.1':>10}{'':>9}{'':>8}"
            )
            continue
        print(
            f"  {role:<20}{headcount[role]:6.1f}{have:12.1f}{need:10.1f}"
            f"{have - need:9.1f}{have / need:8.0%}"
        )

    # --- shrinkage range for one back-office role --------------------------
    role = "tier_1"
    weekly = ledger[ledger.role == role].groupby("week_index")["shrinkage"].mean()
    daily = result["daily"]
    daily_role = daily[daily.role == role]
    lines = result["shrinkage_lines"][role]

    stated = assumptions["fte"]["paid_shrinkage"][role]
    print(f"\nShrinkage across the plan year, {role} (spec 5.2):")
    print(
        f"  annual average      {weekly.mean():7.2%}  from the 5.2 lines, "
        f"matching the {stated:.0%} in 5.1"
    )
    print(f"  weekly range        {weekly.min():7.2%} to {weekly.max():.2%}")
    print(f"  busiest week        {weekly.idxmax()} ({ledger[ledger.week_index == weekly.idxmax()]['week_start'].iloc[0]})")
    print(f"  daily range         {daily_role['shrinkage'].min():7.2%} to {daily_role['shrinkage'].max():.2%}")
    print("  by line, annual average:")
    for code, values in sorted(lines.items(), key=lambda kv: -kv[1].mean()):
        print(f"    {code:<20}{values.mean():7.2%}")

    by_day = daily_role.groupby("day_of_week")["shrinkage"].mean()
    print("  by weekday, averaged over the year:")
    for day in WEEKDAYS:
        print(f"    {day:<20}{by_day[day]:7.2%}")

    # --- Records Specialists help the counter (spec 4.5) -------------------
    onsite = intervals.onsite_intervals(assumptions, sites)["intervals"]
    spare = records_specialist_spare_hours(result["daily"], required, assumptions)
    helped = apply_records_specialist_help(onsite, spare, assumptions)

    week_spare = spare[spare.week_index == first_week]
    week_helped = helped[helped.week_index == first_week]

    print(f"\nRecords Specialists helping the counter, first plan week (spec 4.5):")
    print(
        f"  {'site':<6}{'productive':>11}{'own work':>10}{'spare':>8}"
        f"{'offered':>9}{'used':>8}{'OA before':>11}{'OA after':>10}{'saved':>8}"
    )
    for site in sorted(week_spare["site"].unique()):
        s = week_spare[week_spare.site == site]
        h = week_helped[week_helped.site == site]
        before = h["scheduled_staff_before_help"].sum() * 0.5
        after = h["scheduled_hours"].sum()
        print(
            f"  {site:<6}{s['productive_hours'].sum():11.1f}{s['own_work_hours'].sum():10.1f}"
            f"{s['spare_hours'].sum():8.1f}{s['help_hours_offered'].sum():9.1f}"
            f"{h['help_hours'].sum():8.1f}{before:11.1f}{after:10.1f}"
            f"{before - after:8.1f}"
        )
    before_all = week_helped["scheduled_staff_before_help"].sum() * 0.5
    after_all = week_helped["scheduled_hours"].sum()
    print(
        f"  {'ALL':<6}{week_spare['productive_hours'].sum():11.1f}"
        f"{week_spare['own_work_hours'].sum():10.1f}{week_spare['spare_hours'].sum():8.1f}"
        f"{week_spare['help_hours_offered'].sum():9.1f}{week_helped['help_hours'].sum():8.1f}"
        f"{before_all:11.1f}{after_all:10.1f}{before_all - after_all:8.1f}"
    )
    print(
        f"  coverage required is unchanged: "
        f"{week_helped['required_staff'].sum():,.0f} staff-intervals either way"
    )

    # --- the gap, quarter by quarter, with no hiring (spec 6.1) -----------
    gaps = gap_by_quarter(assumptions, sites)
    by_quarter = gaps.groupby(["role", "quarter"]).agg(
        gap_fte=("gap_fte", "mean"), coverage=("coverage", "mean")
    )

    print("\nGap by role and quarter, tuned and with NO hiring (spec 6.1):")
    print(f"  {'role':<20}" + "".join(f"{'Q' + str(q):>10}" for q in (1, 2, 3, 4)))
    for role in sorted(gaps["role"].unique()):
        cells = "".join(
            f"{by_quarter.loc[(role, q), 'gap_fte']:+10.2f}" for q in (1, 2, 3, 4)
        )
        print(f"  {role:<20}{cells}")
    print(f"  {'':<20}" + "".join(f"{'':>10}" for _ in range(4)))
    print(f"  {'coverage':<20}" + "".join(f"{'Q' + str(q):>10}" for q in (1, 2, 3, 4)))
    for role in sorted(gaps["role"].unique()):
        cells = "".join(
            f"{by_quarter.loc[(role, q), 'coverage']:10.1%}" for q in (1, 2, 3, 4)
        )
        print(f"  {role:<20}{cells}")

    worst = gaps.groupby("week_index")["gap_fte"].sum().idxmin()
    print(
        f"\n  the network is thinnest in week {worst} "
        f"({gaps[gaps.week_index == worst]['gap_fte'].sum():+.1f} FTE across all roles)"
    )


if __name__ == "__main__":
    main()
