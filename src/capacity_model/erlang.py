"""Erlang C for live queues: phones and the walk-in counter (spec 4.1).

Session 3.

The WFM idea: when a person is waiting in real time - on hold, or standing at
the counter - you cannot staff to the average. Arrivals are random, so some
half-hours pile up while others sit idle, and the pile-ups are what the
customer experiences. Erlang C prices that randomness: it answers "given this
much traffic and this many people, what share of contacts get served inside
the target?".

    Formula (spec 4.1):
        Traffic A (Erlangs) = arrivals per interval x handle time / interval
        Probability of waiting = Erlang C of (staff, A)
        Service level = 1 - P(wait) x e ^ (-(N - A) x target / handle time)

Teaching note from the spec: dividing weekly volume x AHT by 40 understates
staff, because Erlang C staffs to each PEAK, and a small single-site queue
needs proportionally more cushion than one large pooled queue. Six separate
site queues therefore cost more people than one network queue at the same
service level - which is what the centralized-phones scenario in 7 measures.

Two things in here are worth knowing about:

  THE ERLANG B RECURSION. The textbook Erlang C formula needs A^N / N!, and
  both halves explode: at 500 Erlangs, A^N overflows a float long before N!
  can cancel it. The recursion below carries only the RATIO, never the pieces,
  so it is exact at any traffic level a records operation could ever see.

  THE OCCUPANCY CAP. Service level alone will happily hand back a team
  running at 95% busy. That is a real answer to the wrong question: it leaves
  no recovery time between contacts and falls over the first time somebody is
  off sick. The cap refuses those answers, and it is often the binding
  constraint in a small queue rather than the service level.

This module is pure math on numbers passed in. It is deliberately not wired
to the work-volume data yet.
"""

import math

import numpy as np

# Comparisons against a goal or a cap are float comparisons, and a case like
# 8.5 Erlangs over 10 staff should count as exactly at the 85% cap rather
# than a hair over it.
TOLERANCE = 1e-12


def traffic(arrivals, handle_time_seconds, interval_seconds):
    """Offered traffic in Erlangs (spec 4.1).

    One Erlang is one contact's worth of work arriving per unit of time - so
    A is "how many people would be busy if nobody ever had to wait". It is
    the floor under the staff count, never the answer: staffing exactly A
    means an infinite queue, which is why every formula below needs N > A.

    Accepts an array of arrival counts as well as a single number, so a
    whole day of intervals converts in one call. Everything downstream of
    here works one interval at a time, because Erlang C has no closed form
    to vectorise.
    """
    counts = np.asarray(arrivals, dtype=float)
    if (counts < 0).any():
        raise ValueError(f"arrivals cannot be negative: {arrivals}")
    if handle_time_seconds < 0:
        raise ValueError(f"handle time cannot be negative: {handle_time_seconds}")
    if interval_seconds <= 0:
        raise ValueError(f"interval length must be positive: {interval_seconds}")

    offered = counts * handle_time_seconds / interval_seconds
    return float(offered) if offered.ndim == 0 else offered


def erlang_b(staff, offered_traffic):
    """Blocking probability, by the recursion rather than by factorials.

        B(0, A) = 1
        B(n, A) = A x B(n-1, A) / (n + A x B(n-1, A))

    Erlang B on its own answers a different question (a caller who finds
    everyone busy hangs up instead of waiting), but it is the numerically
    safe route to Erlang C: every step is a ratio bounded in [0, 1], so
    nothing overflows no matter how large A or N grow.
    """
    staff = _check_staff(staff)
    if offered_traffic < 0:
        raise ValueError(f"traffic cannot be negative: {offered_traffic}")
    if offered_traffic == 0:
        # No work offered: with any server at all nothing is ever blocked.
        return 1.0 if staff == 0 else 0.0

    blocking = 1.0
    for n in range(1, staff + 1):
        busy = offered_traffic * blocking
        blocking = busy / (n + busy)
    return blocking


def probability_of_waiting(staff, offered_traffic):
    """Erlang C: the share of contacts that find every server busy.

    Derived from Erlang B so it inherits the recursion's stability:

        C = B / (1 - rho x (1 - B)),  rho = A / N

    Below N = A the queue is unstable - work arrives faster than it can be
    cleared, so the backlog grows without limit and every caller eventually
    waits. That returns 1.0 rather than raising, because a staffing search
    needs to be able to ask about understaffed options and get a sensible
    "this does not work" instead of an exception.
    """
    staff = _check_staff(staff)
    if offered_traffic <= 0:
        return 0.0
    if staff <= offered_traffic:
        return 1.0

    blocking = erlang_b(staff, offered_traffic)
    utilisation = offered_traffic / staff
    return blocking / (1.0 - utilisation * (1.0 - blocking))


def service_level(staff, offered_traffic, target_seconds, handle_time_seconds):
    """Share of contacts answered within the target (spec 4.1).

        SL = 1 - P(wait) x e ^ (-(N - A) x target / handle time)

    The exponential is the chance that a caller who DID have to wait is
    still waiting when the target expires. (N - A) is the spare capacity
    working the queue down, so service level improves steeply once staffing
    clears the traffic - and barely at all before it.

    Staff at or below the traffic returns 0.0: an unstable queue serves
    nobody inside a fixed target once it has been running long enough.
    """
    staff = _check_staff(staff)
    if handle_time_seconds <= 0:
        raise ValueError(f"handle time must be positive: {handle_time_seconds}")
    if target_seconds < 0:
        raise ValueError(f"answer target cannot be negative: {target_seconds}")
    if offered_traffic <= 0:
        return 1.0
    if staff <= offered_traffic:
        return 0.0

    waiting = probability_of_waiting(staff, offered_traffic)
    decay = math.exp(-(staff - offered_traffic) * target_seconds / handle_time_seconds)
    return 1.0 - waiting * decay


def occupancy(staff, offered_traffic):
    """Share of paid time the team spends actually handling contacts.

    A / N, and the reason the cap exists. A queue can hit its service level
    while every person is busy 95% of the half-hour, which looks efficient on
    paper and is unworkable in practice - no recovery between contacts, no
    slack for a sick day, and attrition that shows up two quarters later.
    """
    staff = _check_staff(staff)
    if offered_traffic <= 0:
        return 0.0
    if staff == 0:
        return math.inf
    return offered_traffic / staff


def required_staff(
    offered_traffic,
    target_seconds,
    handle_time_seconds,
    service_level_goal,
    occupancy_cap,
    max_staff=5000,
):
    """The smallest team meeting the service goal without breaching the cap.

    Both constraints move one way as staff are added - service level only
    rises, occupancy only falls - so the first N satisfying both is the
    answer, and the search can start at the lowest N the cap could possibly
    allow (ceil(A / cap)) instead of at 1.

    Which constraint binds is worth watching. In a busy queue the service
    level decides; in a small or slow one the cap decides, and the plan ends
    up staffing above what the goal alone would ask for. Both are correct
    answers to the question actually being asked.
    """
    if offered_traffic < 0:
        raise ValueError(f"traffic cannot be negative: {offered_traffic}")
    if not 0 < occupancy_cap <= 1:
        raise ValueError(f"occupancy cap must be in (0, 1]: {occupancy_cap}")
    if not 0 <= service_level_goal <= 1:
        raise ValueError(f"service level goal must be in [0, 1]: {service_level_goal}")
    if offered_traffic == 0:
        return 0

    staff = max(1, math.ceil(offered_traffic / occupancy_cap))
    while staff <= max_staff:
        within_cap = occupancy(staff, offered_traffic) <= occupancy_cap + TOLERANCE
        meets_goal = (
            service_level(staff, offered_traffic, target_seconds, handle_time_seconds)
            >= service_level_goal - TOLERANCE
        )
        if within_cap and meets_goal:
            return staff
        staff += 1

    raise ValueError(
        f"no staff count up to {max_staff} meets a {service_level_goal:.0%} goal "
        f"at {offered_traffic:.2f} Erlangs; check the inputs"
    )


def staff_for_interval(
    arrivals,
    handle_time_seconds,
    interval_seconds,
    target_seconds,
    service_level_goal,
    occupancy_cap,
    max_staff=5000,
):
    """Convenience: arrivals straight through to a staff number.

    Returns (staff, traffic, service level achieved, occupancy) so a caller
    can see not just the answer but how much headroom it bought - which is
    what tells you whether the cap or the goal was the binding constraint.
    """
    offered = traffic(arrivals, handle_time_seconds, interval_seconds)
    staff = required_staff(
        offered,
        target_seconds,
        handle_time_seconds,
        service_level_goal,
        occupancy_cap,
        max_staff,
    )
    return (
        staff,
        offered,
        service_level(staff, offered, target_seconds, handle_time_seconds),
        occupancy(staff, offered),
    )


def _check_staff(staff):
    if isinstance(staff, float) and not staff.is_integer():
        raise ValueError(f"staff must be a whole number of people: {staff}")
    staff = int(staff)
    if staff < 0:
        raise ValueError(f"staff cannot be negative: {staff}")
    return staff
