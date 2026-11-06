"""Intraday pattern, live-queue staffing, breaks and lunches (spec 3.4, 4.1, 4.2).

Session 3. The interval curves were approved by Siri on 10/22, and this
module now carries the onsite live-work build end to end: weekly walk-in and
call volume split down to half-hours, staffed with Erlang C, and grossed up
for the people on break.

The WFM idea, in two steps:

  ACROSS DAYS, then ACROSS HALF-HOURS. A weekly volume tells you almost
  nothing about staffing, because Erlang C staffs to the peak and a week's
  average hides every peak there is. Monday morning and Friday afternoon are
  different problems at the same weekly total, so the weekly figure is split
  by day (spec 3.4) and then by half-hour before it ever reaches a queue.

  SCHEDULED IS NOT THE SAME AS NEEDED. Erlang C gives the staff who must be
  AT the counter in an interval. Scheduled staff has to be higher, because
  some of the team is on break or at lunch right then (spec 4.2):

      Scheduled staff = staff needed / (1 - share away in that interval)

The away share is DERIVED from the break and lunch rules rather than typed
in. The derivation is simpler than it first looks: if every person takes one
relief of M minutes somewhere inside a window W minutes long, and the team is
staggered evenly across that window, then the team collectively spends
(staff x M) minutes away out of (staff x W) minutes of capacity - so the
share away in any interval of the window is just M / W, whatever the interval
length happens to be.

    15-minute break in a 2-hour window  = 15 / 120 = 12.5%
    30-minute lunch  in a 3-hour window = 30 / 180 = 16.7%

Teaching note from spec 4.2: an unpaid lunch costs no pay but opens a
coverage hole midday. That is why small sites end up staffed for their peaks
rather than for their hours - and why the chart puts the away share directly
under the arrival curves, on the same time axis, so the overlap is visible.

Run it:
    .venv\\Scripts\\python -m capacity_model.intervals
"""

import datetime

import numpy as np

import pandas as pd

from . import automation, config, erlang, generate_data, work_volume

MINUTES_PER_HOUR = 60


# ---------------------------------------------------------------------------
# The day's shape
# ---------------------------------------------------------------------------


def _parse_clock(text):
    hours, minutes = str(text).split(":")
    return datetime.time(int(hours), int(minutes))


def _minutes_since_midnight(moment):
    return moment.hour * MINUTES_PER_HOUR + moment.minute


def interval_starts(assumptions=None):
    """The start time of each half-hour the site is open (spec 3.4).

    Mon to Fri, 8:00 to 17:00, 18 intervals. Checked against the closing
    time rather than trusted, because an interval count that disagrees with
    the opening hours would silently shorten or stretch every curve built
    on top of it.
    """
    assumptions = assumptions or config.load_assumptions()
    hours = assumptions["intraday"]["site_hours"]

    opens = _parse_clock(hours["opens"])
    closes = _parse_clock(hours["closes"])
    step = hours["interval_minutes"]
    count = hours["intervals_per_day"]

    span = _minutes_since_midnight(closes) - _minutes_since_midnight(opens)
    if span != step * count:
        raise ValueError(
            f"site hours {hours['opens']}-{hours['closes']} span {span} minutes, "
            f"which is not {count} intervals of {step} minutes"
        )

    base = datetime.datetime.combine(datetime.date.min, opens)
    return [(base + datetime.timedelta(minutes=step * i)).time() for i in range(count)]


def interval_labels(assumptions=None):
    """Interval start times as "HH:MM", the keys the curves are written with."""
    return [t.strftime("%H:%M") for t in interval_starts(assumptions)]


def day_of_week_shares(assumptions=None):
    """How a week's volume splits across Monday to Friday (spec 3.4).

    Requests build up over the weekend, so Monday carries 23% and the week
    tapers to 17% on Friday. Must sum to 1.0.
    """
    assumptions = assumptions or config.load_assumptions()
    shares = assumptions["intraday"]["day_of_week_share"]
    total = sum(shares.values())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"day-of-week shares sum to {total}, not 1.0")
    return dict(shares)


def arrival_curve(name, assumptions=None):
    """One interval curve as an array, in interval order (spec 3.4).

    The YAML keys the curve by clock time so the shape is readable; this
    looks each interval up by name rather than trusting file order, so a
    missing or mis-keyed row fails loudly instead of shifting the afternoon
    along by one slot.
    """
    assumptions = assumptions or config.load_assumptions()
    curves = assumptions["intraday"]["interval_curves"]
    if name not in curves:
        raise ValueError(f"no interval curve named {name!r}; have {sorted(curves)}")

    curve = curves[name]
    labels = interval_labels(assumptions)
    missing = [label for label in labels if label not in curve]
    if missing:
        raise ValueError(f"curve {name!r} is missing intervals: {missing}")
    extra = [key for key in curve if key not in labels]
    if extra:
        raise ValueError(f"curve {name!r} has intervals outside site hours: {extra}")

    values = np.array([float(curve[label]) for label in labels])
    total = values.sum()
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"curve {name!r} sums to {total}, not 1.0")
    return values


def curve_names(assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    return sorted(assumptions["intraday"]["interval_curves"])


# ---------------------------------------------------------------------------
# Breaks and lunch (spec 4.2)
# ---------------------------------------------------------------------------


def live_work_types(assumptions=None):
    """The two work types that form a live queue, keyed by curve name.

    Walk-ins and calls are the only work where somebody waits in real time
    (spec 4.1). Everything else queues as backlog and is planned weekly in
    4.3, so only these two reach Erlang C.
    """
    assumptions = assumptions or config.load_assumptions()
    by_code = {w["code"]: w for w in assumptions["work_types"]}
    mapping = {}
    for curve, code in assumptions["breaks_and_lunch"]["relief_placement"][
        "load_curves"
    ].items():
        work_type = by_code[code]
        if "handle_time_minutes" not in work_type:
            raise ValueError(f"{code} needs a handle_time_minutes to drive a live queue")
        mapping[curve] = work_type
    return mapping


def combined_load(assumptions=None):
    """Relative load per interval for a typical week (spec 3.4).

    Load is volume x handle time, not contact count - that distinction
    decides the answer here. A call costs 5 minutes and arrives 1.0 per
    request; a walk-in costs 8 minutes but only 0.15 per request. So calls
    carry about four times the onsite live load, and ranking the day by
    contacts alone would point relief at the wrong half-hours.

    This is the curve-level version, for the standalone intraday chart,
    which has no volumes behind it. onsite_intervals() uses the real
    per-interval traffic instead; the two agree in shape for a week whose
    walk-in and call volumes sit in their usual ratio.
    """
    assumptions = assumptions or config.load_assumptions()
    total = np.zeros(len(interval_starts(assumptions)))
    for curve, work_type in live_work_types(assumptions).items():
        minutes_per_request = work_type["rate"] * work_type["handle_time_minutes"]
        total = total + minutes_per_request * arrival_curve(curve, assumptions)
    return total


def _allocate_relief(weights, mass, cap):
    """Spread `mass` of relief over a window, by weight, capped per interval.

    Water-filling: share it out in proportion to weight, and wherever that
    would push an interval past its cap, peg it at the cap and push the
    excess back into the ones still below theirs. Repeat until nothing is
    over.

    `cap` is a number, or one headroom per interval. The per-interval form
    is what makes the COMBINED cap work (spec 3.4, decided 10/27): a relief
    placed after another sees only the capacity the first one left, so the
    two cannot add up past the cap where their windows overlap.

    The total never changes - the same people take the same breaks - so the
    window average is preserved whatever the cap does to the distribution.
    """
    weights = np.asarray(weights, dtype=float)
    count = len(weights)
    caps = np.broadcast_to(np.asarray(cap, dtype=float), weights.shape).astype(float)
    if mass > caps.sum() + 1e-12:
        raise ValueError(
            f"cannot place {mass:.4f} of relief across {count} intervals with "
            f"{caps.sum():.4f} of headroom between them; widen the window, "
            "raise the cap, or move an overlapping relief"
        )

    shares = np.zeros(count)
    # An interval with no headroom left is not a candidate at all.
    free = caps > 1e-15
    remaining = mass

    while free.any() and remaining > 1e-15:
        live = np.where(free, weights, 0.0)
        if live.sum() <= 0:
            # No weight left to distribute by: spread what remains evenly
            # across what headroom is left.
            shares[free] += np.minimum(remaining / free.sum(), caps[free])
            break
        tentative = remaining * live / live.sum()
        over = free & (tentative > caps + 1e-15)
        if not over.any():
            shares[free] += tentative[free]
            break
        shares[over] = caps[over]
        remaining -= caps[over].sum()
        free = free & ~over

    return shares


def relief_windows(assumptions=None):
    """Which intervals each relief's window covers, as boolean masks.

    Kept separate from the shares because a window is a fact about the rule,
    while a share of zero is a fact about the placement. Reading the window
    back off the shares would mislead the moment the cap pushed an interval
    to zero.
    """
    assumptions = assumptions or config.load_assumptions()
    starts = interval_starts(assumptions)
    step = assumptions["intraday"]["site_hours"]["interval_minutes"]

    windows = {}
    for relief in assumptions["breaks_and_lunch"]["reliefs"]:
        window_from = _parse_clock(relief["window"]["from"])
        window_to = _parse_clock(relief["window"]["to"])
        window_minutes = _minutes_since_midnight(window_to) - _minutes_since_midnight(window_from)
        if window_minutes <= 0:
            raise ValueError(f"{relief['code']}: window must end after it starts")
        if window_minutes % step:
            raise ValueError(
                f"{relief['code']}: a {window_minutes}-minute window is not a whole "
                f"number of {step}-minute intervals"
            )

        inside = np.array([window_from <= start < window_to for start in starts])
        if not inside.any():
            raise ValueError(f"{relief['code']}: window falls outside the site's open hours")
        windows[relief["code"]] = inside
    return windows


def relief_shares(assumptions=None, load=None):
    """Share of the team away in each interval, per relief (spec 4.2).

    One entry per break and lunch, each an array over the day's intervals.

    The window AVERAGE is fixed by the rule itself: a relief of M minutes
    taken somewhere inside a window of W minutes puts M / W of the team away
    across that window, whatever the placement. 15/120 = 12.5% for a break,
    30/180 = 16.7% at lunch.

    Where inside the window the relief falls is the placement method:

      even          - the same share in every interval. Simple, and the
                      last thing a real team would do.
      inverse_load  - share inversely proportional to how busy the interval
                      is, so the quiet half-hours absorb the relief and the
                      busy ones keep their cover. This is what a supervisor
                      does by eye. Capped per interval so the rule cannot
                      send the whole team to lunch in one slot.

    THE CAP IS COMBINED (spec 3.4, decided 10/27): it limits everyone away
    in a half-hour across breaks AND lunch, not each relief on its own. A
    cap that counted reliefs separately would allow 25% at lunch plus 25%
    on a break in the same half-hour, which is half the team gone and
    exactly the rota the cap exists to forbid.

    It is implemented by giving the reliefs a shared headroom and placing
    them IN THE ORDER THEY FALL IN THE DAY, each into what the earlier ones
    left. That ordering is a modelling choice and it has a consequence
    worth knowing: where two windows overlap, the earlier relief has first
    call on the shared half-hour and the later one works around it. It is
    the order a rota is actually built - the morning break is settled
    before anyone argues about lunch - and it reduces exactly to the old
    per-relief behaviour whenever the windows are disjoint, which is why
    the baseline is untouched by the change.
    """
    assumptions = assumptions or config.load_assumptions()
    starts = interval_starts(assumptions)
    step = assumptions["intraday"]["site_hours"]["interval_minutes"]
    placement = assumptions["breaks_and_lunch"]["relief_placement"]
    method = placement["method"]
    cap = float(placement["max_share_per_interval"])
    if not 0 < cap <= 1:
        raise ValueError(f"max_share_per_interval must be in (0, 1]: {cap}")

    if method == "inverse_load" and load is None:
        load = combined_load(assumptions)
    windows = relief_windows(assumptions)

    shares = {}
    # Headroom shared across every relief, so the cap is a limit on the
    # TOTAL away in a half-hour (spec 3.4, 10/27).
    headroom = np.full(len(starts), cap)
    in_day_order = sorted(
        assumptions["breaks_and_lunch"]["reliefs"],
        key=lambda relief: _minutes_since_midnight(
            _parse_clock(relief["window"]["from"])
        ),
    )
    for relief in in_day_order:
        inside = windows[relief["code"]]
        window_from = _parse_clock(relief["window"]["from"])
        window_to = _parse_clock(relief["window"]["to"])
        window_minutes = _minutes_since_midnight(window_to) - _minutes_since_midnight(window_from)

        average = relief["minutes"] / window_minutes
        mass = average * int(inside.sum())

        if method == "even":
            weights = np.ones(int(inside.sum()))
        elif method == "inverse_load":
            window_load = load[inside]
            if (window_load <= 0).any():
                raise ValueError(
                    f"{relief['code']}: an interval in its window has no load, so "
                    "inverse-load placement is undefined there"
                )
            weights = 1.0 / window_load
        else:
            raise ValueError(f"unknown relief placement method: {method!r}")

        placed = np.zeros(len(starts))
        try:
            placed[inside] = _allocate_relief(weights, mass, headroom[inside])
        except ValueError as problem:
            raise ValueError(f"{relief['code']}: {problem}") from problem
        shares[relief["code"]] = placed
        headroom = headroom - placed

    # Returned in the file's own order, not the day's, so a caller reading
    # the result does not have to know how it was built.
    return {
        relief["code"]: shares[relief["code"]]
        for relief in assumptions["breaks_and_lunch"]["reliefs"]
    }


def relief_window_average(code, assumptions=None, load=None):
    """The average share away across one relief's own window.

    Fixed by the rule (minutes / window minutes) and unchanged by placement,
    which is the property the placement change has to preserve.
    """
    shares = relief_shares(assumptions, load)[code]
    inside = relief_windows(assumptions)[code]
    return float(shares[inside].mean())


def away_share(assumptions=None, load=None):
    """Total share of staff on break or at lunch in each interval (spec 4.2).

    Reliefs add where their windows overlap, and the 25% cap is a limit on
    that TOTAL (spec 3.4, decided 10/27). relief_shares guarantees it by
    construction; this checks it anyway, because it is the one property the
    whole placement rule exists to deliver and an invariant nobody checks
    is an invariant nobody notices breaking.
    """
    assumptions = assumptions or config.load_assumptions()
    shares = relief_shares(assumptions, load)
    total = np.sum(list(shares.values()), axis=0)
    labels = interval_labels(assumptions)

    # Two guards, and both are needed. A total at or above 1.0 means the
    # whole team is away at once and spec 4.2's formula divides by zero -
    # reachable only by lifting the cap to 1.0, but reachable.
    if (total >= 1.0).any():
        worst = labels[int(np.argmax(total))]
        raise ValueError(
            f"breaks and lunch put {total.max():.1%} of the team away at {worst}; "
            "the whole team cannot be on relief at once"
        )

    # And the cap itself, which relief_shares guarantees by construction
    # (spec 3.4, decided 10/27: across all reliefs combined). Checked
    # anyway, because it is the one property the placement rule exists to
    # deliver and an invariant nobody checks is one nobody notices break.
    cap = float(
        assumptions["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"]
    )
    if (total > cap + 1e-9).any():
        worst = labels[int(np.argmax(total))]
        raise ValueError(
            f"breaks and lunch put {total.max():.2%} of the team away at {worst}, "
            f"over the {cap:.0%} combined cap"
        )
    return total


def scheduled_staff(staff_needed, away):
    """Spec 4.2: scheduled staff = staff needed / (1 - share away).

    Provided here because it is the formula this section exists to state.
    Nothing calls it yet - wiring it to Erlang C is the next task.
    """
    away = np.asarray(away, dtype=float)
    if (away < 0).any() or (away >= 1).any():
        raise ValueError("away share must be in [0, 1)")
    return np.asarray(staff_needed, dtype=float) / (1.0 - away)


# ---------------------------------------------------------------------------
# The review chart
# ---------------------------------------------------------------------------


def plot_intraday(assumptions=None):
    """The approved day shape (spec 3.4 asks for sign-off from a chart).

    Two panels on one time axis. Arrival curves on top, because they are
    what the reader came for; the share of staff away underneath, because
    the thing worth seeing is where the two line up. The lunch window sits
    directly under the walk-in trough and the start of the afternoon rise -
    the midday coverage hole from spec 4.2, drawn rather than described.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assumptions = assumptions or config.load_assumptions()
    labels = interval_labels(assumptions)
    positions = np.arange(len(labels))
    away = away_share(assumptions)

    calls = arrival_curve("calls", assumptions)
    walk_ins = arrival_curve("walk_ins", assumptions)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 7), dpi=150, sharex=True, height_ratios=[2, 1]
    )
    fig.patch.set_facecolor(generate_data.SURFACE)

    # The three relief windows run back to back (9-11, 11-14, 14-16), so
    # shading them would merge into one block and read as a single event.
    # Dotted dividers at the boundaries instead, carried down both panels so
    # the eye can take a time from an arrival peak to the coverage below.
    starts = interval_starts(assumptions)
    windows = []
    for relief in assumptions["breaks_and_lunch"]["reliefs"]:
        window_from = _parse_clock(relief["window"]["from"])
        window_to = _parse_clock(relief["window"]["to"])
        inside = [i for i, start in enumerate(starts) if window_from <= start < window_to]
        # "Morning paid break" -> "Morning break"; "Unpaid lunch" -> "Lunch".
        # Paid-vs-unpaid is in the YAML and in the summary; on the chart it
        # would just crowd the label.
        label = relief["name"].replace(" paid", "").replace("Unpaid ", "").capitalize()
        windows.append((inside[0], inside[-1], label))

    boundaries = sorted({edge for first, last, _ in windows for edge in (first - 0.5, last + 0.5)})
    for axis in (top, bottom):
        for edge in boundaries:
            axis.axvline(
                edge, color=generate_data.BASELINE, linewidth=1, linestyle=":", zorder=1
            )

    for axis in (top, bottom):
        axis.set_facecolor(generate_data.SURFACE)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=generate_data.GRIDLINE, linewidth=0.8)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(generate_data.BASELINE)
        axis.tick_params(colors=generate_data.INK_MUTED, labelsize=9, length=0)

    top.plot(
        positions, calls * 100, color=generate_data.SERIES, linewidth=2,
        marker="o", markersize=5, label="Status and help calls (W20)", zorder=3,
    )
    top.plot(
        positions, walk_ins * 100, color=generate_data.HOLIDAY, linewidth=2,
        marker="o", markersize=5, label="Walk-in requesters (W19)", zorder=3,
    )
    top.set_ylim(0, max(calls.max(), walk_ins.max()) * 100 * 1.25)
    top.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
    top.set_ylabel("Share of the day's arrivals", color=generate_data.INK_SECONDARY, fontsize=10)

    legend = top.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(generate_data.INK_SECONDARY)

    bottom.fill_between(
        positions, away * 100, step="mid", color=generate_data.INK_MUTED, alpha=0.30, zorder=2
    )
    bottom.step(
        positions, away * 100, where="mid", color=generate_data.INK_SECONDARY,
        linewidth=2, zorder=3,
    )
    cap = assumptions["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"] * 100
    bottom.set_ylim(0, max(cap * 1.45, away.max() * 100 * 1.9))
    bottom.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
    bottom.set_ylabel("Staff away", color=generate_data.INK_SECONDARY, fontsize=10)

    # The cap is a rule, not data, so it is drawn as a rule: a dashed line
    # showing the headroom the placement is working inside.
    bottom.axhline(cap, color=generate_data.HOLIDAY, linewidth=1.5, linestyle="--", zorder=4)
    bottom.annotate(
        f"cap {cap:.0f}%",
        (len(labels) - 0.7, cap),
        textcoords="offset points",
        xytext=(0, 4),
        va="bottom",
        ha="right",
        fontsize=9,
        color=generate_data.HOLIDAY,
    )

    # Name each window and its AVERAGE in the panel the levels belong to.
    # The average is what the rule fixes; the steps show where inside the
    # window the placement actually put the relief.
    averages = {
        relief["code"]: relief_window_average(relief["code"], assumptions)
        for relief in assumptions["breaks_and_lunch"]["reliefs"]
    }
    for (first, last, label), code in zip(windows, averages):
        middle = (first + last) / 2
        bottom.text(
            middle,
            0.95,
            f"{label}  ·  avg {averages[code]:.1%}",
            transform=bottom.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=generate_data.INK_SECONDARY,
        )

    bottom.set_xticks(positions)
    bottom.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    bottom.set_xlim(-0.6, len(labels) - 0.4)

    top.set_title(
        "Within-day pattern and break coverage",
        color=generate_data.INK,
        fontsize=14,
        loc="left",
        pad=34,
    )
    top.text(
        0.0,
        1.022,
        "SYNTHETIC. Interval curves approved 10/22 under spec 3.4. "
        "Break and lunch coverage is derived from spec 4.2.",
        transform=top.transAxes,
        color=generate_data.INK_SECONDARY,
        fontsize=10,
    )

    fig.tight_layout()
    out_dir = config.resolve_path(assumptions["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / assumptions["paths"]["intraday_chart_png"]
    fig.savefig(out_path, facecolor=generate_data.SURFACE)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# The onsite live-work build (spec 3.4 -> 4.1 -> 4.2)
#
# This is where the three sections meet. A weekly volume becomes a staffing
# number in four steps, and every one of them matters:
#
#   WEEK -> DAY     by the day-of-week split. Monday carries 23% and Friday
#                   17%, so a flat fifth would understaff the start of the
#                   week and overstaff the end.
#   DAY -> HALF-HOUR by the approved curves. This is the step that makes
#                   Erlang C possible at all: it needs arrivals in an
#                   interval, and a daily total hides every peak.
#   ARRIVALS -> STAFF by Erlang C, counter and phones costed SEPARATELY and
#                   then added. They are different queues with different
#                   goals and different handle times, and a person cannot
#                   be at the counter and on a call at the same moment.
#   STAFF -> SCHEDULED by the break and lunch cover.
#
# Teaching note worth keeping in view: the sum of these interval staff
# numbers is far more than weekly volume x handle time / 40 would suggest.
# That gap is what Erlang C is for.
# ---------------------------------------------------------------------------

SECONDS_PER_MINUTE = 60


def pooled_required_staff(
    walk_in_arrivals, call_arrivals, settings, interval_seconds, cap, max_staff
):
    """One blended team per site and interval (spec 4.1, decided 10/22).

    At a site taking a handful of walk-ins an hour, two separate queues each
    need somebody parked on them and both run near 20% occupancy - two
    people mostly waiting. In the real office the person at the counter
    answers the phone between requesters, so the two are pooled.

    Solved at the PHONES' goal - 80% in 20 seconds, the stricter of the two -
    on a volume-weighted handle time, then topped up until the walk-in goal
    (90% within 10 minutes) also holds.

    Checking the second goal matters even though the first usually binds.
    The weighted handle time sits between 5 and 8 minutes, and a longer
    handle time makes a 20-second target harder but a 10-minute target
    easier, so which goal binds shifts with the channel mix. Solving for one
    alone would let a blended team quietly fail the counter to flatter the
    phones.

    Returns (staff, traffic, weighted handle time in seconds).
    """
    walk_ins, calls = settings["walk_ins"], settings["calls"]
    work_seconds = (
        walk_in_arrivals * walk_ins["handle_time_seconds"]
        + call_arrivals * calls["handle_time_seconds"]
    )
    contacts = walk_in_arrivals + call_arrivals
    if contacts <= 0 or work_seconds <= 0:
        return 0, 0.0, 0.0

    handle_time = work_seconds / contacts
    offered = work_seconds / interval_seconds

    staff = erlang.required_staff(
        offered,
        calls["target_seconds"],
        handle_time,
        calls["service_level_goal"],
        cap,
        max_staff,
    )
    # Top up until the counter goal holds as well. Service level only rises
    # with staff, so this terminates at the first adequate number.
    while staff < max_staff and erlang.service_level(
        staff, offered, walk_ins["target_seconds"], handle_time
    ) < walk_ins["service_level_goal"] - erlang.TOLERANCE:
        staff += 1

    return staff, offered, handle_time


def support_work_hours(work, assumptions=None):
    """Weekly email and portal hours per site (spec 4.2, W21).

    Backlog work, so it has no interval curve of its own - it is whatever
    the team can fit around the live queues.
    """
    assumptions = assumptions or config.load_assumptions()
    support = assumptions["onsite_team"]["support_work"]
    by_code = {w["code"]: w for w in assumptions["work_types"]}
    work_type = by_code[support["work_type"]]
    if "handle_time_minutes" not in work_type:
        raise ValueError(f"{support['work_type']} needs a handle_time_minutes")

    rows = work[work["work_type"] == support["work_type"]]
    if rows.empty:
        raise ValueError(f"work volume has no rows for {support['work_type']}")
    weekly = rows.groupby(["week_index", "site"])["work_items"].sum()
    # Spec 7.1's demand-side uplift reaches email too: the simple messages
    # are the ones a portal answers, so what is left takes longer. Exactly
    # 1.0 per week in the baseline.
    uplift = np.array(
        [
            automation.deflection_handle_time_multiple(
                support["work_type"], week, assumptions
            )
            for week, _site in weekly.index
        ]
    )
    return weekly * work_type["handle_time_minutes"] * uplift / MINUTES_PER_HOUR


def live_weekly_volume(work, assumptions=None):
    """Weekly walk-in and call volume per site, from the work-volume build.

    Takes the lagged volumes as they come out of spec 3.3 rather than
    re-deriving them from requests. Walk-ins land in the week the request
    arrives; calls are split 50/50 over that week and the next - so the two
    channels drift apart week to week, which is exactly why the live load
    has to be rebuilt per week rather than assumed constant.
    """
    assumptions = assumptions or config.load_assumptions()
    codes = {curve: wt["code"] for curve, wt in live_work_types(assumptions).items()}

    wanted = work[work["work_type"].isin(codes.values())]
    weekly = wanted.pivot_table(
        index=["week_index", "week_start", "period", "site"],
        columns="work_type",
        values="work_items",
        aggfunc="sum",
    ).reset_index()
    missing = set(codes.values()) - set(weekly.columns)
    if missing:
        raise ValueError(f"work volume has no rows for {sorted(missing)}")
    return weekly, codes


# ---------------------------------------------------------------------------
# Centralized phones (spec 7, decided 10/27)
#
# The baseline pools the counter and the phones at each SITE (4.1). This
# scenario pools the phones across the NETWORK instead and leaves each
# counter to staff walk-ins alone.
#
# Everything here is built so the four figures a leader needs come out as
# separate lines rather than one net number:
#
#   sum of six phone-only queues   what the phones cost unpooled
#   central, service level only    what pooling alone would save
#   central, with the 85% cap      how much of that the cap takes back
#   central, with the +5% uplift   what the lost local knowledge costs
#
# Sites share one local time (decided 10/27), which is what makes pooling
# possible at all: six queues in six time zones do not peak together and
# could not be answered by one team at one goal.
# ---------------------------------------------------------------------------


def central_phones_settings(assumptions=None):
    """The central-phones block (spec 7)."""
    assumptions = assumptions or config.load_assumptions()
    return assumptions["scenario"]["central_phones"]


def central_phones_enabled(assumptions=None):
    """Whether the phones are centralized. False in the baseline."""
    return bool(central_phones_settings(assumptions)["enabled"])


def central_phones_from_week(assumptions=None):
    """The absolute week index the central team takes over (spec 7: week 13)."""
    assumptions = assumptions or config.load_assumptions()
    return config.scenario_starts_week(assumptions)


def _queue_settings(assumptions):
    """Handle time, target and goal for each live channel (4.1)."""
    live = live_work_types(assumptions)
    goals = {g["code"]: g for g in assumptions["live_queues"]["goals"]}
    settings = {}
    for curve, work_type in live.items():
        goal = next(g for g in goals.values() if g["work_type"] == work_type["code"])
        settings[curve] = {
            "handle_time_seconds": work_type["handle_time_minutes"] * SECONDS_PER_MINUTE,
            "target_seconds": goal["target_seconds"],
            "service_level_goal": goal["service_level"],
        }
    return settings


def central_phone_queue(assumptions=None, sites=None, work=None):
    """The network phone queue, every half-hour (spec 7, decided 10/27).

    One row per week, weekday and interval - no site, because there is one
    queue. Calls are summed across the six sites first and solved once,
    which is the whole mechanism: Erlang C's spare capacity is shared
    rather than duplicated six times.

    Four staffing answers on every row, so the decomposition in the report
    is arithmetic rather than assertion:

        required_separate     the six per-site queues added up
        required_no_cap       central, service level only
        required_capped       central, with the 85% occupancy cap
        required_staff        central, capped, at the uplifted handle time

    required_separate is computed at the BASE handle time, because the
    comparison spec Section 9 asks for is pooling against not pooling at
    the same goal and the same handle time. The uplift is a separate line.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if work is None:
        work = work_volume.build(assumptions, sites)["work"]

    weekly, _codes = live_weekly_volume(work, assumptions)
    live = live_work_types(assumptions)
    settings = _queue_settings(assumptions)
    phones = settings["calls"]
    queues = assumptions["live_queues"]
    interval_seconds = queues["interval_seconds"]
    cap = queues["occupancy_cap"]
    max_staff = queues["max_staff_search"]
    uplift = float(central_phones_settings(assumptions)["handle_time_uplift"])

    labels = interval_labels(assumptions)
    curve = arrival_curve("calls", assumptions)
    days = list(day_of_week_shares(assumptions).items())
    call_code = live["calls"]["code"]

    base_handle = phones["handle_time_seconds"]
    central_handle = base_handle * (1.0 + uplift)

    # Calls per site-week, so the network total and each site's share of
    # it are both available.
    by_week = weekly.groupby(["week_index", "week_start", "period"], as_index=False).agg(
        calls=(call_code, "sum")
    )

    rows = []
    for record in by_week.to_dict("records"):
        site_calls = weekly[weekly["week_index"] == record["week_index"]]
        for day_name, day_share in days:
            network = record["calls"] * day_share * curve
            offered_base = erlang.traffic(network, base_handle, interval_seconds)
            offered_up = erlang.traffic(network, central_handle, interval_seconds)

            def solve(offered, handle, occupancy_cap):
                return np.array(
                    [
                        erlang.required_staff(
                            value,
                            phones["target_seconds"],
                            handle,
                            phones["service_level_goal"],
                            occupancy_cap,
                            max_staff,
                        )
                        for value in offered
                    ]
                )

            # The six per-site queues, at the BASE handle time, added up.
            separate = np.zeros(len(labels))
            for site_row in site_calls.to_dict("records"):
                site_arrivals = site_row[call_code] * day_share * curve
                site_offered = erlang.traffic(
                    site_arrivals, base_handle, interval_seconds
                )
                separate = separate + solve(site_offered, base_handle, cap)

            rows.append(
                pd.DataFrame(
                    {
                        "week_index": record["week_index"],
                        "week_start": record["week_start"],
                        "period": record["period"],
                        "day_of_week": day_name,
                        "interval": np.arange(len(labels)),
                        "interval_start": labels,
                        "call_arrivals": network,
                        "traffic_base": offered_base,
                        "traffic_uplifted": offered_up,
                        "handle_seconds_base": base_handle,
                        "handle_seconds_central": central_handle,
                        "required_separate": separate,
                        # Service level only: what pooling alone buys.
                        "required_no_cap": solve(offered_base, base_handle, 1.0),
                        # Then the cap, which bites hardest here (spec 7).
                        "required_capped": solve(offered_base, base_handle, cap),
                        # Then the price of the lost local knowledge.
                        "required_staff": solve(offered_up, central_handle, cap),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _onsite_intervals_blended(assumptions, sites, work):
    """The blended onsite team, every half-hour (spec 3.4 -> 4.1 -> 4.2).

    The baseline. One row per week, site, weekday and interval across the
    208-week timeline.

    Staffing is the BLENDED queue (spec 4.1, decided 10/22): counter and
    phones pooled into one team. The separate-queue answer is kept beside it
    in required_separate, because the difference between the two is the
    whole point of the decision and the plan should be able to show it.

    Relief placement uses the REAL load (spec 3.4): traffic in Erlangs,
    volume x handle time over the interval length. That differs from the
    curve-level proxy whenever a week's walk-in and call volumes drift from
    their usual ratio, which the 50/50 call lag guarantees they will.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if work is None:
        work = work_volume.build(assumptions, sites)["work"]

    weekly, codes = live_weekly_volume(work, assumptions)
    live = live_work_types(assumptions)
    goals = {g["code"]: g for g in assumptions["live_queues"]["goals"]}
    queues = assumptions["live_queues"]
    interval_seconds = queues["interval_seconds"]
    cap = queues["occupancy_cap"]
    max_staff = queues["max_staff_search"]

    labels = interval_labels(assumptions)
    n_intervals = len(labels)
    days = list(day_of_week_shares(assumptions).items())
    curves = {curve: arrival_curve(curve, assumptions) for curve in live}

    # Pair each curve with the goal that governs it, via the work type.
    base_settings = {}
    for curve, work_type in live.items():
        goal = next(g for g in goals.values() if g["work_type"] == work_type["code"])
        base_settings[curve] = {
            "handle_time_seconds": work_type["handle_time_minutes"] * SECONDS_PER_MINUTE,
            "target_seconds": goal["target_seconds"],
            "service_level_goal": goal["service_level"],
        }

    rows = []
    for record in weekly.to_dict("records"):
        volumes = {curve: record[live[curve]["code"]] for curve in live}

        # Spec 7.1's demand-side uplift: the contacts left after deflection
        # are the harder ones, so they take longer. Per WEEK, because
        # deflection starts at the scenario's own week - and exactly 1.0 in
        # the baseline, where the multiply moves nothing.
        settings = {
            curve: {
                **base,
                "handle_time_seconds": base["handle_time_seconds"]
                * automation.deflection_handle_time_multiple(
                    live[curve]["code"], record["week_index"], assumptions
                ),
            }
            for curve, base in base_settings.items()
        }

        # The relief shape depends only on the RATIO of the two channels
        # within the week, because the day share scales every interval
        # equally and cancels out of the ranking. So it is computed once per
        # site-week and reused across the five days.
        shape = np.zeros(n_intervals)
        for curve, setting in settings.items():
            shape = shape + volumes[curve] * curves[curve] * setting["handle_time_seconds"]
        away = away_share(assumptions, load=shape) if shape.sum() > 0 else np.zeros(n_intervals)

        for day_name, day_share in days:
            arrivals = {
                curve: volumes[curve] * day_share * curves[curve] for curve in live
            }
            traffic = {
                curve: erlang.traffic(
                    arrivals[curve], settings[curve]["handle_time_seconds"], interval_seconds
                )
                for curve in live
            }
            needed = {
                curve: np.array(
                    [
                        erlang.required_staff(
                            offered,
                            settings[curve]["target_seconds"],
                            settings[curve]["handle_time_seconds"],
                            settings[curve]["service_level_goal"],
                            cap,
                            max_staff,
                        )
                        for offered in traffic[curve]
                    ]
                )
                for curve in live
            }

            # The "before" answer, kept for comparison: two queues staffed
            # independently and added. Spec 4.1 replaced this on 10/22.
            separate = sum(needed.values())

            # The blended team. Pooling cannot need more people than running
            # the two queues apart, and at these volumes it needs far fewer.
            pooled = np.empty(n_intervals, dtype=int)
            pooled_traffic = np.empty(n_intervals)
            pooled_handle = np.empty(n_intervals)
            for i in range(n_intervals):
                pooled[i], pooled_traffic[i], pooled_handle[i] = pooled_required_staff(
                    arrivals["walk_ins"][i],
                    arrivals["calls"][i],
                    settings,
                    interval_seconds,
                    cap,
                    max_staff,
                )

            required = pooled
            scheduled = scheduled_staff(required, away)
            live_handle_hours = (
                arrivals["walk_ins"] * settings["walk_ins"]["handle_time_seconds"]
                + arrivals["calls"] * settings["calls"]["handle_time_seconds"]
            ) / 3600.0

            rows.append(
                pd.DataFrame(
                    {
                        "week_index": record["week_index"],
                        "week_start": record["week_start"],
                        "period": record["period"],
                        "site": record["site"],
                        "day_of_week": day_name,
                        "interval": np.arange(n_intervals),
                        "interval_start": labels,
                        "walk_in_arrivals": arrivals["walk_ins"],
                        "call_arrivals": arrivals["calls"],
                        "traffic_counter": traffic["walk_ins"],
                        "traffic_phones": traffic["calls"],
                        "required_counter": needed["walk_ins"],
                        "required_phones": needed["calls"],
                        "required_separate": separate,
                        "traffic_pooled": pooled_traffic,
                        "pooled_handle_seconds": pooled_handle,
                        "required_staff": required,
                        "occupancy": np.where(required > 0, pooled_traffic / np.maximum(required, 1), 0.0),
                        "away_share": away,
                        "scheduled_staff": scheduled,
                        "scheduled_hours": scheduled * interval_seconds / 3600.0,
                        "live_handle_hours": live_handle_hours,
                    }
                )
            )

    table = pd.concat(rows, ignore_index=True)
    return _summarise_site_days(table, assumptions, work)


def _summarise_site_days(table, assumptions, work):
    """Per site-day hours, and the email that fits into idle time (4.2).

    Shared by both builders, because email absorption is a property of the
    schedule rather than of how the schedule was solved: whatever shape the
    live queues take, the hours they already pay for have idle time in them
    and 4.2 works email there first.
    """

    days_frame = (
        table.groupby(
            ["week_index", "week_start", "period", "site", "day_of_week"], as_index=False
        )
        .agg(
            live_work_hours=("scheduled_hours", "sum"),
            live_handle_hours=("live_handle_hours", "sum"),
            separate_staff_intervals=("required_separate", "sum"),
            pooled_staff_intervals=("required_staff", "sum"),
        )
    )

    # --- email and portal work in idle time (spec 4.2, decided 10/22) -----
    #
    # The live schedule already pays for every interval it covers, and the
    # team is not handling contacts for all of it. That idle time is real
    # paid capacity, so email is worked there first and costs nothing extra.
    # Only what will not fit adds hours.
    support = assumptions["onsite_team"]["support_work"]
    weekly_email = support_work_hours(work, assumptions)
    day_share = day_of_week_shares(assumptions)

    days_frame["idle_hours"] = (
        days_frame["live_work_hours"] - days_frame["live_handle_hours"]
    ).clip(lower=0.0)
    days_frame["email_hours"] = [
        weekly_email.get((week, site), 0.0) * day_share[day]
        for week, site, day in zip(
            days_frame["week_index"], days_frame["site"], days_frame["day_of_week"]
        )
    ]

    if support.get("absorb_in_idle_time", True):
        absorbed = np.minimum(days_frame["email_hours"], days_frame["idle_hours"])
    else:
        absorbed = np.zeros(len(days_frame))
    days_frame["email_absorbed_hours"] = absorbed
    days_frame["email_spillover_hours"] = days_frame["email_hours"] - absorbed

    # Only the spillover carries the paid-break allowance. Hours absorbed
    # into idle time sit inside a schedule whose breaks 4.2 already paid
    # for, so charging them again would buy the same break twice.
    allowance = support["spillover_break_allowance"]
    days_frame["email_paid_hours"] = days_frame["email_spillover_hours"] / (1.0 - allowance)
    days_frame["total_hours"] = (
        days_frame["live_work_hours"] + days_frame["email_paid_hours"]
    )

    return {"intervals": table, "hours": days_frame}


def onsite_intervals(assumptions=None, sites=None, work=None):
    """The onsite team, every half-hour, in whichever shape is configured.

    Two shapes, and the dispatch is the whole difference between them:

      BLENDED (the baseline, spec 4.1 decided 10/22). Counter and phones
        pooled at each site. One person answers the phone between
        requesters, which is what a small site actually does.
      CENTRAL PHONES (spec 7, decided 10/27). One network phone queue,
        and each counter staffs walk-ins alone at 90% in 10 minutes.

    The baseline takes the first branch and nothing else, so it runs the
    same code it always did - which is what keeps every generated file
    byte-identical.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if work is None:
        work = work_volume.build(assumptions, sites)["work"]

    if central_phones_enabled(assumptions):
        return _onsite_intervals_central(assumptions, sites, work)
    return _onsite_intervals_blended(assumptions, sites, work)


def _onsite_intervals_central(assumptions, sites, work):
    """Central phone queue, counters on walk-ins alone (spec 7, 10/27).

    The counter is solved at the WALK-IN goal only - 90% within 10 minutes
    on an 8-minute handle time - because the blend is gone. That is the
    effect the scenario exists to measure: a counter taking a handful of
    walk-ins an hour still needs somebody standing at it, and today that
    somebody is answering the phone between requesters.

    The central team's requirement is charged back to the sites pro rata
    by the calls each one sends the queue, because the hours come out of
    the Onsite Associates (spec 7) and they sit at six sites. Pro rata by
    calls is also what shares the pooling gain rather than letting one site
    capture it.

    Before the changeover week the sites run blended, so the scenario's
    first twelve weeks match the baseline exactly.
    """
    settings = _queue_settings(assumptions)
    walk_ins, phones = settings["walk_ins"], settings["calls"]
    queues = assumptions["live_queues"]
    interval_seconds = queues["interval_seconds"]
    cap = queues["occupancy_cap"]
    max_staff = queues["max_staff_search"]
    interval_hours = interval_seconds / 3600.0

    live = live_work_types(assumptions)
    call_code, walk_code = live["calls"]["code"], live["walk_ins"]["code"]
    labels = interval_labels(assumptions)
    n_intervals = len(labels)
    curves = {curve: arrival_curve(curve, assumptions) for curve in live}
    days = list(day_of_week_shares(assumptions).items())
    from_week = central_phones_from_week(assumptions)

    central = (
        central_phone_queue(assumptions, sites, work)
        .set_index(["week_index", "day_of_week", "interval"])
        .sort_index()
    )
    weekly, _codes = live_weekly_volume(work, assumptions)
    network_calls = weekly.groupby("week_index")[call_code].sum()

    # Weeks before the changeover run blended, so they come out of the
    # existing builder untouched rather than out of a special case here.
    before = work[work["week_index"] < from_week]
    after = work[work["week_index"] >= from_week]
    pieces = []
    if len(before):
        pieces.append(_onsite_intervals_blended(assumptions, sites, before)["intervals"])

    rows = []
    for record in weekly[weekly["week_index"] >= from_week].to_dict("records"):
        week = record["week_index"]
        share = (
            record[call_code] / network_calls[week] if network_calls[week] > 0 else 0.0
        )
        walk_volume, call_volume = record[walk_code], record[call_code]

        # Two relief shapes now, because there are two teams: the counter's
        # load is walk-ins and the central team's is calls. The windows and
        # the combined 25% cap are unchanged - they are roster policy.
        counter_load = walk_volume * curves["walk_ins"] * walk_ins["handle_time_seconds"]
        phone_load = call_volume * curves["calls"] * phones["handle_time_seconds"]
        counter_away = (
            away_share(assumptions, load=counter_load)
            if counter_load.sum() > 0
            else np.zeros(n_intervals)
        )
        phone_away = (
            away_share(assumptions, load=phone_load)
            if phone_load.sum() > 0
            else np.zeros(n_intervals)
        )

        for day_name, day_share in days:
            walk_arrivals = walk_volume * day_share * curves["walk_ins"]
            call_arrivals = call_volume * day_share * curves["calls"]
            counter_traffic = erlang.traffic(
                walk_arrivals, walk_ins["handle_time_seconds"], interval_seconds
            )
            counter_needed = np.array(
                [
                    erlang.required_staff(
                        offered,
                        walk_ins["target_seconds"],
                        walk_ins["handle_time_seconds"],
                        walk_ins["service_level_goal"],
                        cap,
                        max_staff,
                    )
                    for offered in counter_traffic
                ]
            )
            phone_traffic = erlang.traffic(
                call_arrivals, phones["handle_time_seconds"], interval_seconds
            )
            phone_needed = np.array(
                [
                    erlang.required_staff(
                        offered,
                        phones["target_seconds"],
                        phones["handle_time_seconds"],
                        phones["service_level_goal"],
                        cap,
                        max_staff,
                    )
                    for offered in phone_traffic
                ]
            )

            block = central.loc[(week, day_name)]
            central_staff = block["required_staff"].to_numpy()
            central_share_staff = central_staff * share
            counter_scheduled = scheduled_staff(counter_needed, counter_away)
            central_scheduled = central_share_staff / (1.0 - phone_away)

            rows.append(
                pd.DataFrame(
                    {
                        "week_index": week,
                        "week_start": record["week_start"],
                        "period": record["period"],
                        "site": record["site"],
                        "day_of_week": day_name,
                        "interval": np.arange(n_intervals),
                        "interval_start": labels,
                        "walk_in_arrivals": walk_arrivals,
                        "call_arrivals": call_arrivals,
                        "traffic_counter": counter_traffic,
                        "traffic_phones": phone_traffic,
                        "required_counter": counter_needed,
                        "required_phones": phone_needed,
                        "required_separate": counter_needed + phone_needed,
                        # The counter's own queue is what the site staffs,
                        # so it is the one the pooled columns describe.
                        "traffic_pooled": counter_traffic,
                        "pooled_handle_seconds": walk_ins["handle_time_seconds"],
                        "required_staff": counter_needed + central_share_staff,
                        "occupancy": np.where(
                            counter_needed > 0,
                            counter_traffic / np.maximum(counter_needed, 1),
                            0.0,
                        ),
                        "away_share": counter_away,
                        "scheduled_staff": counter_scheduled + central_scheduled,
                        "scheduled_hours": (counter_scheduled + central_scheduled)
                        * interval_hours,
                        "live_handle_hours": (
                            walk_arrivals * walk_ins["handle_time_seconds"]
                            + call_arrivals * block["handle_seconds_central"].to_numpy()
                        )
                        / 3600.0,
                        # The central queue, repeated on each site's rows so
                        # the service-level recompute can reach it.
                        "central_required_staff": central_staff,
                        "central_traffic": block["traffic_uplifted"].to_numpy(),
                        "central_handle_seconds": block[
                            "handle_seconds_central"
                        ].to_numpy(),
                        "central_call_share": share,
                        "central_share_staff": central_share_staff,
                        "central_scheduled_staff": central_scheduled,
                        "counter_scheduled_staff": counter_scheduled,
                    }
                )
            )

    pieces.append(pd.concat(rows, ignore_index=True))
    # NOT sorted. Both builders emit rows in the same nested order - each
    # week-site, then each weekday in the day-share order, then each
    # interval - and the blended weeks here come straight out of the
    # blended builder. Sorting by day_of_week as a STRING would reorder
    # them to friday-monday-thursday, which lines up against nothing and
    # is the kind of mismatch that only shows up when somebody compares
    # two runs positionally.
    table = pd.concat(pieces, ignore_index=True)
    return _summarise_site_days(table, assumptions, work)



def write_onsite(result, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    paths = assumptions["paths"]
    out_dir = config.resolve_path(paths["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / paths["onsite_intervals_csv"]
    result["intervals"].to_csv(path, index=False, float_format="%.6f")

    # The day summary is where the email absorption lives, and it is the
    # grain Session 4 will take hours from.
    days_path = out_dir / paths["onsite_days_csv"]
    result["hours"].to_csv(days_path, index=False, float_format="%.6f")
    return path, days_path


def plot_onsite_day(result, assumptions=None, site="N1", week_index=None, day="monday"):
    """One site, one day: what Erlang C asks for and what has to be rostered.

    The gap between the two lines is the break cover, and it is widest where
    the away share underneath is deepest. Reading the two panels together is
    the point - a staffing number that ignores breaks is not a roster.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assumptions = assumptions or config.load_assumptions()
    if week_index is None:
        week_index = assumptions["calendar"]["history_weeks"]

    table = result["intervals"]
    chosen = table[
        (table.site == site)
        & (table.week_index == week_index)
        & (table.day_of_week == day)
    ].sort_values("interval")
    if chosen.empty:
        raise ValueError(f"no rows for {site} week {week_index} {day}")

    labels = list(chosen["interval_start"])
    positions = np.arange(len(labels))
    required = chosen["required_staff"].to_numpy()
    scheduled = chosen["scheduled_staff"].to_numpy()
    away = chosen["away_share"].to_numpy()

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 7), dpi=150, sharex=True, height_ratios=[2, 1]
    )
    fig.patch.set_facecolor(generate_data.SURFACE)
    for axis in (top, bottom):
        axis.set_facecolor(generate_data.SURFACE)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=generate_data.GRIDLINE, linewidth=0.8)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(generate_data.BASELINE)
        axis.tick_params(colors=generate_data.INK_MUTED, labelsize=9, length=0)

    # The pre-blend answer as a dashed benchmark rather than a third series:
    # it is what the model used to say, not something the plan now staffs.
    separate = chosen["required_separate"].to_numpy()
    top.step(
        positions, separate, where="mid", color=generate_data.INK_MUTED,
        linewidth=1.5, linestyle="--",
        label="Before: counter and phones staffed apart", zorder=2,
    )
    top.step(
        positions, scheduled, where="mid", color=generate_data.HOLIDAY,
        linewidth=2, label="Scheduled, blended (covers breaks)", zorder=3,
    )
    top.step(
        positions, required, where="mid", color=generate_data.SERIES,
        linewidth=2, label="Required, blended team (Erlang C)", zorder=4,
    )
    top.fill_between(
        positions, required, scheduled, step="mid",
        color=generate_data.HOLIDAY, alpha=0.18, zorder=2,
    )
    top.set_ylim(0, max(scheduled.max(), separate.max()) * 1.35)
    top.set_ylabel("Onsite Associates", color=generate_data.INK_SECONDARY, fontsize=10)
    legend = top.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(generate_data.INK_SECONDARY)

    bottom.fill_between(
        positions, away * 100, step="mid", color=generate_data.INK_MUTED,
        alpha=0.30, zorder=2,
    )
    bottom.step(
        positions, away * 100, where="mid", color=generate_data.INK_SECONDARY,
        linewidth=2, zorder=3,
    )
    bottom.set_ylim(0, max(away.max() * 100 * 1.6, 5))
    bottom.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
    bottom.set_ylabel("Staff away", color=generate_data.INK_SECONDARY, fontsize=10)
    bottom.set_xticks(positions)
    bottom.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    bottom.set_xlim(-0.6, len(labels) - 0.4)

    monday = chosen["week_start"].iloc[0]
    top.set_title(
        f"SYNTHETIC: onsite live work at {site}, {day.title()} of the week of {monday}",
        color=generate_data.INK, fontsize=14, loc="left", pad=34,
    )
    occupancy = (
        chosen["live_handle_hours"].sum() / chosen["scheduled_hours"].sum()
        if chosen["scheduled_hours"].sum() > 0
        else 0.0
    )
    top.text(
        0.0, 1.022,
        f"One blended team: counter and phones pooled, solved at the phones' goal "
        f"and topped up for the counter. {chosen['scheduled_hours'].sum():.1f} scheduled "
        f"hours, {occupancy:.0%} of them spent on live contacts.",
        transform=top.transAxes, color=generate_data.INK_SECONDARY, fontsize=10,
    )

    fig.tight_layout()
    out_dir = config.resolve_path(assumptions["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / assumptions["paths"]["onsite_day_chart_png"]
    fig.savefig(out_path, facecolor=generate_data.SURFACE)
    plt.close(fig)
    return out_path


def main():
    assumptions = config.load_assumptions()
    labels = interval_labels(assumptions)
    away = away_share(assumptions)
    shares = relief_shares(assumptions)

    print("Day of week (spec 3.4):")
    for day, share in day_of_week_shares(assumptions).items():
        print(f"  {day:<10} {share:6.1%}")

    print(f"\nInterval curves, {len(labels)} half-hours (spec 3.4, approved 10/22):")
    curves = {name: arrival_curve(name, assumptions) for name in curve_names(assumptions)}
    header = "  " + "interval".ljust(10) + "".join(f"{name:>12}" for name in curves)
    print(header + f"{'staff away':>12}")
    for i, label in enumerate(labels):
        row = "  " + label.ljust(10)
        row += "".join(f"{curve[i]:>11.1%}" for curve in curves.values())
        print(row + f"{away[i]:>12.1%}")
    row = "  " + "TOTAL".ljust(10)
    print(row + "".join(f"{curve.sum():>11.1%}" for curve in curves.values()))

    placement = assumptions["breaks_and_lunch"]["relief_placement"]
    windows = relief_windows(assumptions)
    print(
        f"\nRelief windows (spec 4.2), placement: {placement['method']}, "
        f"cap {placement['max_share_per_interval']:.0%}"
    )
    for relief in assumptions["breaks_and_lunch"]["reliefs"]:
        code = relief["code"]
        inside = shares[code][windows[code]]
        window = relief["window"]
        paid = "paid" if relief["paid"] else "unpaid"
        print(
            f"  {relief['name']:<22} {window['from']}-{window['to']}  "
            f"{relief['minutes']:>2} min {paid:<6} -> avg {inside.mean():.1%} "
            f"(range {inside.min():.1%} to {inside.max():.1%}) over {len(inside)} intervals"
        )
    print(
        f"  busiest single interval sends {away.max():.1%} away, against a "
        f"{placement['max_share_per_interval']:.0%} cap"
    )

    png = plot_intraday(assumptions)
    print(f"\nWrote the day-shape chart to {png}")

    # --- the onsite live-work build (spec 3.4 -> 4.1 -> 4.2) ---------------
    sites = config.load_sites()
    print("\nBuilding onsite live work from the work volume...")
    result = onsite_intervals(assumptions, sites)
    csv_path, days_path = write_onsite(result, assumptions)
    first_plan_week = assumptions["calendar"]["history_weeks"]
    chart = plot_onsite_day(result, assumptions, site="N1", week_index=first_plan_week)

    table, hours = result["intervals"], result["hours"]
    print(f"  {len(table):,} interval rows -> {csv_path}")
    print(f"  day summary -> {days_path}")
    print(f"  chart -> {chart}")

    week = hours[hours.week_index == first_plan_week]
    rows = table[table.week_index == first_plan_week]
    monday = week["week_start"].iloc[0]

    # FTE on the spec 5.1 basis: 40 paid hours less 21.75% shrinkage. The
    # breaks are NOT in that shrinkage - 4.2 already built them into the
    # interval schedule - so counting them here would pay twice.
    paid = assumptions["fte"]["paid_hours_per_week"]
    shrinkage = assumptions["fte"]["paid_shrinkage"]["onsite_associate"]
    productive = paid * (1 - shrinkage)

    print(f"\nFirst plan week ({monday}): Onsite Associate hours")
    print(
        f"  FTE basis (spec 5.1): {paid:.0f} paid hours x (1 - {shrinkage:.4f}) "
        f"= {productive:.2f} productive hours per FTE"
    )
    print(
        f"  {'site':<6}{'live h':>9}{'email h':>9}{'total h':>9}{'FTE':>7}"
        f"{'occ':>7}{'was occ':>9}{'was h':>9}"
    )
    # Occupancy is the Erlang measure on both sides - offered traffic over
    # the staff AT THE QUEUE - so before and after compare like with like.
    # The "was" hours are rebuilt exactly, by grossing the separate-queue
    # staff up through the same break cover, not by scaling the new total.
    interval_hours = assumptions["live_queues"]["interval_seconds"] / 3600.0
    rows = rows.assign(
        was_hours=rows["required_separate"] / (1 - rows["away_share"]) * interval_hours
    )

    def line(name, hour_rows, interval_rows):
        live = hour_rows["live_work_hours"].sum()
        email = hour_rows["email_paid_hours"].sum()
        traffic = interval_rows["traffic_pooled"].sum()
        occupancy = traffic / interval_rows["required_staff"].sum()
        was_occupancy = traffic / interval_rows["required_separate"].sum()
        print(
            f"  {name:<6}{live:9.1f}{email:9.1f}{live + email:9.1f}"
            f"{(live + email) / productive:7.2f}{occupancy:7.1%}"
            f"{was_occupancy:9.1%}{interval_rows['was_hours'].sum():9.1f}"
        )

    for site in sorted(week["site"].unique()):
        line(site, week[week.site == site], rows[rows.site == site])
    line("ALL", week, rows)

    print(
        f"\nBlending (spec 4.1): {rows['required_separate'].sum():,.0f} staff-intervals "
        f"apart -> {rows['required_staff'].sum():,.0f} pooled "
        f"({1 - rows['required_staff'].sum() / rows['required_separate'].sum():.0%} fewer)"
    )

    print("\nEmail and portal work (spec 4.2):")
    for column, label in (
        ("live_handle_hours", "live contact handling"),
        ("idle_hours", "idle time inside the live schedule"),
        ("email_hours", "email work to place"),
        ("email_absorbed_hours", "  absorbed into idle time"),
        ("email_spillover_hours", "  spilling over"),
        ("email_paid_hours", "  paid hours added (with 6.25% breaks)"),
    ):
        print(f"  {label:<42}{week[column].sum():8.1f} h")

    peak = rows.loc[rows["scheduled_staff"].idxmax()]
    lunch = [
        labels[i]
        for i, inside in enumerate(relief_windows(assumptions)["lunch"])
        if inside
    ]
    peaks = rows.loc[rows.groupby(["site", "day_of_week"])["scheduled_staff"].idxmax()]
    share = peaks["interval_start"].isin(lunch).mean()
    print(
        f"\nPeak scheduled staff: {peak['scheduled_staff']:.2f} at {peak['site']} on "
        f"{peak['day_of_week']} {peak['interval_start']} "
        f"(required {peak['required_staff']:.0f}, {peak['away_share']:.1%} away)"
    )
    print(f"  daily peak falls inside the lunch window in {share:.0%} of site-days")


if __name__ == "__main__":
    main()
