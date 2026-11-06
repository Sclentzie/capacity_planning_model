"""Session 3, task 2a: the intraday draft and the break rules (spec 3.4, 4.2).

The curves were approved by Siri on 10/22. These tests still do not pin their
individual values: the numbers are tuning, and freezing them would turn every
future adjustment into a test failure rather than a decision.

What they pin instead is everything that must hold whatever the curves are
set to: the shares add up, the shape matches the peaks spec 3.4 describes in
words, the away share is genuinely derived from the break rules, and the
onsite build reproduces erlang.py exactly.
"""

import copy

import numpy as np
import pytest

from capacity_model import config, erlang, intervals, work_volume


@pytest.fixture(scope="module")
def assumptions():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def labels(assumptions):
    return intervals.interval_labels(assumptions)


@pytest.fixture(scope="module")
def away(assumptions):
    return intervals.away_share(assumptions)


def index_of(labels, clock):
    return labels.index(clock)


# ---------------------------------------------------------------------------
# The day's shape
# ---------------------------------------------------------------------------


def test_day_of_week_shares_sum_to_one(assumptions):
    """Spec 3.4: Mon 23, Tue 21, Wed 20, Thu 19, Fri 17.

    A week's work has to land somewhere inside the week. If these did not
    sum to 1.0 the model would quietly gain or lose volume at the point it
    splits a weekly forecast into days.
    """
    shares = intervals.day_of_week_shares(assumptions)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
    assert list(shares) == ["monday", "tuesday", "wednesday", "thursday", "friday"]
    assert shares["monday"] == 0.23
    assert shares["friday"] == 0.17


def test_the_week_tapers_from_monday_to_friday(assumptions):
    """Requests build up over the weekend (spec 3.4), so Monday is heaviest.

    Pinning the ORDER rather than the numbers: the five values can be tuned
    without breaking this, but a Friday heavier than a Monday would mean the
    reasoning behind the split had been lost.
    """
    shares = list(intervals.day_of_week_shares(assumptions).values())
    assert shares == sorted(shares, reverse=True)
    assert shares[0] > shares[-1]


def test_site_hours_give_eighteen_half_hours(assumptions, labels):
    """Spec 3.4: Mon-Fri 8:00 to 17:00, 18 half-hour intervals."""
    assert len(labels) == 18
    assert labels[0] == "08:00"
    assert labels[-1] == "16:30"
    assert labels[1] == "08:30"


def test_interval_count_must_agree_with_opening_hours(assumptions):
    """A count that contradicts the hours is refused, not silently used.

    This is the kind of edit that would otherwise stretch or truncate every
    curve in the model without anything visibly breaking.
    """
    broken = copy.deepcopy(assumptions)
    broken["intraday"]["site_hours"]["intervals_per_day"] = 17
    with pytest.raises(ValueError, match="not 17 intervals"):
        intervals.interval_starts(broken)


# ---------------------------------------------------------------------------
# The DRAFT curves
# ---------------------------------------------------------------------------


def test_every_curve_sums_to_one(assumptions):
    """Each curve is a share of its own day, so it must total 100%."""
    names = intervals.curve_names(assumptions)
    assert set(names) == {"calls", "walk_ins"}
    for name in names:
        curve = intervals.arrival_curve(name, assumptions)
        assert len(curve) == 18
        assert curve.sum() == pytest.approx(1.0, abs=1e-9), f"{name} sums to {curve.sum()}"
        assert (curve > 0).all(), f"{name} has an empty interval; the site is open all day"


def test_calls_peak_between_eight_and_ten(assumptions, labels):
    """Spec 3.4: calls peak 8:00 to 10:00.

    Measured as a block rather than a single point, so the draft can be
    reshaped without this becoming brittle: the first two hours must carry
    more than any other two-hour block in the day.
    """
    calls = intervals.arrival_curve("calls", assumptions)
    blocks = {labels[i]: calls[i : i + 4].sum() for i in range(0, len(calls) - 3)}
    assert max(blocks, key=blocks.get) == "08:00"
    assert blocks["08:00"] == pytest.approx(0.38)
    assert calls.argmax() <= index_of(labels, "10:00")


def test_walk_ins_peak_late_morning_and_early_afternoon(assumptions, labels):
    """Spec 3.4: walk-ins peak late morning AND early afternoon.

    Two humps with a dip between them, which is the shape the words
    describe. Somebody travelling to a counter arrives later than somebody
    picking up a phone, and the midday trough is the judgement call in this
    draft - an operation whose requesters come on their own lunch break
    would show a bump there instead. Worth Siri's eye on the chart.
    """
    walk_ins = intervals.arrival_curve("walk_ins", assumptions)
    late_morning = slice(index_of(labels, "10:30"), index_of(labels, "12:00"))
    midday = slice(index_of(labels, "12:00"), index_of(labels, "13:00"))
    early_afternoon = slice(index_of(labels, "13:30"), index_of(labels, "15:00"))

    assert walk_ins[late_morning].max() > walk_ins[midday].max()
    assert walk_ins[early_afternoon].max() > walk_ins[midday].max()
    assert walk_ins.argmax() in range(late_morning.start, late_morning.stop)


def test_the_two_curves_really_are_different_shapes(assumptions):
    """Calls and walk-ins must not be the same curve under two names.

    If they were, the model would be claiming a counter behaves exactly like
    a phone line, and the whole point of splitting W19 from W20 would be
    lost. Their peaks must fall in different halves of the day.
    """
    calls = intervals.arrival_curve("calls", assumptions)
    walk_ins = intervals.arrival_curve("walk_ins", assumptions)
    assert not np.allclose(calls, walk_ins)
    assert calls.argmax() < 9 <= walk_ins.argmax() or calls.argmax() < walk_ins.argmax()


def test_a_curve_that_does_not_sum_to_one_is_refused(assumptions):
    broken = copy.deepcopy(assumptions)
    broken["intraday"]["interval_curves"]["calls"]["08:00"] = 0.5
    with pytest.raises(ValueError, match="sums to"):
        intervals.arrival_curve("calls", broken)


def test_a_curve_missing_an_interval_is_refused(assumptions):
    """Better a loud failure than an afternoon silently shifted by a slot."""
    broken = copy.deepcopy(assumptions)
    del broken["intraday"]["interval_curves"]["walk_ins"]["13:00"]
    with pytest.raises(ValueError, match="missing intervals"):
        intervals.arrival_curve("walk_ins", broken)


def test_a_curve_outside_opening_hours_is_refused(assumptions):
    broken = copy.deepcopy(assumptions)
    broken["intraday"]["interval_curves"]["calls"]["17:30"] = 0.0
    with pytest.raises(ValueError, match="outside site hours"):
        intervals.arrival_curve("calls", broken)


# ---------------------------------------------------------------------------
# Breaks and lunch (spec 4.2)
# ---------------------------------------------------------------------------


def test_each_break_window_averages_twelve_and_a_half_percent(assumptions):
    """Spec 4.2: a 15-minute break in a 2-hour window leaves 12.5% away.

    15 minutes of each person's time out of 120 minutes of window. Since
    10/22 the relief is placed where the floor is quiet rather than spread
    evenly, so this is the window AVERAGE - the same people taking the same
    breaks, just not at the same moment.
    """
    for code in ("morning_break", "afternoon_break"):
        assert intervals.relief_window_average(code, assumptions) == pytest.approx(
            0.125, abs=1e-9
        ), code


def test_away_share_is_one_sixth_at_lunch(assumptions, labels, away):
    """Spec 4.2: a 30-minute lunch over a 3-hour window leaves 16.7% away.

    30 minutes out of 180. This is the deepest hole in the day and it lands
    squarely on the walk-in trough and the start of the afternoon rise -
    the midday coverage problem the teaching note in 4.2 is about.
    """
    average = intervals.relief_window_average("lunch", assumptions)
    assert average == pytest.approx(1 / 6, abs=1e-9)
    assert round(average, 3) == 0.167


def test_nobody_is_away_outside_the_relief_windows(assumptions, labels, away):
    """Open before 9:00 and after 16:00 with the whole team on the floor."""
    for clock in ("08:00", "08:30", "16:00", "16:30"):
        assert away[index_of(labels, clock)] == 0.0

    assert np.count_nonzero(away) == 14
    assert np.count_nonzero(away == 0.0) == 4


def test_away_share_is_derived_from_the_rules_not_typed_in(assumptions):
    """Change a rule and the curve must move with it.

    A hard-coded 12.5% would pass every test above. These two edits prove
    the shares are computed: halving the lunch window doubles the share
    away in it, and lengthening a break lengthens nothing but its own level.
    """
    shorter_window = copy.deepcopy(assumptions)
    for relief in shorter_window["breaks_and_lunch"]["reliefs"]:
        if relief["code"] == "morning_break":
            relief["window"]["to"] = "10:30"  # 2 hours -> 1.5 hours
    assert intervals.relief_window_average("morning_break", shorter_window) == pytest.approx(
        15 / 90, abs=1e-9
    )

    longer_lunch = copy.deepcopy(assumptions)
    for relief in longer_lunch["breaks_and_lunch"]["reliefs"]:
        if relief["code"] == "lunch":
            relief["minutes"] = 40  # 30 -> 40 in the same 3-hour window
    assert intervals.relief_window_average("lunch", longer_lunch) == pytest.approx(
        40 / 180, abs=1e-9
    )


def test_relief_windows_must_be_whole_intervals(assumptions):
    """A window that does not line up with the grid is refused."""
    broken = copy.deepcopy(assumptions)
    broken["breaks_and_lunch"]["reliefs"][0]["window"]["to"] = "10:45"
    with pytest.raises(ValueError, match="whole number"):
        intervals.relief_shares(broken)


def test_relief_that_does_not_fit_under_the_cap_is_refused(assumptions):
    """The cap is the first guard: a relief too big to place is refused.

    A 180-minute lunch inside a 180-minute window needs the whole team away
    the whole time, which no 25% cap can absorb. The message names the
    numbers rather than leaving a silent over-cap result.
    """
    broken = copy.deepcopy(assumptions)
    for relief in broken["breaks_and_lunch"]["reliefs"]:
        if relief["code"] == "lunch":
            relief["minutes"] = 180
    with pytest.raises(ValueError, match="cannot place"):
        intervals.away_share(broken)


def test_the_whole_team_cannot_be_away_at_once(assumptions):
    """Spec 4.2's formula divides by (1 - share away), so 100% is refused.

    The second guard, reachable once the cap is lifted high enough to let
    an impossible total through. Caught here with a clear message rather
    than downstream as a division by zero or infinite staff.
    """
    broken = copy.deepcopy(assumptions)
    broken["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"] = 1.0
    for relief in broken["breaks_and_lunch"]["reliefs"]:
        if relief["code"] == "lunch":
            relief["minutes"] = 180
    with pytest.raises(ValueError, match="cannot be on relief at once"):
        intervals.away_share(broken)


def test_breaks_are_built_into_intervals_only_for_the_onsite_team(assumptions):
    """Spec 4.2: every break is counted exactly once.

    Onsite Associates get theirs here, by interval. Back-office teams get
    theirs through shrinkage in 5.2. Recording which pool is which is what
    stops the same break being paid for twice in Session 4.
    """
    pools = assumptions["breaks_and_lunch"]["built_into_intervals_for"]
    assert pools == ["onsite_associate"]

    roles = {r["code"] for r in config.load_skills()["roles"]}
    assert set(pools) <= roles, "the named pool must be a real role in skills.yaml"


def test_paid_and_unpaid_reliefs_are_labelled(assumptions):
    """The paid flag drives where the cost lands, so it must be explicit.

    Paid breaks cost paid hours; the unpaid lunch costs none but stretches
    the shift and opens the coverage hole. Session 4 needs to tell them
    apart to avoid double-counting against shrinkage.
    """
    reliefs = {r["code"]: r for r in assumptions["breaks_and_lunch"]["reliefs"]}
    assert reliefs["morning_break"]["paid"] is True
    assert reliefs["afternoon_break"]["paid"] is True
    assert reliefs["lunch"]["paid"] is False


# ---------------------------------------------------------------------------
# The scheduled-staff formula (spec 4.2) - defined but not yet wired up
# ---------------------------------------------------------------------------


def test_scheduled_staff_matches_the_spec_formula(away):
    """Spec 4.2: scheduled = needed / (1 - away).

    Spec 9 states the case by hand: 4 needed at the counter while 20% of
    staff are away gives 5 scheduled.
    """
    assert intervals.scheduled_staff(4, 0.20) == pytest.approx(5.0)

    needed = np.full(18, 10.0)
    scheduled = intervals.scheduled_staff(needed, away)
    assert (scheduled >= needed).all(), "cover can never be below the staff needed"
    assert scheduled[0] == pytest.approx(10.0), "no uplift when nobody is away"
    np.testing.assert_allclose(scheduled, needed / (1 - away))


def test_scheduled_staff_refuses_an_impossible_away_share():
    with pytest.raises(ValueError):
        intervals.scheduled_staff(4, 1.0)


# ---------------------------------------------------------------------------
# The review chart
# ---------------------------------------------------------------------------


def test_the_day_shape_chart_is_written(assumptions):
    """Spec 3.4 wants the curves signed off from a chart, so one must exist.

    Approved 10/22, so the filename no longer carries DRAFT - a stale draft
    label on an approved input is its own kind of wrong.
    """
    name = assumptions["paths"]["intraday_chart_png"]
    assert "DRAFT" not in name, "the curves are approved; drop the draft label"

    path = intervals.plot_intraday(assumptions)
    assert path.exists()
    assert path.stat().st_size > 10_000


# ---------------------------------------------------------------------------
# The 10/22 revision: lunch-hour bumps and load-aware relief placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["calls", "walk_ins"])
def test_both_curves_bump_at_lunch(assumptions, labels, name):
    """Revised 10/22: a lunch-hour bump replaces the old midday trough.

    The first draft had both channels at their quietest over lunch, which
    assumed requesters behave like office workers. They are patients and
    carers, and the hour they are free to chase a records request is their
    own lunch break.

    Measured against the shoulders either side rather than as an absolute
    value, so the draft can still be reshaped: every interval in 12:00-13:00
    must sit above both 11:30 and 13:30.
    """
    curve = intervals.arrival_curve(name, assumptions)
    before = curve[index_of(labels, "11:30")]
    after = curve[index_of(labels, "13:30")]
    for clock in ("12:00", "12:30"):
        value = curve[index_of(labels, clock)]
        assert value > before, f"{name} at {clock} ({value:.3f}) not above 11:30 ({before:.3f})"
        assert value > after, f"{name} at {clock} ({value:.3f}) not above 13:30 ({after:.3f})"


def test_the_lunch_bump_stays_modest_for_walk_ins(assumptions, labels):
    """Spec 3.4 keeps BOTH walk-in peaks, so the bump must not become one.

    A bump that outgrew the late-morning and early-afternoon peaks would
    have quietly rewritten the shape the spec describes in words.
    """
    curve = intervals.arrival_curve("walk_ins", assumptions)
    bump = max(curve[index_of(labels, c)] for c in ("12:00", "12:30"))
    assert bump < curve[index_of(labels, "11:00")], "bump outgrew the late-morning peak"
    assert bump < curve[index_of(labels, "14:00")], "bump outgrew the afternoon peak"


def test_calls_still_peak_in_the_first_two_hours(assumptions, labels):
    """The bump must not have cost calls their 8:00-10:00 peak (spec 3.4)."""
    calls = intervals.arrival_curve("calls", assumptions)
    bump = max(calls[index_of(labels, c)] for c in ("12:00", "12:30"))
    assert calls.argmax() <= index_of(labels, "10:00")
    assert bump < calls[index_of(labels, "08:00")]


def test_no_interval_exceeds_the_cap(assumptions, away):
    """Revised 10/22: no single half-hour may send more than the cap away.

    Concentrating relief into the quietest slot would look optimal to the
    arithmetic and be unworkable on a rota, so the placement is bounded.
    """
    cap = assumptions["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"]
    assert cap == 0.25
    assert away.max() <= cap + 1e-12, f"worst interval sends {away.max():.1%} away"


def test_relief_goes_where_the_floor_is_quietest(assumptions):
    """The point of the 10/22 change: relief follows the load, inversely.

    Within each window the quieter half-hours must carry more relief than
    the busier ones. Checked as a strict ordering rather than a tolerance:
    sort the window's intervals by load and the relief shares must come out
    in exactly the opposite order.
    """
    load = intervals.combined_load(assumptions)
    shares = intervals.relief_shares(assumptions)
    windows = intervals.relief_windows(assumptions)

    for code, inside in windows.items():
        window_load = load[inside]
        window_relief = shares[code][inside]
        order = np.argsort(window_load)
        ranked = window_relief[order]
        assert list(ranked) == sorted(ranked, reverse=True), (
            f"{code}: relief does not fall as load rises ({window_relief})"
        )
        assert window_relief.max() > window_relief.min(), f"{code}: relief is still flat"


def test_placement_moves_relief_without_changing_the_total(assumptions):
    """The same people take the same breaks; only the timing moves.

    Compared head to head against the old even placement: every window
    average is identical, and the distribution inside it is not.
    """
    evenly = copy.deepcopy(assumptions)
    evenly["breaks_and_lunch"]["relief_placement"]["method"] = "even"

    even_shares = intervals.relief_shares(evenly)
    load_shares = intervals.relief_shares(assumptions)
    windows = intervals.relief_windows(assumptions)

    for code, inside in windows.items():
        assert even_shares[code][inside].mean() == pytest.approx(
            load_shares[code][inside].mean(), abs=1e-12
        ), f"{code}: placement changed the window total"
        assert not np.allclose(even_shares[code], load_shares[code]), (
            f"{code}: inverse-load placement made no difference"
        )

    # Even placement is the flat 12.5 / 16.7 the first draft produced.
    assert np.allclose(even_shares["morning_break"][windows["morning_break"]], 0.125)
    assert np.allclose(even_shares["lunch"][windows["lunch"]], 1 / 6)


def test_the_cap_binds_and_still_preserves_the_window_average(assumptions):
    """A case built so the cap has to bite, checking it redistributes.

    With a tight cap the quietest intervals fill to the limit and the
    excess is pushed back into the next-quietest, rather than being
    dropped. The window average must survive that intact - otherwise the
    cap would be quietly deleting somebody's lunch.
    """
    tight = copy.deepcopy(assumptions)
    tight["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"] = 0.18

    shares = intervals.relief_shares(tight)
    windows = intervals.relief_windows(tight)

    lunch = shares["lunch"][windows["lunch"]]
    assert lunch.max() == pytest.approx(0.18, abs=1e-12), "the cap should bind at lunch"
    assert (lunch <= 0.18 + 1e-12).all()
    assert lunch.mean() == pytest.approx(1 / 6, abs=1e-9), "the cap lost some relief"

    for code, inside in windows.items():
        expected = intervals.relief_window_average(code, assumptions)
        assert shares[code][inside].mean() == pytest.approx(expected, abs=1e-9)


def test_an_unknown_placement_method_is_refused(assumptions):
    broken = copy.deepcopy(assumptions)
    broken["breaks_and_lunch"]["relief_placement"]["method"] = "whenever"
    with pytest.raises(ValueError, match="unknown relief placement method"):
        intervals.relief_shares(broken)


def test_load_is_volume_times_handle_time_not_contact_count(assumptions):
    """Spec 3.4: busy means minutes of work, not number of contacts.

    A call costs 5 minutes and arrives 1.0 per request; a walk-in costs 8
    minutes but only 0.15 per request. So a call carries 5 minutes of load
    per request against a walk-in's 1.2 - calls are about four times the
    onsite live load, and ranking the day by contacts alone would point
    relief at the wrong half-hours.

    Checked by rebuilding the weighting from the work types rather than
    trusting a constant.
    """
    live = intervals.live_work_types(assumptions)
    assert set(live) == {"calls", "walk_ins"}
    assert live["calls"]["code"] == "W20"
    assert live["walk_ins"]["code"] == "W19"

    expected = sum(
        work_type["rate"]
        * work_type["handle_time_minutes"]
        * intervals.arrival_curve(curve, assumptions)
        for curve, work_type in live.items()
    )
    np.testing.assert_allclose(intervals.combined_load(assumptions), expected)

    minutes = {c: wt["rate"] * wt["handle_time_minutes"] for c, wt in live.items()}
    assert minutes["calls"] == pytest.approx(5.0)
    assert minutes["walk_ins"] == pytest.approx(1.2)
    assert minutes["calls"] > 3 * minutes["walk_ins"], "calls should dominate the load"


def test_a_live_queue_work_type_must_declare_a_handle_time(assumptions):
    """Erlang C cannot run without one, so a missing value fails loudly."""
    broken = copy.deepcopy(assumptions)
    for work_type in broken["work_types"]:
        if work_type["code"] == "W20":
            del work_type["handle_time_minutes"]
    with pytest.raises(ValueError, match="handle_time_minutes"):
        intervals.live_work_types(broken)


# ---------------------------------------------------------------------------
# Task 2b: the onsite live-work build (spec 3.4 -> 4.1 -> 4.2)
# ---------------------------------------------------------------------------

SAMPLE_WEEKS = [0, 100, 156]   # two history weeks and the first plan week


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def work(assumptions, sites):
    return work_volume.build(assumptions, sites)["work"]


@pytest.fixture(scope="module")
def onsite(assumptions, sites, work):
    """A few weeks only - the rules are the same in all 208."""
    sample = work[work["week_index"].isin(SAMPLE_WEEKS)]
    return intervals.onsite_intervals(assumptions, sites, sample)


def test_scheduled_is_never_below_required(onsite):
    """Spec 4.2: scheduled = required / (1 - away), and away is in [0, 1).

    Cover can only ever add people. A scheduled number below the required
    one would mean the model had quietly decided somebody could be on their
    lunch break and at the counter at the same time.
    """
    table = onsite["intervals"]
    assert (table["scheduled_staff"] >= table["required_staff"] - 1e-12).all()

    # Equal exactly where nobody is away, strictly greater where somebody is.
    no_relief = table[table["away_share"] == 0]
    with_relief = table[(table["away_share"] > 0) & (table["required_staff"] > 0)]
    np.testing.assert_allclose(
        no_relief["scheduled_staff"], no_relief["required_staff"], rtol=1e-12
    )
    assert (with_relief["scheduled_staff"] > with_relief["required_staff"]).all()


def test_interval_arrivals_add_back_to_the_weekly_volume(onsite, work, assumptions):
    """Splitting a week into days and half-hours must not change the total.

    Day shares sum to 1.0 and each curve sums to 1.0, so the 90 intervals of
    a week have to add back to the volume that went in. If they did not, the
    model would be inventing or losing contacts at the point it goes
    intraday, and every staffing number after that would be wrong.
    """
    table = onsite["intervals"]
    codes = {c: wt["code"] for c, wt in intervals.live_work_types(assumptions).items()}

    weekly = (
        work[work["week_index"].isin(SAMPLE_WEEKS)]
        .groupby(["week_index", "site", "work_type"])["work_items"]
        .sum()
    )
    split = table.groupby(["week_index", "site"])[
        ["walk_in_arrivals", "call_arrivals"]
    ].sum()

    for (week, site), row in split.iterrows():
        assert row["walk_in_arrivals"] == pytest.approx(
            weekly[(week, site, codes["walk_ins"])], rel=1e-9
        ), f"walk-ins at {site} week {week}"
        assert row["call_arrivals"] == pytest.approx(
            weekly[(week, site, codes["calls"])], rel=1e-9
        ), f"calls at {site} week {week}"


def test_the_day_of_week_split_is_applied(onsite, assumptions):
    """Each weekday must carry its own share of the week, not a flat fifth.

    Monday takes 23% and Friday 17% (spec 3.4), so a model that split the
    week evenly would understaff the start and overstaff the end - the exact
    error the day shares exist to prevent.
    """
    table = onsite["intervals"]
    shares = intervals.day_of_week_shares(assumptions)

    by_day = table.groupby(["week_index", "site", "day_of_week"])["call_arrivals"].sum()
    totals = table.groupby(["week_index", "site"])["call_arrivals"].sum()

    for (week, site, day), value in by_day.items():
        assert value / totals[(week, site)] == pytest.approx(shares[day], rel=1e-9)


def test_relief_still_averages_the_window_rule_under_real_load(onsite, assumptions):
    """Switching to real load must move relief about, not change how much.

    Placement now ranks intervals by volume x handle time rather than by the
    curve-share proxy, which reorders the day. The window averages are fixed
    by the rule itself (15/120 and 30/180), so they have to survive the
    change untouched for every site and week.
    """
    table = onsite["intervals"]
    windows = intervals.relief_windows(assumptions)
    expected = {"morning_break": 0.125, "lunch": 1 / 6, "afternoon_break": 0.125}

    one_day = table[table["day_of_week"] == "monday"]
    for (week, site), rows in one_day.groupby(["week_index", "site"]):
        away = rows.sort_values("interval")["away_share"].to_numpy()
        for code, mask in windows.items():
            assert away[mask].mean() == pytest.approx(expected[code], abs=1e-9), (
                f"{code} at {site} week {week}"
            )


def test_no_interval_exceeds_the_cap_under_real_load(onsite, assumptions):
    cap = assumptions["breaks_and_lunch"]["relief_placement"]["max_share_per_interval"]
    assert onsite["intervals"]["away_share"].max() <= cap + 1e-12


def test_real_load_reorders_the_day_against_the_curve_proxy(onsite, assumptions):
    """The switch to real load has to actually change something.

    If the two agreed everywhere, switching to real load would have been a
    no-op and this suite would be proving nothing. They differ because the
    two channels drift apart week to week: walk-ins land in the week the
    request arrives, calls are split 50/50 over that week and the next.
    """
    proxy = intervals.away_share(assumptions)
    table = onsite["intervals"]
    monday = table[table.day_of_week == "monday"]

    differences = []
    for _key, rows in monday.groupby(["week_index", "site"]):
        away = rows.sort_values("interval")["away_share"].to_numpy()
        differences.append(np.abs(away - proxy).max())
    assert max(differences) > 1e-6, "real load made no difference to placement"




def test_live_work_hours_are_scheduled_staff_times_half_an_hour(onsite, assumptions):
    """Spec 4.2: hours are the sum of scheduled staff x 0.5 hour across the day."""
    interval_hours = assumptions["live_queues"]["interval_seconds"] / 3600.0
    assert interval_hours == 0.5

    table, hours = onsite["intervals"], onsite["hours"]
    np.testing.assert_allclose(
        table["scheduled_hours"], table["scheduled_staff"] * interval_hours, rtol=1e-12
    )

    rebuilt = table.groupby(["week_index", "site", "day_of_week"])["scheduled_hours"].sum()
    stated = hours.set_index(["week_index", "site", "day_of_week"])["live_work_hours"]
    np.testing.assert_allclose(rebuilt.sort_index(), stated.sort_index(), rtol=1e-12)


def test_the_build_covers_every_site_day_and_interval(onsite, assumptions, sites):
    table = onsite["intervals"]
    expected = (
        len(SAMPLE_WEEKS)
        * len(sites["sites"])
        * len(assumptions["intraday"]["day_of_week_share"])
        * assumptions["intraday"]["site_hours"]["intervals_per_day"]
    )
    assert len(table) == expected
    assert set(table["period"]) == {"history", "plan"}
    assert table["required_staff"].min() >= 1, "a live queue always needs somebody there"


def test_erlang_traffic_accepts_a_whole_day_of_intervals():
    """traffic() is elementwise, so a day converts in one call.

    Added for this build. Everything after it still works one interval at a
    time, because Erlang C has no closed form to vectorise.
    """
    arrivals = np.array([0.0, 1.5, 12.0])
    offered = erlang.traffic(arrivals, 300, 1800)
    np.testing.assert_allclose(offered, arrivals * 300 / 1800)
    assert isinstance(erlang.traffic(10, 300, 1800), float)
    with pytest.raises(ValueError):
        erlang.traffic(np.array([1.0, -1.0]), 300, 1800)


# ---------------------------------------------------------------------------
# Task 2c: the blended team, support work in idle time, and the FTE basis
# ---------------------------------------------------------------------------


def test_pooling_never_needs_more_than_two_separate_queues(onsite):
    """Spec 9: pooled staff is never above the two queues added together.

    Pooling can only help. One team of N absorbs a burst on either channel,
    where two teams of N/2 each have to carry their own cushion against
    randomness - the same reason one big contact centre beats six small
    ones. If this ever reversed, the blended answer would be costing money
    rather than saving it.
    """
    table = onsite["intervals"]
    assert (table["required_staff"] <= table["required_separate"]).all()
    assert (table["required_staff"] >= 1).all(), "a live queue always needs somebody"

    # And it must actually be saving something, or the 10/22 decision was a
    # no-op dressed up as a change.
    saved = table["required_separate"].sum() - table["required_staff"].sum()
    assert saved > 0
    assert saved / table["required_separate"].sum() > 0.10


def test_the_pooled_queue_meets_the_walk_in_goal_in_every_interval(onsite, assumptions):
    """Spec 9: the counter goal holds everywhere, not just on average.

    The pooled queue is SOLVED at the phones' 20-second goal, so the
    walk-in goal is the one that could quietly slip. Checked on every row
    rather than a sample, because a blended team that fails the counter to
    flatter the phones is exactly the failure this decision risks.
    """
    goals = {g["code"]: g for g in assumptions["live_queues"]["goals"]}
    walk_ins, phones = goals["walk_ins"], goals["phones"]
    table = onsite["intervals"]

    for row in table.itertuples():
        offered, handle = row.traffic_pooled, row.pooled_handle_seconds
        staff = int(row.required_staff)
        assert erlang.service_level(
            staff, offered, walk_ins["target_seconds"], handle
        ) >= walk_ins["service_level"] - erlang.TOLERANCE, f"counter goal at {row.site}"
        assert erlang.service_level(
            staff, offered, phones["target_seconds"], handle
        ) >= phones["service_level"] - erlang.TOLERANCE, f"phones goal at {row.site}"
        assert erlang.occupancy(staff, offered) <= (
            assumptions["live_queues"]["occupancy_cap"] + erlang.TOLERANCE
        )


def test_the_pooled_handle_time_is_volume_weighted(onsite):
    """Spec 4.1: a volume-weighted handle time, not an average of the two.

    Calls outnumber walk-ins roughly four to one, so a plain mean of 5 and 8
    minutes would overstate the blended handle time badly and inflate every
    staffing number that follows.
    """
    table = onsite["intervals"].sample(500, random_state=0)
    contacts = table["walk_in_arrivals"] + table["call_arrivals"]
    expected = (
        table["walk_in_arrivals"] * 8 * 60 + table["call_arrivals"] * 5 * 60
    ) / contacts
    np.testing.assert_allclose(table["pooled_handle_seconds"], expected, rtol=1e-12)

    # Between the two handle times, and nearer the call end because calls
    # dominate the volume.
    assert (table["pooled_handle_seconds"] > 5 * 60).all()
    assert (table["pooled_handle_seconds"] < 8 * 60).all()
    assert table["pooled_handle_seconds"].mean() < 6.5 * 60


def test_the_walk_in_top_up_is_live_code():
    """The counter goal never binds on this data - prove it still can.

    At these handle times (5 to 8 minutes) a 10-minute target is so much
    looser than a 20-second one that the phones goal always decides, so the
    top-up loop never fires in any of the 112,320 real intervals. That makes
    it exactly the kind of guard that rots unnoticed.

    This drives it with an hour-long handle time, where the 10-minute target
    stops being generous, and checks it does the right thing.
    """
    settings = {
        "walk_ins": {
            "handle_time_seconds": 3600,
            "target_seconds": 600,
            "service_level_goal": 0.90,
        },
        "calls": {
            "handle_time_seconds": 3600,
            "target_seconds": 20,
            "service_level_goal": 0.80,
        },
    }
    # 1.0 contact per half hour at an hour each is 2 Erlangs offered.
    staff, offered, handle = intervals.pooled_required_staff(
        walk_in_arrivals=0.5, call_arrivals=0.5, settings=settings,
        interval_seconds=1800, cap=0.85, max_staff=500,
    )
    assert handle == pytest.approx(3600.0)
    assert offered == pytest.approx(2.0)

    phones_only = erlang.required_staff(offered, 20, handle, 0.80, 0.85)
    assert staff > phones_only, "the top-up should have raised staff here"
    assert erlang.service_level(phones_only, offered, 600, handle) < 0.90
    assert erlang.service_level(staff, offered, 600, handle) >= 0.90
    assert erlang.service_level(staff - 1, offered, 600, handle) < 0.90


def test_one_interval_matches_erlang_directly(onsite, assumptions):
    """The hand-checkable case, now for the blended team (spec 4.1).

    N1, Monday of the first plan week, 12:00. Taken through erlang.py by
    hand: 1.151 walk-ins and 4.743 calls give a volume-weighted handle time
    of 335.2 seconds and 1.097 Erlangs pooled. Three people clear the
    phones' 80%-in-20-seconds goal; two do not.

    The same interval needed 5 when the two queues were staffed apart - 2
    at the counter and 3 on the phones - which is the blending decision in
    one row.
    """
    table = onsite["intervals"]
    row = table[
        (table.site == "N1")
        & (table.week_index == 156)
        & (table.day_of_week == "monday")
        & (table.interval_start == "12:00")
    ]
    assert len(row) == 1
    row = row.iloc[0]

    queues = assumptions["live_queues"]
    cap = queues["occupancy_cap"]
    goals = {g["work_type"]: g for g in queues["goals"]}

    walk_ins, calls = row["walk_in_arrivals"], row["call_arrivals"]
    work_seconds = walk_ins * 8 * 60 + calls * 5 * 60
    handle = work_seconds / (walk_ins + calls)
    offered = erlang.traffic(walk_ins + calls, handle, 1800)

    assert handle == pytest.approx(335.159969, abs=1e-5)
    assert offered == pytest.approx(1.097458, abs=1e-6)
    assert offered == pytest.approx(row["traffic_pooled"], rel=1e-12)
    assert handle == pytest.approx(row["pooled_handle_seconds"], rel=1e-12)

    staff = erlang.required_staff(
        offered, goals["W20"]["target_seconds"], handle,
        goals["W20"]["service_level"], cap,
    )
    assert staff == 3
    assert staff == row["required_staff"]
    assert erlang.service_level(2, offered, 20, handle) < 0.80
    assert erlang.service_level(3, offered, 20, handle) >= 0.80
    assert erlang.service_level(3, offered, 600, handle) >= 0.90

    # The separate-queue answer, kept beside it for the comparison.
    counter = erlang.required_staff(row["traffic_counter"], 600, 8 * 60, 0.90, cap)
    phones = erlang.required_staff(row["traffic_phones"], 20, 5 * 60, 0.80, cap)
    assert (counter, phones) == (2, 3)
    assert counter + phones == row["required_separate"] == 5

    assert row["occupancy"] == pytest.approx(offered / staff, rel=1e-12)
    assert row["scheduled_staff"] == pytest.approx(staff / (1 - row["away_share"]), rel=1e-12)


def test_blending_lifts_occupancy_off_the_floor(onsite):
    """The reason for the decision, measured (spec 4.1).

    Two tiny separate queues each need somebody parked on them, so the team
    sits near 20% busy - mostly waiting. Pooling raises that without
    touching either service goal.
    """
    table = onsite["intervals"]
    traffic = table["traffic_pooled"].sum()
    before = traffic / table["required_separate"].sum()
    after = traffic / table["required_staff"].sum()

    assert before < 0.25, "the separate queues should be running near 20%"
    assert after > before
    assert after < 0.85, "still comfortably inside the cap"


# ---------------------------------------------------------------------------
# Email and portal work in idle time (spec 4.2, decided 10/22)
# ---------------------------------------------------------------------------


def test_email_spillover_is_bounded_and_accounted_for(onsite):
    """Spec 9: spillover never exceeds the email work, and the two halves add up.

    Absorbed plus spillover must equal the email hours exactly - the idle
    time can soak work up, but it cannot make any disappear.
    """
    days = onsite["hours"]
    assert (days["email_spillover_hours"] >= -1e-12).all()
    assert (days["email_spillover_hours"] <= days["email_hours"] + 1e-12).all()
    np.testing.assert_allclose(
        days["email_absorbed_hours"] + days["email_spillover_hours"],
        days["email_hours"],
        rtol=1e-9,
        atol=1e-12,
    )
    assert (days["email_absorbed_hours"] <= days["idle_hours"] + 1e-9).all()


def test_email_spills_over_only_when_idle_time_runs_out(onsite):
    """Zero spillover wherever idle time covers the work, and vice versa."""
    days = onsite["hours"]
    fits = days[days["email_hours"] <= days["idle_hours"]]
    assert (fits["email_spillover_hours"] <= 1e-12).all()
    np.testing.assert_allclose(
        fits["email_absorbed_hours"], fits["email_hours"], rtol=1e-9
    )

    overflows = days[days["email_hours"] > days["idle_hours"] + 1e-9]
    if len(overflows):
        np.testing.assert_allclose(
            overflows["email_absorbed_hours"], overflows["idle_hours"], rtol=1e-9
        )


def test_idle_time_is_the_schedule_minus_the_live_handling(onsite):
    """Idle hours are paid hours nobody is on a contact for.

    The live schedule pays for every interval it covers, and the team is not
    handling contacts for all of it - at these volumes, not for most of it.
    That is the capacity email work is placed into.
    """
    days = onsite["hours"]
    expected = (days["live_work_hours"] - days["live_handle_hours"]).clip(lower=0.0)
    np.testing.assert_allclose(days["idle_hours"], expected, rtol=1e-12)
    assert (days["live_handle_hours"] <= days["live_work_hours"] + 1e-9).all()


def test_only_spillover_hours_carry_the_break_allowance(onsite, assumptions):
    """Spec 4.2: hours absorbed into idle time must not be charged breaks twice.

    The live schedule's breaks are already paid for by the interval build.
    Adding the 6.25% allowance to absorbed hours would buy the same break a
    second time; it belongs only on hours the schedule did not already cover.
    """
    allowance = assumptions["onsite_team"]["support_work"]["spillover_break_allowance"]
    assert allowance == 0.0625

    days = onsite["hours"]
    np.testing.assert_allclose(
        days["email_paid_hours"],
        days["email_spillover_hours"] / (1 - allowance),
        rtol=1e-12,
    )
    absorbed_only = days[days["email_spillover_hours"] <= 1e-12]
    assert (absorbed_only["email_paid_hours"] <= 1e-12).all()


def test_total_hours_are_the_live_schedule_plus_paid_spillover(onsite):
    days = onsite["hours"]
    np.testing.assert_allclose(
        days["total_hours"], days["live_work_hours"] + days["email_paid_hours"], rtol=1e-12
    )
    assert (days["total_hours"] >= days["live_work_hours"] - 1e-12).all()


# ---------------------------------------------------------------------------
# The FTE basis (spec 5.1)
# ---------------------------------------------------------------------------


def test_the_fte_basis_comes_from_the_spec_not_a_round_number(assumptions):
    """Spec 5.1: 40 paid hours, Onsite Associate shrinkage 21.75%.

    Productive hours per FTE are 40 x (1 - 0.2175) = 31.30, not a generic
    37.5. The difference is not cosmetic: it moves a 1,000-hour week from
    26.7 FTE to 31.9, which is the kind of gap that changes a hiring
    decision.

    The 21.75% is itself the point - it is 28% less the 6.25% paid-break
    line, because 4.2 builds this team's breaks into the interval schedule.
    Carrying them in both places would pay for the same break twice.
    """
    fte = assumptions["fte"]
    assert fte["paid_hours_per_week"] == 40
    shrinkage = fte["paid_shrinkage"]["onsite_associate"]
    assert shrinkage == 0.2175

    productive = fte["paid_hours_per_week"] * (1 - shrinkage)
    assert productive == pytest.approx(31.30)

    # The other roles keep their breaks in shrinkage, so they sit higher.
    for role in ("records_specialist", "tier_1", "tier_2", "qa_specialist"):
        assert fte["paid_shrinkage"][role] == 0.28
    assert shrinkage == pytest.approx(0.28 - 0.0625, abs=1e-9)


def test_onsite_shrinkage_excludes_the_breaks_built_into_intervals(assumptions):
    """The two mechanisms must not overlap (spec 4.2 and 5.2).

    Onsite Associates are the only pool whose breaks are built into the
    interval schedule, and they are the only pool whose shrinkage is below
    28%. If that ever stopped lining up, some break would be paid for twice
    or not at all.
    """
    by_interval = assumptions["breaks_and_lunch"]["built_into_intervals_for"]
    assert by_interval == ["onsite_associate"]

    shrinkage = assumptions["fte"]["paid_shrinkage"]
    for role, value in shrinkage.items():
        if role in by_interval:
            assert value < 0.28, f"{role} builds breaks by interval, so must sit below 28%"
        else:
            assert value == 0.28, f"{role} carries breaks in shrinkage"
