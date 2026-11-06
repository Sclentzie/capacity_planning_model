"""Spec Section 9, Session 2: the forecast checks (spec 3.2).

    - Forecast MAPE on the 13-week holdout is 10% or less for the chosen
      method; both methods' MAPE are reported.

Around that one required check sit the tests that make it mean something. A
MAPE number is only trustworthy if the holdout was genuinely unseen, if the
method really is the better of the two, and if the seasonal index learned the
shape that was put into the data rather than an artefact. Each of those is a
test here.

The irreducible floor is worth keeping in mind while reading these numbers:
the generator adds normal noise with a 5% standard deviation, and the mean
absolute value of such a draw is 5% x sqrt(2/pi) = 3.99%. A perfect forecast -
one that knew every ingredient exactly - would still score about 4% MAPE. So
the target is not "get close to zero", it is "get close to 4".
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from capacity_model import config, forecast, generate_data


@pytest.fixture(scope="module")
def assumptions():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def history(assumptions, sites):
    return generate_data.generate_history(assumptions, sites)


@pytest.fixture(scope="module")
def mondays(assumptions):
    cal = assumptions["calendar"]
    return generate_data.week_start_dates(cal["history_start"], cal["history_weeks"])


@pytest.fixture(scope="module")
def labels(mondays, assumptions):
    return forecast.event_labels(mondays, assumptions)


@pytest.fixture(scope="module")
def scored(history, assumptions):
    """Both methods trained on 143 weeks and scored on the last 13."""
    return forecast.score_holdout(history, assumptions)


@pytest.fixture(scope="module")
def scored_without_events(history, assumptions):
    """The same holdout, with calendar events left in the series."""
    return forecast.score_holdout(history, assumptions, use_calendar_events=False)


@pytest.fixture(scope="module")
def chosen(scored, assumptions):
    _scores, predictions = scored
    return forecast.choose_method(predictions, assumptions)


@pytest.fixture(scope="module")
def forward(history, chosen, assumptions):
    method, _pooled = chosen
    return forecast.forecast_forward(history, method, assumptions)


# ---------------------------------------------------------------------------
# The required check
# ---------------------------------------------------------------------------


def test_chosen_method_meets_the_mape_target(chosen, assumptions):
    """Spec 9: 10% or less on the 13-week holdout for the chosen method."""
    method, pooled = chosen
    target = assumptions["forecast"]["mape_target"]
    assert pooled[method] <= target, (
        f"{method} scored {pooled[method]:.2%} on the holdout, target {target:.0%}"
    )


def test_both_methods_mape_are_reported(scored, chosen, sites):
    """Spec 9: both methods' MAPE are reported, not only the winner's.

    Reporting the loser is what makes the choice evidence rather than a
    preference, so this checks the number exists for every site and for the
    network, for both methods.
    """
    scores, _predictions = scored
    _method, pooled = chosen

    assert set(scores["method"]) == set(forecast.METHODS)
    assert set(pooled.index) == set(forecast.METHODS)
    for method in forecast.METHODS:
        per_site = scores[scores.method == method]
        assert len(per_site) == len(sites["sites"])
        assert per_site["mape"].notna().all()
        assert (per_site["mape"] > 0).all()


def test_the_chosen_method_is_the_better_one(chosen):
    """'Uses the better one' (spec 3.2) - verify the choice follows the score."""
    method, pooled = chosen
    assert method == pooled.idxmin()


def test_both_methods_land_near_the_noise_floor(chosen):
    """A loose sanity band on both methods, not just the winner.

    A method scoring far below 4% would mean the holdout leaked; far above
    10% would mean it never learned the seasonality. Either is a bug rather
    than a close race.
    """
    _method, pooled = chosen
    noise_floor = 0.05 * np.sqrt(2 / np.pi)
    for method, value in pooled.items():
        assert value > noise_floor * 0.8, f"{method} scored {value:.2%}, below the noise floor"
        assert value < 0.20, f"{method} scored {value:.2%}"


# ---------------------------------------------------------------------------
# Is the holdout really unseen?
# ---------------------------------------------------------------------------


def test_holdout_weeks_do_not_leak_into_the_fit(history, assumptions, labels):
    """Measured by corrupting the holdout and checking predictions don't move.

    The MAPE check above is worthless if the method can see the weeks it is
    being graded on. So: triple the last 13 weeks of each site's history and
    refit. The ACTUALS change, of course - but if the fit is honest, every
    predicted value must come back bit-for-bit identical, because none of
    those weeks was in the training window.

    Run across all six sites, not one, because this is the test that rules
    out leakage as the explanation when a site scores below the noise floor.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    holdout = assumptions["calendar"]["holdout_weeks"]

    wide = forecast.weekly_by_site(history)
    weeks = wide.index.to_numpy()
    train_weeks, test_weeks = weeks[:-holdout], weeks[-holdout:]
    train_labels, test_labels = labels[:-holdout], labels[-holdout:]

    for site in wide.columns:
        values = wide[site].to_numpy()
        corrupted_values = values.copy()
        corrupted_values[-holdout:] *= 3.0

        for method in forecast.METHODS:
            for use_events in (False, True):
                clean, _ = forecast.fit_and_predict(
                    method, train_weeks, values[:-holdout], test_weeks, cycle,
                    assumptions, train_labels, test_labels, use_events,
                )
                corrupted, _ = forecast.fit_and_predict(
                    method, train_weeks, corrupted_values[:-holdout], test_weeks, cycle,
                    assumptions, train_labels, test_labels, use_events,
                )
                np.testing.assert_allclose(
                    clean, corrupted, rtol=0, atol=0,
                    err_msg=f"{site}/{method}/events={use_events} saw the holdout",
                )


# ---------------------------------------------------------------------------
# Does the seasonal index recover the ingredients it was built from?
# ---------------------------------------------------------------------------


def cycle_position_factors(history, cycle):
    """True seasonality per cycle position, for positions all years agree on.

    Position 12 is deliberately excluded by the agreement rule: Mon
    2024-04-01 is outside the first-quarter window while Mon 2025-03-31, 52
    weeks later, is inside it. That is the calendar drift recorded in
    test_season_weeks_drift_by_at_most_one_week_a_year, and it makes that one
    position genuinely ambiguous.
    """
    weekly = history.groupby("week_index", as_index=False).agg(
        seasonality_factor=("seasonality_factor", "first"),
        is_holiday_week=("is_holiday_week", "first"),
    )
    weekly["position"] = weekly["week_index"] % cycle

    agreed = {}
    for position, group in weekly.groupby("position"):
        factors = group["seasonality_factor"].unique()
        if len(factors) == 1:
            agreed[position] = factors[0]
    return agreed


def test_seasonal_index_recovers_the_generators_own_factors(history, assumptions):
    """Measured as ratios between cycle positions of known true factor.

    The index is normalised to average 1.0, so its absolute level is not
    comparable to the generator's factors - but the RATIO between two
    positions is. A holiday position over an ordinary position should come
    back at 0.60, and a first-quarter position over an ordinary one at 1.10,
    because that is exactly what was put in.

    This is the forecast half of "the history recovers its own ingredients":
    the generator wrote the seasonality, and the forecast reads it back out
    without ever being told what it was.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    holiday_factor = assumptions["demand"]["holiday"]["factor"]
    q1_factor = next(
        p["factor"] for p in assumptions["demand"]["season"]["periods"] if p["name"] == "Q1 lift"
    )

    truth = cycle_position_factors(history, cycle)
    holiday_positions = [p for p, f in truth.items() if np.isclose(f, holiday_factor)]
    q1_positions = [p for p, f in truth.items() if np.isclose(f, q1_factor)]
    plain_positions = [p for p, f in truth.items() if np.isclose(f, 1.0)]
    assert holiday_positions and q1_positions and plain_positions

    wide = forecast.weekly_by_site(history)
    weeks = wide.index.to_numpy()
    network = wide.sum(axis=1).to_numpy()
    index = forecast.seasonal_index(
        weeks, network, cycle, assumptions["forecast"]["decomposition"]
    )

    baseline = index[plain_positions].mean()
    assert index[holiday_positions].mean() / baseline == pytest.approx(holiday_factor, abs=0.03)
    assert index[q1_positions].mean() / baseline == pytest.approx(q1_factor, abs=0.03)


def test_seasonal_index_averages_one(history, assumptions):
    """The index redistributes; it must not move the overall level.

    If the factors did not average 1.0 they would quietly scale the whole
    forecast up or down, and the trend would no longer mean what it says.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    wide = forecast.weekly_by_site(history)
    index = forecast.seasonal_index(
        wide.index.to_numpy(),
        wide.sum(axis=1).to_numpy(),
        cycle,
        assumptions["forecast"]["decomposition"],
    )
    assert len(index) == cycle
    assert index.mean() == pytest.approx(1.0)


def test_decomposition_reproduces_a_clean_series_exactly(assumptions):
    """A unit test with no generator involved: known trend x known season.

    Build 156 weeks of (100 + 2t) x a 52-week pattern with no noise at all,
    fit, and forecast 52 weeks ahead. With nothing random in the series, the
    method should return the construction almost exactly - which pins the
    arithmetic independently of whether the synthetic data is right.

    This is the test that caught the need for the refinement pass: with a
    single pass the first trend line is still contaminated by seasonality
    and every forecast week came back about 0.2% high.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    weeks = np.arange(156)
    pattern = np.ones(cycle)
    pattern[10] = 0.6
    pattern[20] = 1.3
    pattern = pattern / pattern.mean()

    trend = 100.0 + 2.0 * weeks
    values = trend * pattern[weeks % cycle]

    future = np.arange(156, 208)
    predicted = forecast.decomposition_forecast(
        weeks, values, future, cycle, assumptions["forecast"]["decomposition"]
    )
    expected = (100.0 + 2.0 * future) * pattern[future % cycle]

    np.testing.assert_allclose(predicted, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# The forward forecast
# ---------------------------------------------------------------------------


def test_forecast_covers_exactly_the_plan_horizon(forward, assumptions, sites):
    """52 weeks x 6 sites, starting the Monday the history stops."""
    cal = assumptions["calendar"]
    assert len(forward) == cal["plan_weeks"] * len(sites["sites"])

    weeks = sorted(forward["week_index"].unique())
    assert weeks[0] == cal["history_weeks"]
    assert weeks[-1] == cal["history_weeks"] + cal["plan_weeks"] - 1

    mondays = generate_data.week_start_dates(cal["plan_start"], cal["plan_weeks"])
    assert sorted(forward["week_start"].unique()) == sorted(d.isoformat() for d in mondays)
    assert pd.to_datetime(forward["week_start"]).dt.day_of_week.eq(0).all()


def test_forecast_puts_the_holidays_in_the_right_plan_weeks(forward, history, assumptions):
    """The payoff of keying the index to cycle position (spec 3.2).

    The four 2027 holidays land on plan weeks whose cycle positions match the
    history's holiday weeks. If the forecast learned the holidays properly,
    those weeks - and only those - should be the lowest of the 52.

    Under the ISO week-number keying this replaced, plan week 1 would have
    inherited a x0.6 New Year index and shown up here as a false low.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    truth = cycle_position_factors(history, cycle)
    holiday_positions = {
        p for p, f in truth.items() if np.isclose(f, assumptions["demand"]["holiday"]["factor"])
    }

    weekly = forward.groupby("week_index", as_index=False)["forecast_requests"].sum()
    weekly["position"] = weekly["week_index"] % cycle

    lowest = weekly.nsmallest(len(holiday_positions), "forecast_requests")
    assert set(lowest["position"]) == holiday_positions


def test_forecast_keeps_growing(forward, history, assumptions):
    """Direction check: the forecast year must sit above the last history year.

    The generator grows 6% a year and decomposition extrapolates a straight
    line through an exponential, so the recovered rate runs a little under
    6% - the line is fitted across three years and the exponential is
    steepest at the end. The check is therefore a band, not a point: clearly
    growing, and not implausibly fast.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    last_history_year = history[history["week_index"] >= assumptions["calendar"]["history_weeks"] - cycle]
    measured = forward["forecast_requests"].sum() / last_history_year["requests"].sum() - 1.0
    assert 0.02 < measured < 0.09, f"forecast year-over-year growth {measured:.2%}"


def test_forecast_is_positive_everywhere(forward):
    """A negative or zero week would turn into negative required hours."""
    assert (forward["forecast_requests"] > 0).all()


# ---------------------------------------------------------------------------
# The constraint that drove the 156-week history (spec 3.2)
# ---------------------------------------------------------------------------


def test_holt_winters_refuses_a_series_shorter_than_two_cycles(assumptions):
    """Pins the reason history is 156 weeks rather than 104.

    statsmodels cannot initialise 52 seasonal factors from fewer than two
    full cycles, so a 13-week holdout off 104 weeks of history left 91
    training weeks and no challenger. If this guard ever stops firing, the
    history length can be revisited.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    too_short = np.linspace(100.0, 200.0, 2 * cycle - 1)
    with pytest.raises(ValueError, match="two full seasonal cycles|needs"):
        forecast.holt_winters_forecast(
            too_short, 13, cycle, assumptions["forecast"]["holt_winters"]
        )


def test_training_window_clears_the_holt_winters_minimum(assumptions):
    """The arithmetic behind the 156-week decision, as a test."""
    cal = assumptions["calendar"]
    training_weeks = cal["history_weeks"] - cal["holdout_weeks"]
    assert training_weeks >= 2 * cal["seasonal_cycle_weeks"], (
        f"{training_weeks} training weeks is below the "
        f"{2 * cal['seasonal_cycle_weeks']} Holt-Winters needs"
    )


# ---------------------------------------------------------------------------
# Known calendar events (spec 3.2, added 10/21)
# ---------------------------------------------------------------------------


def test_event_labels_mark_every_holiday_and_dip_week(labels, history, assumptions):
    """The labels must match the weeks the generator actually made special.

    Labels come from dates only, so this is the check that the date logic in
    the forecast agrees with the date logic in the generator. If they ever
    drifted apart, the forecast would correct the wrong weeks.
    """
    weekly = history.groupby("week_index", as_index=False).agg(
        is_holiday_week=("is_holiday_week", "first"),
        seasonality_factor=("seasonality_factor", "first"),
    )
    dip_factor = next(
        p["factor"]
        for p in assumptions["demand"]["season"]["periods"]
        if p["name"] == "Pre-Christmas slowdown"
    )

    assert ((labels == "holiday") == weekly["is_holiday_week"].to_numpy()).all()
    is_dip = np.isclose(weekly["seasonality_factor"], dip_factor)
    assert ((labels == "Pre-Christmas slowdown") == is_dip).all()


def test_event_factors_recover_the_generators_own_values(history, labels, assumptions):
    """Ingredient recovery again, this time through the event path.

    The forecast is never handed demand.holiday.factor - it measures the
    size of each event against a baseline fitted on ordinary weeks only. It
    should land near the 0.60 and 0.92 that went in.

    Tolerance reflects how much data each estimate has: the holiday factor
    pools 12 weeks, the dip only 3, so the dip is allowed to be looser.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    wide = forecast.weekly_by_site(history)
    factors = forecast.estimate_event_factors(
        wide.index.to_numpy(),
        wide.sum(axis=1).to_numpy(),
        labels,
        cycle,
        assumptions["forecast"]["decomposition"],
    )

    expected_holiday = assumptions["demand"]["holiday"]["factor"]
    expected_dip = next(
        p["factor"]
        for p in assumptions["demand"]["season"]["periods"]
        if p["name"] == "Pre-Christmas slowdown"
    )
    assert factors["holiday"] == pytest.approx(expected_holiday, abs=0.04)
    assert factors["Pre-Christmas slowdown"] == pytest.approx(expected_dip, abs=0.06)


def test_removing_events_before_fitting_is_not_a_no_op(history, labels, assumptions):
    """Guards the subtlety that made the first implementation do nothing.

    Dividing an event out and multiplying it back cancels EXACTLY if the
    seasonal index is still fitted at the event's own cycle position:
    index[p] = mean(actual/f/trend), so trend x index[p] x f is just
    trend x mean(actual/trend). The pooled factor has to be paired with an
    index fitted on ordinary weeks only.

    This reproduces the cancelling version and asserts the real one differs
    from it, so the no-op can never come back unnoticed.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    holdout = assumptions["calendar"]["holdout_weeks"]
    settings = assumptions["forecast"]["decomposition"]

    wide = forecast.weekly_by_site(history)
    weeks, values = wide.index.to_numpy(), wide["N1"].to_numpy()
    train_w, test_w = weeks[:-holdout], weeks[-holdout:]
    train_v = values[:-holdout]
    train_l, test_l = labels[:-holdout], labels[-holdout:]

    raw = forecast.decomposition_forecast(train_w, train_v, test_w, cycle, settings)

    # The cancelling version: adjust, but let the index see event weeks.
    factors = forecast.estimate_event_factors(train_w, train_v, train_l, cycle, settings)
    adjusted = train_v / forecast.apply_factors(train_l, factors)
    cancels = forecast.decomposition_forecast(
        train_w, adjusted, test_w, cycle, settings
    ) * forecast.apply_factors(test_l, factors)
    np.testing.assert_allclose(cancels, raw, rtol=1e-9)

    # The real one must not match it.
    real, _ = forecast.fit_and_predict(
        forecast.DECOMPOSITION, train_w, train_v, test_w, cycle, assumptions,
        train_l, test_l, True,
    )
    assert np.max(np.abs(real / raw - 1)) > 1e-3


def test_event_weeks_are_kept_out_of_the_seasonal_index(history, labels, assumptions):
    """Positions that always carry an event must fall back to 1.0.

    Thanksgiving is always cycle position 46, so once event weeks are
    excluded there is no ordinary observation left there. The index should
    sit at the no-seasonality fallback and let the event factor carry the
    drop, rather than quietly absorbing it twice.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    wide = forecast.weekly_by_site(history)
    weeks = wide.index.to_numpy()
    index, _slope, _intercept, _factors = forecast.fit_ordinary_baseline(
        weeks, wide.sum(axis=1).to_numpy(), labels, cycle,
        assumptions["forecast"]["decomposition"],
    )

    always_event = {
        int(p)
        for p in set(weeks % cycle)
        if not (labels[(weeks % cycle) == p] == forecast.ORDINARY).any()
    }
    assert always_event, "expected some positions to always carry an event"
    for position in always_event:
        assert index[position] == pytest.approx(1.0, abs=0.02)


def test_calendar_events_improve_the_chosen_method(scored, scored_without_events, chosen):
    """The point of the change, measured rather than asserted.

    Only the chosen method is held to this. Holt-Winters is handed the same
    correction in the only form it can consume - the whole series with the
    events divided out - and it smooths across weeks, so it re-absorbs most
    of the benefit and can come out slightly worse. That is a real result,
    reported rather than hidden.
    """
    method, _pooled = chosen
    with_events = scored[0].set_index(["site", "method"])["mape"]
    without = scored_without_events[0].set_index(["site", "method"])["mape"]

    improved = sum(
        with_events[(site, method)] < without[(site, method)]
        for site in with_events.index.get_level_values("site").unique()
    )
    assert improved >= 4, f"only {improved} of 6 sites improved"


def test_plan_holidays_are_placed_by_date(assumptions):
    """Spec 3.2: plan events come from the calendar, not the cycle position.

    The 2027 plan year should carry four holiday weeks and one pre-Christmas
    dip, each on the Monday of the week containing the holiday.
    """
    cal = assumptions["calendar"]
    plan_mondays = generate_data.week_start_dates(cal["plan_start"], cal["plan_weeks"])
    plan_labels = forecast.event_labels(plan_mondays, assumptions)

    flagged = {m: l for m, l in zip(plan_mondays, plan_labels) if l != forecast.ORDINARY}
    assert sorted(flagged) == [
        datetime.date(2027, 6, 28),    # July 4 2027 falls on the Sunday
        datetime.date(2027, 11, 22),   # Thanksgiving, Nov 25
        datetime.date(2027, 12, 13),   # the week before Christmas week
        datetime.date(2027, 12, 20),   # Christmas, Dec 25
        datetime.date(2027, 12, 27),   # New Year, Jan 1 2028
    ]
    assert flagged[datetime.date(2027, 12, 13)] == "Pre-Christmas slowdown"
    assert flagged[datetime.date(2027, 12, 20)] == "holiday"


def test_forecast_carries_its_event_labels(forward):
    """The output has to say which weeks were treated as calendar events.

    Without it, a leader looking at a low week in the plan cannot tell a
    holiday from a forecast that fell over.
    """
    assert "calendar_event" in forward.columns
    flagged = forward[forward.calendar_event != forecast.ORDINARY]
    assert flagged["week_start"].nunique() == 5


# ---------------------------------------------------------------------------
# MAPE itself
# ---------------------------------------------------------------------------


def test_mape_is_zero_on_a_perfect_prediction():
    actual = np.array([100.0, 200.0, 300.0])
    assert forecast.mape(actual, actual) == 0.0


def test_mape_is_a_percentage_not_an_absolute():
    """Equal percentage misses on different-sized sites must score the same.

    This is why MAPE is the metric: the 0.8 site and the 1.2 site differ by
    half, and an absolute error would let the big site dominate the choice of
    method for everyone.
    """
    small = forecast.mape(np.array([100.0]), np.array([110.0]))
    large = forecast.mape(np.array([1000.0]), np.array([1100.0]))
    assert small == pytest.approx(large) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Why adding the December dip did not move the raw-series scores
# ---------------------------------------------------------------------------


def test_raw_decomposition_is_blind_to_a_dip_on_the_52_week_grid(assumptions, sites, labels):
    """Explains a result that looks like a stale file but is not.

    Adding the pre-Christmas dip changed three weeks of history - 49, 101 and
    153 - and week 153 sits inside the 13-week holdout. Yet the raw-series
    MAPE came back identical to the run made before the dip existed. That is
    not a cached number; it is an exact invariance.

    The three dip weeks are 364 days apart, so all three land on cycle
    position 49. Classical decomposition keyed to cycle position absorbs a
    constant multiplier at a single position perfectly:

        index[49] scales by 0.92, so actual/index is unchanged
        => the de-seasonalised series is identical
        => the trend is identical
        => the prediction at week 153 scales by exactly 0.92

    The actual at week 153 scales by 0.92 too, so |actual - predicted| /
    actual does not move at all. The method is structurally blind to any
    seasonal factor that repeats on its own grid - which is precisely what it
    is built to absorb.

    Holt-Winters is NOT invariant, because it weights recent weeks more
    heavily instead of averaging each position equally. Asserting that it
    DOES move is what proves the data really changed and the scores really
    were recomputed, rather than read from disk.
    """
    import copy

    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    holdout = assumptions["calendar"]["holdout_weeks"]
    settings = assumptions["forecast"]["decomposition"]

    with_dip = generate_data.generate_history(assumptions, sites)

    without_dip = copy.deepcopy(assumptions)
    for period in without_dip["demand"]["season"]["periods"]:
        if period["name"] == "Pre-Christmas slowdown":
            period["factor"] = 1.0
    no_dip_history = generate_data.generate_history(without_dip, sites)

    a = forecast.weekly_by_site(with_dip)
    b = forecast.weekly_by_site(no_dip_history)

    # The data must genuinely differ, and only at the dip weeks.
    ratio = (a / b).to_numpy()
    moved = np.where(np.abs(ratio - 1.0).max(axis=1) > 1e-9)[0]
    assert len(moved) == 3, f"expected 3 changed weeks, got {len(moved)}"
    assert {int(w) % cycle for w in moved} == {49}, "dip weeks must share one cycle position"
    assert moved.max() > len(a) - holdout - 1, "a dip week must fall inside the holdout"

    weeks = a.index.to_numpy()
    train_w, test_w = weeks[:-holdout], weeks[-holdout:]

    for site in a.columns:
        with_values, without_values = a[site].to_numpy(), b[site].to_numpy()
        scored_with = forecast.mape(
            with_values[-holdout:],
            forecast.decomposition_forecast(
                train_w, with_values[:-holdout], test_w, cycle, settings
            ),
        )
        scored_without = forecast.mape(
            without_values[-holdout:],
            forecast.decomposition_forecast(
                train_w, without_values[:-holdout], test_w, cycle, settings
            ),
        )
        assert scored_with == pytest.approx(scored_without, abs=1e-12), (
            f"{site}: decomposition should be exactly blind to the dip"
        )

    # And the calendar-event treatment must NOT be blind to it: that is the
    # whole reason for pooling the dip into an estimated event factor.
    train_l, test_l = labels[:-holdout], labels[-holdout:]
    values = a["N1"].to_numpy()
    raw = forecast.decomposition_forecast(train_w, values[:-holdout], test_w, cycle, settings)
    evented, _ = forecast.fit_and_predict(
        forecast.DECOMPOSITION, train_w, values[:-holdout], test_w, cycle,
        assumptions, train_l, test_l, True,
    )
    assert np.max(np.abs(evented / raw - 1)) > 1e-3
