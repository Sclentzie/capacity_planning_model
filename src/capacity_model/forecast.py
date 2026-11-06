"""Demand forecast: classical decomposition and Holt-Winters (spec 3.2).

Session 2.

The WFM idea: forecast the one demand driver - requests received per site -
and let every work type's volume fall out of it, so the whole plan moves from
a single forecast rather than from many disconnected guesses.

Two methods race, and that is the point. A single method with nothing to
compare it against has no evidence behind it; "we tested it against an
alternative on data it had never seen" is the answer to "why this forecast?".

    Baseline  - classical decomposition: a seasonal index by position in the
                52-week cycle, a linear trend on the de-seasonalised series,
                then the two recombined.
    Challenger- Holt-Winters (statsmodels), additive trend, multiplicative
                season.

Both train on the first 143 weeks and are scored by MAPE on the last 13 weeks
of history, which neither has seen. The better one is then refitted on all 156
weeks to produce the 52-week forward forecast.

Three decisions from spec 3.2, all made 10/21 and all living in
assumptions.yaml rather than here:

  SEASONAL INDEX KEY. "Week of year" means position in a 52-week cycle
  anchored at the history start, not the ISO week number - 2026 is a 53-week
  ISO year, and ISO keying would hand the first plan week a seasonal index
  built from a single x0.6 New Year week.

  HISTORY LENGTH. 156 weeks, because Holt-Winters needs two full seasonal
  cycles just to initialise, and a 13-week holdout off 104 weeks leaves only
  91 training weeks.

  METHOD SELECTION. One method for all six sites, on pooled holdout MAPE.

Run it:
    .venv\\Scripts\\python -m capacity_model.forecast
"""

import datetime
import warnings

import numpy as np
import pandas as pd

from . import config, generate_data

DECOMPOSITION = "decomposition"
HOLT_WINTERS = "holt_winters"
METHODS = (DECOMPOSITION, HOLT_WINTERS)


# ---------------------------------------------------------------------------
# Shaping the history
# ---------------------------------------------------------------------------


def weekly_by_site(history):
    """Requests per week per site: the series the forecast actually works on.

    Request types are summed away here. The forecast runs on total requests
    and the type mix is reapplied afterwards (spec 3: one forecast drives
    everything), so a rare request type's noise never destabilises the trend.
    """
    wide = history.pivot_table(
        index="week_index", columns="site", values="requests", aggfunc="sum"
    )
    return wide.sort_index()


# ---------------------------------------------------------------------------
# Known calendar events (spec 3.2, added 10/21)
#
# A holiday is not a pattern to be discovered. We know Thanksgiving 2027 is
# the week of November 22 the same way we know it today, so making the
# forecast rediscover it from two or three noisy observations wastes
# information and puts it on the wrong week whenever a holiday drifts off the
# 52-week grid.
#
# The split that keeps this honest: take the DATES from the calendar, and
# learn the SIZE of each effect from the training data. The forecast is never
# handed demand.holiday.factor, so the holdout still grades something the
# model had to work out for itself.
# ---------------------------------------------------------------------------

ORDINARY = ""


def event_labels(mondays, assumptions):
    """Label each week with the calendar event it carries, or "" for none.

    Holiday weeks take the holiday's own name; a season rule anchored to a
    holiday (the pre-Christmas dip) takes the period's name. Everything else
    is ordinary. Only the dates come from here - no factors.
    """
    demand = assumptions["demand"]
    _factors, holiday_names = generate_data.week_factors(mondays, demand)

    relative = generate_data.relative_season_mondays(
        demand["season"],
        demand["holiday"]["weeks"],
        range(mondays[0].year, mondays[-1].year + 2),
    )

    labels = []
    for monday, holiday in zip(mondays, holiday_names):
        if holiday is not None:
            # Every holiday shares one factor in spec 3.1, and pooling them
            # gives roughly 9 training observations instead of 2 or 3 each,
            # which is the difference between a usable estimate and noise.
            labels.append("holiday")
            continue
        label = ORDINARY
        for name, mondays_for_name in relative.items():
            if monday in mondays_for_name:
                label = name
                break
        labels.append(label)
    return np.array(labels, dtype=object)


def fit_ordinary_baseline(weeks, values, labels, cycle, settings):
    """Trend and seasonality from ORDINARY weeks, then each event's size.

    Excluding the event weeks from the seasonal index is the whole point,
    and it is subtle enough to be worth spelling out. If the index is still
    fitted at an event's own cycle position, dividing the event out and
    multiplying it back is EXACTLY a no-op:

        index[p] = mean(actual / factor / trend)
        prediction = trend x index[p] x factor = trend x mean(actual / trend)

    The factor cancels and the positional index just re-learns what was
    removed - measured at 1.7e-13 relative difference, so not approximately
    a no-op but precisely one. Pooling only buys anything when the event
    weeks are kept out of the index.

    With them out, cycle positions that ALWAYS carry an event (Thanksgiving
    is always position 46) have no observation left and keep a factor of
    1.0. That is the right answer rather than a gap: the baseline there is
    whatever the trend says, and the event factor supplies the rest.

    Returns (index, slope, intercept, factors).
    """
    ordinary = labels == ORDINARY
    index, slope, intercept = fit_decomposition(
        weeks[ordinary], values[ordinary], cycle, settings
    )
    baseline = (intercept + slope * weeks) * index[weeks % cycle]

    factors = {}
    for label in sorted(set(labels) - {ORDINARY}):
        seen = labels == label
        factors[label] = float(np.mean(values[seen] / baseline[seen]))
    return index, slope, intercept, factors


def estimate_event_factors(weeks, values, labels, cycle, settings):
    """Just the event sizes, for inspection and testing."""
    return fit_ordinary_baseline(weeks, values, labels, cycle, settings)[3]


def apply_factors(labels, factors):
    """The multiplier for each week; 1.0 where there is no event."""
    return np.array([factors.get(label, 1.0) for label in labels], dtype=float)


# ---------------------------------------------------------------------------
# Method 1: classical decomposition (the baseline)
# ---------------------------------------------------------------------------


def fit_decomposition(weeks, values, cycle, settings):
    """Fit a multiplicative seasonal index and a linear trend together.

    Classical decomposition, in the order spec 3.2 sets out:

      1. fit a straight line through the series - a trend to measure against,
         because you cannot see how high a January sits until you know where
         the underlying level was that week
      2. divide the actual by that line, leaving a ratio per week
      3. average the ratios at each cycle position (3 observations each here)
      4. normalise so the factors average 1.0, which keeps the index a pure
         redistribution and leaves the overall level to the trend
      5. de-seasonalise the series with that index and refit the trend

    Then repeat until the index stops moving. The REFINEMENT matters: that
    first straight line is fitted through a series that still has its
    seasonality in it, so it is pulled off by where the busy and quiet weeks
    happen to fall, and the index inherits the error. Each pass re-measures
    the index against a trend that is a little cleaner than the last.

    On a clean, noiseless series the error shrinks by roughly 7x a pass -
    0.19% after one, 0.03% after two - so it converges quickly but not
    instantly, which is why this iterates to a tolerance rather than running
    a fixed number of passes. The settled answer is exact: once the index
    equals the true seasonality, de-seasonalising returns the trend exactly
    and nothing moves again.

    A position with no observation keeps a factor of 1.0 - no seasonality
    known - rather than borrowing a neighbour's, which would invent a
    pattern that is not in the data.

    Returns (index, slope, intercept).
    """
    positions = weeks % cycle
    index = np.ones(cycle)
    slope, intercept = np.polyfit(weeks, values, 1)

    for _ in range(settings["max_refinement_passes"]):
        previous = index

        ratios = values / (intercept + slope * weeks)
        index = previous.copy()
        for k in range(cycle):
            at_k = ratios[positions == k]
            if len(at_k):
                index[k] = at_k.mean()
        index = index / index.mean()

        deseasonalised = values / index[positions]
        slope, intercept = np.polyfit(weeks, deseasonalised, 1)

        if np.max(np.abs(index - previous)) < settings["refinement_tolerance"]:
            break

    return index, slope, intercept


def seasonal_index(weeks, values, cycle, settings):
    """Just the seasonal half of the fit, for inspection and testing."""
    return fit_decomposition(weeks, values, cycle, settings)[0]


def decomposition_forecast(weeks, values, future_weeks, cycle, settings):
    """Fit decomposition on (weeks, values) and predict at future_weeks.

    Week numbers are ABSOLUTE throughout - week 0 is the first week of
    history and the plan carries on from 156 - so a cycle position always
    means the same point in the year whether it is being fitted or predicted.
    """
    index, slope, intercept = fit_decomposition(weeks, values, cycle, settings)
    # Recombine: where the trend says the level will be, times that week's
    # seasonal factor.
    return (intercept + slope * future_weeks) * index[future_weeks % cycle]


# ---------------------------------------------------------------------------
# Method 2: Holt-Winters (the challenger)
# ---------------------------------------------------------------------------


def holt_winters_forecast(values, horizon, cycle, settings):
    """Exponentially weighted level, trend, and season, via statsmodels.

    The contrast with decomposition is the point of running both: where
    decomposition gives every year equal weight, Holt-Winters discounts older
    weeks, so it adapts faster to a change in the business and is more easily
    pulled around by noise.

    Needs at least two full cycles of data to initialise its 52 seasonal
    factors, which is why the history is 156 weeks (spec 3.2).
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    minimum = 2 * cycle
    if len(values) < minimum:
        raise ValueError(
            f"Holt-Winters needs {minimum} weeks to initialise {cycle} seasonal "
            f"factors; got {len(values)}. See spec 3.2."
        )

    with warnings.catch_warnings():
        # Optimiser chatter on a 52-season fit is expected and not a result.
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            values,
            trend=settings["trend"],
            seasonal=settings["seasonal"],
            seasonal_periods=cycle,
            damped_trend=settings["damped_trend"],
            initialization_method=settings["initialization_method"],
        ).fit()
    return np.asarray(model.forecast(horizon))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def mape(actual, predicted):
    """Mean absolute percentage error.

    Percentage rather than absolute error because the six sites differ in
    size by half: a 40-request miss is a bad week at the 0.8 site and a
    rounding error at the 1.2 site, and MAPE scores both on the same scale.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / actual)))


def forecast_one(method, weeks, values, future_weeks, cycle, assumptions):
    """Run one named method over one site's series."""
    if method == DECOMPOSITION:
        return decomposition_forecast(
            weeks, values, future_weeks, cycle, assumptions["forecast"]["decomposition"]
        )
    if method == HOLT_WINTERS:
        return holt_winters_forecast(
            values, len(future_weeks), cycle, assumptions["forecast"]["holt_winters"]
        )
    raise ValueError(f"unknown forecast method: {method!r}")


def fit_and_predict(
    method, weeks, values, future_weeks, cycle, assumptions,
    labels, future_labels, use_events,
):
    """Run one method, with calendar events removed before the fit (spec 3.2).

    The order is the whole idea:
      1. fit trend and seasonality on the ordinary weeks, so holidays never
         enter the seasonal index
      2. measure how big each known event is against that baseline
      3. project the baseline forward
      4. multiply the event factors back onto the target weeks, by date

    The two methods need step 3 done differently. Decomposition can simply
    skip the event weeks, because its seasonal index is a lookup by cycle
    position. Holt-Winters cannot - it walks the series one week at a time
    and needs an unbroken run - so it is handed the series with the events
    divided out, which is the same correction in the form it can consume.

    With events disabled this is just the method on the raw series, which is
    what the comparison in main() reports as the "raw series" column.
    """
    settings = assumptions["forecast"]["calendar_events"]
    if not use_events:
        return forecast_one(method, weeks, values, future_weeks, cycle, assumptions), {}

    if not settings.get("estimate_factors_from_history", True):
        raise NotImplementedError(
            "only estimated event factors are built; reading the generator's own "
            "factor would hand the forecast its answer. See spec 3.2."
        )

    decomposition_settings = assumptions["forecast"]["decomposition"]
    index, slope, intercept, factors = fit_ordinary_baseline(
        weeks, values, labels, cycle, decomposition_settings
    )

    if method == DECOMPOSITION:
        baseline = (intercept + slope * future_weeks) * index[future_weeks % cycle]
        return baseline * apply_factors(future_labels, factors), factors

    adjusted = values / apply_factors(labels, factors)
    predicted = forecast_one(method, weeks, adjusted, future_weeks, cycle, assumptions)
    return predicted * apply_factors(future_labels, factors), factors


def score_holdout(history, assumptions=None, use_calendar_events=None):
    """Train both methods on all but the last 13 weeks, score them on those.

    This is the honest test: the holdout weeks are never seen during fitting,
    so the MAPE reported is what the method would have done forecasting a
    quarter ahead, not how well it can describe weeks it already knows.

    Returns a tidy table of one row per site and method, plus the predictions
    themselves for charting.
    """
    assumptions = assumptions or config.load_assumptions()
    cal = assumptions["calendar"]
    cycle = cal["seasonal_cycle_weeks"]
    holdout = cal["holdout_weeks"]
    if use_calendar_events is None:
        use_calendar_events = assumptions["forecast"]["calendar_events"]["enabled"]

    wide = weekly_by_site(history)
    weeks = wide.index.to_numpy()
    train_weeks, test_weeks = weeks[:-holdout], weeks[-holdout:]

    # Event labels come from the dates of the history's own Mondays, so the
    # holdout weeks are labelled the same way the plan weeks will be.
    mondays = generate_data.week_start_dates(cal["history_start"], cal["history_weeks"])
    labels = event_labels(mondays, assumptions)
    train_labels, test_labels = labels[:-holdout], labels[-holdout:]

    rows, predictions = [], []
    for site in wide.columns:
        actual = wide[site].to_numpy()
        train, test = actual[:-holdout], actual[-holdout:]
        for method in METHODS:
            predicted, _factors = fit_and_predict(
                method, train_weeks, train, test_weeks, cycle, assumptions,
                train_labels, test_labels, use_calendar_events,
            )
            rows.append({"site": site, "method": method, "mape": mape(test, predicted)})
            predictions.append(
                pd.DataFrame(
                    {
                        "week_index": test_weeks,
                        "site": site,
                        "method": method,
                        "actual": test,
                        "predicted": predicted,
                    }
                )
            )

    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def choose_method(holdout_predictions, assumptions=None):
    """Pick one method for the whole network (spec 3.2, decided 10/21).

    Pooled MAPE over every site-week in the holdout - 6 sites x 13 weeks - so
    each site carries equal weight and no single site's bad quarter decides
    the method for all of them.
    """
    assumptions = assumptions or config.load_assumptions()
    if assumptions["forecast"]["method_selection"] != "network":
        raise NotImplementedError(
            "only network-level method selection is built; see spec 3.2"
        )

    pooled = (
        holdout_predictions.groupby("method")
        .apply(lambda g: mape(g["actual"], g["predicted"]), include_groups=False)
        .rename("mape")
        .sort_values()
    )
    return pooled.index[0], pooled


# ---------------------------------------------------------------------------
# The forward forecast
# ---------------------------------------------------------------------------


def forecast_forward(history, method, assumptions=None, use_calendar_events=None):
    """Refit the chosen method on ALL the history and forecast 52 weeks out.

    The holdout was spent proving the method. Once that is done, throwing
    away the 13 most recent weeks would be throwing away the most relevant
    data there is, so the model refits on everything before forecasting.

    Plan holidays are placed by DATE (spec 3.2, added 10/21), not by cycle
    position. For 2027 the two agree, but holidays move against a 52-week
    grid - Thanksgiving slides a day or two a year, and sooner or later that
    pushes it into the neighbouring week - so the date is the thing to trust.
    """
    assumptions = assumptions or config.load_assumptions()
    cal = assumptions["calendar"]
    cycle = cal["seasonal_cycle_weeks"]
    if use_calendar_events is None:
        use_calendar_events = assumptions["forecast"]["calendar_events"]["enabled"]

    wide = weekly_by_site(history)
    weeks = wide.index.to_numpy()
    plan_weeks = np.arange(cal["history_weeks"], cal["history_weeks"] + cal["plan_weeks"])
    plan_mondays = generate_data.week_start_dates(cal["plan_start"], cal["plan_weeks"])

    history_mondays = generate_data.week_start_dates(cal["history_start"], cal["history_weeks"])
    labels = event_labels(history_mondays, assumptions)
    plan_labels = event_labels(plan_mondays, assumptions)

    frames = []
    for site in wide.columns:
        predicted, _factors = fit_and_predict(
            method, weeks, wide[site].to_numpy(), plan_weeks, cycle, assumptions,
            labels, plan_labels, use_calendar_events,
        )
        frames.append(
            pd.DataFrame(
                {
                    "week_index": plan_weeks,
                    "week_start": [d.isoformat() for d in plan_mondays],
                    "site": site,
                    "method": method,
                    "calendar_event": plan_labels,
                    "forecast_requests": predicted,
                }
            )
        )

    forward = pd.concat(frames, ignore_index=True)
    if (forward["forecast_requests"] <= 0).any():
        raise ValueError("forecast produced a non-positive week; check assumptions.yaml")
    return forward


def run(assumptions=None, sites=None):
    """The whole of spec 3.2: score, choose, refit, forecast.

    Scores the holdout BOTH ways - with calendar events removed before
    fitting and without - so the gain from treating holidays as known is
    measured on the same data rather than asserted. The live setting in
    assumptions.yaml decides which one the plan actually uses.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    live = assumptions["forecast"]["calendar_events"]["enabled"]

    history = generate_data.generate_history(assumptions, sites)

    both = {}
    for use_events in (False, True):
        scores, predictions = score_holdout(history, assumptions, use_calendar_events=use_events)
        method, pooled = choose_method(predictions, assumptions)
        both[use_events] = {
            "scores": scores,
            "predictions": predictions,
            "pooled": pooled,
            "method": method,
        }

    chosen = both[live]
    forward = forecast_forward(
        history, chosen["method"], assumptions, use_calendar_events=live
    )
    return forward, chosen, both, history


# ---------------------------------------------------------------------------
# Saving and charting
# ---------------------------------------------------------------------------


def write_outputs(forward, both, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    paths = assumptions["paths"]
    out_dir = config.resolve_path(paths["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = out_dir / paths["requests_forecast_csv"]
    forward.to_csv(forecast_path, index=False, float_format="%.6f")

    scores_path = out_dir / paths["forecast_scores_csv"]
    scores_table(both).to_csv(scores_path, index=False, float_format="%.6f")
    return forecast_path, scores_path


def scores_table(both):
    """Per-site and pooled MAPE, both methods, both treatments (spec 3.2).

    One row per treatment, scope and method, so the whole decision is
    traceable from the table alone: which method won, by how much, and
    whether treating holidays as known actually helped.

    Pulled out of write_outputs 11/03 so the CSV and the workbook's
    Forecast tab are the same table rather than two renderings of it. A
    workbook that recomputed this would eventually disagree with the file
    beside it, and the spreadsheet would win the argument.
    """
    tables = []
    for use_events, result in both.items():
        label = "calendar_events" if use_events else "no_calendar_events"
        per_site = result["scores"].rename(columns={"site": "scope"}).copy()
        pooled_rows = pd.DataFrame(
            {
                "scope": "NETWORK",
                "method": result["pooled"].index,
                "mape": result["pooled"].to_numpy(),
            }
        )
        table = pd.concat([per_site, pooled_rows], ignore_index=True)
        table["treatment"] = label
        tables.append(table[["treatment", "scope", "method", "mape"]])
    return pd.concat(tables, ignore_index=True)


def plot_forecast(history, forward, method, assumptions=None):
    """History and forecast on one axis, with the holdout window marked.

    The holdout band is the honest part of the picture: it shows the stretch
    of weeks the chosen method was graded on before it was trusted with the
    52 weeks to the right of the history.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    assumptions = assumptions or config.load_assumptions()
    cal = assumptions["calendar"]
    holdout = cal["holdout_weeks"]

    hist_weekly = history.groupby(["week_index", "week_start"], as_index=False)["requests"].sum()
    fwd_weekly = forward.groupby(["week_index", "week_start"], as_index=False)[
        "forecast_requests"
    ].sum()

    hist_dates = pd.to_datetime(hist_weekly["week_start"])
    fwd_dates = pd.to_datetime(fwd_weekly["week_start"])

    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=150)
    fig.patch.set_facecolor(generate_data.SURFACE)
    ax.set_facecolor(generate_data.SURFACE)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=generate_data.GRIDLINE, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(generate_data.BASELINE)

    ax.axvspan(
        hist_dates.iloc[-holdout],
        hist_dates.iloc[-1],
        color=generate_data.GRIDLINE,
        zorder=0,
    )
    ax.text(
        hist_dates.iloc[-holdout],
        0.04,
        f"  {holdout}-week holdout",
        transform=ax.get_xaxis_transform(),
        color=generate_data.INK_SECONDARY,
        fontsize=9,
    )

    ax.plot(
        hist_dates,
        hist_weekly["requests"],
        color=generate_data.SERIES,
        linewidth=2,
        label="History (synthetic)",
        zorder=2,
    )
    ax.plot(
        fwd_dates,
        fwd_weekly["forecast_requests"],
        color=generate_data.HOLIDAY,
        linewidth=2,
        label=f"Forecast, {method.replace('_', ' ')}",
        zorder=2,
    )

    ax.set_ylim(0, max(hist_weekly["requests"].max(), fwd_weekly["forecast_requests"].max()) * 1.12)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:,.0f}")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(colors=generate_data.INK_MUTED, labelsize=9, length=0)
    ax.set_ylabel("Requests received", color=generate_data.INK_SECONDARY, fontsize=10)

    ax.set_title(
        "SYNTHETIC: weekly records requests, history and 52-week forecast",
        color=generate_data.INK,
        fontsize=14,
        loc="left",
        pad=34,
    )
    ax.text(
        0.0,
        1.022,
        f"All 6 sites. {cal['history_weeks']} weeks of generated history, then "
        f"{cal['plan_weeks']} weeks forecast. Not real data.",
        transform=ax.transAxes,
        color=generate_data.INK_SECONDARY,
        fontsize=10,
    )

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(generate_data.INK_SECONDARY)

    fig.tight_layout()
    out_dir = config.resolve_path(assumptions["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / assumptions["paths"]["forecast_chart_png"]
    fig.savefig(out_path, facecolor=generate_data.SURFACE)
    plt.close(fig)
    return out_path


def main():
    assumptions, sites, _skills = config.load_all()
    forward, chosen, both, history = run(assumptions, sites)
    method = chosen["method"]
    forecast_path, scores_path = write_outputs(forward, both, assumptions)
    png_path = plot_forecast(history, forward, method, assumptions)

    target = assumptions["forecast"]["mape_target"]
    before, after = both[False]["pooled"], both[True]["pooled"]

    print("Holdout MAPE, pooled over all site-weeks (spec 3.2):")
    print(f"  {'method':<14} {'raw series':>11} {'events removed':>16} {'change':>9}")
    for name in METHODS:
        mark = "  <- chosen" if name == method else ""
        change_points = (after[name] - before[name]) * 100
        print(
            f"  {name:<14} {before[name]:10.2%} {after[name]:15.2%} "
            f"{change_points:+7.2f}pp{mark}"
        )
    print(f"  target for the chosen method: {target:.0%}")

    print("\nPer site, chosen method, events removed:")
    per_site = both[True]["scores"]
    raw_site = both[False]["scores"].set_index(["site", "method"])["mape"]
    for _, row in per_site[per_site.method == method].sort_values("site").iterrows():
        was = raw_site[(row["site"], method)]
        print(f"  {row['site']}  {was:6.2%} -> {row['mape']:6.2%}")

    events = forward[forward.calendar_event != ORDINARY]
    print(f"\nPlan weeks carrying a known calendar event: {events['week_start'].nunique()}")
    for monday, group in events.groupby("week_start"):
        print(f"  {monday}  {group['calendar_event'].iloc[0]}")

    weekly = forward.groupby("week_index")["forecast_requests"].sum()
    print(f"\nWrote {len(forward):,} forecast rows to {forecast_path}")
    print(f"Wrote holdout scores to {scores_path}")
    print(f"Wrote chart to {png_path}")
    print(
        f"Forecast weeks {weekly.index.min()}-{weekly.index.max()}, "
        f"network {weekly.min():,.0f} to {weekly.max():,.0f} requests per week"
    )


if __name__ == "__main__":
    main()
