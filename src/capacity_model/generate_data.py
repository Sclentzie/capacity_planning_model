"""Synthetic history: 104 weeks of records requests per site (spec 3.1).

Session 2.

The WFM idea: we invent the demand history from ingredients we choose on
purpose - a base volume, a growth rate, a seasonal shape, holiday dips, and
random noise. Because we know the true answer that went in, we can later grade
the forecast against it (spec Section 9, Session 2 checks). A forecast tested
only on real data can never be scored this way, because nobody knows the real
signal underneath.

    Formula (spec 3.1):
        Requests (site, week) = base
                                x site size
                                x (1 + annual growth) ^ (week / 52)
                                x season
                                x holiday
                                x (1 + noise)

Nothing here is real data. Everything comes from the fixed seed in
assumptions.yaml, so every run reproduces the same history exactly.

Coverage events (spec 6.7) are NOT generated here yet: an event is logged when
a site is short of staff, and staff supply does not exist until Session 4.

Run it:
    .venv\\Scripts\\python -m capacity_model.generate_data
"""

import calendar as _calendar
import datetime

import numpy as np
import pandas as pd

from . import config

WEEKDAY_NUMBERS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

DAYS_PER_WEEK = 7


# ---------------------------------------------------------------------------
# Calendar helpers
#
# Every week is identified by its Monday. Season windows and holidays are
# resolved against real calendar dates rather than week numbers, because a
# 53-week year would slowly slide a week-number rule off the calendar
# (spec 3.1, decided 10/21).
# ---------------------------------------------------------------------------


def week_start_dates(start, n_weeks):
    """The Monday that labels each of n_weeks consecutive weeks."""
    return [start + datetime.timedelta(days=DAYS_PER_WEEK * w) for w in range(n_weeks)]


def holiday_date(spec, year):
    """Resolve one holiday rule from assumptions.yaml to a date in `year`."""
    rule = spec["rule"]
    if rule == "fixed":
        return datetime.date(year, spec["month"], spec["day"])
    if rule == "nth_weekday":
        # e.g. Thanksgiving: the 4th Thursday of November.
        weekday = WEEKDAY_NUMBERS[spec["weekday"]]
        first = datetime.date(year, spec["month"], 1)
        offset = (weekday - first.weekday()) % DAYS_PER_WEEK
        day = 1 + offset + DAYS_PER_WEEK * (spec["nth"] - 1)
        if day > _calendar.monthrange(year, spec["month"])[1]:
            raise ValueError(f"{spec['name']}: no {spec['nth']}th weekday in that month of {year}")
        return datetime.date(year, spec["month"], day)
    raise ValueError(f"unknown holiday rule: {rule!r}")


def holidays_by_date(holiday_specs, years):
    """Map every holiday date in `years` to its name."""
    found = {}
    for year in years:
        for spec in holiday_specs:
            found[holiday_date(spec, year)] = spec["name"]
    return found


def _within_window(day_of_year_tuple, start, end):
    """Is (month, day) inside the inclusive [start, end] window?

    Windows that wrap past December into January are supported, so a future
    assumption like "Dec 18 to Jan 5" would work without a code change.
    """
    if start <= end:
        return start <= day_of_year_tuple <= end
    return day_of_year_tuple >= start or day_of_year_tuple <= end


def monday_of(day):
    """The Monday that starts the week containing `day`."""
    return day - datetime.timedelta(days=day.weekday())


def relative_season_mondays(season_cfg, holiday_specs, years):
    """Mondays matched by each `weeks_before_holiday` season rule.

    Revised 10/21 (spec 3.1). The December dip used to be a fixed date
    window, and it never fired: every Monday in Dec 18 to 31 is also a
    Christmas or New Year week, and the holiday factor replaces the season
    factor. Anchoring the dip to Christmas week instead means it tracks the
    holiday as it moves through the calendar, so it lands every year.

    Returns {period name: set of Mondays}.
    """
    matched = {}
    for period in season_cfg["periods"]:
        if period.get("rule", "date_window") != "weeks_before_holiday":
            continue
        spec = next(s for s in holiday_specs if s["name"] == period["holiday"])
        offset = datetime.timedelta(days=DAYS_PER_WEEK * period["weeks_before"])
        matched[period["name"]] = {
            monday_of(holiday_date(spec, year)) - offset for year in years
        }
    return matched


def season_factor(monday, season_cfg, relative_mondays):
    """The season multiplier for the week labelled by `monday` (spec 3.1).

    Periods are tried in the order they appear in assumptions.yaml and the
    first match wins. A date_window week belongs to its window when its
    MONDAY falls inside it, so each week lands in exactly one season
    regardless of how it straddles a month boundary.
    """
    here = (monday.month, monday.day)
    for period in season_cfg["periods"]:
        rule = period.get("rule", "date_window")
        if rule == "date_window":
            start = (period["from"]["month"], period["from"]["day"])
            end = (period["to"]["month"], period["to"]["day"])
            if _within_window(here, start, end):
                return period["factor"]
        elif rule == "weeks_before_holiday":
            if monday in relative_mondays.get(period["name"], ()):
                return period["factor"]
        else:
            raise ValueError(f"unknown season rule: {rule!r}")
    return season_cfg["default_factor"]


def week_seasonality(monday, season_cfg, holiday_cfg, holiday_lookup, relative_mondays):
    """Return (factor, holiday_name) for one week.

    Spec 3.1, decided 10/21: the holiday factor REPLACES the season factor in
    a holiday week; the two never stack. A Christmas week that also sat in a
    dip window would be x0.6, not x0.6 x 0.92.

    The WFM reason this matters: stacking would push that week's volume to
    about 55% of normal, deepening an already deep trough. The forecast would
    then have to learn a one-week hole the business does not actually see.
    """
    week_days = [monday + datetime.timedelta(days=d) for d in range(DAYS_PER_WEEK)]
    for day in week_days:
        if day in holiday_lookup:
            if holiday_cfg.get("replaces_season", True):
                return holiday_cfg["factor"], holiday_lookup[day]
            season = season_factor(monday, season_cfg, relative_mondays)
            return season * holiday_cfg["factor"], holiday_lookup[day]
    return season_factor(monday, season_cfg, relative_mondays), None


def week_factors(mondays, demand, extra_years=2):
    """Seasonality factor and holiday name for each Monday in `mondays`.

    Shared by the generator and by the forecast's calendar-event logic
    (spec 3.2), so both agree on which weeks are special and when - the
    forecast reads the DATES from here, never the magnitudes.
    """
    years = range(mondays[0].year, mondays[-1].year + extra_years)
    holiday_lookup = holidays_by_date(demand["holiday"]["weeks"], years)
    relative_mondays = relative_season_mondays(
        demand["season"], demand["holiday"]["weeks"], years
    )

    factors, names = [], []
    for monday in mondays:
        factor, name = week_seasonality(
            monday, demand["season"], demand["holiday"], holiday_lookup, relative_mondays
        )
        factors.append(factor)
        names.append(name)
    return np.array(factors), names


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


def generate_history(assumptions=None, sites=None):
    """Build the 156-week synthetic request history (spec 3.1).

    Returns one long DataFrame: a row per week x site x request type, with the
    ingredient that produced it carried alongside, so any number in the file
    can be traced back to the assumption it came from.
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()

    cal = assumptions["calendar"]
    demand = assumptions["demand"]
    request_types = assumptions["request_types"]
    site_rows = sites["sites"]

    n_weeks = cal["history_weeks"]
    mondays = week_start_dates(cal["history_start"], n_weeks)

    # Holidays can land in any year the history touches, including the year
    # after the last Monday (the week of Mon Dec 28 2026 contains Jan 1 2027).
    seasonality, holiday_names = week_factors(mondays, demand)

    # Growth compounds weekly off a 0-based week index, so week 0 sits exactly
    # at the base rate and the series grows 6% over any 52 weeks (spec 3.1).
    week_index = np.arange(n_weeks)
    growth = (1.0 + demand["annual_growth"]) ** (week_index / 52.0)

    size_factors = np.array([s["size_factor"] for s in site_rows])
    base_shares = np.array([rt["share"] for rt in request_types])
    charts_per_request = np.array([rt["charts_per_request"] for rt in request_types])

    # One generator, seeded once, drawn in a fixed order. This is what makes
    # "same seed produces identical data on every run" (spec 9) true.
    rng = np.random.default_rng(assumptions["random"]["seed"])

    # Noise is per site-week: each site has its own good and bad weeks.
    noise_cfg = demand["noise"]
    noise = rng.normal(noise_cfg["mean"], noise_cfg["std_dev"], size=(n_weeks, len(site_rows)))

    # Mix drift is per WEEK, shared by every site (spec 3.1 says "each week",
    # with no site dimension): the mix of why people request records is a
    # property of the market that week, not of one facility.
    # Relative drift - each share wobbles by 5% OF ITS OWN SIZE - so R1 at 35%
    # moves in points while R11 at 2% moves in tenths of a point, instead of a
    # flat wobble swamping the rare types.
    drift_cfg = demand["mix_drift"]
    drift = rng.normal(0.0, drift_cfg["relative_std_dev"], size=(n_weeks, len(request_types)))
    weekly_shares = base_shares[None, :] * (1.0 + drift)
    if drift_cfg.get("renormalize_to_one", True):
        weekly_shares = weekly_shares / weekly_shares.sum(axis=1, keepdims=True)

    # The 3.1 formula, all six ingredients, as one outer product.
    site_requests = (
        demand["base_requests_per_site_per_week"]
        * size_factors[None, :]
        * growth[:, None]
        * seasonality[:, None]
        * (1.0 + noise)
    )

    if (site_requests <= 0).any() or (weekly_shares <= 0).any():
        # 5% noise would need a ~20-sigma draw to flip a sign, so this means
        # an assumption was edited into an impossible range, not bad luck.
        raise ValueError("Generated a non-positive request count or share; check assumptions.yaml")

    # Requests stay FRACTIONAL, like headcount in spec 5.3: these are expected
    # values. Rounding 6,864 cells to whole requests would quietly lose or
    # invent work every week and break the conservation checks downstream.
    rows = []
    for w in range(n_weeks):
        for s_i, site in enumerate(site_rows):
            for t_i, rtype in enumerate(request_types):
                requests = site_requests[w, s_i] * weekly_shares[w, t_i]
                rows.append(
                    {
                        "week_index": w,
                        "week_start": mondays[w].isoformat(),
                        "cluster": site["cluster"],
                        "site": site["code"],
                        "site_size_factor": size_factors[s_i],
                        "growth_factor": growth[w],
                        "seasonality_factor": seasonality[w],
                        "is_holiday_week": holiday_names[w] is not None,
                        "holiday_name": holiday_names[w] or "",
                        "site_requests": site_requests[w, s_i],
                        "request_type": rtype["code"],
                        "skill_tier": rtype["skill_tier"],
                        "type_share": weekly_shares[w, t_i],
                        "requests": requests,
                        "charts_per_request": charts_per_request[t_i],
                        "charts": requests * charts_per_request[t_i],
                    }
                )

    return pd.DataFrame(rows)


def write_history(history, assumptions=None):
    """Save the history to data/synthetic/ and return the path."""
    assumptions = assumptions or config.load_assumptions()
    paths = assumptions["paths"]
    out_dir = config.resolve_path(paths["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / paths["requests_history_csv"]
    history.to_csv(out_path, index=False, float_format="%.6f")
    return out_path


# ---------------------------------------------------------------------------
# Chart: weekly network requests over the 104 weeks, holiday weeks marked
# ---------------------------------------------------------------------------

# Light-mode chart palette, validated for colorblind separation before use.
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"      # categorical slot 1 - the request line
HOLIDAY = "#eb6834"     # categorical slot 2 - holiday-week points
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def plot_history(history, assumptions=None):
    """Chart weekly network requests, with holiday weeks called out.

    One series over time, so a line. Holiday weeks carry a second mark rather
    than a second line: they are the same measure, flagged - and seeing the
    x0.6 dips land on exactly those weeks is the quickest visual check that
    the holiday ingredient went in correctly.
    """
    import matplotlib

    matplotlib.use("Agg")  # file output only; never opens a window
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    assumptions = assumptions or config.load_assumptions()

    weekly = (
        history.groupby(["week_index", "week_start", "is_holiday_week"], as_index=False)["requests"]
        .sum()
        .sort_values("week_index")
    )
    dates = pd.to_datetime(weekly["week_start"])
    values = weekly["requests"]
    holiday_mask = weekly["is_holiday_week"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Recessive grid: horizontal only, so it reads as a backdrop not a cage.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.plot(dates, values, color=SERIES, linewidth=2, label="Weekly requests, network", zorder=2)
    # A surface-colored ring keeps the holiday dots legible where the line
    # passes under them.
    ax.scatter(
        dates[holiday_mask],
        values[holiday_mask],
        s=70,
        color=HOLIDAY,
        edgecolors=SURFACE,
        linewidths=2,
        zorder=3,
        label="Holiday week (x0.6)",
    )

    ax.set_ylim(0, float(values.max()) * 1.12)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:,.0f}")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_ylabel("Requests received", color=INK_SECONDARY, fontsize=10)

    ax.set_title(
        "SYNTHETIC: weekly records requests received, all 6 sites",
        color=INK,
        fontsize=14,
        loc="left",
        pad=34,
    )
    ax.text(
        0.0,
        1.022,
        f"{len(weekly)} weeks of generated history, "
        f"{weekly['week_start'].iloc[0]} to {weekly['week_start'].iloc[-1]}. "
        "Not real data.",
        transform=ax.transAxes,
        color=INK_SECONDARY,
        fontsize=10,
    )

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()

    paths = assumptions["paths"]
    out_dir = config.resolve_path(paths["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / paths["history_chart_png"]
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main():
    assumptions, sites, _skills = config.load_all()
    history = generate_history(assumptions, sites)
    csv_path = write_history(history, assumptions)
    png_path = plot_history(history, assumptions)

    weekly = history.groupby("week_index")["requests"].sum()
    print(f"Wrote {len(history):,} rows to {csv_path}")
    print(f"Wrote chart to {png_path}")
    print(f"Weeks {weekly.index.min()}-{weekly.index.max()}, "
          f"network requests {weekly.min():,.0f} to {weekly.max():,.0f} per week")
    print(f"Holiday weeks: {int(history.groupby('week_index')['is_holiday_week'].first().sum())}")


if __name__ == "__main__":
    main()
