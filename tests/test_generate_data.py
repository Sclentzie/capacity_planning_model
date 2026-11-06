"""Spec Section 9, Session 2: the generated history recovers its ingredients.

The point of synthetic data is that we KNOW the answer. We chose the growth,
the season, the holiday dips, and the site size factors, so each test here
reaches into the generated history, measures one ingredient back out, and
checks it against the assumption that produced it. If a test fails, either the
generator has a bug or an assumption was edited - and both are things we want
to hear about before a forecast is graded on this data.

Two measurement habits make these tests sharp rather than vague:

  DETREND FIRST. Growth lifts every later week, so a raw average of Q1 weeks
  in 2026 beats a raw average of Q3 weeks in 2025 even with no seasonality at
  all. Dividing each week by its own growth factor removes that, leaving only
  the ingredient under test.

  COMPARE LIKE WITH LIKE. Holiday weeks are x0.6, so they are held out of any
  comparison that is not about holidays; otherwise their dips leak into the
  season and growth measurements.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from capacity_model import config, generate_data


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
def weekly(history):
    """Network requests per week, with the week's ingredients attached."""
    return history.groupby(
        ["week_index", "week_start", "is_holiday_week", "growth_factor", "seasonality_factor"],
        as_index=False,
    )["requests"].sum()


def detrended(weekly_df):
    """Weekly network requests with the growth ingredient divided out."""
    return weekly_df["requests"] / weekly_df["growth_factor"]


# ---------------------------------------------------------------------------
# Shape and conservation
# ---------------------------------------------------------------------------


def test_history_has_a_row_per_week_site_and_request_type(history, assumptions, sites):
    expected = (
        assumptions["calendar"]["history_weeks"]
        * len(sites["sites"])
        * len(assumptions["request_types"])
    )
    assert len(history) == expected == 10296  # 156 weeks x 6 sites x 11 types


def test_history_runs_straight_into_the_plan(assumptions):
    """No gap and no overlap between the last history week and plan week 1.

    The forecast is handed to the plan as if it were the next observation.
    A one-week gap here would silently shift all 52 plan weeks against the
    seasonal calendar they were forecast from.
    """
    cal = assumptions["calendar"]
    weeks = generate_data.week_start_dates(cal["history_start"], cal["history_weeks"] + 1)
    assert weeks[-1] == cal["plan_start"]
    assert cal["history_weeks"] % cal["seasonal_cycle_weeks"] == 0, (
        "history should be whole seasonal cycles, so each cycle position has equal weight"
    )


def test_charts_equal_requests_times_charts_per_request(history):
    """Charts are the retrieval-side volume driver (spec Section 1).

    One bulk payer-audit request (R5) is 1 intake but 10 charts to retrieve,
    review, and release. If this multiplication ever drifts, every downstream
    hour estimate for retrieval and review is wrong, so it is pinned here.
    """
    expected = history["requests"] * history["charts_per_request"]
    np.testing.assert_allclose(history["charts"], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Ingredient 1: annual growth, 6% (spec 3.1; Section 9 allows +/- 1.5 points)
# ---------------------------------------------------------------------------


def year_blocks(weekly, cycle):
    """Split the history into consecutive whole-year blocks of `cycle` weeks."""
    n_years = (weekly["week_index"].max() + 1) // cycle
    return [
        weekly[(weekly.week_index >= y * cycle) & (weekly.week_index < (y + 1) * cycle)]
        for y in range(n_years)
    ]


def test_annual_growth_is_recovered_within_tolerance(weekly, assumptions):
    """Measured by comparing each year with the one before it, week-matched.

    Why this measurement and not a regression: week w and week w+52 are the
    same point in the calendar, so they carry the same holiday status and
    very nearly the same season factor. Dividing one 52-week block by the
    previous one therefore cancels seasonality and leaves the growth
    ingredient alone. Noise averages away over 52 weeks x 6 sites, so what is
    left is 1 + growth. With 156 weeks there are two such year-over-year
    comparisons, and both must land.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    expected = assumptions["demand"]["annual_growth"]
    blocks = year_blocks(weekly, cycle)
    assert len(blocks) == 3

    for earlier, later in zip(blocks, blocks[1:]):
        measured = later["requests"].sum() / earlier["requests"].sum() - 1.0
        assert measured == pytest.approx(expected, abs=0.015), (
            f"recovered annual growth {measured:.2%}, expected {expected:.0%} +/- 1.5 points"
        )


def test_holiday_weeks_line_up_52_weeks_apart(weekly, assumptions):
    """Guard for the test above, and for the forecast's seasonal index.

    Holiday weeks are the biggest swings in the series (x0.6). The growth
    measurement only cancels them if each one has a partner exactly 52 weeks
    later, and the forecast's seasonal index (spec 3.2, keyed to cycle
    position) only learns them for the same reason. If a holiday ever drifted
    off the 52-week grid, both would quietly degrade.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    blocks = year_blocks(weekly, cycle)

    positions = [set(b.loc[b.is_holiday_week, "week_index"] % cycle) for b in blocks]
    assert positions[0], "no holiday weeks found at all"
    for later in positions[1:]:
        assert later == positions[0], "holiday weeks do not repeat on the 52-week cycle"


def test_season_weeks_drift_by_at_most_one_week_a_year(weekly, assumptions):
    """The calendar does NOT line up as perfectly as the holidays do.

    The first-quarter window is a fixed date range (Jan 1 to Mar 31), but a
    52-week year is 364 days, so each year's Mondays slide a day or two
    earlier. One week can therefore fall inside the window in one year and
    outside it the next - in this history, Mon 2024-04-01 is not a Q1 week
    while Mon 2025-03-31, 52 weeks later, is.

    That is real calendar drift, not a bug, and it is small: one week in 52
    differing by 10% moves a year-over-year ratio about 0.2 points, well
    inside the growth tolerance. This test pins the drift at no more than one
    week a year, so a larger misalignment would be caught.
    """
    cycle = assumptions["calendar"]["seasonal_cycle_weeks"]
    blocks = year_blocks(weekly, cycle)
    lifted = [int((~b.is_holiday_week & (b.seasonality_factor > 1.0)).sum()) for b in blocks]
    assert max(lifted) - min(lifted) <= 1, f"first-quarter weeks per year: {lifted}"


# ---------------------------------------------------------------------------
# Ingredient 2: first-quarter lift, +10% (spec 3.1)
# ---------------------------------------------------------------------------


def test_first_quarter_lift_is_visible(weekly, assumptions):
    """Measured as detrended Q1 weeks over detrended ordinary weeks.

    "Ordinary" means weeks carrying no season and no holiday - the 1.0 weeks.
    Dividing the Q1 average by the ordinary average should return the season
    factor itself, because growth is already divided out and noise averages
    down over 26 Q1 weeks.
    """
    q1_factor = next(
        p["factor"] for p in assumptions["demand"]["season"]["periods"] if p["name"] == "Q1 lift"
    )
    level = detrended(weekly)
    is_plain = (~weekly["is_holiday_week"]) & np.isclose(weekly["seasonality_factor"], 1.0)
    is_q1 = (~weekly["is_holiday_week"]) & np.isclose(weekly["seasonality_factor"], q1_factor)

    # Cross-check the flag against the calendar itself rather than against a
    # magic count: the lifted weeks must be exactly the non-holiday weeks
    # whose Monday falls in January, February, or March.
    monday = pd.to_datetime(weekly["week_start"])
    from_calendar = monday.dt.month.isin([1, 2, 3]) & ~weekly["is_holiday_week"]
    assert (is_q1 == from_calendar).all()

    measured = level[is_q1].mean() / level[is_plain].mean()

    assert measured > 1.05, "the first-quarter lift is not visible in the data"
    assert measured == pytest.approx(q1_factor, abs=0.02)


# ---------------------------------------------------------------------------
# Ingredient 3: holiday weeks, x0.6, replacing the season (spec 3.1)
# ---------------------------------------------------------------------------


def test_holiday_weeks_are_the_lowest_weeks(weekly):
    """Measured by ranking. A x0.6 week should be unmistakable.

    Ranking rather than thresholding is the stronger check: it fails if ANY
    ordinary week ever dips below a holiday week, which is what would happen
    if the holiday factor were applied to the wrong weeks or watered down.
    """
    n_holiday = int(weekly["is_holiday_week"].sum())
    assert n_holiday == 12, "4 holidays a year over 3 years of history"

    lowest = weekly.nsmallest(n_holiday, "requests")
    assert lowest["is_holiday_week"].all()


def dip_period(assumptions):
    return next(
        p
        for p in assumptions["demand"]["season"]["periods"]
        if p["name"] == "Pre-Christmas slowdown"
    )


def test_pre_christmas_dip_fires_once_in_every_december(weekly, assumptions):
    """The fix for a dip that used to never fire at all (spec 3.1, 10/21).

    The original rule was a Dec 18 to 31 window on the week's Monday, and it
    was dead input: every Monday in that range is also a Christmas or New
    Year week, and the holiday factor replaces the season factor. Anchoring
    the dip to Christmas week instead makes it land every year.

    So: exactly one dip week per December in the history, and no Decembers
    missed.
    """
    dip = dip_period(assumptions)
    monday = pd.to_datetime(weekly["week_start"])
    is_dip = np.isclose(weekly["seasonality_factor"], dip["factor"])

    assert is_dip.any(), "the pre-Christmas dip is not firing at all"

    dip_years = monday[is_dip].dt.year
    assert (monday[is_dip].dt.month == 12).all(), "a dip week landed outside December"

    decembers = sorted(monday[monday.dt.month == 12].dt.year.unique())
    assert sorted(dip_years) == decembers, (
        f"dip weeks in {sorted(dip_years)}, Decembers in the history {decembers}"
    )
    assert len(dip_years) == len(set(dip_years)) == 3


def test_the_dip_sits_in_the_week_before_christmas_week(weekly, assumptions):
    """Measured against Christmas itself, not against a date range.

    The point of the revised rule is that it tracks the holiday: Christmas
    lands on a different weekday each year, so the week before it moves too.
    This checks the dip Monday is exactly 7 days before the Monday of the
    week containing December 25, in every year.
    """
    dip = dip_period(assumptions)
    monday = pd.to_datetime(weekly["week_start"])
    dip_mondays = set(monday[np.isclose(weekly["seasonality_factor"], dip["factor"])].dt.date)

    expected = set()
    for year in sorted(monday.dt.year.unique()):
        christmas = datetime.date(year, 12, 25)
        christmas_monday = generate_data.monday_of(christmas)
        candidate = christmas_monday - datetime.timedelta(days=7 * dip["weeks_before"])
        if candidate in set(monday.dt.date):
            expected.add(candidate)

    assert dip_mondays == expected


def test_the_dip_is_a_real_drop_in_the_data(weekly, assumptions):
    """Not just a flag: the dip weeks must actually be low.

    Measured detrended against ordinary weeks, the same way the Q1 lift and
    the holiday factor are measured, so it recovers 0.92.
    """
    dip = dip_period(assumptions)
    level = detrended(weekly)
    is_dip = np.isclose(weekly["seasonality_factor"], dip["factor"])
    is_plain = (~weekly["is_holiday_week"]) & np.isclose(weekly["seasonality_factor"], 1.0)

    measured = level[is_dip].mean() / level[is_plain].mean()
    assert measured == pytest.approx(dip["factor"], abs=0.03)


def test_holiday_factor_is_recovered_and_does_not_stack_on_the_season(weekly, assumptions):
    """Measured as detrended holiday weeks over detrended ordinary weeks.

    Spec 3.1 (decided 10/21): the holiday factor REPLACES the season factor.
    Four of the eight holiday weeks sit inside the late-December dip, so if
    the two ever stacked those weeks would read 0.6 x 0.92 = 0.552 and this
    ratio would land near 0.57 instead of 0.60.
    """
    holiday_factor = assumptions["demand"]["holiday"]["factor"]
    level = detrended(weekly)
    is_plain = (~weekly["is_holiday_week"]) & np.isclose(weekly["seasonality_factor"], 1.0)

    measured = level[weekly["is_holiday_week"]].mean() / level[is_plain].mean()
    assert measured == pytest.approx(holiday_factor, abs=0.02)

    # And no holiday week carries a season factor at all.
    assert np.allclose(
        weekly.loc[weekly["is_holiday_week"], "seasonality_factor"], holiday_factor
    )


# ---------------------------------------------------------------------------
# Ingredient 4: site size factors (spec 3.1)
# ---------------------------------------------------------------------------


def test_each_site_is_scaled_by_its_size_factor(history, sites):
    """Measured as each site's share of network volume.

    Growth, season, and holidays hit every site in the same week by the same
    amount, so they cancel out of a SHARE. What is left is the size factor
    over the sum of the size factors - the one ingredient that distinguishes
    the sites. Noise averages down over 104 weeks, so the tolerance is tight.
    """
    size_factors = {s["code"]: s["size_factor"] for s in sites["sites"]}
    total_factor = sum(size_factors.values())

    by_site = history.groupby("site")["requests"].sum()
    measured_share = by_site / by_site.sum()

    for site, factor in size_factors.items():
        expected_share = factor / total_factor
        assert measured_share[site] == pytest.approx(expected_share, rel=0.02), (
            f"{site}: share {measured_share[site]:.4f}, expected {expected_share:.4f}"
        )

    # Ordering must hold for every pair of sites whose factors actually
    # differ. Sites that share a factor (N2 and S3 are both 1.0) are left out
    # of this: which of the two lands higher is decided by noise alone, so
    # asserting an order between them would be testing the random draw.
    for site_a, factor_a in size_factors.items():
        for site_b, factor_b in size_factors.items():
            if factor_a > factor_b:
                assert measured_share[site_a] > measured_share[site_b], (
                    f"{site_a} ({factor_a}) should out-volume {site_b} ({factor_b})"
                )


# ---------------------------------------------------------------------------
# Ingredient 5: the request-type mix (spec Section 1 shares, 3.1 drift)
# ---------------------------------------------------------------------------


def test_request_type_shares_stay_close_to_the_spec_mix(history, assumptions):
    """Measured as each type's realized share of all requests over 104 weeks.

    The weekly mix drifts by 5% of each share's own size, but drift is
    zero-mean, so averaging 104 weeks should return the spec share. A type
    that came back off by more than 2% of itself would mean the drift is
    biased - for example, renormalisation systematically favouring the large
    types - which would quietly change the Tier 1 / Tier 2 workload split.
    """
    expected = {rt["code"]: rt["share"] for rt in assumptions["request_types"]}

    by_type = history.groupby("request_type")["requests"].sum()
    measured = by_type / by_type.sum()

    for code, share in expected.items():
        assert measured[code] == pytest.approx(share, rel=0.02), (
            f"{code}: realized share {measured[code]:.4f}, spec share {share:.4f}"
        )


def test_weekly_shares_always_sum_to_one(history):
    """Rescaling to 100% (spec 3.1) must hold in EVERY week, not on average.

    If it did not, the site's weekly requests would not equal the sum of its
    request types, and work would appear or vanish at the point demand is
    split by type.
    """
    per_week = history.groupby(["week_index", "site"])["type_share"].sum()
    np.testing.assert_allclose(per_week, 1.0, rtol=1e-12)

    totals = history.groupby(["week_index", "site"]).agg(
        summed=("requests", "sum"), site_total=("site_requests", "first")
    )
    np.testing.assert_allclose(totals["summed"], totals["site_total"], rtol=1e-12)


def test_drift_actually_moves_the_mix(history):
    """The mirror of the test above: the shares must not be frozen either.

    A generator that silently dropped the drift would pass every average-based
    check here. This fails if the weekly shares never move.
    """
    spread = history.groupby("request_type")["type_share"].std()
    assert (spread > 0).all(), "request-type shares are not drifting at all"


# ---------------------------------------------------------------------------
# Spec Section 9: same seed produces identical data on every run
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_data(assumptions, sites, history):
    """Measured by generating twice in one process and comparing cell by cell.

    Reproducibility is what makes a forecast score meaningful: if the data
    moved between runs, a MAPE improvement could be luck rather than method.
    """
    again = generate_data.generate_history(assumptions, sites)
    pd.testing.assert_frame_equal(history, again)


def test_a_different_seed_produces_different_data(assumptions, sites, history):
    """Guard for the test above: identical output must come from the SEED.

    A generator that ignored its noise inputs would pass the reproducibility
    test perfectly. This one fails if the seed is not actually in use.
    """
    other = {**assumptions, "random": {**assumptions["random"], "seed": 999}}
    different = generate_data.generate_history(other, sites)
    assert not np.allclose(history["requests"], different["requests"])


def test_saved_csv_matches_a_fresh_generation(history, assumptions):
    """The committed file on disk must be the file this code produces.

    Catches a stale data/synthetic/ CSV left behind after an assumption
    changed - the quiet failure where tests pass against fresh data while
    every chart and downstream run reads the old file.
    """
    paths = assumptions["paths"]
    csv_path = config.resolve_path(paths["synthetic_data_dir"], paths["requests_history_csv"])
    if not csv_path.exists():
        # The CSV is generated, not committed, so a fresh clone has none yet.
        # Skipping keeps a clean clone green; the check still bites on every
        # run after the data exists.
        pytest.skip("no generated history on disk yet; run capacity_model.generate_data")

    on_disk = pd.read_csv(csv_path)
    assert len(on_disk) == len(history)
    for column in ("requests", "charts", "type_share", "site_requests"):
        np.testing.assert_allclose(
            on_disk[column], history[column], rtol=1e-5, atol=1e-6,
            err_msg=f"{column} on disk differs from a fresh run",
        )
