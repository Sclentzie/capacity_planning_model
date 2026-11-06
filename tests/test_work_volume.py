"""Spec Section 9, Session 2: the lag check, plus the rules it rests on.

    - Sum of lagged work volumes equals requests x ratio over the full
      horizon (no work lost or double-counted by the lag).

That one check is the headline, and the tests around it are what make it mean
something. A lag that conserves work can still put it on the wrong week, hang
digitization off nothing, or bill a fee to a request type that never pays one,
so each of those rules is pinned separately.

Every expected figure here is rebuilt from the source series INDEPENDENTLY of
work_volume.py - from the generated history and the forecast - rather than
read back out of the module's own conservation table. A module that agrees
with itself proves nothing.
"""

import numpy as np
import pandas as pd
import pytest

from capacity_model import config, forecast, generate_data, work_volume


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
def forward(assumptions, sites):
    forecast_rows, _chosen, _both, _history = forecast.run(assumptions, sites)
    return forecast_rows


@pytest.fixture(scope="module")
def result(assumptions, sites, history, forward):
    return work_volume.build(assumptions, sites, history, forward)


@pytest.fixture(scope="module")
def source(assumptions, sites, history, forward):
    """Requests and charts per request type over all 208 weeks.

    Rebuilt here from the history and the forecast so the conservation tests
    have an expected value that work_volume.py had no hand in producing.
    """
    hist = history.groupby("request_type")["requests"].sum()

    base_shares = {rt["code"]: rt["share"] for rt in assumptions["request_types"]}
    charts_per = {rt["code"]: rt["charts_per_request"] for rt in assumptions["request_types"]}
    plan_total = forward["forecast_requests"].sum()

    requests = {code: hist[code] + plan_total * share for code, share in base_shares.items()}
    charts = {code: value * charts_per[code] for code, value in requests.items()}
    return {"requests": pd.Series(requests), "charts": pd.Series(charts)}


@pytest.fixture(scope="module")
def totals(result):
    """Every work item produced anywhere: in horizon plus spill.

    The spill half matters. Work lagged off the last plan week has to be
    counted somewhere or the conservation check passes by losing it.
    """
    both = pd.concat([result["work"], result["spill"]], ignore_index=True)
    return both.groupby(["work_type", "media", "request_type"], dropna=False)["work_items"].sum()


def total_for(totals, work_type, request_type=None):
    chunk = totals.xs(work_type, level="work_type")
    if request_type is not None:
        chunk = chunk.xs(request_type, level="request_type")
    return float(chunk.sum())


# ---------------------------------------------------------------------------
# The required check: the lag conserves work
# ---------------------------------------------------------------------------


def test_every_work_type_conserves_its_source_volume(result, source, assumptions):
    """Spec 9: total lagged work = source volume x ratio, over the horizon.

    Checked per work type against an expected value rebuilt from the source
    series, with spill counted in. A lag profile sums to 1.0, so moving work
    between weeks must not change how much there is.
    """
    media_share = {m["code"]: m["share"] for m in assumptions["record_media"]}
    by_code = {w["code"]: w for w in assumptions["work_types"]}
    totals = pd.concat([result["work"], result["spill"]], ignore_index=True)
    grouped = totals.groupby(["work_type", "media"], dropna=False)["work_items"].sum()

    def expected_for(code):
        spec = by_code[code]
        if spec["basis"] == work_volume.CHAINED:
            return expected_for(spec["follows"]) * float(spec.get("rate", 1.0))
        basis = source["requests"] if spec["basis"] == "per_request" else source["charts"]
        rates = work_volume.request_type_rates(spec, list(basis.index), by_code)
        return float((basis.to_numpy() * rates).sum())

    def media_for(code):
        # A chained row inherits the media of whatever it follows, so a
        # microfiche retrieval yields a microfiche digitization item.
        spec = by_code[code]
        if "media" in spec:
            return spec["media"]
        if spec["basis"] == work_volume.CHAINED:
            return media_for(spec["follows"])
        return [work_volume.NO_MEDIA]

    for code, spec in by_code.items():
        media = media_for(code)
        total_expected = expected_for(code)
        for medium in media:
            share = media_share[medium] if medium != work_volume.NO_MEDIA else 1.0
            key = (code, medium if medium != work_volume.NO_MEDIA else np.nan)
            actual = grouped.get(key, grouped.get((code, medium), 0.0))
            assert actual == pytest.approx(total_expected * share, rel=1e-9), (
                f"{code}/{medium or 'n/a'}: {actual:,.2f} vs expected "
                f"{total_expected * share:,.2f}"
            )


def test_conservation_table_agrees_with_the_rows_it_describes(result):
    """The spill report must describe the file beside it, not a parallel sum."""
    cons = result["conservation"]
    assert cons["difference"].abs().max() < 1e-6

    rows = result["work"].groupby(["work_type", "media"], dropna=False)["work_items"].sum()
    for _, row in cons.iterrows():
        key = (row["work_type"], row["media"])
        landed = rows.get(key, rows.get((row["work_type"], np.nan), 0.0))
        assert landed == pytest.approx(row["in_horizon"], rel=1e-9)


def test_unlagged_work_never_spills(result, assumptions):
    """Work that lands in the arrival week cannot fall off the end.

    A non-zero spill on an in_week row would mean the lag machinery is
    shifting work it was told not to shift.
    """
    in_week = {
        w["code"] for w in assumptions["work_types"] if assumptions["lag_profiles"][w["lag"]] == [1.0]
    }
    assert in_week, "expected some work types to land in the arrival week"
    cons = result["conservation"].set_index("work_type")
    for code in in_week:
        assert cons.loc[code, "spill"] == pytest.approx(0.0, abs=1e-9)


def test_lagged_work_does_spill_and_is_reported(result):
    """The mirror: spill must be real and written down, not quietly zero.

    If this ever reported nothing, the conservation check above would be
    passing because the lag had stopped working, not because it balanced.
    """
    cons = result["conservation"]
    assert cons["spill"].sum() > 0
    assert not result["spill"].empty
    assert (result["spill"]["week_index"] >= 208).all()


def test_unlagged_rows_land_exactly_on_the_arrival_week(result, history, assumptions):
    """W1 is one per request, in the week it arrives - so it IS the requests.

    Pins the lag's zero case against the history week by week, which catches
    an off-by-one that a total-only check would sail past.
    """
    n_history = assumptions["calendar"]["history_weeks"]
    w1 = (
        result["work"]
        .query("work_type == 'W1' and period == 'history'")
        .groupby(["week_index", "site"])["work_items"]
        .sum()
    )
    requests = history.groupby(["week_index", "site"])["requests"].sum()
    requests = requests[requests.index.get_level_values("week_index") < n_history]
    np.testing.assert_allclose(w1.sort_index(), requests.sort_index(), rtol=1e-9)


# ---------------------------------------------------------------------------
# Chained work: it only exists because something else happened
# ---------------------------------------------------------------------------


def weekly(frame, work_type):
    both = frame.query("work_type == @work_type")
    return both.groupby("week_index")["work_items"].sum().sort_index()


def test_digitization_follows_its_onsite_retrieval_by_one_week(result):
    """W12 in week w must equal W11 in week w-1 (spec 3.3).

    Not "about equal": every non-digital item pulled off a shelf becomes
    exactly one digitization item the following week. Spec 9 asks that none
    is lost and none is digitized twice, and a week-by-week identity is the
    way to see it.
    """
    both = pd.concat([result["work"], result["spill"]], ignore_index=True)
    w11, w12 = weekly(both, "W11"), weekly(both, "W12")

    assert w12.get(0, 0.0) == pytest.approx(0.0), "nothing to digitize in week 0"
    shifted = w11.reindex(range(w11.index.min(), w11.index.max() + 2)).fillna(0.0)
    shifted.index = shifted.index + 1
    common = w12.index.intersection(shifted.index)
    np.testing.assert_allclose(w12.loc[common], shifted.loc[common], rtol=1e-9, atol=1e-9)
    assert w12.sum() == pytest.approx(w11.sum(), rel=1e-9)


def test_digitization_keeps_the_media_of_the_record_it_came_from(result):
    """A microfiche pull makes a microfiche digitization item, not a paper one.

    Media has to survive the chain, because Session 3 prices onsite retrieval
    at 25, 35 and 45 minutes by medium and the non-digital tracker in 4.6
    reports open items by medium.
    """
    both = pd.concat([result["work"], result["spill"]], ignore_index=True)
    w11 = both.query("work_type == 'W11'").groupby("media")["work_items"].sum()
    w12 = both.query("work_type == 'W12'").groupby("media")["work_items"].sum()

    assert set(w11.index) == {"paper", "microfiche", "archive_room"}
    assert set(w12.index) == set(w11.index)
    np.testing.assert_allclose(w12.sort_index(), w11.sort_index(), rtol=1e-9)


def test_rework_is_three_percent_of_quality_review_a_week_later(result, assumptions):
    """W22 in week w = 3% of W14 in week w-1 (spec Section 1 and 3.3)."""
    rate = next(w["rate"] for w in assumptions["work_types"] if w["code"] == "W22")
    both = pd.concat([result["work"], result["spill"]], ignore_index=True)
    w14, w22 = weekly(both, "W14"), weekly(both, "W22")

    shifted = (w14 * rate).reindex(range(w14.index.min(), w14.index.max() + 2)).fillna(0.0)
    shifted.index = shifted.index + 1
    common = w22.index.intersection(shifted.index)
    np.testing.assert_allclose(w22.loc[common], shifted.loc[common], rtol=1e-9, atol=1e-9)
    assert w22.sum() == pytest.approx(w14.sum() * rate, rel=1e-9)


def test_chained_work_never_exceeds_its_parent(result):
    """A sanity bound that holds whatever the rates are set to.

    Digitization cannot outnumber the retrievals that caused it, and rework
    cannot outnumber the reviews. If an assumption is ever edited so that it
    does, that is a modelling error rather than a new scenario.
    """
    both = pd.concat([result["work"], result["spill"]], ignore_index=True)
    totals = both.groupby("work_type")["work_items"].sum()
    assert totals["W12"] <= totals["W11"] * (1 + 1e-9)
    assert totals["W22"] <= totals["W14"] * (1 + 1e-9)


# ---------------------------------------------------------------------------
# Request-specific rows
# ---------------------------------------------------------------------------


def test_fee_processing_only_touches_billable_request_types(totals, source):
    """W3: billable requests, R4 to R7 (spec Section 1).

    A patient asking for their own records is not invoiced, so W3 must not
    appear for R1 to R3 at all - not a small number, none.
    """
    present = set(totals.xs("W3", level="work_type").index.get_level_values("request_type"))
    assert present == {"R4", "R5", "R6", "R7"}

    expected = source["requests"][["R4", "R5", "R6", "R7"]].sum()
    assert total_for(totals, "W3") == pytest.approx(expected, rel=1e-9)
    # The spec quotes 32% of requests; confirm the mix really delivers that.
    assert total_for(totals, "W3") / source["requests"].sum() == pytest.approx(0.32, abs=0.005)


def test_legal_review_covers_all_subpoenas_plus_a_slice_of_the_rest(totals, source, assumptions):
    """W7: all R8, plus 8% of others (spec Section 1).

    The two-tier rule is the interesting part: a court order always gets a
    legal read, while everything else gets one only when something in the
    request flags it.
    """
    rate = next(w["rate"] for w in assumptions["work_types"] if w["code"] == "W7")

    assert total_for(totals, "W7", "R8") == pytest.approx(source["requests"]["R8"], rel=1e-9)
    for code in source["requests"].index:
        if code == "R8":
            continue
        assert total_for(totals, "W7", code) == pytest.approx(
            source["requests"][code] * rate, rel=1e-9
        )


def test_certified_copies_cover_r7_r8_and_a_fifth_of_r4(totals, source):
    """W17: R7, R8, and 20% of R4, about 14% of requests (spec Section 1)."""
    present = set(totals.xs("W17", level="work_type").index.get_level_values("request_type"))
    assert present == {"R4", "R7", "R8"}

    assert total_for(totals, "W17", "R4") == pytest.approx(
        source["requests"]["R4"] * 0.20, rel=1e-9
    )
    for code in ("R7", "R8"):
        assert total_for(totals, "W17", code) == pytest.approx(source["requests"][code], rel=1e-9)
    assert total_for(totals, "W17") / source["requests"].sum() == pytest.approx(0.14, abs=0.01)


def test_amendment_tracking_is_r10_only(totals, source):
    """W23: all R10 and nothing else (spec Section 1)."""
    present = set(totals.xs("W23", level="work_type").index.get_level_values("request_type"))
    assert present == {"R10"}
    assert total_for(totals, "W23") == pytest.approx(source["requests"]["R10"], rel=1e-9)


def test_amendment_tracking_spreads_evenly_over_four_weeks(result, assumptions):
    """W23 is the only profile longer than two weeks, so it gets its own check.

    An amendment is tracked to closure, which takes a while; spreading it
    evenly over four weeks is what keeps one week's amendments from looking
    like a spike the plan has to staff for.
    """
    profile = assumptions["lag_profiles"]["spread_4_weeks"]
    assert profile == [0.25, 0.25, 0.25, 0.25]

    both = pd.concat([result["work"], result["spill"]], ignore_index=True)
    w23 = weekly(both, "W23")
    # Week 0 can only have received the first quarter of week 0's requests.
    assert w23.loc[0] > 0
    assert w23.loc[1] > w23.loc[0]


# ---------------------------------------------------------------------------
# Per request vs per chart, and the media partition
# ---------------------------------------------------------------------------


def test_bulk_requests_multiply_charts_but_not_intake(totals, assumptions):
    """The charts-per-request lever from Section 1, measured end to end.

    R5 is a bulk payer audit: one request, ten charts. So a per-chart step
    must be exactly ten times a per-request step for R5, and exactly equal
    to it for every single-chart type. This is what lets one bulk request
    load retrieval and review without loading the intake desk.

    Measured on W24, the fulfiller self-check, which is the per-chart row
    that covers every chart. Quality review used to serve this purpose and
    no longer can: since 10/22 it samples (see the QA coverage tests), so
    its ratio to intake is a coverage rule rather than the chart multiple.
    """
    charts_per = {rt["code"]: rt["charts_per_request"] for rt in assumptions["request_types"]}
    for code, multiplier in charts_per.items():
        intake = total_for(totals, "W1", code)
        per_chart = total_for(totals, "W24", code)
        assert per_chart == pytest.approx(intake * multiplier, rel=1e-9), (
            f"{code}: self-check/intake = {per_chart / intake:.4f}, expected {multiplier}"
        )
    assert charts_per["R5"] == 10


def test_retrieval_media_partition_covers_every_chart_exactly_once(totals, source, assumptions):
    """W8 + W9 + W11 must equal the charts, no more and no less.

    Every chart lives somewhere, so the three retrieval rows partition the
    population. Double-counting here would inflate retrieval hours across the
    whole plan, and under-counting would hide onsite work.
    """
    retrieved = sum(total_for(totals, code) for code in ("W8", "W9", "W11"))
    assert retrieved == pytest.approx(source["charts"].sum(), rel=1e-9)

    media_share = {m["code"]: m["share"] for m in assumptions["record_media"]}
    assert sum(media_share.values()) == pytest.approx(1.0)
    for code, medium in (("W8", "digital"), ("W9", "email_archive")):
        assert total_for(totals, code) / source["charts"].sum() == pytest.approx(
            media_share[medium], rel=1e-9
        )


def test_onsite_retrieval_splits_into_its_three_media(totals, source, assumptions):
    """Paper 7%, microfiche 2%, archive room 2% - 11% of charts in total."""
    media_share = {m["code"]: m["share"] for m in assumptions["record_media"]}
    onsite = [m["code"] for m in assumptions["record_media"] if m["onsite"]]
    assert set(onsite) == {"paper", "microfiche", "archive_room"}

    w11 = totals.xs("W11", level="work_type").groupby("media").sum()
    charts = source["charts"].sum()
    for medium in onsite:
        assert w11[medium] / charts == pytest.approx(media_share[medium], rel=1e-9)
    assert w11.sum() / charts == pytest.approx(0.11, rel=1e-9)


def test_additional_chart_passes_sit_on_top_of_the_partition(totals, source):
    """W10, W13 and W15 are extra passes, not slices of the media split.

    Multi-source compilation happens as well as retrieving the chart, not
    instead of it, so these are measured against total charts and are allowed
    to push the per-chart work above 100%.
    """
    charts = source["charts"].sum()
    for code, rate in (("W10", 0.10), ("W13", 0.01), ("W15", 0.20)):
        assert total_for(totals, code) / charts == pytest.approx(rate, rel=1e-9)


# ---------------------------------------------------------------------------
# Shape of the output
# ---------------------------------------------------------------------------


def test_output_covers_history_and_plan_with_no_gap(result, assumptions):
    cal = assumptions["calendar"]
    work = result["work"]
    weeks = np.sort(work["week_index"].unique())
    assert weeks[0] == 0
    assert weeks[-1] == cal["history_weeks"] + cal["plan_weeks"] - 1
    assert len(weeks) == cal["history_weeks"] + cal["plan_weeks"]

    periods = work.groupby("period")["week_index"].agg(["min", "max"])
    assert periods.loc["history", "max"] == cal["history_weeks"] - 1
    assert periods.loc["plan", "min"] == cal["history_weeks"]


def test_every_work_type_and_request_type_is_represented(result, assumptions):
    work = result["work"]
    assert set(work["work_type"]) == {w["code"] for w in assumptions["work_types"]}
    assert set(work["request_type"]) == {rt["code"] for rt in assumptions["request_types"]}
    assert (work["work_items"] > 0).all(), "zero rows should be dropped, not stored"


def test_volumes_are_never_negative(result):
    assert (result["work"]["work_items"] >= 0).all()
    assert (result["spill"]["work_items"] >= 0).all()


def test_no_handle_times_or_automation_have_leaked_in(result):
    """Session 2 is volumes only (spec 3.3); hours arrive in Session 3."""
    forbidden = {"handle_time", "minutes", "hours", "fte", "automation", "deflection"}
    assert not forbidden & set(result["work"].columns)


# ---------------------------------------------------------------------------
# Risk-based QA coverage and the fulfiller self-check (revised 10/22)
# ---------------------------------------------------------------------------


def test_quality_review_covers_the_high_risk_request_types_in_full(totals, source, assumptions):
    """Spec 9: 100% of charts for R5, R7, R8, and every redaction chart.

    Full coverage where an error costs most - a bulk payer audit, or
    anything going to a lawyer or a court - and the redaction charts,
    because that is where a disclosure mistake would actually be made.
    """
    coverage = next(
        w for w in assumptions["work_types"] if w["code"] == "W14"
    )["qa_coverage"]
    assert coverage["full_for_request_types"] == ["R5", "R7", "R8"]
    assert coverage["full_for_work_type"] == "W15"
    assert coverage["sample_rate"] == 0.10

    for code in coverage["full_for_request_types"]:
        assert total_for(totals, "W14", code) == pytest.approx(
            source["charts"][code], rel=1e-9
        ), f"{code} should be reviewed in full"


def test_other_request_types_get_the_redaction_charts_plus_a_sample(
    totals, source, assumptions
):
    """Spec 9: a 10% sample of all other charts, on top of redaction.

    The rate is two populations added, not one number:
        redaction share + (1 - redaction share) x sample
        0.20            + 0.80                  x 0.10   = 0.28
    and the redaction share is read from W15 so the two cannot drift.
    """
    by_code = {w["code"]: w for w in assumptions["work_types"]}
    redaction = by_code["W15"]["rate"]
    sample = by_code["W14"]["qa_coverage"]["sample_rate"]
    expected = redaction + (1 - redaction) * sample
    assert expected == pytest.approx(0.28)

    full = set(by_code["W14"]["qa_coverage"]["full_for_request_types"])
    for code in source["charts"].index:
        if code in full:
            continue
        assert total_for(totals, "W14", code) / source["charts"][code] == pytest.approx(
            expected, rel=1e-9
        ), code


def test_retuning_redaction_retunes_the_qa_that_follows_it(assumptions):
    """The link to W15 has to be live, not decorative.

    If redaction ever covered half the charts, QA would automatically cover
    half plus a tenth of the rest. A hard-coded 0.28 would pass every test
    above and silently miss that.
    """
    import copy

    by_code = {w["code"]: w for w in assumptions["work_types"]}
    codes = [rt["code"] for rt in assumptions["request_types"]]

    rates = work_volume.request_type_rates(by_code["W14"], codes, by_code)
    assert rates[codes.index("R1")] == pytest.approx(0.28)

    changed = copy.deepcopy(by_code)
    changed["W15"]["rate"] = 0.50
    moved = work_volume.request_type_rates(changed["W14"], codes, changed)
    assert moved[codes.index("R1")] == pytest.approx(0.50 + 0.50 * 0.10)
    assert moved[codes.index("R5")] == pytest.approx(1.0), "full coverage is unaffected"


def test_every_chart_gets_exactly_one_fulfiller_self_check(totals, source, assumptions):
    """Spec 9: every chart gets exactly one W24 self-check.

    Not a sample and not a multiple - one per chart, which is what makes a
    10% QA sample defensible on the charts that no longer get a second
    pair of eyes.
    """
    w24 = next(w for w in assumptions["work_types"] if w["code"] == "W24")
    assert w24["basis"] == "per_chart"
    assert w24["rate"] == 1.0
    assert w24["owner"] == "by_request_type"
    assert w24["handle_time_minutes"] == 1
    assert w24["lag"] == "split_50_50", "spec 3.3: lagged like review"

    for code in source["charts"].index:
        assert total_for(totals, "W24", code) == pytest.approx(
            source["charts"][code], rel=1e-9
        ), code
    assert total_for(totals, "W24") == pytest.approx(source["charts"].sum(), rel=1e-9)


def test_quality_review_now_covers_less_than_every_chart(totals, source):
    """The change has to have changed something.

    Review used to be one per chart. If it still were, the coverage rule
    would be inert and every test above would pass on a no-op.
    """
    reviewed = total_for(totals, "W14")
    charts = source["charts"].sum()
    assert reviewed < charts
    assert 0.5 < reviewed / charts < 0.8, f"{reviewed / charts:.1%} of charts reviewed"


def test_rework_follows_the_charts_actually_reviewed(totals, assumptions):
    """W22 stays 3% of W14, so lighter review means lighter rework."""
    rate = next(w for w in assumptions["work_types"] if w["code"] == "W22")["rate"]
    assert rate == 0.03
    assert total_for(totals, "W22") == pytest.approx(total_for(totals, "W14") * rate, rel=1e-9)
