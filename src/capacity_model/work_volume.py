"""From requests to work: weekly volume by site and work type (spec 3.3).

Session 2. (Not in the spec Section 2 file list - the lag logic belongs with
Demand, and putting it in forecast.py would mix two different jobs in one
module. Same judgement call as config.py.)

The WFM idea: one forecast drives everything. A request arriving in week w
does not generate all of its work in week w - retrieval trails it, review
trails retrieval, a status call comes days later - so each work type carries a
lag profile that places its volume in the week it actually lands. Staffing to
the arrival week would build the plan around a peak that has already passed.

    Formula (spec 3.3):
        Work volume (type, site, week) = the sum, over each lag week, of
            requests in that earlier week
            x work per request for the type
            x the type's lag share for that week

Three things decide how much work a request makes:

  PER REQUEST OR PER CHART. Intake happens once per request; retrieval and
  review happen once per CHART. A bulk payer-audit request (R5) is ten
  charts, so it multiplies retrieval, review and release without multiplying
  intake. That is the charts-per-request lever from Section 1 doing its job.

  WHERE THE RECORD LIVES. W8, W9 and W11 partition the charts by medium -
  digital 85%, email archive 4%, onsite 11% split paper 7 / microfiche 2 /
  archive room 2 - so they add to exactly 100% of charts and no chart is
  retrieved twice. W10 (multi-source), W13 (missing record) and W15
  (redaction) are ADDITIONAL passes over charts, not part of that partition.

  WHAT ALREADY HAPPENED. Digitization (W12) exists only because an onsite
  retrieval happened, and rework (W22) only because a quality review failed.
  These chain off another work type's output rather than off requests, so
  they can never be larger than their parent.

Every volume carries the request type it came from, so Session 3 can route it
to the right tier (skills.yaml) without re-deriving the mix.

This module is VOLUMES ONLY. No handle times, no deflection, no automation -
those arrive in Sessions 3 and 5.

Run it:
    .venv\\Scripts\\python -m capacity_model.work_volume
"""

import numpy as np
import pandas as pd

from . import automation, config, digitization, forecast, generate_data

# Work lagged off the final weeks lands past the end of the plan. Rather than
# dropping it - which would quietly destroy work and break the conservation
# check in spec 9 - it is computed on an axis this many weeks longer and
# reported separately. The deepest chain is W23's 4-week spread, so 8 is
# comfortable headroom.
PAD_WEEKS = 8

PER_REQUEST = "per_request"
PER_CHART = "per_chart"
CHAINED = "chained"

NO_MEDIA = ""


# ---------------------------------------------------------------------------
# The source series: requests and charts, history then plan, as one timeline
# ---------------------------------------------------------------------------


def source_volumes(history, forward, assumptions, sites):
    """Requests and charts per week x site x request type, for all 208 weeks.

    History and plan are stitched into ONE continuous timeline, because the
    lag has to flow across the join: work from the last history week lands in
    the first plan week. Treating them as two separate series would leave a
    seam where a week of retrieval simply vanished.

    The plan half splits the forecast by the long-run request mix from
    assumptions.yaml rather than by a forecast of the mix itself. The forecast
    models total requests per site; nothing in it predicts next year's
    week-to-week drift between request types, and inventing one would be
    making up precision we do not have.

    Returns (requests, charts, weeks, site_codes, type_codes) with the arrays
    shaped (weeks, sites, request types).
    """
    cal = assumptions["calendar"]
    request_types = assumptions["request_types"]
    site_codes = [s["code"] for s in sites["sites"]]
    type_codes = [rt["code"] for rt in request_types]
    charts_per_request = np.array([rt["charts_per_request"] for rt in request_types], dtype=float)

    n_history = cal["history_weeks"]
    n_plan = cal["plan_weeks"]
    n_total = n_history + n_plan

    requests = np.zeros((n_total, len(site_codes), len(type_codes)))

    # History: the generator already produced requests per type, drift and all.
    hist = history.pivot_table(
        index="week_index", columns=["site", "request_type"], values="requests", aggfunc="sum"
    )
    for s_i, site in enumerate(site_codes):
        for t_i, rtype in enumerate(type_codes):
            requests[:n_history, s_i, t_i] = hist[(site, rtype)].to_numpy()

    # Plan: the forecast is a site total, split by the spec mix.
    base_shares = np.array([rt["share"] for rt in request_types])
    plan = forward.pivot_table(
        index="week_index", columns="site", values="forecast_requests", aggfunc="sum"
    ).sort_index()

    # A scenario's demand multiplier (spec 7) lands HERE and nowhere else:
    # on the plan forecast, from the scenario's starts week onward. Never on
    # the history, because the forecast is fitted to what actually happened
    # and inflating it would move the trend and the seasonal index - the
    # scenario would then be answering a different question from the one it
    # asks. The baseline multiplier is 1.0 and this block does nothing.
    multiplier = config.scenario_multiplier("demand", assumptions)
    scale = np.ones(n_plan)
    if multiplier != 1.0:
        starts = config.scenario_starts_week(assumptions)
        plan_weeks = np.array(plan.index)
        scale = np.where(plan_weeks >= starts, multiplier, 1.0)

    for s_i, site in enumerate(site_codes):
        totals = plan[site].to_numpy()
        if multiplier != 1.0:
            totals = totals * scale
        requests[n_history:, s_i, :] = totals[:, None] * base_shares[None, :]

    charts = requests * charts_per_request[None, None, :]
    return requests, charts, n_history, n_total, site_codes, type_codes


# ---------------------------------------------------------------------------
# The lag
# ---------------------------------------------------------------------------


def apply_lag(source, profile, n_axis):
    """Spread `source` forward over the weeks in `profile`.

    The profile sums to 1.0, so this moves work between weeks without
    creating or destroying any. That invariant is checked here rather than
    only in the tests, because a silent clip at the end of the axis is
    exactly the bug spec 9 asks us to rule out.
    """
    out = np.zeros((n_axis,) + source.shape[1:])
    for offset, share in enumerate(profile):
        if share == 0.0:
            continue
        end = min(n_axis, offset + source.shape[0])
        if end <= offset:
            continue
        out[offset:end] += source[: end - offset] * share

    expected = source.sum() * float(np.sum(profile))
    if not np.isclose(out.sum(), expected, rtol=1e-9, atol=1e-9):
        raise ValueError(
            f"lag clipped work off the end of the axis: kept {out.sum():.6f} "
            f"of {expected:.6f}. Increase PAD_WEEKS."
        )
    return out


def qa_coverage_rates(work_type, type_codes, by_code):
    """Risk-based QA coverage for W14 (spec Section 1, revised 10/22).

    Quality review stopped being "every chart" on 10/22. It is now full
    coverage where an error costs most - bulk payer audits, and anything
    going to a lawyer or a court - plus every chart that needed redaction,
    because that is where a disclosure mistake would be made. Everything
    else gets a sample.

    For an ordinary request type the rate is therefore not one number but
    two populations added:

        redaction share + (1 - redaction share) x sample rate
        0.20            + 0.80                  x 0.10         = 0.28

    The redaction share is READ from W15 rather than repeated here, so
    retuning redaction automatically retunes the QA that follows it.
    """
    coverage = work_type["qa_coverage"]
    full_types = set(coverage.get("full_for_request_types", []))
    sample_rate = float(coverage["sample_rate"])

    always_reviewed = 0.0
    linked = coverage.get("full_for_work_type")
    if linked is not None:
        parent = by_code[linked]
        if parent["basis"] != work_type["basis"]:
            raise ValueError(
                f"{work_type['code']} links QA coverage to {linked}, which counts "
                f"{parent['basis']} not {work_type['basis']}"
            )
        always_reviewed = float(parent.get("rate", 1.0))
        if not 0.0 <= always_reviewed <= 1.0:
            raise ValueError(f"{linked} rate {always_reviewed} cannot be a chart share")

    sampled = always_reviewed + (1.0 - always_reviewed) * sample_rate
    return np.array(
        [1.0 if code in full_types else sampled for code in type_codes]
    )


def request_type_rates(work_type, type_codes, by_code=None):
    """The per-request-type rate for one work type (spec Section 1).

    Layers, most specific last:
      rate            applies to every request type
      request_types   a whitelist - anything not listed generates nothing
      rate_overrides  a named rate for one type
      qa_coverage     the risk-based rule W14 uses (see above)

    So W7 is "8% of others, but all of R8", and W17 is "R7 and R8 in full,
    plus 20% of R4" - both expressed as data rather than as a branch in code.
    """
    if "qa_coverage" in work_type:
        if by_code is None:
            raise ValueError(f"{work_type['code']} needs the work-type table to resolve")
        return qa_coverage_rates(work_type, type_codes, by_code)

    base = float(work_type.get("rate", 1.0))
    allowed = work_type.get("request_types")
    overrides = work_type.get("rate_overrides", {})

    rates = np.zeros(len(type_codes))
    for i, code in enumerate(type_codes):
        if allowed is not None and code not in allowed:
            continue
        rates[i] = float(overrides.get(code, base))
    return rates


# ---------------------------------------------------------------------------
# Building the volumes
# ---------------------------------------------------------------------------


def media_share_arrays(assumptions, sites, n_weeks, site_codes):
    """Each medium's share of charts, as an array over (week, site, 1).

    The baseline's media mix is FIXED: 85% digital, 11% onsite, and the
    same every week of the plan. Reactive digitization (W12) does the work
    of converting a record but never feeds the result back into this
    table, so the non-digital share a leader is quoted does not move all
    year. That is a deliberate baseline property, not an oversight - see
    the record_media note in assumptions.yaml.

    Coordinated digitization (spec 7) is the scenario that makes it move.
    Each site's onsite shares are multiplied by what is LEFT of its onsite
    request share, and every point taken off them is added to digital:
    a converted chart still gets fetched, it is just not in a cabinet.

    Returns None when the scenario is off, so the caller keeps its scalar
    arithmetic and the baseline stays byte-identical.
    """
    if not digitization.enabled(assumptions):
        return None

    onsite = set(digitization.onsite_media(assumptions))
    digital = digitization.digital_media(assumptions)
    base = {m["code"]: float(m["share"]) for m in assumptions["record_media"]}
    factors = digitization.onsite_factor_by_week(assumptions, sites)

    # Shaped to the SOURCE weeks, not the padded axis: this multiplies the
    # chart basis before the lag is applied, and apply_lag does the
    # extending afterwards.
    #
    # Default 1.0, so every history week is untouched and nothing before
    # the starts week moves - which is what Section 9's "weeks 1 to 12
    # match baseline" check tests.
    left = np.ones((n_weeks, len(site_codes), 1), dtype=float)
    for (week, site), factor in factors.items():
        if 0 <= week < n_weeks and site in site_codes:
            left[week, site_codes.index(site), 0] = factor

    arrays = {}
    for code, share in base.items():
        if code in onsite:
            arrays[code] = share * left
        else:
            arrays[code] = np.full((n_weeks, len(site_codes), 1), share)
    # Everything taken off the onsite media lands on digital, so the three
    # retrieval work types still partition the charts exactly once (the
    # conservation check in Section 9).
    freed = sum(base[code] for code in onsite) * (1.0 - left)
    arrays[digital] = arrays[digital] + freed
    return arrays


def build(assumptions=None, sites=None, history=None, forward=None):
    """Compute weekly work volume for every work type (spec 3.3).

    Returns a dict with:
      work          in-horizon rows, week 0 to the last plan week
      spill         work that lands past the last plan week, kept not dropped
      conservation  per work type: what the lag should have produced, what
                    landed in horizon, what spilled, and the difference
    """
    assumptions = assumptions or config.load_assumptions()
    sites = sites or config.load_sites()
    if history is None:
        history = generate_data.generate_history(assumptions, sites)
    if forward is None:
        forward, chosen, _both, _hist = forecast.run(assumptions, sites)

    requests, charts, n_history, n_total, site_codes, type_codes = source_volumes(
        history, forward, assumptions, sites
    )
    n_axis = n_total + PAD_WEEKS

    media_share = {m["code"]: float(m["share"]) for m in assumptions["record_media"]}
    media_arrays = media_share_arrays(assumptions, sites, n_total, site_codes)
    profiles = assumptions["lag_profiles"]
    work_types = {w["code"]: w for w in assumptions["work_types"]}

    basis_arrays = {PER_REQUEST: requests, PER_CHART: charts}

    computed = {}      # code -> {medium: array (n_axis, S, T)}
    expected = {}      # (code, medium) -> expected total
    pending = list(assumptions["work_types"])

    # Non-chained rows first, then anything chaining off them. Looping until
    # nothing new resolves handles a chain of any depth without assuming one.
    while pending:
        progressed = False
        still_pending = []
        for wt in pending:
            code = wt["code"]
            profile = profiles[wt["lag"]]

            if wt["basis"] == CHAINED:
                parent = wt.get("follows")
                if parent not in computed:
                    still_pending.append(wt)
                    continue
                rate = float(wt.get("rate", 1.0))
                computed[code] = {}
                for medium, parent_series in computed[parent].items():
                    # The parent's FULL series is the source, spill included,
                    # so a digitization item is never lost just because its
                    # retrieval happened in the last week of the plan.
                    computed[code][medium] = apply_lag(parent_series * rate, profile, n_axis)
                    expected[(code, medium)] = parent_series.sum() * rate
                progressed = True
                continue

            basis = basis_arrays[wt["basis"]]
            rates = request_type_rates(wt, type_codes, work_types)
            media = wt.get("media", [NO_MEDIA])

            computed[code] = {}
            for medium in media:
                if medium == NO_MEDIA:
                    share = 1.0
                elif media_arrays is not None:
                    # Coordinated digitization (spec 7): the share moves
                    # week by week and site by site, so it is an array.
                    share = media_arrays[medium]
                else:
                    share = media_share[medium]
                source = basis * rates[None, None, :] * share
                computed[code][medium] = apply_lag(source, profile, n_axis)
                expected[(code, medium)] = source.sum()
            progressed = True

        if not progressed and still_pending:
            missing = {w["code"]: w.get("follows") for w in still_pending}
            raise ValueError(f"work types chain off something that does not exist: {missing}")
        pending = still_pending

    frames, spill_frames, conservation = [], [], []
    week_starts = [
        d.isoformat()
        for d in generate_data.week_start_dates(
            assumptions["calendar"]["history_start"], n_axis
        )
    ]

    for wt in assumptions["work_types"]:
        code = wt["code"]
        rates = request_type_rates(wt, type_codes, work_types)
        if wt["basis"] == CHAINED:
            # A chained row inherits its parent's request-type reach.
            rates = request_type_rates(work_types[wt["follows"]], type_codes, work_types)
        keep = [i for i, r in enumerate(rates) if r > 0]
        kept_types = [type_codes[i] for i in keep]

        for medium, series in computed[code].items():
            trimmed = series[:, :, keep]
            in_horizon = trimmed[:n_total]
            spilled = trimmed[n_total:]

            frames.append(
                _to_rows(in_horizon, 0, week_starts, site_codes, kept_types, wt, medium, n_history)
            )
            if spilled.sum() > 0:
                spill_frames.append(
                    _to_rows(
                        spilled, n_total, week_starts, site_codes, kept_types, wt, medium, n_history
                    )
                )

            conservation.append(
                {
                    "work_type": code,
                    "media": medium,
                    "expected": expected[(code, medium)],
                    "in_horizon": float(in_horizon.sum()),
                    "spill": float(spilled.sum()),
                }
            )

    work = pd.concat(frames, ignore_index=True)
    # Deflection (spec 7.1) removes contacts that never arrive, before
    # anything downstream counts them. A no-op in the baseline.
    work = apply_deflection(work, assumptions)
    work = work[work["work_items"] > 0].reset_index(drop=True)
    spill = (
        pd.concat(spill_frames, ignore_index=True)
        if spill_frames
        else pd.DataFrame(columns=work.columns)
    )
    spill = spill[spill["work_items"] > 0].reset_index(drop=True)

    cons = pd.DataFrame(conservation)
    cons["total"] = cons["in_horizon"] + cons["spill"]
    cons["difference"] = cons["total"] - cons["expected"]

    return {"work": work, "spill": spill, "conservation": cons, "forward": forward}


def _to_rows(series, week_offset, week_starts, site_codes, type_codes, work_type, medium, n_history):
    """Flatten a (weeks, sites, types) block into tidy rows."""
    n_weeks, n_sites, n_types = series.shape
    weeks = np.arange(week_offset, week_offset + n_weeks)
    return pd.DataFrame(
        {
            "week_index": np.repeat(weeks, n_sites * n_types),
            "week_start": np.repeat([week_starts[w] for w in weeks], n_sites * n_types),
            "period": np.where(
                np.repeat(weeks, n_sites * n_types) < n_history, "history", "plan"
            ),
            "site": np.tile(np.repeat(site_codes, n_types), n_weeks),
            "work_type": work_type["code"],
            "stage": work_type["stage"],
            "owner": work_type["owner"],
            "request_type": np.tile(type_codes, n_weeks * n_sites),
            "media": medium,
            "work_items": series.ravel(),
        }
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def apply_deflection(table, assumptions=None):
    """Remove the contacts that never arrive (spec 7.1).

    Applied to the finished work table rather than inside the lag, because
    deflection is about CONTACTS and the contact types it reaches - status
    calls, emails and portal messages - carry no lag of their own. A call
    that is never made is simply not there.

    Only from the scenario's starts week, and only in the plan: the
    history is what happened, and no AI was answering the phone in 2024.
    """
    assumptions = assumptions or config.load_assumptions()
    if not automation.deflection_enabled(assumptions):
        return table

    share = automation.deflected_share(assumptions)
    codes = set(automation.deflection_work_types(assumptions))
    from_week = automation.deflection_from_week(assumptions)

    hit = table["work_type"].isin(codes) & (table["week_index"] >= from_week)
    if not hit.any():
        return table
    out = table.copy()
    out.loc[hit, "work_items"] = out.loc[hit, "work_items"] * (1.0 - share)
    return out


def write_outputs(result, assumptions=None):
    assumptions = assumptions or config.load_assumptions()
    paths = assumptions["paths"]
    out_dir = config.resolve_path(paths["synthetic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    work_path = out_dir / paths["work_volume_csv"]
    result["work"].to_csv(work_path, index=False, float_format="%.6f")

    # The spill report is the other half of the conservation check: work that
    # the lag pushed past the last plan week is written down here rather than
    # silently dropped, so the totals still tie out (spec 9, Session 2).
    spill_path = out_dir / paths["work_volume_spill_csv"]
    result["conservation"].to_csv(spill_path, index=False, float_format="%.6f")
    return work_path, spill_path


def main():
    assumptions, sites, _skills = config.load_all()
    result = build(assumptions, sites)
    work_path, spill_path = write_outputs(result, assumptions)

    work, cons = result["work"], result["conservation"]
    cal = assumptions["calendar"]
    first_plan_week = cal["history_weeks"]

    print(f"Wrote {len(work):,} work-volume rows to {work_path}")
    print(f"Wrote the conservation and spill report to {spill_path}")
    print(
        f"  worst conservation difference: {cons['difference'].abs().max():.3e} "
        f"work items (should be ~0)"
    )
    print(f"  total work spilling past the last plan week: {cons['spill'].sum():,.1f} items")

    plan_week = work[work.week_index == first_plan_week]
    monday = plan_week["week_start"].iloc[0]
    print(f"\nFirst plan week ({monday}), total work items by stage, all 6 sites:")
    by_stage = plan_week.groupby("stage")["work_items"].sum().sort_values(ascending=False)
    for stage, items in by_stage.items():
        print(f"  {stage:<12} {items:10,.0f}")
    print(f"  {'TOTAL':<12} {by_stage.sum():10,.0f}")

    requests = result["forward"]
    first = requests[requests.week_index == first_plan_week]["forecast_requests"].sum()
    print(f"\n  driven by {first:,.0f} forecast requests that week")


if __name__ == "__main__":
    main()
