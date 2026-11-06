"""Spec Section 9, Sessions 6 and 7: the last four tabs tie to the model.

Task 6c. Next Dollar, Coverage Events, Non-Digital Tracker, Forecast -
which completes spec Section 8's ten.

These four are harder to tie out than 6b's, and for one reason: three of
them are NOT the baseline run. Next Dollar is priced at the AI 100% run's
volumes, the Forecast is upstream of every scenario, and only Coverage
Events and the Non-Digital Tracker read the baseline. So the first thing
each test here establishes is WHICH run a figure came from, because a
plausible number from the wrong run is the failure mode this whole file
exists to catch - and it is the one that actually happened on this
project. Found 11/03 in Session 5: the Next Dollar ranking was once
re-priced by handing it a required-hours table from a different scenario,
and the answer looked entirely reasonable.

WHAT EACH TAB CAN GET WRONG, AND THE TEST FOR IT

    NEXT DOLLAR can be priced at the wrong volumes. Pinned to the figures
      spec Section 7 records for the AI 100% run - W8 +$1.58, W5 +$0.378,
      W14 -$0.069 - and checked to be DIFFERENT from the baseline's own
      pricing, so the test cannot pass on a tab that quietly used the
      baseline.

    COVERAGE EVENTS can lose the F-numbering, which is the whole point of
      the schema, or can present a suggested cause as a confirmed one.
      Every spec 6.7 column is checked present with its field number, and
      the summary block is reconciled to the event log it summarises.

    THE NON-DIGITAL TRACKER can draw a falling line. The baseline's
      non-digital share is FLAT, and that is spec 7's finding rather than
      a modelling shortcut - so the test asserts it is flat and asserts
      the tab says it is flat.

    THE FORECAST can report a MAPE from the weeks it was fitted to. The
      holdout is 13 weeks the methods never saw, and the MAPE on the tab
      is recomputed here from the predictions beside it, which is the only
      way to know the number is the holdout's and not the training set's.
"""

import copy
import math

import pytest
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from capacity_model import config, coverage, digitization, export, plan, scenarios

from test_export import evaluate


CENT = 0.005


@pytest.fixture(scope="module")
def baseline_assumptions():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def result(baseline_assumptions, sites):
    return scenarios.run(None, baseline_assumptions, sites)


@pytest.fixture(scope="module")
def run(result):
    return result["runs"][export.BASELINE_KEY]


@pytest.fixture(scope="module")
def book(result, baseline_assumptions, tmp_path_factory):
    path = tmp_path_factory.mktemp("workbook_6c") / "wb.xlsx"
    export.build(result, baseline_assumptions).save(path)
    return load_workbook(path)


@pytest.fixture(scope="module")
def tables(run, baseline_assumptions, sites):
    scores, weeks, facts = export.forecast_tables(baseline_assumptions, sites)
    return {
        "Next Dollar": export.next_dollar_table(baseline_assumptions, sites),
        "Coverage Events": export.coverage_events_table(run, baseline_assumptions),
        "coverage by type": export.by_event_type_table(run, baseline_assumptions),
        "coverage summary": export.coverage_summary_table(run, baseline_assumptions),
        "Non-Digital Tracker": export.non_digital_table(run, baseline_assumptions),
        "digitization backlog": export.digitization_backlog_table(
            run, baseline_assumptions
        ),
        "non-digital share": export.non_digital_share_table(
            run, baseline_assumptions
        ),
        "forecast scores": scores,
        "forecast weeks": weeks,
        "forecast facts": facts,
    }


TAB_COLUMNS = {
    "Next Dollar": export.NEXT_DOLLAR_COLUMNS,
    "Coverage Events": export.COVERAGE_EVENT_COLUMNS,
    "Non-Digital Tracker": export.NON_DIGITAL_COLUMNS,
}


def written_rows(book, name, columns, table, header_row=export.DETAIL_HEADER_ROW):
    sheet = book[name]
    first = header_row + 1
    out = []
    for offset in range(len(table)):
        row = first + offset
        out.append(
            {
                header: sheet.cell(row=row, column=index).value
                for index, (header, _s, _f, _w, _t) in enumerate(columns, start=1)
            }
        )
    return out


def find_header(sheet, first_header, after):
    """Where a second block starts on a tab that carries more than one."""
    for row in range(after, sheet.max_row + 1):
        if sheet[f"A{row}"].value == first_header:
            return row
    raise AssertionError(f"no block header {first_header!r} after row {after}")


# ---------------------------------------------------------------------------
# Layer 1: the writer put the builder's frame in the right cells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TAB_COLUMNS))
def test_every_written_cell_is_the_builder_s_own_value(book, tables, name):
    """Cell by cell, every row, to the cent."""
    columns = TAB_COLUMNS[name]
    table = tables[name]
    rows = written_rows(book, name, columns, table)
    assert len(rows) == len(table) > 0, name

    for position, (record, written) in enumerate(
        zip(table.to_dict("records"), rows)
    ):
        for header, source, _fmt, _width, _total in columns:
            if source is None:
                continue
            expected = record.get(source)
            got = written[header]
            if expected is None or (
                isinstance(expected, float) and math.isnan(expected)
            ):
                assert got is None, (name, position, header, got)
            elif isinstance(expected, (str, bool)):
                assert got == expected, (name, position, header)
            elif isinstance(expected, (int, float)):
                assert got == pytest.approx(float(expected), abs=CENT), (
                    name, position, header, got, expected,
                )
            else:
                assert got == str(expected), (name, position, header)


def block_headers(sheet, first_header, after):
    """Every row at or after `after` whose column A holds this header.

    Two of these tabs carry blocks whose first column header is the same
    string, so the blocks are found in order rather than by searching for
    a unique label - and in order is what a reader sees anyway.
    """
    return [
        row
        for row in range(after, sheet.max_row + 1)
        if sheet[f"A{row}"].value == first_header
    ]


def test_the_second_blocks_are_written_too(book, tables):
    """The blocks below the first table, which are the easy ones to lose.

    Each sits at a row offset nothing else depends on, so a change above
    it moves it silently - which makes them exactly the blocks that stop
    being written without anybody noticing.
    """
    checks = (
        ("Coverage Events", 0, export.BY_EVENT_TYPE_COLUMNS, "coverage by type",
         "Event type (F2)"),
        ("Coverage Events", 0, export.COVERAGE_SUMMARY_COLUMNS,
         "coverage summary", "Cause (F13)"),
        ("Non-Digital Tracker", 0, export.DIGITIZATION_BACKLOG_COLUMNS,
         "digitization backlog", "Plan week"),
        ("Non-Digital Tracker", 1, export.NON_DIGITAL_SHARE_COLUMNS,
         "non-digital share", "Plan week"),
    )
    for name, which, columns, key, first_header in checks:
        sheet = book[name]
        after = export.DETAIL_HEADER_ROW + 1 + len(tables[name]) + 1
        found = block_headers(sheet, first_header, after)
        assert len(found) > which, (key, found)
        header = found[which]

        table = tables[key]
        rows = written_rows(book, name, columns, table, header)
        assert len(rows) == len(table) > 0, key
        for record, written in zip(table.to_dict("records"), rows):
            for label, source, _f, _w, _t in columns:
                expected = record[source]
                if isinstance(expected, (str, bool)):
                    assert written[label] == expected, (key, label)
                elif expected is None or (
                    isinstance(expected, float) and math.isnan(expected)
                ):
                    assert written[label] is None, (key, label)
                else:
                    assert written[label] == pytest.approx(
                        float(expected), abs=CENT
                    ), (key, label)


# ---------------------------------------------------------------------------
# Next Dollar: the right volumes, and the ranking it implies
# ---------------------------------------------------------------------------


def test_next_dollar_is_priced_at_the_ai_100_run_not_the_baseline(
    tables, baseline_assumptions, sites
):
    """The failure that actually happened, in test form.

    The ranking is not a property of a work type: setup is fixed and the
    saving scales with items, so the per-item figures - and the order -
    depend on the volumes the machine actually sees. Pricing this tab at
    the baseline's volumes would produce an entirely plausible table that
    answered a different question.
    """
    table = tables["Next Dollar"].set_index("work_type")

    # Spec Section 7's recorded figures for the AI 100% run, re-measured
    # 11/03. If the tab were priced on another run these would all move.
    for code, saving in (
        ("W8", 1.580785),
        ("W5", 0.377731),
        ("W14", -0.069239),
        ("W1", -0.460811),
        ("W12", -3.198602),
        ("W7", -5.606526),
    ):
        assert table.loc[code, "savings_per_item_after_fixed"] == pytest.approx(
            saving, abs=5e-6
        ), code

    # And the proof that this is not the baseline's own pricing is
    # stronger than a difference: the baseline runs NO machine, so it
    # cannot price a fixed cost per item at all. Every row would be
    # blank, and the tab would be empty where it matters most.
    baseline_table = plan.next_dollar(baseline_assumptions, sites)
    assert baseline_table["fixed_cost_per_item"].isna().all()
    assert int(table["fixed_cost_per_item"].notna().sum()) == 19


def test_a_dimension_that_does_not_apply_says_so_rather_than_being_blank(
    tables, book
):
    """A blank cell cannot be told apart from an exporter that failed.

    W5's handle time does not vary by medium, so the model carries an
    empty medium for it. Written straight through, that becomes a blank
    cell on the tab - the same trap the Assumptions tab hit in 6b, and
    caught here by the same cell-by-cell tie-out.
    """
    table = tables["Next Dollar"].set_index("work_type")
    assert table.loc["W5", "media"] == export.NOT_APPLICABLE
    assert table.loc["W8", "media"] == "digital"
    assert table.loc["W12", "media"] == "paper"

    # No row is left blank in that column.
    assert not any(
        not str(medium or "").strip() for medium in tables["Next Dollar"]["media"]
    )

    # And the tab says what n/a means, so it is not read as missing data.
    sheet = book["Next Dollar"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert "does not vary by medium" in text
    assert "not a missing value" in text


def test_only_the_two_profitable_rows_rank_positive(tables):
    """The finding the AI targeted scenario was picked on.

    Two rows of nineteen clear their own setup fee, and W14 is the near
    miss - seven cents an item on the second-highest volume in the
    network. If a third row went positive, the targeted list would be
    wrong and so would the Summary's AI targeted finding.
    """
    table = tables["Next Dollar"]
    positive = table[table["savings_per_item_after_fixed"] > 0]
    assert set(positive["work_type"]) == {"W5", "W8"}

    ranked = table[table["savings_per_dollar"].notna()]
    assert len(ranked) == 19
    # Ranked best first, and the rank column agrees with the order.
    assert list(ranked["rank"]) == list(range(1, 20))
    assert list(ranked["work_type"])[:2] == ["W8", "W5"]
    assert ranked["savings_per_dollar"].is_monotonic_decreasing

    near_miss = table.set_index("work_type").loc["W14"]
    assert -0.08 < near_miss["savings_per_item_after_fixed"] < -0.06


def test_the_rows_automation_is_forbidden_from_touching_are_not_ranked(
    tables, book
):
    """W11 and W24 are reported, not ranked, and their cells are BLANK.

    Spec Section 1 pins them to 0%, so the machine never touches an item
    and there is no cost per item to divide by. Calling it zero once made
    them look sixty times more profitable than anything real and put them
    at the top of this ranking. Not measurable is not free.
    """
    table = tables["Next Dollar"].set_index("work_type")
    for code in ("W11", "W24"):
        assert code in table.index
        assert math.isnan(float(table.loc[code, "fixed_cost_per_item"])), code
        assert math.isnan(float(table.loc[code, "savings_per_dollar"])), code
        assert table.loc[code, "rank"] != table.loc[code, "rank"]  # NaN

    # And they sort last, so a reader's eye never lands on them first.
    assert list(tables["Next Dollar"]["work_type"])[-2:] == ["W11", "W24"]

    # The written cells really are empty, not zero.
    columns = export.NEXT_DOLLAR_COLUMNS
    rows = written_rows(book, "Next Dollar", columns, tables["Next Dollar"])
    for row in rows:
        if row["Work type"] in ("W11", "W24"):
            assert row["Saving per dollar spent"] is None
            assert row["Fixed cost per item"] is None
            assert row["Plan-year saving on the items automated"] is None
            assert row["Rank"] is None


def test_the_plan_year_saving_formula_is_per_item_times_volume(book, tables):
    """The one product the exporter writes, checked on every ranked row.

    It is the figure a leader would otherwise reach for a calculator to
    get, which is exactly why it belongs in the formula bar rather than
    as a written number: the derivation is the point.
    """
    columns = export.NEXT_DOLLAR_COLUMNS
    headers = [header for header, *_rest in columns]
    sheet = book["Next Dollar"]
    saving = get_column_letter(
        headers.index("Plan-year saving on the items automated") + 1
    )
    first = export.DETAIL_HEADER_ROW + 1

    checked = 0
    for offset, record in enumerate(tables["Next Dollar"].to_dict("records")):
        row = first + offset
        formula = sheet[f"{saving}{row}"].value
        per_item = record["savings_per_item_after_fixed"]
        if per_item != per_item:  # NaN: an unranked row
            assert formula is None
            continue
        assert isinstance(formula, str) and formula.startswith("=")
        expected = per_item * record["automated_items"]
        assert evaluate(book, "Next Dollar", formula) == pytest.approx(
            expected, abs=CENT
        ), record["work_type"]
        checked += 1
    assert checked == 19


def test_the_saving_is_on_the_items_a_machine_actually_did(tables, result):
    """The defect Siri caught reading the tab, in test form.

    The total row used to multiply each per-item saving by every item of
    that work type that ARRIVES, which describes a machine doing 100% of
    the work. At spec 7.1's defaults it does about 8% of the eligible
    simple items. The old figure was -$1,459,131 against an honest
    -$87,770: seventeen times too large, and not conservative in a safe
    direction either, because it scaled up the losing rows as well as the
    two winning ones.
    """
    table = tables["Next Dollar"]
    automated = result["runs"]["ai_and_automation"]["prepared"]["automated"]
    assert automated is not None and len(automated)

    # The volumes are the run's own, not a share of the arriving items.
    from capacity_model import supply

    assumptions = result["runs"]["ai_and_automation"]["assumptions"]
    plan_weeks, _mondays = supply.plan_window(assumptions)
    rows = automated[automated["week_index"].isin(plan_weeks)]
    by_code = rows.groupby("work_type")["automated_items"].sum()

    for record in table.to_dict("records"):
        expected = float(by_code.get(record["work_type"], 0.0))
        assert record["automated_items"] == pytest.approx(expected, abs=1e-6), (
            record["work_type"]
        )

    assert table["automated_items"].sum() == pytest.approx(
        float(rows["automated_items"].sum()), abs=1e-6
    )

    # And it really is a small share of what arrives, so the two figures
    # are nowhere near each other and the fix was not cosmetic.
    arriving = float(table["plan_year_items"].sum())
    did = float(table["automated_items"].sum())
    assert did < 0.10 * arriving

    on_automated = float(table["saving_on_automated_items"].sum())
    on_everything = float(
        (table["savings_per_item_after_fixed"] * table["plan_year_items"]).sum()
    )
    assert on_automated == pytest.approx(-87769.69, abs=0.01)
    assert on_everything < 10 * on_automated  # both negative; the old one was far worse

    # W11 and W24 are pinned to 0%, so they automated nothing at all.
    pinned = table.set_index("work_type")
    for code in ("W11", "W24"):
        assert pinned.loc[code, "automated_items"] == 0.0


def test_the_machine_side_reconciles_to_the_run_s_own_invoice(tables, result):
    """To the cent, through two completely different routes.

    The tab reaches the machine's cost as a per-item fee plus a fixed
    cost per item, each multiplied by the items that work type automated.
    The scenario reaches it by adding up the automation plan's own weekly
    item, setup and platform lines. If those agree there is no arithmetic
    left to doubt on the machine half of the ranking.
    """
    table = tables["Next Dollar"].copy()
    priced = table[table["automated_items"] > 0]

    fee = float((priced["automated_item_cost"] * priced["automated_items"]).sum())
    fixed = float((priced["fixed_cost_per_item"] * priced["automated_items"]).sum())

    run = result["runs"]["ai_and_automation"]
    spend = scenarios.machine_spend(run)
    assert fee == pytest.approx(spend["item_cost"], abs=CENT)
    assert fixed == pytest.approx(
        spend["setup_cost"] + spend["platform_cost"], abs=CENT
    )
    assert fee + fixed == pytest.approx(spend["total"], abs=CENT)

    # Setup dominates, which is the whole finding of the AI scenarios.
    assert fixed > 5 * fee


def test_the_total_does_not_reconcile_to_the_labour_saving_and_the_gap_is_named(
    tables, result, book
):
    """The reconciliation Siri expected, measured - and it does NOT hold.

    Asked for: the total should reconcile to the AI 100% run's labour
    saving less its machine cost. It does not. The run's figure is
    -$141,312 and the tab's is -$87,770, a gap of $53,543 on a $141,000
    number - 38%, far too large to be rounding.

    The gap is exactly identifiable, and it is this project's recurring
    finding rather than an error:

        gap = the hours value this table frees
              - the exception and QA labour the machine hands back
              - the labour saving the payroll actually banked

    The table prices every freed hour at its loaded rate. The run only
    banks a saving where the hour cancels an overtime hour or a hiring
    class, and on a held roster most of them do not. So this test pins
    the identity to the cent rather than asserting a reconciliation that
    is not true, and checks the tab says so in words - because a reader
    comparing this total to Capacity Scenarios will otherwise think one
    of the two tabs is wrong.
    """
    table = tables["Next Dollar"]
    priced = table[table["automated_items"] > 0]

    ai = result["runs"]["ai_and_automation"]
    baseline = result["runs"][export.BASELINE_KEY]
    ai_summary = scenarios.summarise(ai, ai["assumptions"], ai["sites"])
    base_summary = scenarios.summarise(
        baseline, baseline["assumptions"], baseline["sites"]
    )

    labour_saving = base_summary["labour_cost"] - ai_summary["labour_cost"]
    machine_cost = ai_summary["machine_cost"]
    run_figure = labour_saving - machine_cost
    tab_figure = float(priced["saving_on_automated_items"].sum())

    # It does not reconcile, and the shortfall is material.
    assert run_figure == pytest.approx(-141312.31, abs=0.01)
    assert tab_figure == pytest.approx(-87769.69, abs=0.01)
    gap = tab_figure - run_figure
    assert gap > 0.25 * abs(run_figure), "the gap is too small to need a note"

    # But the gap is exactly accounted for, to the cent.
    def total(column):
        return float((priced[column] * priced["automated_items"]).sum())

    human = total("human_cost_per_item")
    exception = total("automated_exception_cost")
    qa = total("automated_qa_cost")
    assert gap == pytest.approx(human - exception - qa - labour_saving, abs=CENT)

    # Which is to say: the value of the hours freed, net of what the
    # machine hands back, against what the payroll actually released.
    freed = human - exception - qa
    assert freed > labour_saving
    assert freed - labour_saving == pytest.approx(gap, abs=CENT)

    # And the human side of the table ties to the required hours the run
    # frees, priced at each role's loaded rate - so the discrepancy sits
    # entirely between REQUIRED HOURS and PAYROLL, which is the held
    # roster and nothing else.
    hours_value = _required_hours_value(baseline, ai)
    assert human == pytest.approx(hours_value, abs=0.05)

    sheet = book["Next Dollar"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert "DOES NOT RECONCILE TO THE RUN'S LABOUR SAVING" in text.upper()
    assert "ITEMS ACTUALLY AUTOMATED" in text.upper()


def _required_hours_value(baseline, ai):
    """Hours the AI run frees, priced at each role's loaded rate.

    The other end of the per-item human figure: if the two agree, the
    table's human side is the model's own required-hours reduction and
    not a second computation of it.
    """
    from capacity_model import plan as plan_module, supply

    LOOP_KINDS = export.LOOP_KINDS

    assumptions = baseline["assumptions"]
    plan_weeks, _mondays = supply.plan_window(assumptions)

    def by_role(run):
        hours = run["prepared"]["required"]
        hours = hours[hours["week_index"].isin(plan_weeks)]
        # The human rows only. An automation run's required table also
        # carries the hours the machine HANDS BACK under their own work
        # kind (spec 5.5), and those are not hours it freed.
        if "work_kind" in hours.columns:
            hours = hours[~hours["work_kind"].isin(LOOP_KINDS)]
        return hours.groupby("role")["required_hours"].sum()

    before, after = by_role(baseline), by_role(ai)
    value = 0.0
    for role in before.index:
        saved = float(before.get(role, 0.0)) - float(after.get(role, 0.0))
        value += saved * plan_module.productive_hour_cost(role, assumptions)
    return value


def test_the_tab_says_a_freed_hour_has_to_be_on_the_margin(book):
    """Siri's note, and the most important sentence on the tab.

    The table prices a human hour at its loaded rate, which ASSUMES the
    hour can be released. On a held roster it often cannot, and then a
    positive per-item saving is worth exactly nothing. W14 at +15% volume
    is the worked example, and the note carries it.
    """
    sheet = book["Next Dollar"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert "ONLY MONEY IF THE HOUR IT FREES IS ON THE MARGIN" in text.upper()
    assert "W14" in text


def test_the_tab_header_names_the_run_it_is_priced_at(book):
    """And never says "Baseline run", which is what it said at first.

    The shared detail-tab writer labels its source line "Baseline run",
    which is right for the four tabs in 6b and wrong here: the baseline
    runs no machine and could not price a single row of this table. A
    header contradicting the notes three rows below it would be the one
    mistake this tab cannot afford, because the basis IS the finding -
    the ranking changes with volume.
    """
    sheet = book["Next Dollar"]
    source = str(sheet[f"A{export.DETAIL_SOURCE_ROW}"].value)
    assert "Baseline run" not in source
    assert "AI and automation" in source
    assert "full program" in source

    # The other 6b tabs still say what they are.
    assert "Baseline run" in str(
        book["Capacity Plan"][f"A{export.DETAIL_SOURCE_ROW}"].value
    )


# ---------------------------------------------------------------------------
# Coverage Events: the schema, and the summary that has to reconcile
# ---------------------------------------------------------------------------


def test_every_spec_six_seven_field_is_on_the_tab_with_its_number(book):
    """F1 to F24, because the numbering IS the schema.

    A reader comparing this tab to spec 6.7 should not have to count
    columns, and a field dropped from the log should show up here as a
    missing header rather than as a column nobody noticed had gone.
    """
    sheet = book["Coverage Events"]
    headers = [
        sheet.cell(row=export.DETAIL_HEADER_ROW, column=index).value
        for index in range(1, len(export.COVERAGE_EVENT_COLUMNS) + 1)
    ]
    joined = " | ".join(str(header) for header in headers)

    fields = {
        field
        for field, _columns, _types, _grain in coverage.MODEL_FIELDS
    } | {field for field, _columns in coverage.PERSON_FIELDS}
    for field in fields:
        assert f"{field} " in joined, field

    # Every logged column appears exactly once.
    sources = [source for _h, source, _f, _w, _t in export.COVERAGE_EVENT_COLUMNS]
    for column in coverage.EVENT_COLUMNS:
        if column == "week_index":
            continue
        assert sources.count(column) == 1, column


def test_the_tab_says_which_fields_a_person_fills(book):
    """F13 is a suggestion; F15, F23 and F24 belong to a person.

    Presenting a suggested cause as a confirmed one is how an operation
    stops trusting its own log after the first argument about a week
    nobody remembers.
    """
    sheet = book["Coverage Events"]
    headers = " | ".join(
        str(sheet.cell(row=export.DETAIL_HEADER_ROW, column=index).value)
        for index in range(1, len(export.COVERAGE_EVENT_COLUMNS) + 1)
    )
    assert "F13 Cause (model suggests)" in headers
    for field in ("F15", "F23", "F24"):
        assert f"{field} " in headers
        position = headers.index(f"{field} ")
        assert "(person)" in headers[position:position + 60], field


def test_the_summary_blocks_reconcile_to_the_event_log(tables, run):
    """Three routes to the same events, and they have to agree.

    The log counts 1,284 events one at a time; one block groups them by
    event type and another by cause and action. If the three disagree,
    one of them is describing a year that did not happen.
    """
    events = tables["Coverage Events"]
    by_type = tables["coverage by type"]
    summary = tables["coverage summary"]

    assert len(events) == len(run["events"])
    assert by_type["events"].sum() == len(events)
    assert summary["events"].sum() == len(events)

    for column in ("hours_short_before_cover", "hours_covered",
                   "hours_uncovered", "overtime_cost", "travel_cost"):
        assert by_type[column].sum() == pytest.approx(
            events[column].sum(), abs=CENT
        ), column
        assert summary[column].sum() == pytest.approx(
            events[column].sum(), abs=CENT
        ), column

    pairs = set(
        zip(events["cause"], events["planned_or_unplanned"], events["action_taken"])
    )
    assert set(
        zip(summary["cause"], summary["planned_or_unplanned"],
            summary["action_taken"])
    ) == pairs


def test_the_hours_columns_are_never_totalled_down_the_log(book, tables):
    """The defect this tab shipped with for an afternoon, in test form.

    The SAME uncovered hour is logged twice - once on the E2 overtime
    event as what was still short after it, and again on the E3 that
    accepted the gap. A total row summing the column gives 839.9 hours
    against a true 420.0: a plausible, quotable, wrong number on the one
    tab a leader reads to find out how much work did not get done.

    So the log totals only what belongs to exactly one event type, and
    the hours are totalled in the by-event-type block instead.
    """
    events = tables["Coverage Events"]
    doubled = float(events["hours_uncovered"].sum())
    true = float(
        events[events["event_type"] == export.ACCEPTED_GAP][
            "hours_uncovered"
        ].sum()
    )
    # The double count is real, which is what makes the rule necessary.
    assert doubled == pytest.approx(2 * true, abs=1e-6)

    totalled = {
        header for header, _s, _f, _w, is_total
        in export.COVERAGE_EVENT_COLUMNS if is_total
    }
    for header in totalled:
        assert "Hours" not in header, header
    # Travel and overtime belong to one event type each, so they do add.
    assert any("Travel cost" in header for header in totalled)
    assert any("Overtime cost" in header for header in totalled)

    sheet = book["Coverage Events"]
    total_row = export.DETAIL_HEADER_ROW + 1 + len(events) + 1
    headers = [header for header, *_rest in export.COVERAGE_EVENT_COLUMNS]
    for header in headers:
        if "Hours" in header:
            letter = get_column_letter(headers.index(header) + 1)
            assert sheet[f"{letter}{total_row}"].value is None, header


def test_the_tab_warns_about_the_double_count_with_measured_figures(book, run):
    """And it says so in words, with both figures measured rather than typed.

    A note reading "839.9 against a true 420.0" would be two numbers in
    the logic, stale the first time a default moved. They are computed
    from the log the note sits under.
    """
    note = export.coverage_notes(run)[-1]
    assert "DO NOT ADD DOWN THE LOG" in note
    events = run["events"]
    doubled = float(events["hours_uncovered"].sum())
    true = float(
        events[events["event_type"] == export.ACCEPTED_GAP][
            "hours_uncovered"
        ].sum()
    )
    assert f"{doubled:,.1f}" in note
    assert f"{true:,.1f}" in note

    sheet = book["Coverage Events"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert "DO NOT ADD DOWN THE LOG" in text


def test_an_accepted_gap_leaves_hours_uncovered_and_a_borrow_does_not(tables):
    """The log has to distinguish what was fixed from what was absorbed.

    An E3 accepted gap is work that did not get done. An E1 borrow is work
    that got done by somebody who drove there. A log that reported them
    the same way would make an operation look like it had no uncovered
    hours at all.
    """
    events = tables["Coverage Events"]
    by_type = events.groupby("event_type")["hours_uncovered"].sum()

    assert by_type.get("E3", 0.0) > 0
    assert by_type.get("E1", 0.0) == pytest.approx(0.0, abs=1e-9)

    borrows = events[events["event_type"] == "E1"]
    assert len(borrows) > 0
    # Somebody drove: a borrow carries travel hours and a mileage cost.
    assert (borrows["travel_hours"] > 0).all()
    assert (borrows["travel_cost"] > 0).all()
    # And only a borrow names a covering site (F7).
    assert borrows["covering_site"].notna().all()


def test_the_event_log_ties_to_the_capacity_plan_per_event_type(book, tables):
    """Two cross-tab ties, both exact, and each on its own event type.

    Capacity Plan reaches 2,767.6 overtime hours and 420.0 uncovered by
    adding weekly role rows. The event log reaches the same two figures
    by adding what each E2 covered and what each E3 accepted. Different
    code, different frames, so agreement is evidence rather than
    arithmetic - and it only works per event type, which is the point of
    the block it is read from.
    """
    by_type = tables["coverage by type"].set_index("event_type")
    headers = [header for header, *_rest in export.CAPACITY_PLAN_COLUMNS]
    plan_sheet = book["Capacity Plan"]
    total_row = export.DETAIL_HEADER_ROW + 1 + 780 + 1

    def plan_total(header):
        letter = get_column_letter(headers.index(header) + 1)
        return evaluate(book, "Capacity Plan", plan_sheet[f"{letter}{total_row}"].value)

    assert float(
        by_type.loc[export.OVERTIME_EVENT, "hours_covered"]
    ) == pytest.approx(plan_total("Overtime hours"), abs=1e-6)
    assert float(
        by_type.loc[export.ACCEPTED_GAP, "hours_uncovered"]
    ) == pytest.approx(plan_total("Uncovered hours"), abs=1e-6)

    # A borrow covers hours and leaves none uncovered, so the two ties
    # cannot be the same number reached twice.
    assert float(by_type.loc["E1", "hours_uncovered"]) == pytest.approx(0.0)
    assert plan_total("Overtime hours") != pytest.approx(
        plan_total("Uncovered hours")
    )


# ---------------------------------------------------------------------------
# Non-Digital Tracker: the flat line is the finding
# ---------------------------------------------------------------------------


def test_only_non_digital_media_are_tracked(tables, baseline_assumptions, run):
    """Paper, microfiche and the archive room - the ones somebody walks to.

    Digital and the email archive do not belong on this tab: nobody
    retrieves them from a cabinet, and including them would make the
    non-digital backlog look twelve times larger than it is.
    """
    table = tables["Non-Digital Tracker"]
    onsite = set(digitization.onsite_media(baseline_assumptions))
    assert onsite == {"paper", "microfiche", "archive_room"}
    assert set(table["media"]) == onsite
    assert set(table["work_type"]) == {"W11", "W12"}
    assert set(table["scope"]) == {"N1", "N2", "N3", "S1", "S2", "S3"}
    assert len(table) == 52 * 6 * 2 * 3

    # Against the model's own queue frame, not a second computation.
    queues = run["queues"]
    source = queues[queues["media"].isin(onsite)]
    assert len(table) == len(source)
    for column in ("arrivals", "ending_backlog", "completed"):
        assert table[column].sum() == pytest.approx(
            source[column].sum(), abs=1e-6
        ), column


def test_the_digitization_backlog_is_measured_against_its_ten_day_goal(
    tables, baseline_assumptions, run
):
    """W12's goal is 10 days because it is OFF the request's path (spec 4.4).

    Nobody is waiting on it, which is exactly why it is the queue a short
    week quietly borrows from - "pause digitization" is one of spec 6.7's
    logged actions. The days of inventory is rebuilt from network totals,
    because the mean of six sites' ratios is not the ratio of the sums.
    """
    backlog = tables["digitization backlog"]
    assert len(backlog) == 52
    assert set(backlog["goal_business_days"]) == {10.0}

    queues = run["queues"]
    source = queues[queues["work_type"] == "W12"]
    assert backlog["arrivals"].sum() == pytest.approx(
        source["arrivals"].sum(), abs=1e-6
    )
    assert backlog["ending_backlog"].sum() == pytest.approx(
        source["ending_backlog"].sum(), abs=1e-6
    )

    # The ratio of the sums, checked on every week.
    days = float(baseline_assumptions["calendar"]["business_days_per_week"])
    for record in backlog.to_dict("records"):
        assert record["days_of_inventory"] == pytest.approx(
            record["ending_backlog"] / (record["arrivals"] / days), abs=1e-9
        ), record["plan_week"]

    # And it is not the same as averaging the six sites' own ratios.
    #
    # NOT checked on plan week 1, which is the week the two methods must
    # agree: spec 4.3 PLACES every queue at its target backlog in week 1,
    # so each site's days of inventory is exactly its goal and the mean of
    # six tens is ten. Picking week 1 for this was the first version of
    # this test, and it failed for the right reason. The weeks that
    # separate the two methods are the ones carrying what the week before
    # actually left.
    naive = source.groupby("week_index")["days_of_inventory"].mean()
    rebuilt = backlog.set_index("plan_week")["days_of_inventory"]
    history = int(baseline_assumptions["calendar"]["history_weeks"])
    differences = [
        abs(float(naive.loc[week]) - float(rebuilt.loc[week - history + 1]))
        for week in naive.index
    ]
    assert max(differences) > 1e-6, (
        "the per-site mean and the ratio of the sums never differ, so this "
        "test proves nothing"
    )
    assert differences[0] == pytest.approx(0.0, abs=1e-9), (
        "plan week 1 should agree: every queue is placed at its target"
    )

    assert bool(backlog["goal_met"].all())


def test_the_non_digital_share_is_flat_and_the_tab_says_so(tables, book,
                                                           baseline_assumptions):
    """The flat line is spec 7's finding, not a modelling shortcut.

    Reactive digitization converts about 21,000 pulled charts a plan year
    out of a 2,000,000-chart archive, and three years of it moves the
    share by under half a point. A tracker drawing a falling line here
    would be the chart that justified a project this model says has about
    a 39-year payback.
    """
    share = tables["non-digital share"]
    assert len(share) == 52
    assert share["non_digital_share"].nunique() == 1
    assert float(share["non_digital_share"].iloc[0]) == pytest.approx(
        digitization.non_digital_share(baseline_assumptions), abs=1e-12
    )
    # And it is the sum of its parts, read off record_media.
    parts = share[["paper_share", "microfiche_share", "archive_room_share"]]
    assert parts.iloc[0].sum() == pytest.approx(
        float(share["non_digital_share"].iloc[0]), abs=1e-12
    )

    sheet = book["Non-Digital Tracker"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert "FLAT IN THE BASELINE" in text.upper()
    assert "39-year payback" in text


# ---------------------------------------------------------------------------
# Forecast: the MAPE has to be the holdout's
# ---------------------------------------------------------------------------


def test_the_scoreboard_is_the_model_s_own_holdout_scores(
    tables, baseline_assumptions, sites
):
    """The MAPE on the tab is score_holdout's, recomputed independently.

    Both methods are fitted on all but the last 13 weeks and graded on
    those 13, which they never saw. This refits from the model rather
    than trusting the tab, so a scoreboard reporting a fit statistic
    dressed up as a forecast error would fail here.
    """
    from capacity_model import forecast, generate_data

    history = generate_data.generate_history(baseline_assumptions, sites)
    scores = tables["forecast scores"].set_index(["treatment", "scope", "method"])

    for use_events, treatment in ((False, "no_calendar_events"),
                                  (True, "calendar_events")):
        per_site, predictions = forecast.score_holdout(
            history, baseline_assumptions, use_calendar_events=use_events
        )
        pooled = forecast.choose_method(predictions, baseline_assumptions)[1]
        for record in per_site.to_dict("records"):
            key = (treatment, record["site"], record["method"])
            assert float(scores.loc[key, "mape"]) == pytest.approx(
                record["mape"], abs=1e-12
            ), key
        for method, value in pooled.to_dict().items():
            key = (treatment, "NETWORK", method)
            assert float(scores.loc[key, "mape"]) == pytest.approx(
                value, abs=1e-12
            ), key


def test_the_holdout_predictions_are_only_on_the_holdout_weeks(tables):
    """13 weeks with a prediction from each method, and none anywhere else.

    A prediction appearing on a week outside the holdout would mean a
    method was being shown forecasting weeks it was fitted to, which is
    the claim the holdout exists to avoid making.
    """
    weeks = tables["forecast weeks"]
    facts = tables["forecast facts"]

    holdout = weeks[weeks["period"] == "holdout"]
    assert len(holdout) == facts["holdout_weeks"] == 13
    for method in ("decomposition", "holt_winters"):
        assert holdout[method].notna().all(), method
        assert weeks[weeks["period"] != "holdout"][method].isna().all(), method
        errors = (
            (holdout[method] - holdout["actual_requests"]).abs()
            / holdout["actual_requests"]
        )
        assert 0 < float(errors.mean()) < 0.10, method


def test_aggregating_the_network_can_rank_the_methods_the_other_way(
    tables, book
):
    """Found 11/03, and the reason the tab carries a note about it.

    The choice is made on POOLED SITE-WEEK MAPE so every site carries
    equal weight (spec 3.2, decided 10/21). The holdout columns on the
    tab are network TOTALS, and adding six sites lets their misses cancel
    - so the aggregate error is smaller than the pooled figure and, on
    this history, actually prefers the other method. Both facts are true
    and neither is a bug; a reader who did not know would think the tab
    contradicted itself.
    """
    weeks = tables["forecast weeks"]
    scores = tables["forecast scores"]
    facts = tables["forecast facts"]
    holdout = weeks[weeks["period"] == "holdout"]

    aggregate = {}
    for method in ("decomposition", "holt_winters"):
        errors = (
            (holdout[method] - holdout["actual_requests"]).abs()
            / holdout["actual_requests"]
        )
        aggregate[method] = float(errors.mean())

    pooled = scores[
        (scores["treatment"] == facts["treatment"]) & (scores["scope"] == "NETWORK")
    ].set_index("method")["mape"]

    # The decision stands on the pooled figure.
    assert pooled.idxmin() == facts["method"]
    # Aggregation makes every method look better than it is...
    for method in aggregate:
        assert aggregate[method] < pooled[method], method
    # ...and on this history it reverses the ranking, which is exactly
    # why the tab says so in words.
    assert min(aggregate, key=aggregate.get) != facts["method"]

    sheet = book["Forecast"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert "NETWORK TOTALS" in text.upper()
    assert "cancel" in text


def test_both_methods_and_both_treatments_are_scored(tables):
    """Spec 3.2 asks for the MAPE of BOTH methods, not just the winner.

    A scoreboard with one row is a claim. Two methods across two
    treatments and seven scopes is a measurement, and it is what makes
    the choice arguable - the gain from treating holidays as known is
    measured on the same data rather than asserted.
    """
    scores = tables["forecast scores"]
    assert set(scores["method"]) == {"decomposition", "holt_winters"}
    assert set(scores["treatment"]) == {"calendar_events", "no_calendar_events"}
    assert set(scores["scope"]) == {"N1", "N2", "N3", "S1", "S2", "S3", "NETWORK"}
    assert len(scores) == 2 * 2 * 7 == 28
    assert (scores["mape"] > 0).all()
    assert (scores["mape"] < 0.5).all()

    # Exactly one row is marked as the one the plan uses.
    assert list(scores["chosen"]).count("chosen") == 1

    # Treating holidays as known helps at network level, which is why the
    # live setting has it on - measured, not assumed.
    pooled = scores[scores["scope"] == "NETWORK"].set_index(
        ["treatment", "method"]
    )["mape"]
    assert (
        pooled[("calendar_events", "decomposition")]
        < pooled[("no_calendar_events", "decomposition")]
    )


def test_a_per_site_row_may_disagree_with_the_network_choice(tables):
    """One method for the whole network, chosen on pooled MAPE (10/21).

    So the per-site rows are ALLOWED to disagree, and some do. This is
    worth a test because the obvious bug - picking a method per site and
    reporting it as one - would make every per-site row agree, and the
    table would look tidier and be wrong.
    """
    scores = tables["forecast scores"]
    facts = tables["forecast facts"]
    live = scores[
        (scores["treatment"] == facts["treatment"]) & (scores["scope"] != "NETWORK")
    ]
    winners = live.loc[live.groupby("scope")["mape"].idxmin()]
    assert set(winners["method"]) == {"decomposition", "holt_winters"}, (
        "no site prefers the challenger, so this test proves nothing"
    )
    disagreeing = winners[winners["method"] != facts["method"]]
    assert len(disagreeing) > 0


def test_the_history_and_forecast_span_every_week_once(tables,
                                                       baseline_assumptions):
    """156 history weeks plus 52 plan weeks, each once, with no overlap."""
    weeks = tables["forecast weeks"]
    cal = baseline_assumptions["calendar"]
    assert len(weeks) == cal["history_weeks"] + cal["plan_weeks"] == 208
    assert weeks["week_index"].is_unique
    assert weeks["period"].value_counts().to_dict() == {
        "history": 143, "forecast": 52, "holdout": 13,
    }

    # A history row has an actual and no forecast; a forecast row the
    # reverse. A row with both would mean the two series overlap, and the
    # chart would be drawing a forecast of a week it already knew.
    history = weeks[weeks["period"] != "forecast"]
    forward = weeks[weeks["period"] == "forecast"]
    assert history["actual_requests"].notna().all()
    assert history["forecast_requests"].isna().all()
    assert forward["forecast_requests"].notna().all()
    assert forward["actual_requests"].isna().all()

    # Two week columns, each blank on the other's rows, because one
    # column would show "1" for both the second week of 2024 and the
    # first week of the plan.
    assert history["plan_week"].isna().all()
    assert forward["history_week"].isna().all()
    assert sorted(forward["plan_week"]) == list(range(1, 53))


def test_the_forecast_chart_is_embedded(book):
    """Spec Section 8 asks for history and forecast; a table is half of it.

    The chart carries the one thing a table cannot: the shape, and the
    shaded holdout band that shows which stretch the method was graded on
    before it was trusted with the 52 weeks to its right.
    """
    sheet = book["Forecast"]
    assert len(sheet._images) == 1
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert export.FORECAST_CHART in text
    assert "holdout" in text.lower()


def test_the_tab_draws_the_chart_when_the_file_is_not_there(
    tables, baseline_assumptions, tmp_path
):
    """The embed test must not be able to pass on a leftover file.

    The outputs folder is not committed, so on a fresh clone there is no
    chart at all - and the test above passed for a while only because a
    PNG from an earlier run happened to be sitting on this machine. That
    is the same trap as a test that checks a frame which never carried the
    column. So this one points the model at an empty outputs folder, which
    is what a clean checkout looks like, and asserts the tab produces the
    picture itself rather than skipping it.
    """
    assumptions = copy.deepcopy(baseline_assumptions)
    assumptions["paths"]["outputs_dir"] = str(tmp_path)
    facts = tables["forecast facts"]

    expected = tmp_path / assumptions["paths"]["forecast_chart_png"]
    assert not expected.exists()

    drawn = export._forecast_chart(assumptions, facts)

    assert drawn == expected and drawn.exists()
    # A real PNG, not an empty file: the signature bytes every PNG starts
    # with, and a size only a drawn figure reaches.
    assert drawn.read_bytes()[:4] == bytes.fromhex("89504e47")
    assert drawn.stat().st_size > 10_000

    # Second call returns the file that is now there, without redrawing it.
    stamp = drawn.stat().st_mtime_ns
    assert export._forecast_chart(assumptions, facts) == expected
    assert drawn.stat().st_mtime_ns == stamp


def test_the_forecast_tab_names_the_method_and_both_pooled_scores(book, tables):
    """The decision, stated on the tab rather than left to be inferred."""
    sheet = book["Forecast"]
    facts = tables["forecast facts"]
    labels = {}
    for row in range(1, sheet.max_row + 1):
        label = sheet[f"A{row}"].value
        if isinstance(label, str):
            labels[label] = sheet[f"C{row}"].value

    assert labels.get("Method used for the plan") == facts["method"]
    assert labels.get("Calendar events treated as known") == facts["treatment"]
    assert labels.get("Holdout length, weeks") == pytest.approx(13.0)

    pooled = [
        key for key in labels if key.startswith("Pooled MAPE")
    ]
    assert len(pooled) == 4  # two methods x two treatments
    for key in pooled:
        assert 0 < float(labels[key]) < 0.5, key


# ---------------------------------------------------------------------------
# Readability and the Hiring Plan note
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TAB_COLUMNS) + ["Forecast"])
def test_each_new_tab_is_labelled_filtered_and_frozen(book, name):
    """A thousand-row table with no filter and no frozen header is unusable."""
    sheet = book[name]
    assert sheet[f"A{export.DETAIL_TITLE_ROW}"].value == f"{name} - SYNTHETIC"
    assert "SYNTHETIC" in str(sheet[f"A{export.DETAIL_NOTE_ROW}"].value)
    assert sheet.freeze_panes is not None
    assert sheet.auto_filter.ref is not None, name


@pytest.mark.parametrize("name", list(TAB_COLUMNS))
def test_each_new_column_carries_its_own_number_format(book, tables, name):
    """Cents on Next Dollar, because the finding is seven cents an item."""
    sheet = book[name]
    first = export.DETAIL_HEADER_ROW + 1
    table = tables[name]
    for index, (header, source, number_format, _w, _t) in enumerate(
        TAB_COLUMNS[name], start=1
    ):
        if number_format is None:
            continue
        for offset in range(min(len(table), 40)):
            cell = sheet.cell(row=first + offset, column=index)
            if cell.value is None:
                continue
            assert cell.number_format == number_format, (name, header, offset)


def test_next_dollar_is_shown_in_cents_because_the_finding_is_in_cents(book):
    """W14 misses by seven cents, so a dollar format would erase it."""
    assert export.MONEY_CENTS.startswith("$#,##0.00")
    assert export.MONEY_CENTS != export.MONEY
    headers = [header for header, *_rest in export.NEXT_DOLLAR_COLUMNS]
    sheet = book["Next Dollar"]
    letter = get_column_letter(
        headers.index("Saving per item, after fixed") + 1
    )
    cell = sheet[f"{letter}{export.DETAIL_HEADER_ROW + 1}"]
    assert cell.number_format == export.MONEY_CENTS

    # And savings per dollar is a ratio, never shown with a dollar sign:
    # "$0.68" and "0.68 per dollar spent" are different claims.
    letter = get_column_letter(headers.index("Saving per dollar spent") + 1)
    assert sheet[
        f"{letter}{export.DETAIL_HEADER_ROW + 1}"
    ].number_format == export.PER_DOLLAR
    assert "$" not in export.PER_DOLLAR


def test_the_hiring_plan_carries_the_summer_gap_note(book):
    """Siri's wording, 11/03, now a finding in spec 6.5.

    The sharpest thing on the tab: the plan hires permanent people for a
    seasonal dip and carries the surplus into the fall. It sits under the
    surplus flags, where a reader has just seen both halves.
    """
    sheet = book["Hiring Plan"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert (
        "The 8 hires cover a summer gap from week 23 (the vacation season) "
        "while tier 2 is in surplus in weeks 14-22 and tier 1 from week 36; "
        "see spec 6.5." in text
    )
    assert export.HIRING_PLAN_NOTES[-1].startswith("The 8 hires")
