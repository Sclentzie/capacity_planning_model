"""Spec Section 9, Sessions 6 and 7: the four baseline tabs tie to the model.

Task 6b. The Section 9 check this file extends:

    - Workbook and dashboard totals tie to the model's own totals to the
      cent.

Task 6a tied the two scenario tabs. These four are the DETAIL tabs -
Capacity Plan, Capacity Ledger, Hiring Plan, Assumptions - and detail is
where a tie-out earns its keep. A headline figure gets checked by whoever
quotes it; row 537 of an 780-row table gets checked by nobody, which is
exactly why it has to be checked here.

THREE LAYERS, AND THEY ARE NOT THE SAME TEST

A detail tab can be wrong in two independent places, and testing only one
of them proves nothing:

    THE WRITER can put the right frame in the wrong cells - a column
      offset, a row offset, a number format applied to the neighbour. So
      every written cell is compared to the builder's own frame.

    THE BUILDER can hand the writer a frame that is not the model's
      answer - a merge that dropped rows, a fillna that papered over a
      missing join, an aggregation at the wrong grain. So the builder's
      frame is compared to the model's own tables, recomputed here from
      `run[...]` rather than by calling the builder again. A test that
      called the builder twice and compared the results would pass on any
      bug the builder has.

    THE TOTALS can be right in each cell and still not add up, if a SUM
      range is one row short. So every total formula is evaluated, and
      then matched to the figure the model itself reports - and for cost,
      overtime and uncovered hours, to the SAME figure Capacity Scenarios
      shows on its baseline row. That last one is the strongest form of
      the Section 9 check: the detail tab and the headline tab have to
      agree, and neither is allowed to be the reference.

WHAT THE BLANKS ARE FOR

A good part of this file checks that cells are EMPTY. That is not
pedantry. A service level on a back-office row, or days of inventory on
the counter, would be a number the model never computed, and a workbook
whose empty cells are filled with plausible zeros is less trustworthy
than one that admits what it does not measure.
"""

import pytest
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string, get_column_letter

from capacity_model import config, export, scenarios

# One evaluator for the whole workbook, imported rather than copied: two
# formula evaluators would eventually disagree, and then the tests would
# be arguing with each other instead of with the exporter.
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
    """Every scenario once, because build() needs the comparison table."""
    return scenarios.run(None, baseline_assumptions, sites)


@pytest.fixture(scope="module")
def run(result):
    """The baseline run itself - the model's own answer, unmediated."""
    return result["runs"][export.BASELINE_KEY]


@pytest.fixture(scope="module")
def book(result, baseline_assumptions, tmp_path_factory):
    path = tmp_path_factory.mktemp("workbook_6b") / "wb.xlsx"
    export.build(result, baseline_assumptions).save(path)
    return load_workbook(path)


@pytest.fixture(scope="module")
def tables(run, baseline_assumptions):
    """The builders' frames, built once."""
    return {
        "Capacity Plan": export.capacity_plan_table(run, baseline_assumptions),
        "Capacity Ledger": export.capacity_ledger_table(run, baseline_assumptions),
        "Hiring Plan": export.hiring_plan_table(run, baseline_assumptions),
        "Assumptions": export.assumptions_table(run, baseline_assumptions),
        "surplus": export.surplus_table(run, baseline_assumptions),
        "machine": export.machine_week_table(run, baseline_assumptions),
    }


TAB_COLUMNS = {
    "Capacity Plan": export.CAPACITY_PLAN_COLUMNS,
    "Capacity Ledger": export.CAPACITY_LEDGER_COLUMNS,
    "Hiring Plan": export.HIRING_PLAN_COLUMNS,
    "Assumptions": export.ASSUMPTION_COLUMNS,
}


def written_rows(book, name, columns, table, header_row=export.DETAIL_HEADER_ROW):
    """The tab's data block, read back as a list of dicts keyed by column."""
    sheet = book[name]
    first = header_row + 1
    out = []
    for offset in range(len(table)):
        row = first + offset
        record = {}
        for index, (header, _source, _fmt, _width, _total) in enumerate(
            columns, start=1
        ):
            record[header] = sheet.cell(row=row, column=index).value
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Layer 1: the writer put the builder's frame in the right cells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TAB_COLUMNS))
def test_every_written_cell_is_the_builder_s_own_value(book, tables, name):
    """Cell by cell, to the cent, on all four tabs.

    The loop is over every row rather than a sample, because the failure
    this catches is an OFFSET - and an offset is invisible in a sample,
    where every value you look at is some real value from the table.
    """
    columns = TAB_COLUMNS[name]
    table = tables[name]
    rows = written_rows(book, name, columns, table)
    assert len(rows) == len(table), name

    for position, (record, written) in enumerate(
        zip(table.to_dict("records"), rows)
    ):
        for header, source, _fmt, _width, _total in columns:
            if source is None:
                continue  # a live formula; checked in its own test
            expected = record.get(source)
            got = written[header]
            if expected is None or (
                isinstance(expected, float) and expected != expected
            ):
                assert got is None, (name, position, header, got)
            elif isinstance(expected, str):
                assert got == expected, (name, position, header)
            elif isinstance(expected, bool):
                assert got == expected, (name, position, header)
            elif isinstance(expected, (int, float)):
                assert got == pytest.approx(float(expected), abs=CENT), (
                    name, position, header, got, expected,
                )
            else:
                # A date, for instance. The exporter writes it as text
                # rather than as an Excel serial date, because a key's
                # VALUE on the Assumptions tab is there to be read and
                # compared with the YAML, not calculated with.
                assert got == str(expected), (name, position, header)


def test_the_surplus_block_is_written_too(book, tables, run):
    """The second block on the Hiring Plan, which is easy to forget.

    A hiring plan that only ever says "hire" is a budget request, so the
    pause-hiring flags are part of the output, not a footnote - and they
    sit below the classes at a row offset nothing else depends on, which
    is precisely the kind of block that silently stops being written.
    """
    surplus = tables["surplus"]
    assert len(surplus) == len(run["surplus"]) > 0

    sheet = book["Hiring Plan"]
    header = None
    for row in range(1, sheet.max_row + 1):
        if sheet[f"A{row}"].value == "Role" and row > export.DETAIL_HEADER_ROW:
            header = row
            break
    assert header, "the surplus block has no header row"

    rows = written_rows(book, "Hiring Plan", export.SURPLUS_COLUMNS, surplus, header)
    for record, written in zip(surplus.to_dict("records"), rows):
        for label, source, _fmt, _width, _total in export.SURPLUS_COLUMNS:
            expected = record[source]
            if isinstance(expected, str):
                assert written[label] == expected
            else:
                assert written[label] == pytest.approx(float(expected), abs=CENT)


# ---------------------------------------------------------------------------
# Layer 2: the builder's frame is the model's answer
#
# Recomputed from the model's own tables here, never by calling the
# builder a second time - that would pass on any bug the builder has.
# ---------------------------------------------------------------------------


def test_the_capacity_plan_has_one_row_per_week_role_and_scope(tables, run):
    """780 rows: 52 plan weeks x 15 (role, scope) pairs.

    Fifteen, not five: spec 6.7 puts the two onsite roles at a site and
    the back office on the network, so Onsite Associate and Records
    Specialist appear six times each and the three remote teams once.
    """
    table = tables["Capacity Plan"]
    pairs = run["gap_fte"][["role", "scope"]].drop_duplicates()
    weeks = run["gap_fte"]["week_index"].nunique()
    assert len(pairs) == 15
    assert weeks == 52
    assert len(table) == 780

    assert set(map(tuple, table[["role", "scope"]].drop_duplicates().values)) == set(
        map(tuple, pairs.values)
    )
    assert sorted(table["plan_week"].unique()) == list(range(1, 53))


def test_the_capacity_plan_measures_come_from_the_model_s_own_tables(tables, run):
    """Each column traced back to the frame that produced it."""
    table = tables["Capacity Plan"].set_index(["week_index", "role", "scope"])

    gap = run["gap_fte"].set_index(["week_index", "role", "scope"])
    for column in ("required_hours", "covered_hours", "short_fte", "surplus_fte"):
        assert table[column].sum() == pytest.approx(gap[column].sum(), abs=1e-6)
        for key in list(gap.index)[:40]:
            assert table.loc[key, column] == pytest.approx(
                gap.loc[key, column], abs=1e-9
            ), (column, key)

    cost = run["role_cost"].set_index(["week_index", "role", "scope"])
    for column in ("paid_headcount", "overtime_hours", "total_cost"):
        assert table[column].sum() == pytest.approx(cost[column].sum(), abs=CENT)

    # Available and uncovered hours are coalesced from two sources, which
    # is the merge most likely to drop rows without saying so.
    back_office = run["cost"].set_index(["week_index", "role", "scope"])
    for key in list(back_office.index)[:40]:
        assert table.loc[key, "available_hours"] == pytest.approx(
            back_office.loc[key, "available_hours"], abs=1e-9
        )
        assert table.loc[key, "days_of_inventory_after"] == pytest.approx(
            back_office.loc[key, "days_of_inventory_after"], abs=1e-9
        )
    assert table["available_hours"].notna().all()
    assert table["uncovered_hours"].notna().all()


def test_the_onsite_week_is_the_sum_of_its_five_days(tables, run):
    """Hours add across the week; the counter is planned by DAY (spec 6.2).

    The onsite frame is daily because a walk-in cannot wait until
    Thursday, so the weekly figure on this tab is a roll-up - and a
    roll-up at the wrong grain is a five-times error that still looks like
    a plausible number.
    """
    table = tables["Capacity Plan"]
    onsite = table[table["role"] == "onsite_associate"]
    daily = run["onsite"]

    assert onsite["available_hours"].sum() == pytest.approx(
        daily["available_hours"].sum(), abs=1e-6
    )
    assert onsite["uncovered_hours"].sum() == pytest.approx(
        daily["uncovered_hours"].sum(), abs=1e-6
    )
    assert len(onsite) == 52 * 6


def test_a_weekly_service_level_is_volume_weighted_not_averaged(tables, run):
    """Monday carries 23% of the week, so the mean of five days is wrong.

    A service level is a SHARE of contacts answered in time. Averaging
    five daily percentages weights Friday - the quietest day, where the
    level looks best - the same as Monday, and the tab would report a
    service level nobody experienced. This checks the weighting on a real
    week rather than asserting the intent in a comment.
    """
    table = tables["Capacity Plan"]
    daily = run["onsite"]
    week = int(daily["week_index"].min())
    group = daily[(daily["week_index"] == week) & (daily["site"] == "N1")]

    weighted = (
        group["service_level_phones_after"] * group["calls"]
    ).sum() / group["calls"].sum()
    plain = group["service_level_phones_after"].mean()

    row = table[
        (table["week_index"] == week)
        & (table["role"] == "onsite_associate")
        & (table["scope"] == "N1")
    ]
    assert len(row) == 1
    assert float(row["phone_service_level"].iloc[0]) == pytest.approx(
        weighted, abs=1e-12
    )
    # And the two really do differ, so the test is not passing by luck.
    assert abs(weighted - plain) > 1e-6


def test_the_ledger_is_the_model_s_roster_movement(tables, run):
    """Starting headcount, hires, attrition, ramp, shrinkage, hours."""
    table = tables["Capacity Ledger"].set_index(["week_index", "role", "scope"])
    ledger = run["prepared"]["ledger"]["ledger"].set_index(
        ["week_index", "role", "scope"]
    )
    assert len(table) == len(ledger) == 780
    for column in ("starting_headcount", "hires", "attrition", "ending_headcount",
                   "ramped_fte", "shrinkage", "productive_hours"):
        assert table[column].sum() == pytest.approx(ledger[column].sum(), abs=1e-6)
        for key in list(ledger.index)[:40]:
            assert table.loc[key, column] == pytest.approx(
                ledger.loc[key, column], abs=1e-12
            ), (column, key)


def test_ending_headcount_is_starting_plus_hires_minus_attrition(book, tables):
    """The identity, checked on all 780 WRITTEN rows.

    Ending headcount is a written value, not a formula, because only
    totals and differences are formulas (decided 11/03). That makes it the
    one column a reader might reasonably want to re-derive by hand, so the
    identity is asserted here instead of being left to a spot-check - and
    on the written cells, not the frame, because the frame is not what
    gets forwarded.
    """
    columns = export.CAPACITY_LEDGER_COLUMNS
    rows = written_rows(book, "Capacity Ledger", columns, tables["Capacity Ledger"])
    assert len(rows) == 780
    for position, row in enumerate(rows):
        expected = (
            row["Starting headcount"] + row["Hires"] - row["Attrition"]
        )
        assert row["Ending headcount"] == pytest.approx(expected, abs=1e-9), position


def test_the_hiring_plan_is_the_model_s_classes_in_plan_weeks(tables, run,
                                                              baseline_assumptions):
    """Three classes, eight people, every date converted to a plan week.

    The model counts weeks from the start of the generated history, so a
    hire-by week of 164 is meaningless to a recruiter. The conversion is
    one subtraction and therefore exactly the kind of thing that is off by
    one in a workbook nobody checks.
    """
    table = tables["Hiring Plan"]
    classes = run["hiring"]
    history = baseline_assumptions["calendar"]["history_weeks"]

    assert len(table) == len(classes) == 3
    assert table["class_size"].sum() == classes["class_size"].sum() == 8

    for record in table.to_dict("records"):
        match = classes[
            (classes["role"] == record["role"])
            & (classes["hire_by_week"] == record["hire_by_week"])
        ]
        assert len(match) == 1
        source = match.iloc[0]
        assert record["plan_hire_by"] == source["hire_by_week"] - history + 1
        assert record["plan_requisition"] == source["requisition_week"] - history + 1
        assert record["plan_first_productive"] == (
            source["first_productive_week"] - history + 1
        )
        assert 1 <= record["plan_hire_by"] <= 52
        # Requisition never comes after the week it is meant to cover.
        assert record["plan_requisition"] <= record["plan_hire_by"]
        assert record["weeks_to_productive"] == (
            source["first_productive_week"] - source["requisition_week"]
        )


def test_the_baseline_runs_no_machine_and_the_tab_says_so(tables, book):
    """Automation is off in the baseline, and off is not zero.

    Automated items and human-in-the-loop hours are genuinely zero - every
    hour was worked by a person. Uptime is BLANK, because a 0% uptime
    would say the automation was down all year, which is a different and
    false claim. The tab writes a sentence instead of 52 empty rows.
    """
    assert len(tables["machine"]) == 0
    assert (tables["Capacity Ledger"]["human_in_the_loop_hours"] == 0).all()

    sheet = book["Capacity Ledger"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert "NO AUTOMATION IN THE BASELINE RUN" in text
    assert "there is no machine to be down" in text
    # And no uptime column was written with a zero in it anywhere.
    assert "Machine uptime" not in text


def test_human_in_the_loop_hours_are_found_when_there_is_a_machine(
    baseline_assumptions, sites
):
    """The column is zero on the baseline, so it is proved on a run that isn't.

    A test that only ever saw the baseline would pass if the builder
    returned zero unconditionally, which is the whole failure mode. AI
    targeted automates two work types, so its ledger must carry exception,
    QA-sample and sign-off hours (spec 5.5) - and they must land on the
    roles that actually do them, not on a single network row.
    """
    scenario = scenarios.load_scenario(
        scenarios.scenario_dir(baseline_assumptions) / "ai_targeted.yaml"
    )
    run = scenarios.run_scenario(scenario, baseline_assumptions, sites)
    table = export.capacity_ledger_table(run, run["assumptions"])

    loop = table["human_in_the_loop_hours"]
    assert loop.sum() > 0

    # Against the model's own figure, from the required-hours table.
    required = run["prepared"]["required"]
    expected = required[
        required["work_kind"].isin(export.LOOP_KINDS)
    ]["required_hours"].sum()
    assert loop.sum() == pytest.approx(expected, abs=1e-6)

    # More than one role carries them: the QA sample goes to QA (spec 5.5)
    # and the exceptions stay with the role that owned the item.
    carrying = table[loop > 0]["role"].unique()
    assert len(carrying) > 1

    # And the machine block is no longer empty on this run.
    machine = export.machine_week_table(run, run["assumptions"])
    assert len(machine) > 0
    assert machine["automated_items"].sum() == pytest.approx(
        run["prepared"]["automated"]["automated_items"].sum(), abs=1e-6
    )


# ---------------------------------------------------------------------------
# Layer 3: the totals
# ---------------------------------------------------------------------------


TOTAL_SOURCES = {
    "Capacity Plan": (
        ("Required hours", "required_hours"),
        ("Available hours", "available_hours"),
        ("Covered hours", "covered_hours"),
        ("Uncovered hours", "uncovered_hours"),
        ("Overtime hours", "overtime_hours"),
        ("Labour cost", "total_cost"),
    ),
    "Capacity Ledger": (
        ("Hires", "hires"),
        ("Attrition", "attrition"),
        ("Productive hours", "productive_hours"),
        ("Overtime hours", "overtime_hours"),
        ("Human-in-the-loop hours", "human_in_the_loop_hours"),
    ),
    "Hiring Plan": (("Class size", "class_size"),),
}


@pytest.mark.parametrize("name", list(TOTAL_SOURCES))
def test_every_total_formula_evaluates_to_the_model_s_total(book, tables, name):
    """The SUM formulas, evaluated, against the column they total.

    A total is the one cell on a detail tab that gets quoted, and a SUM
    range one row short is the error that looks right: the figure is
    plausible, the tab is long, and nobody adds 780 rows by hand.
    """
    columns = TAB_COLUMNS[name]
    table = tables[name]
    sheet = book[name]
    headers = [header for header, *_rest in columns]

    first = export.DETAIL_HEADER_ROW + 1
    total_row = first + len(table) + 1
    assert str(sheet[f"A{total_row}"].value or "").startswith(
        ("PLAN-YEAR TOTAL", "TOTAL")
    ), (name, sheet[f"A{total_row}"].value)

    checked = 0
    for header, source in TOTAL_SOURCES[name]:
        letter = get_column_letter(headers.index(header) + 1)
        formula = sheet[f"{letter}{total_row}"].value
        assert isinstance(formula, str) and formula.startswith("=SUM("), (
            name, header, formula,
        )
        got = evaluate(book, name, formula)
        assert got == pytest.approx(float(table[source].sum()), abs=CENT), (
            name, header,
        )
        checked += 1
    assert checked == len(TOTAL_SOURCES[name])


def test_a_stock_is_never_totalled(book, tables):
    """Headcount, service levels and days of inventory have no total.

    Adding a headcount across 52 weeks counts the same ten people
    fifty-two times, and adding a service level is meaningless in a
    different way again. A workbook that put a number there would be
    offering a figure somebody would eventually quote, so the cell is left
    empty on purpose and this is the test that keeps it empty.
    """
    for name in ("Capacity Plan", "Capacity Ledger"):
        columns = TAB_COLUMNS[name]
        sheet = book[name]
        total_row = export.DETAIL_HEADER_ROW + 1 + len(tables[name]) + 1
        for index, (header, _source, _fmt, _width, totalled) in enumerate(
            columns, start=1
        ):
            if totalled or index == 1:
                continue
            cell = sheet.cell(row=total_row, column=index)
            assert cell.value is None, (name, header, cell.value)

    plan = TAB_COLUMNS["Capacity Plan"]
    never = {
        "Paid headcount", "Short FTE", "Surplus FTE", "Phone service level",
        "Walk-in service level", "Days of inventory",
    }
    assert never <= {
        header for header, _s, _f, _w, totalled in plan if not totalled
    }
    ledger_never = {"Starting headcount", "Ending headcount", "Ramped FTE",
                    "Shrinkage"}
    assert ledger_never <= {
        header for header, _s, _f, _w, totalled
        in TAB_COLUMNS["Capacity Ledger"] if not totalled
    }


def test_the_hours_gap_formula_is_available_minus_required(book, tables):
    """The one live difference on a detail tab, checked on every row.

    A reader should be able to see the subtraction rather than trust it,
    and a shortfall has to come out NEGATIVE - a gap written the other way
    round would make every short week look like spare capacity.
    """
    columns = export.CAPACITY_PLAN_COLUMNS
    headers = [header for header, *_rest in columns]
    sheet = book["Capacity Plan"]
    table = tables["Capacity Plan"]

    gap = get_column_letter(headers.index("Hours gap (available - required)") + 1)
    first = export.DETAIL_HEADER_ROW + 1

    shortfalls = 0
    for offset, record in enumerate(table.to_dict("records")):
        row = first + offset
        formula = sheet[f"{gap}{row}"].value
        assert isinstance(formula, str) and formula.startswith("="), row
        expected = record["available_hours"] - record["required_hours"]
        assert evaluate(book, "Capacity Plan", formula) == pytest.approx(
            expected, abs=1e-9
        ), row
        if expected < 0:
            shortfalls += 1
    # A plan with no short rows at all would make this test vacuous.
    assert shortfalls > 0


def test_the_detail_totals_equal_the_capacity_scenarios_baseline_row(book, tables):
    """The cross-tab tie, and the strongest form of the Section 9 check.

    Capacity Plan reaches its labour cost by adding 780 weekly role rows.
    Capacity Scenarios reaches the same figure through summarise() on the
    whole run. The two routes are independent, so if they agree to the
    cent the figure is the model's and not an artefact of either path -
    and neither tab is allowed to be the reference for the other.
    """
    plan = book["Capacity Plan"]
    scenario_sheet = book["Capacity Scenarios"]
    headers = [header for header, *_rest in export.CAPACITY_PLAN_COLUMNS]
    total_row = export.DETAIL_HEADER_ROW + 1 + len(tables["Capacity Plan"]) + 1
    baseline_row = export.SCENARIO_FIRST_DATA_ROW
    assert scenario_sheet[f"A{baseline_row}"].value == "Baseline"

    for detail, headline, tolerance in (
        ("Labour cost", "Labour cost", CENT),
        ("Overtime hours", "Overtime hours", 1e-6),
        ("Uncovered hours", "Uncovered hours", 1e-6),
    ):
        letter = get_column_letter(headers.index(detail) + 1)
        got = evaluate(book, "Capacity Plan", plan[f"{letter}{total_row}"].value)
        want = scenario_sheet[
            f"{export.SCENARIO_LETTERS[headline]}{baseline_row}"
        ].value
        assert got == pytest.approx(float(want), abs=tolerance), detail


def test_the_ledger_hires_total_is_the_hiring_plan_s_people(book, tables):
    """Two tabs, one hiring decision, and they have to agree.

    The ledger reaches eight people by adding a fractional weekly hires
    column across 780 rows; the Hiring Plan reaches eight by adding three
    class sizes. If those disagree, one of the two tabs is describing a
    roster the model never planned.
    """
    ledger_headers = [header for header, *_r in export.CAPACITY_LEDGER_COLUMNS]
    hiring_headers = [header for header, *_r in export.HIRING_PLAN_COLUMNS]

    ledger = book["Capacity Ledger"]
    ledger_total = export.DETAIL_HEADER_ROW + 1 + len(tables["Capacity Ledger"]) + 1
    letter = get_column_letter(ledger_headers.index("Hires") + 1)
    hires = evaluate(book, "Capacity Ledger", ledger[f"{letter}{ledger_total}"].value)

    plan = book["Hiring Plan"]
    plan_total = export.DETAIL_HEADER_ROW + 1 + len(tables["Hiring Plan"]) + 1
    letter = get_column_letter(hiring_headers.index("Class size") + 1)
    people = evaluate(book, "Hiring Plan", plan[f"{letter}{plan_total}"].value)

    assert hires == pytest.approx(people, abs=1e-9)
    assert hires == pytest.approx(8.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The blanks, which are the honest answer rather than a gap
# ---------------------------------------------------------------------------


def test_a_service_level_is_only_written_where_somebody_is_waiting(book, tables):
    """Live channels have service levels; a back-office queue does not.

    A phone call or a walk-in is answered in time or it is not. Remote
    fulfillment work is allowed to wait, so it has days of inventory
    instead - and a 0.0% service level on a tier 1 row would be a number
    the model never computed.
    """
    columns = export.CAPACITY_PLAN_COLUMNS
    rows = written_rows(book, "Capacity Plan", columns, tables["Capacity Plan"])

    live, backlog = 0, 0
    for row in rows:
        onsite = row["Role"] == "onsite_associate"
        if onsite:
            assert row["Phone service level"] is not None, row["Plan week"]
            assert row["Walk-in service level"] is not None
            assert row["Days of inventory"] is None
            live += 1
        else:
            assert row["Phone service level"] is None, (row["Role"], row["Plan week"])
            assert row["Walk-in service level"] is None
            backlog += 1
    assert live == 52 * 6
    assert backlog == 780 - live


def test_days_of_inventory_is_written_for_every_back_office_row(book, tables):
    """The mirror image: work that is allowed to wait is measured in days.

    Checked separately from the blanks, because "never written" would pass
    the blanks test too - and a column that is always empty is a column
    that silently stopped being written.
    """
    columns = export.CAPACITY_PLAN_COLUMNS
    rows = written_rows(book, "Capacity Plan", columns, tables["Capacity Plan"])
    written = [
        row for row in rows
        if row["Role"] != "onsite_associate" and row["Days of inventory"] is not None
    ]
    assert len(written) == 468
    assert all(row["Days of inventory"] >= 0 for row in written)


# ---------------------------------------------------------------------------
# Assumptions: the tab that makes everything else arguable
# ---------------------------------------------------------------------------


def _label(item):
    """How the exporter names a block inside a list: code, name, or members.

    A travel pair has no code, so it is addressed by the two sites it
    joins - the one thing about it that cannot change without it becoming
    a different pair.
    """
    if item.get("code"):
        return str(item["code"])
    if item.get("name"):
        return str(item["name"])
    for key in ("sites", "roles", "work_types"):
        members = item.get(key)
        if isinstance(members, (list, tuple)) and members:
            return "-".join(str(member) for member in members)
    return None


def resolve(mapping, key):
    """Follow a dotted YAML key through the loaded config.

    Lists of blocks are keyed by `code`, the same way the exporter writes
    them, so this walks the structure a reader would walk.
    """
    current = mapping
    for part in key.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, (list, tuple)):
            match = [
                item for item in current
                if isinstance(item, dict) and _label(item) == part
            ]
            assert len(match) == 1, key
            current = match[0]
        else:
            raise KeyError(key)
    return current


def test_every_assumptions_key_resolves_in_the_yaml_it_names(tables, run):
    """The whole point of the tab: the key is an ADDRESS, not a label.

    A reader who disagrees with a result has to be able to paste the key
    into the file and land on the number that produced the plan. A key
    that does not resolve is worse than no key, because it sends somebody
    looking for an input that is not there.
    """
    table = tables["Assumptions"]
    loaded = {
        "assumptions.yaml": run["assumptions"],
        "sites.yaml": run["sites"],
        "skills.yaml": config.load_skills(),
    }
    checked = 0
    for record in table.to_dict("records"):
        source = record["source"]
        if source not in loaded:
            continue  # the scenario's own three rows
        value = resolve(loaded[source], record["key"])
        # An empty input says WHICH empty it is, because "" and null and
        # [] do not mean the same thing in these files - choose it from
        # the data, no cap at all, every eligible work type.
        if value is None:
            assert record["value"] == export.NO_VALUE, record["key"]
        elif isinstance(value, (list, tuple)) and not len(value):
            assert record["value"] == export.EMPTY_LIST, record["key"]
        elif record["kind"] == "list":
            assert record["value"] == ", ".join(str(item) for item in value)
        elif isinstance(value, str) and not value.strip():
            assert record["value"] == export.EMPTY_STRING, record["key"]
        else:
            assert str(record["value"]) == str(value), record["key"]
        checked += 1
    assert checked > 900


def test_the_assumptions_tab_lists_the_inputs_that_are_switched_off(book, tables):
    """Off is an input too.

    The baseline looks the way it does BECAUSE automation, deflection and
    the records hub are disabled. A traceability tab that listed only the
    settings that were on would be a list of the interesting inputs, not a
    record of the run.
    """
    table = tables["Assumptions"].set_index("key")
    for key in (
        "scenario.deflection.enabled",
        "scenario.records_hub.enabled",
        "scenario.proof_of_concept.enabled",
    ):
        assert key in table.index, key
        assert table.loc[key, "value"] in (False, "False", 0), key

    # The seed is listed, because it is why two runs agree.
    assert any("seed" in key for key in table.index)


def test_an_empty_input_says_which_kind_of_empty_it_is(tables):
    """Blank is the one thing an input value must never be.

    A blank cell on a traceability tab cannot be told apart from an
    exporter that failed to write - and these three empties mean three
    different things: records_hub.site "" means choose the hub from the
    data, capacity_cap null means no cap at all, only_work_types []
    means every eligible type. One blank for all of them would lose
    exactly the distinction the tab is for.

    Caught by the cell-by-cell tie-out, which is the point of running it
    over every row: this is row 39 of 986.
    """
    table = tables["Assumptions"].set_index("key")
    assert table.loc["scenario.records_hub.site", "value"] == export.EMPTY_STRING
    assert table.loc[
        "automation.capacity_cap_items_per_week", "value"
    ] == export.NO_VALUE
    assert table.loc[
        "automation.rates.only_work_types", "value"
    ] == export.EMPTY_LIST
    assert len({export.EMPTY_STRING, export.EMPTY_LIST, export.NO_VALUE}) == 3

    # And no row is left blank anywhere on the tab.
    values = tables["Assumptions"]["value"]
    assert not values.isna().any()
    assert not any(
        isinstance(value, str) and not value.strip() for value in values
    )


def test_the_assumptions_tab_covers_all_three_input_files(tables):
    """One missing file would make the run look reproducible when it isn't."""
    sources = set(tables["Assumptions"]["source"])
    assert {"assumptions.yaml", "sites.yaml", "skills.yaml"} <= sources
    counts = tables["Assumptions"]["source"].value_counts()
    assert counts["assumptions.yaml"] > 500
    assert counts["sites.yaml"] > 20
    assert counts["skills.yaml"] > 10


def test_no_assumptions_key_is_an_index_that_could_move(tables):
    """Keys are addressed by code, never by position.

    `work_types.W8.handle_time_minutes` survives somebody reordering the
    file; `work_types.8.handle_time_minutes` does not, and a traceability
    tab must not hand out a reference that goes stale on a reformat.
    """
    for key in tables["Assumptions"]["key"]:
        for part in key.split("."):
            assert not part.isdigit(), key


# ---------------------------------------------------------------------------
# Readability, on the four new tabs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TAB_COLUMNS))
def test_each_detail_tab_is_labelled_filtered_and_frozen(book, name, tables):
    """A 780-row table with no filter and no frozen header is unusable.

    Not a nicety: a reader who scrolls to row 400 and can no longer see
    which column is which starts guessing, and a guessed column is a wrong
    number quoted with confidence.
    """
    sheet = book[name]
    assert sheet[f"A{export.DETAIL_TITLE_ROW}"].value == f"{name} - SYNTHETIC"
    assert "SYNTHETIC" in str(sheet[f"A{export.DETAIL_NOTE_ROW}"].value)
    assert sheet.freeze_panes is not None

    header_row = export.DETAIL_HEADER_ROW
    last = header_row + len(tables[name])
    expected = f"A{header_row}:{get_column_letter(len(TAB_COLUMNS[name]))}{last}"
    assert sheet.auto_filter.ref == expected, (name, sheet.auto_filter.ref)

    for index, (header, _s, _f, _w, _t) in enumerate(TAB_COLUMNS[name], start=1):
        assert sheet.cell(row=header_row, column=index).value == header


@pytest.mark.parametrize("name", list(TAB_COLUMNS))
def test_each_detail_column_carries_its_own_number_format(book, name, tables):
    """A dollar read as a day is a different kind of wrong from a rounding
    error, so money, hours, FTE, days, counts and percentages never share
    a format - on the detail tabs as much as on the headline ones."""
    sheet = book[name]
    first = export.DETAIL_HEADER_ROW + 1
    table = tables[name]
    for index, (header, source, number_format, _w, _t) in enumerate(
        TAB_COLUMNS[name], start=1
    ):
        if number_format is None:
            continue
        for offset in range(min(len(table), 60)):
            cell = sheet.cell(row=first + offset, column=index)
            if cell.value is None:
                continue
            assert cell.number_format == number_format, (name, header, offset)


def test_a_percentage_is_formatted_as_one(book, tables):
    """Shrinkage of 0.255 shown raw invites somebody to read it as a count.

    One decimal place, because the difference between 80.0% and 80.4% is
    the difference between meeting a service goal and meeting it by luck.
    """
    assert export.PERCENT.startswith("0.0%")
    headers = [header for header, *_r in export.CAPACITY_LEDGER_COLUMNS]
    sheet = book["Capacity Ledger"]
    letter = get_column_letter(headers.index("Shrinkage") + 1)
    cell = sheet[f"{letter}{export.DETAIL_HEADER_ROW + 1}"]
    assert cell.number_format == export.PERCENT
    assert 0.0 < float(cell.value) < 1.0


def test_the_plan_weeks_are_plan_weeks_not_history_weeks(book, tables,
                                                         baseline_assumptions):
    """Nobody plans in week 171.

    The model counts from the start of the generated history, which is
    right for a forecast and meaningless to a reader. Every week on every
    new tab is 1 to 52, and the absolute index appears nowhere.
    """
    history = baseline_assumptions["calendar"]["history_weeks"]
    assert history > 52
    for name in ("Capacity Plan", "Capacity Ledger"):
        columns = TAB_COLUMNS[name]
        rows = written_rows(book, name, columns, tables[name])
        weeks = {row["Plan week"] for row in rows}
        assert weeks == set(range(1, 53)), name
        assert max(weeks) <= 52

    for record in tables["Hiring Plan"].to_dict("records"):
        for column in ("plan_hire_by", "plan_requisition", "plan_start",
                       "plan_first_productive"):
            assert 1 <= record[column] <= 52, column
