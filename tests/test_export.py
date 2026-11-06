"""Spec Section 9, Sessions 6 and 7: the workbook ties to the model.

Task 6a. The Section 9 check this file owns:

    - Workbook and dashboard totals tie to the model's own totals to the
      cent.

WHY THIS IS THE CHECK THAT MATTERS

Every number in this model is already tested. None of those tests say
anything about the spreadsheet, and the spreadsheet is the artefact that
leaves the building. A workbook is the one output that gets forwarded, cut
down, pasted into a deck and quoted back six weeks later, so a figure that
is a little bit wrong there is worse than a figure that is wrong in the
code - in the code somebody would eventually run the tests.

Three ways a workbook can lie, and one test each:

    WRONG NUMBER. A written value that is not the model's value, usually
      from a column read off the wrong key. Pinned cell by cell against
      the comparison table, to the cent.

    WRONG REFERENCE. A formula pointing one row or one column out. This is
      the one that survives review, because a plausible number appears and
      nothing looks broken. Every formula is parsed and EVALUATED here,
      against the values the workbook itself holds, and the result is
      matched to the difference the model computes.

    MISSING CONTEXT. A tab with no SYNTHETIC label, or a Summary with no
      note saying what all-in cost means. Not an arithmetic error, but on
      this project it is the one that would matter most: the data is
      fictional and the workbook has to say so wherever it is opened.

ON EVALUATING THE FORMULAS

openpyxl writes formulas but never computes them, and reading the file
back gives the formula text, not a result - the cached value only exists
if Excel itself has saved the file. So this module parses the small, regular
formula language the exporter emits (`=C5+D5`, `=B6-B$5`) and evaluates it
against the written values.

Stated plainly, because it is a real limit: this proves the references
point where they should and the arithmetic is what it claims. It is not
Excel. What it cannot catch is Excel refusing to open the file at all, and
`test_the_workbook_has_no_formula_errors` covers the part of that which is
checkable here - no #REF!, no formula pointing at a tab, a cell or a blank
that would turn into one, and no reference chain deep enough to be a cycle.
"""

import re

import pytest
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string

from capacity_model import config, export, scenarios


# A reference to one cell, with or without Excel's absolute-$ markers, and
# optionally on another sheet: 'Capacity Scenarios'!J23, B$5, C6.
REFERENCE = re.compile(
    r"^(?:'(?P<sheet>[^']+)'!)?\$?(?P<column>[A-Z]{1,3})\$?(?P<row>\d+)$"
)

# =SUM(F6:F785) - the only function the exporter emits, and it emits it
# only for a plan-year total (task 6b).
SUM_CALL = re.compile(
    r"^=SUM\((?P<start>\$?[A-Z]{1,3}\$?\d+):(?P<end>\$?[A-Z]{1,3}\$?\d+)\)$"
)


@pytest.fixture(scope="module")
def baseline():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def result(baseline, sites):
    """Every scenario, once. The slow fixture in this file, deliberately:
    a tie-out on a subset would not be a tie-out."""
    return scenarios.run(None, baseline, sites)


@pytest.fixture(scope="module")
def comparison(result):
    return result["comparison"].set_index("scenario")


@pytest.fixture(scope="module")
def book(result, baseline, tmp_path_factory):
    """Written to a temp path, not to outputs/.

    The test must not depend on, or clobber, whatever the last real run
    left in outputs/ - and a test that only passes after you remember to
    run main() first is not a test.
    """
    path = tmp_path_factory.mktemp("workbook") / "wb.xlsx"
    export.build(result, baseline).save(path)
    return load_workbook(path)


MAX_DEPTH = 6


def evaluate(book, sheet_name, formula, depth=0):
    """Evaluate one of the exporter's formulas against written values.

    Handles exactly the language the exporter emits - a single cell, or two
    cells with + or - between them - and refuses anything else rather than
    guessing. A silent fallback here would be the one place a wrong formula
    could slip through the test that exists to catch wrong formulas.

    CHAINS ARE RESOLVED, because the workbook has them by design and Excel
    resolves them: all-in cost is a formula over written labour and
    machine, so "all-in vs baseline" differences two formulas, and the
    Summary then references that. Found 11/03 by writing this evaluator
    one level deep first and watching it reject the real workbook - the
    restriction was mine, not Excel's.

    `MAX_DEPTH` is the guard that matters. A reference cycle in a
    spreadsheet is a circular-reference warning in Excel and an infinite
    recursion here, so the depth is capped and the cap is an assertion
    rather than a silent stop.
    """
    assert depth <= MAX_DEPTH, (
        f"{formula!r} on {sheet_name} is more than {MAX_DEPTH} references "
        "deep, which means a cycle or a chain no reader could follow"
    )

    def value_of(token):
        match = REFERENCE.match(token.strip())
        assert match, f"not a cell reference: {token!r} in {formula!r}"
        target = book[match.group("sheet")] if match.group("sheet") else book[sheet_name]
        cell = target.cell(
            row=int(match.group("row")),
            column=column_index_from_string(match.group("column")),
        )
        assert cell.value is not None, (
            f"{formula!r} points at {target.title}!{cell.coordinate}, which "
            "is empty - in Excel that silently reads as zero"
        )
        if isinstance(cell.value, str) and cell.value.startswith("="):
            return evaluate(book, target.title, cell.value, depth + 1)
        return float(cell.value)

    assert formula.startswith("="), formula

    # A plan-year total. Blanks inside the range are SKIPPED rather than
    # treated as an error, because that is what Excel's SUM does - but a
    # range that is entirely empty is a wrong range, not a zero total, and
    # that is worth failing on.
    call = SUM_CALL.match(formula)
    if call:
        start = REFERENCE.match(call.group("start").replace("$", ""))
        end = REFERENCE.match(call.group("end").replace("$", ""))
        assert start and end, formula
        assert start.group("column") == end.group("column"), (
            f"{formula!r} sums across columns, which this exporter never does"
        )
        column = column_index_from_string(start.group("column"))
        sheet = book[sheet_name]
        total, seen = 0.0, 0
        for number in range(int(start.group("row")), int(end.group("row")) + 1):
            cell = sheet.cell(row=number, column=column)
            if cell.value is None:
                continue
            if isinstance(cell.value, str):
                if not cell.value.startswith("="):
                    continue
                total += evaluate(book, sheet_name, cell.value, depth + 1)
            else:
                total += float(cell.value)
            seen += 1
        assert seen, f"{formula!r} sums a range with nothing in it"
        return total

    body = formula[1:]
    for operator, apply in (
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        # Next Dollar's plan-year saving: a per-item figure times the
        # volume it is spread over. The one product the exporter writes,
        # and it is still a total - the figure a leader would otherwise
        # reach for a calculator to get.
        ("*", lambda a, b: a * b),
    ):
        if operator in body:
            left, right = body.split(operator, 1)
            return apply(value_of(left), value_of(right))
    return value_of(body)


# ---------------------------------------------------------------------------
# The shape of the workbook (spec Section 8's tab table)
# ---------------------------------------------------------------------------


def test_every_section_eight_tab_exists_in_the_spec_s_order(book):
    """Spec Section 8 names ten tabs; the workbook has ten, in that order.

    Task 6a's brief said seven. The spec's table is the source of truth and
    lists ten, which export.py's own docstring already said, so all ten are
    created and the eight that are not filled yet are labelled - flagged for
    Siri rather than resolved by picking a number.
    """
    expected = [name for name, _description in export.TABS]
    assert book.sheetnames == expected
    assert len(expected) == 10
    # The workbook opens on the tab a leader should read first.
    assert book.active.title == "Summary"


def test_every_tab_says_it_is_synthetic(book):
    """Including the placeholders.

    A workbook gets forwarded with its provenance stripped off, and a tab
    that looks blank is exactly the one somebody screenshots. So the label
    is on all ten, not on a cover sheet.
    """
    for name in book.sheetnames:
        sheet = book[name]
        assert "SYNTHETIC" in str(sheet["A1"].value), name
        assert "SYNTHETIC" in str(sheet["A2"].value), name
        assert "FICTIONAL" in str(sheet["A2"].value), name


def test_the_file_name_is_labelled_synthetic(baseline):
    assert "SYNTHETIC" in baseline["paths"]["workbook_xlsx"]
    assert export.workbook_path(baseline).name.endswith(".xlsx")
    assert "SYNTHETIC" in export.workbook_path(baseline).name


def test_every_section_eight_tab_is_filled(book):
    """Task 6c filled the last four, so there are no placeholders left.

    This test replaces the one that checked the placeholders were
    labelled. It is kept rather than deleted because the thing worth
    guarding has only moved: a tab that silently stopped being written
    would now leave a BLANK sheet where a placeholder used to say what
    was coming, which is strictly worse.
    """
    assert set(export.BUILT_TABS) == {name for name, _d in export.TABS}
    for name, _description in export.TABS:
        sheet = book[name]
        # A filled tab has a title, the synthetic line, and content below
        # the header row. A blank sheet has none of that.
        assert str(sheet["A1"].value or "").endswith("SYNTHETIC"), name
        assert sheet.max_row > export.DETAIL_HEADER_ROW, name
        values = sum(
            1
            for row in sheet.iter_rows()
            for cell in row
            if cell.value is not None
        )
        assert values > 20, (name, values)


def test_every_written_value_is_the_model_s_own_figure(book, comparison):
    """Spec Section 9, cell by cell, for all 21 runs.

    To the cent means to the cent: `abs=0.005` on money, and exact on the
    week counts, which are integers and have no business being close.
    """
    sheet = book["Capacity Scenarios"]
    first = export.SCENARIO_FIRST_DATA_ROW
    seen = set()

    for offset in range(len(comparison)):
        row = first + offset
        name = sheet[f"{export.SCENARIO_LETTERS['Scenario']}{row}"].value
        assert name in comparison.index, name
        seen.add(name)
        expected = comparison.loc[name]

        for header, source, _fmt, _width in export.SCENARIO_COLUMNS:
            if source in (None, "scenario"):
                continue
            cell = sheet[f"{export.SCENARIO_LETTERS[header]}{row}"]
            assert cell.value == pytest.approx(float(expected[source]), abs=0.005), (
                name,
                header,
            )
            if header == "Weeks over the 5-day goal":
                assert cell.value == float(expected[source])

    # Every run is on the tab, and no row is a duplicate.
    assert seen == set(comparison.index)
    assert len(seen) == 21
    assert sheet.max_row == first + len(comparison) - 1


def test_the_baseline_leads_the_table_because_the_formulas_point_at_it(book):
    """Its row is the one every difference is measured against."""
    sheet = book["Capacity Scenarios"]
    first = export.SCENARIO_FIRST_DATA_ROW
    assert sheet[f"A{first}"].value == "Baseline"
    # And the rest are alphabetical - the only order that does not smuggle
    # in an opinion on a tab whose job is to be neutral.
    rest = [
        sheet[f"A{row}"].value
        for row in range(first + 1, sheet.max_row + 1)
    ]
    assert rest == sorted(rest)


# ---------------------------------------------------------------------------
# Section 9: every live formula evaluates to the model's own difference
# ---------------------------------------------------------------------------


def test_the_all_in_formula_is_labour_plus_machine(book, comparison):
    """The one TOTAL on the tab, and spec 6.4's whole point made visible.

    All-in is deliberately NOT written as a value: it is a formula over the
    two halves, so a reader who doubts that labour alone misleads can see
    the sum in the formula bar.
    """
    sheet = book["Capacity Scenarios"]
    letter = export.SCENARIO_LETTERS["All-in cost"]
    for offset in range(len(comparison)):
        row = export.SCENARIO_FIRST_DATA_ROW + offset
        name = sheet[f"A{row}"].value
        formula = sheet[f"{letter}{row}"].value
        assert isinstance(formula, str) and formula.startswith("="), name
        got = evaluate(book, "Capacity Scenarios", formula)
        assert got == pytest.approx(
            float(comparison.loc[name, "total_cost"]), abs=0.005
        ), name
        # And it really is the sum of the two written halves.
        assert got == pytest.approx(
            float(comparison.loc[name, "labour_cost"])
            + float(comparison.loc[name, "machine_cost"]),
            abs=0.005,
        ), name


def test_every_vs_baseline_formula_evaluates_to_the_model_s_difference(
    book, comparison
):
    """Spec Section 9, on the formulas rather than the values.

    The failure this is for is a reference one row or one column out: a
    plausible number appears, nothing looks broken, and the comparison a
    leader acts on is wrong. So every difference cell is evaluated and
    matched against the model's own subtraction.
    """
    sheet = book["Capacity Scenarios"]
    sources = {
        header: source for header, source, _f, _w in export.SCENARIO_COLUMNS
    }
    baseline = comparison.loc["Baseline"]
    checked = 0

    for offset in range(len(comparison)):
        row = export.SCENARIO_FIRST_DATA_ROW + offset
        name = sheet[f"A{row}"].value
        for difference, measure in export.VS_BASELINE:
            formula = sheet[f"{export.SCENARIO_LETTERS[difference]}{row}"].value
            assert isinstance(formula, str) and formula.startswith("="), (
                name,
                difference,
            )
            got = evaluate(book, "Capacity Scenarios", formula)
            source = sources[measure]
            if source is None:  # the all-in total, itself a formula
                want = float(comparison.loc[name, "total_cost"]) - float(
                    baseline["total_cost"]
                )
            else:
                want = float(comparison.loc[name, source]) - float(baseline[source])
            assert got == pytest.approx(want, abs=0.005), (name, difference)
            checked += 1

    assert checked == len(comparison) * len(export.VS_BASELINE)

    # The baseline's own differences are ZERO, not blank: a blank reads as
    # "not measured" and a zero reads as "measured, and it is the reference".
    first = export.SCENARIO_FIRST_DATA_ROW
    for difference, _measure in export.VS_BASELINE:
        formula = sheet[f"{export.SCENARIO_LETTERS[difference]}{first}"].value
        assert evaluate(book, "Capacity Scenarios", formula) == pytest.approx(0.0)


def test_a_known_figure_survives_the_whole_chain(book, comparison):
    """One hand-checked row, end to end, as a reader would read it.

    The hub: +$980 all-in against the baseline, and 0.14 days of delivery
    added to every request. If every other test in this file passed and
    this one failed, the workbook would be internally consistent and
    quoting the wrong operation.
    """
    sheet = book["Capacity Scenarios"]
    row = None
    for candidate in range(export.SCENARIO_FIRST_DATA_ROW, sheet.max_row + 1):
        if sheet[f"A{candidate}"].value == "Central records hub":
            row = candidate
    assert row is not None

    all_in = evaluate(
        book, "Capacity Scenarios",
        sheet[f"{export.SCENARIO_LETTERS['All-in vs baseline']}{row}"].value,
    )
    turnaround = evaluate(
        book, "Capacity Scenarios",
        sheet[f"{export.SCENARIO_LETTERS['Turnaround vs baseline']}{row}"].value,
    )
    assert all_in == pytest.approx(980.0, abs=1.0)
    assert turnaround == pytest.approx(0.139, abs=0.001)
    # No machine in this scenario, so all-in is labour.
    assert sheet[f"{export.SCENARIO_LETTERS['Machine cost']}{row}"].value == 0.0


# ---------------------------------------------------------------------------
# The Summary (spec Section 8, curated headline block, decided 11/03)
# ---------------------------------------------------------------------------


def test_the_summary_holds_no_numbers_of_its_own(book):
    """Every figure on it is a reference into Capacity Scenarios.

    Which is the point: a Summary that recomputed anything would be a
    second model written in Excel, and when the two disagreed the
    spreadsheet would win the argument because it is the thing in the room.
    """
    sheet = book["Summary"]
    numbers = 0
    for row in sheet.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            if isinstance(cell.value, (int, float)):
                raise AssertionError(
                    f"Summary {cell.coordinate} holds a literal number "
                    f"({cell.value!r}); every figure must reference "
                    f"'Capacity Scenarios'"
                )
            if isinstance(cell.value, str) and cell.value.startswith("="):
                assert "Capacity Scenarios" in cell.value, cell.coordinate
                numbers += 1
    assert numbers > 0


def test_every_summary_formula_evaluates_to_the_model_s_figure(book, comparison):
    """Spec Section 9, across the tab boundary."""
    sheet = book["Summary"]
    scenarios_sheet = book["Capacity Scenarios"]
    rows = {
        scenarios_sheet[f"A{row}"].value: row
        for row in range(
            export.SCENARIO_FIRST_DATA_ROW, scenarios_sheet.max_row + 1
        )
    }
    by_letter = {
        letter: header for header, letter in export.SCENARIO_LETTERS.items()
    }
    sources = {header: source for header, source, _f, _w in export.SCENARIO_COLUMNS}
    differences = dict(export.VS_BASELINE)
    baseline = comparison.loc["Baseline"]
    checked = 0

    for row in sheet.iter_rows():
        for cell in row:
            if not (isinstance(cell.value, str) and cell.value.startswith("=")):
                continue
            match = REFERENCE.match(cell.value[1:])
            assert match, f"{cell.coordinate}: {cell.value!r}"
            header = by_letter[match.group("column")]
            name = scenarios_sheet[f"A{match.group('row')}"].value
            assert name in rows, (cell.coordinate, name)

            got = evaluate(book, "Summary", cell.value)
            if header in differences:
                measure = differences[header]
                source = sources[measure] or "total_cost"
                want = float(comparison.loc[name, source]) - float(baseline[source])
            else:
                source = sources[header] or "total_cost"
                want = float(comparison.loc[name, source])
            assert got == pytest.approx(want, abs=0.005), (cell.coordinate, name, header)
            checked += 1

    assert checked > 0


def test_the_summary_headlines_are_the_curated_eight(book):
    """Spec Section 8 and the 11/03 decision: a curated block, not 21 rows.

    Twenty-one scenarios is a reference table. The Summary is the eight a
    leader would act on, each with the sentence that says what the run
    found - because a difference column tells you a scenario is cheaper and
    does not tell you it bought that by answering the phone 40 seconds
    slower.
    """
    sheet = book["Summary"]
    column_a = [
        str(sheet[f"A{row}"].value)
        for row in range(1, sheet.max_row + 1)
        if sheet[f"A{row}"].value is not None
    ]
    named = [name for name, _finding in export.HEADLINES]
    assert len(named) == 8
    assert len(set(named)) == 8
    for name in named:
        assert name in column_a, name

    # Each has its one-line finding beside it, in the last column.
    last = chr(ord("A") + len(export.SUMMARY_HEADLINE_COLUMNS) - 1)
    findings = {
        str(sheet[f"A{row}"].value): sheet[f"{last}{row}"].value
        for row in range(1, sheet.max_row + 1)
    }
    for name, finding in export.HEADLINES:
        assert findings[name] == finding
        assert 80 < len(finding) < 260, name


def test_the_headlines_name_scenarios_that_actually_ran(comparison):
    """A curated list is a list of literals, so it can go stale.

    If a scenario were renamed, the Summary would reference a row that
    moved. The writer raises rather than writing a wrong reference, and
    this is the test that would have caught it before the raise did.
    """
    for name, _finding in export.HEADLINES:
        assert name in comparison.index, name


def test_the_summary_says_what_all_in_cost_means(book, baseline):
    """The note block, and it is not decoration.

    All-in cost is a definition this project changed on 11/03, and the
    whole reason it changed is that quoting one half misleads. A workbook
    that showed the number without the definition would reintroduce the
    problem it was meant to fix.
    """
    sheet = book["Summary"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    notes = export.summary_notes(baseline, 21)
    for note in notes:
        assert note in text, note[:50]

    # The four things a reader has to know before quoting a figure.
    assert "ALL-IN COST = labour + machine" in text
    assert "SYNTHETIC DATA" in text
    assert "TURNAROUND is the requester's view" in text
    assert "THE ROSTER IS HELD" in text
    assert "Capacity Scenarios" in text

    # And the pointer to the full table counts the runs rather than
    # stating a number that would go stale.
    assert "AND 13 MORE RUNS" in text


# ---------------------------------------------------------------------------
# Readability (task 6a item 4)
# ---------------------------------------------------------------------------


def test_everything_is_arial(book):
    """One font, on every written cell.

    Not a style preference: a workbook in two fonts is how a reader can
    tell which cells were added later, and it is the first thing that makes
    an output look unowned.
    """
    for name in book.sheetnames:
        for row in book[name].iter_rows():
            for cell in row:
                if cell.value is None:
                    continue
                assert cell.font.name == export.FONT, (name, cell.coordinate)


def test_both_built_tabs_freeze_their_headers(book):
    """So row 4's headers stay visible down 21 rows of scenarios."""
    scenarios_sheet = book["Capacity Scenarios"]
    assert scenarios_sheet.freeze_panes == f"B{export.SCENARIO_FIRST_DATA_ROW}"
    assert book["Summary"].freeze_panes == "A5"
    for name in book.sheetnames:
        assert book[name].freeze_panes is not None, name


def test_negatives_are_in_parentheses_on_every_number(book):
    """In a column holding both a saving and a cost, a leading minus is
    easy to miss at a glance and a bracket is not - and the difference
    between (51,569) and 51,569 is the whole recommendation."""
    for fmt in (export.MONEY, export.HOURS, export.FTE, export.DAYS,
                export.COUNT, export.PERCENT, export.MONEY_CENTS,
                export.PER_DOLLAR):
        assert ";" in fmt, fmt
        negative = fmt.split(";", 1)[1]
        assert negative.startswith("(") and negative.endswith(")"), fmt
        assert "-" not in negative, fmt

    sheet = book["Capacity Scenarios"]
    for header, _source, number_format, _width in export.SCENARIO_COLUMNS:
        if number_format is None:
            continue
        cell = sheet[
            f"{export.SCENARIO_LETTERS[header]}{export.SCENARIO_FIRST_DATA_ROW}"
        ]
        assert cell.number_format == number_format, header


def test_money_hours_and_days_each_have_their_own_format(book):
    """A dollar read as a day is a different kind of wrong from a rounding
    error, so the three never share a format."""
    sheet = book["Capacity Scenarios"]
    row = export.SCENARIO_FIRST_DATA_ROW
    fmt = lambda header: sheet[f"{export.SCENARIO_LETTERS[header]}{row}"].number_format

    assert fmt("All-in cost").startswith("$")
    assert fmt("Labour cost").startswith("$")
    assert fmt("Machine cost").startswith("$")
    assert fmt("Overtime hours") == export.HOURS
    assert fmt("Uncovered hours") == export.HOURS
    assert fmt("Requester turnaround, mean days") == export.DAYS
    assert fmt("Network required FTE") == export.FTE
    assert fmt("Weeks over the 5-day goal") == export.COUNT
    assert len({fmt("All-in cost"), fmt("Overtime hours"), fmt(
        "Requester turnaround, mean days"
    )}) == 3


def test_every_column_has_a_width_so_nothing_shows_as_hashes(book):
    """A money column narrower than its number renders as ####.

    Which is not a cosmetic problem: a leader who cannot read the figure
    widens the column, sees a different number of decimal places than the
    person beside them, and the two quote different totals.
    """
    for name in export.BUILT_TABS:
        sheet = book[name]
        used = {cell.column_letter for row in sheet.iter_rows() for cell in row
                if cell.value is not None}
        for letter in used:
            assert sheet.column_dimensions[letter].width, (name, letter)
            assert sheet.column_dimensions[letter].width >= 10, (name, letter)


def test_the_workbook_has_no_formula_errors(book):
    """The checkable half of "open and recalculate".

    No Excel here, so this cannot prove the file opens. What it can prove
    is that nothing the exporter wrote is already an error literal, and
    that every reference resolves to a tab and a cell that exist and hold
    a number - which is what an Excel recalculate would turn into #REF! or
    #VALUE!.
    """
    errors = ("#REF!", "#VALUE!", "#NAME?", "#DIV/0!", "#N/A", "#NULL!", "#NUM!")
    for name in book.sheetnames:
        sheet = book[name]
        for row in sheet.iter_rows():
            for cell in row:
                if not isinstance(cell.value, str):
                    continue
                for error in errors:
                    assert error not in cell.value, (name, cell.coordinate)
                if cell.value.startswith("="):
                    # Resolves, and to a number rather than to text or a blank.
                    got = evaluate(book, name, cell.value)
                    assert isinstance(got, float), (name, cell.coordinate)


def test_the_workbook_is_the_same_twice(result, baseline, tmp_path):
    """A fixed-seed model should write a byte-stable workbook.

    There is deliberately no run timestamp in the file: a wall-clock date
    would make two runs of the same inputs differ, which would be the one
    place in this repository where that is not true. This is the test that
    keeps somebody from adding one.
    """
    first = tmp_path / "one.xlsx"
    second = tmp_path / "two.xlsx"
    export.build(result, baseline).save(first)
    export.build(result, baseline).save(second)
    # The zip container carries per-file timestamps, so compare the cells
    # rather than the bytes - which is the thing that actually has to match.
    one, two = load_workbook(first), load_workbook(second)
    assert one.sheetnames == two.sheetnames
    for name in one.sheetnames:
        a, b = one[name], two[name]
        assert a.max_row == b.max_row and a.max_column == b.max_column, name
        for row in range(1, a.max_row + 1):
            for column in range(1, a.max_column + 1):
                assert a.cell(row=row, column=column).value == b.cell(
                    row=row, column=column
                ).value, (name, row, column)
