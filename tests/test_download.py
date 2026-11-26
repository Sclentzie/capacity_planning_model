"""The Excel a reader downloads from the dashboard (spec Section 9, 11/23).

All data is SYNTHETIC.

WHAT THESE CHECK, AND WHY IT IS NOT THE SAME AS THE WORKBOOK'S TESTS
--------------------------------------------------------------------
The workbook's own tests tie its cells to the model. These tie the
download to the SCREEN: a file that is beautifully formatted and says
something other than the table it was taken from is worse than no file,
because it travels and the page does not. So every figure in the file is
checked against the figure the dashboard is showing, to the cent, and the
things a file needs that a page does not - which run, what was filtered,
and for a custom run what was changed - are checked for presence rather
than assumed.

These run off the SAVED results, like the dashboard does, so they take a
second rather than the eight minutes a scenario sweep costs.
"""

import pandas as pd
import pytest
from openpyxl.utils import get_column_letter

from capacity_model import api, download, export, scenarios, views

# Every error literal Excel can put in a cell, so a formula that does not
# resolve is caught here rather than in front of a reader.
ERRORS = ("#REF!", "#VALUE!", "#NAME?", "#DIV/0!", "#N/A", "#NULL!", "#NUM!")


@pytest.fixture(scope="module")
def saved():
    return api.load_results()


@pytest.fixture(scope="module")
def dimensions():
    return api.dimensions()


@pytest.fixture(scope="module")
def view(saved):
    """The detail table the page shows for one run, unfiltered."""
    plan = saved["scenarios"]["ai_targeted"]["plan"]
    return views.plan_view(views.filter_plan(plan, None, None))


@pytest.fixture(scope="module")
def filters(dimensions):
    choice = views.scope_choices(dimensions)[0]
    roles = [role["code"] for role in dimensions["roles"]]
    return views.filter_lines(dimensions, choice, roles)


@pytest.fixture(scope="module")
def plan_sheet(view, filters):
    book = download.plan_book(view, run_name="AI targeted", filters=filters)
    return book[download.PLAN_SHEET]


@pytest.fixture(scope="module")
def comparison(saved):
    return views.comparison_table(saved)


@pytest.fixture(scope="module")
def comparison_sheet(comparison):
    book = download.comparison_book(comparison)
    return book[download.SCENARIOS_SHEET]


def header_row(sheet, columns):
    """Which row the header landed on, found rather than assumed.

    The preamble above it is as long as the file needs - a custom run
    carries its changed inputs there - so a test that counted rows would
    be testing its own arithmetic.
    """
    first = columns[0][0]
    for row in range(1, 40):
        if sheet[f"A{row}"].value == first:
            return row
    raise AssertionError(f"no header row carrying {first!r}")


def written(sheet, columns, count):
    """The block read back as dicts, the way a reader would read it."""
    start = header_row(sheet, columns) + 1
    rows = []
    for offset in range(count):
        rows.append(
            {
                header: sheet[f"{get_column_letter(index)}{start + offset}"].value
                for index, (header, *_rest) in enumerate(columns, start=1)
            }
        )
    return rows


# ---------------------------------------------------------------------------
# The file says what the screen says
# ---------------------------------------------------------------------------


def test_every_figure_in_the_file_is_the_figure_on_the_screen(plan_sheet, view):
    """To the cent, column by column, row by row.

    THE WHOLE POINT OF THE DOWNLOAD. A reader takes this file into a
    meeting the page is not in, and the moment one number differs the
    file wins the argument - it is the thing in the room. Checked at full
    precision rather than at the precision either one prints, because
    rounding is a display and a difference is a defect.
    """
    rows = written(plan_sheet, download.PLAN_COLUMNS, len(view))
    assert len(rows) == len(view) == 780

    names = export.role_names()
    for index, row in enumerate(rows):
        shown = view.iloc[index]
        for header, source, _format, _width, _total in download.PLAN_COLUMNS:
            if source is None:
                # The hours gap, which is a live formula here as it is on
                # the tab: checked for liveness below, not for value.
                continue
            value = row[header]
            expected = shown[header]
            if expected is None or (
                isinstance(expected, float) and pd.isna(expected)
            ):
                assert value is None, (header, index)
            elif header == export.ROLE_HEADER:
                assert value == names[expected], (header, index)
            elif isinstance(expected, str):
                assert value == expected, (header, index)
            else:
                assert value == pytest.approx(float(expected), abs=1e-9), (
                    header, index,
                )


def test_the_hours_gap_is_still_a_live_formula(plan_sheet, view):
    """A reader sees the subtraction, exactly as the workbook shows it.

    Written out rather than computed into a value, because the one thing
    a spreadsheet can do that a picture of a table cannot is show its
    own arithmetic.
    """
    rows = written(plan_sheet, download.PLAN_COLUMNS, 3)
    for row in rows:
        formula = row[views.HOURS_GAP_HEADER]
        assert isinstance(formula, str) and formula.startswith("=")
        assert "-" in formula


def test_the_comparison_file_is_the_ranked_table_on_screen(
    comparison_sheet, comparison
):
    """Same runs, same order: the biggest saving first, not alphabetical."""
    rows = written(
        comparison_sheet, download.comparison_columns(comparison), len(comparison)
    )
    assert [row[download.SCENARIO_HEADER] for row in rows] == list(
        comparison[download.SCENARIO_HEADER]
    )
    for index, row in enumerate(rows):
        assert row[views.ALL_IN_HEADER] == pytest.approx(
            float(comparison[views.ALL_IN_HEADER].iloc[index]), abs=1e-9
        )


def test_the_difference_column_is_live_against_the_baseline_row(
    comparison_sheet, comparison
):
    """A difference is a subtraction a reader can see and check.

    Pointed at the Baseline ROW wherever the ranking put it, with the row
    locked by a $ so a sort or a fill cannot slide it - which is the one
    way a spreadsheet silently starts differencing the wrong thing.
    """
    columns = download.comparison_columns(comparison)
    rows = written(comparison_sheet, columns, len(comparison))
    baseline = list(comparison[download.SCENARIO_HEADER]).index(views.BASELINE_LABEL)
    baseline_row = header_row(comparison_sheet, columns) + 1 + baseline

    for index, row in enumerate(rows):
        formula = row[views.DIFFERENCE_HEADER]
        assert isinstance(formula, str) and formula.startswith("=")
        assert f"${baseline_row}" in formula, formula
    assert rows[baseline][views.DIFFERENCE_HEADER].endswith(f"${baseline_row}")


# ---------------------------------------------------------------------------
# What a file needs that a page does not
# ---------------------------------------------------------------------------


def test_the_file_says_it_is_synthetic(plan_sheet, comparison_sheet):
    """A file travels further than the page it came from."""
    for sheet in (plan_sheet, comparison_sheet):
        head = " ".join(
            str(sheet[f"A{row}"].value) for row in range(1, 6)
            if sheet[f"A{row}"].value
        )
        assert "SYNTHETIC" in head
        assert "fictional" in head.lower() or "FICTIONAL" in head


def test_the_filters_travel_with_the_file(view, dimensions):
    """A filtered extract that does not say it is filtered is a wrong total.

    Somebody opening this cold has no sidebar to look at. If they sum a
    column of six sites' rows believing it is the network, nothing on the
    page or in the file tells them otherwise - so the filter does.
    """
    north = next(
        choice for choice in views.scope_choices(dimensions)
        if choice["code"] == "north"
    )
    lines = views.filter_lines(dimensions, north, ["tier_1"])
    book = download.plan_book(view, run_name="AI targeted", filters=lines)
    sheet = book[download.PLAN_SHEET]
    head = " ".join(
        str(sheet[f"A{row}"].value) for row in range(1, 10)
        if sheet[f"A{row}"].value
    )
    assert "Cluster North" in head
    assert "N1, N2, N3" in head
    # Named as a person would say it, here as everywhere else.
    assert "Tier 1" in head
    assert "tier_1" not in head


def test_a_custom_run_carries_the_inputs_it_changed(view, filters):
    """A custom run's inputs exist nowhere else.

    The 21 saved runs are described in the workbook and in the spec. A run
    somebody built in the panel is described only by what they typed, so a
    file without it is unreadable a week later - the same rule the custom
    CSV has followed since 7c.
    """
    changes = ["Phone service goal: 80% in 30 seconds (was 80% in 20)"]
    book = download.plan_book(
        view, run_name=api.CUSTOM_RUN_NAME, filters=filters, changes=changes,
    )
    sheet = book[download.PLAN_SHEET]
    head = [
        str(sheet[f"A{row}"].value) for row in range(1, 14)
        if sheet[f"A{row}"].value
    ]
    assert "INPUTS CHANGED" in head
    assert changes[0] in head
    assert head.index("INPUTS CHANGED") < head.index(changes[0])


def test_a_custom_run_that_changed_nothing_says_so(view, filters):
    """A blank section reads as "not recorded"; a sentence reads as a fact."""
    book = download.plan_book(
        view, run_name=api.CUSTOM_RUN_NAME, filters=filters, changes=[],
    )
    sheet = book[download.PLAN_SHEET]
    head = " ".join(
        str(sheet[f"A{row}"].value) for row in range(1, 14)
        if sheet[f"A{row}"].value
    )
    assert "INPUTS CHANGED" in head
    assert "Nothing changed" in head


def test_a_saved_run_has_no_inputs_changed_section(plan_sheet):
    """It would be an empty promise: the workbook already describes it."""
    head = " ".join(
        str(plan_sheet[f"A{row}"].value) for row in range(1, 10)
        if plan_sheet[f"A{row}"].value
    )
    assert "INPUTS CHANGED" not in head


# ---------------------------------------------------------------------------
# The styling, which is the workbook's and not a second set of rules
# ---------------------------------------------------------------------------


def test_the_header_row_is_the_dashboard_slate_in_white_bold(
    plan_sheet, comparison_sheet
):
    """The page's own slate, so the file and the page look like one product.

    White on this slate is 7.36:1, well past the 4.5 AA asks of body
    text, and a header is larger and bolder than body text. It was the
    accent teal until 11/25, at 4.84:1.
    """
    for sheet, columns in (
        (plan_sheet, download.PLAN_COLUMNS),
        (comparison_sheet, (("Scenario",),)),
    ):
        cell = sheet[f"A{header_row(sheet, columns)}"]
        assert cell.fill.fgColor.rgb.endswith(export.HEADER_COLOUR), sheet.title
        assert cell.font.bold
        assert cell.font.color.rgb.endswith("FFFFFF")
        assert cell.font.name == export.FONT

    assert views.contrast_ratio(
        "#ffffff", f"#{export.HEADER_COLOUR}"
    ) >= views.AA_TEXT


def test_the_header_slate_is_the_page_s_own_slate():
    """One colour, named in one place, used in two (Siri, 11/25).

    The workbook records the hex rather than importing it, because views
    imports export and the arrow cannot point both ways. This is what
    keeps the record honest.

    The slate is the back office's own hue - what the NETWORK scope is
    written in and what the CSV button is edged with - so the header row
    sits with the sand, the lilac and the violet in the cells under it.
    The accent teal it replaced is the colour of every slider and focus
    ring on the dashboard, chosen because it means nothing; a workbook
    has no sliders, so there it meant nothing twice.
    """
    assert f"#{export.HEADER_COLOUR}".lower() == views.SCOPE_HUES[
        views.NETWORK_REGION
    ][0].lower()
    # And no longer the accent, which is the change itself.
    assert f"#{export.HEADER_COLOUR}".lower() != views.ACCENT_COLOURS[
        views.LIGHT_MODE
    ].lower()
    # White bold text on it, with room to spare: this is the half of the
    # change a reader actually sees.
    assert views.contrast_ratio(
        "#ffffff", f"#{export.HEADER_COLOUR}"
    ) >= 6.0


def test_the_tints_are_what_the_page_paints_in_light_mode():
    """Excel has no transparency, so the page's washes are mixed down.

    The page tints a cell with a translucent wash on purpose, so the
    browser re-mixes it against whichever palette is showing. A
    spreadsheet cell is one opaque colour and a spreadsheet is a
    light-mode document, so each fill here must be exactly that wash
    composited over the cream page - checked rather than eyeballed,
    because six hexes typed into a second file is six chances to drift.
    """
    page = views.PAGE_COLOURS[views.LIGHT_MODE]
    for region, wash in views.REGION_TINTS.items():
        mixed = views.composite_over(wash, page)
        assert f"#{export.SCOPE_TINT_COLOURS[region]}".lower() == mixed, region
    for kind, wash in views.KIND_TINTS.items():
        mixed = views.composite_over(wash, page)
        assert f"#{export.KIND_TINT_COLOURS[kind]}".lower() == mixed, kind


def test_black_arial_on_every_tint_is_readable():
    """A tint that needs its own text colour is a tint we cannot use.

    Every cell in this workbook is black Arial, so each fill has to clear
    AA under it. All six do, with room: the tightest is 8.5:1.
    """
    text = views.TEXT_COLOURS[views.LIGHT_MODE]
    tints = list(export.SCOPE_TINT_COLOURS.values()) + list(
        export.KIND_TINT_COLOURS.values()
    )
    ratios = [views.contrast_ratio(text, f"#{tint}") for tint in tints]
    assert min(ratios) >= views.AA_TEXT
    assert min(ratios) >= 8.0


def test_the_scope_cell_wears_its_own_region(plan_sheet, view):
    """The tint labels the scope; it does not judge the row.

    One cell, not the whole row, for the reason the page gives: a tinted
    row would read as a verdict on the figures in it.
    """
    headers = [header for header, *_rest in download.PLAN_COLUMNS]
    letter = get_column_letter(headers.index(export.SCOPE_HEADER) + 1)
    money = get_column_letter(headers.index("Required hours") + 1)
    start = header_row(plan_sheet, download.PLAN_COLUMNS) + 1

    seen = set()
    for offset in range(len(view)):
        row = start + offset
        scope = plan_sheet[f"{letter}{row}"].value
        region = export.scope_regions()[scope]
        assert plan_sheet[f"{letter}{row}"].fill.fgColor.rgb.endswith(
            export.SCOPE_TINT_COLOURS[region]
        ), (scope, row)
        # And nothing else in the row is tinted.
        assert plan_sheet[f"{money}{row}"].fill.fgColor.rgb in ("00000000", None)
        seen.add(region)
    assert seen == set(export.SCOPE_TINT_COLOURS)


def test_the_kind_cell_wears_its_own_kind(comparison_sheet, comparison):
    """Choice, risk and sensitivity, coloured as the Compare-all table is.

    And the BASELINE ROW IS NOT ONE OF THEM (11/25). It says "Baseline"
    on screen, because on a table ranked against the baseline it is the
    line the others are measured from rather than something anybody is
    proposing - so the downloaded file says the same word and leaves the
    cell uncoloured, which falls out of the fill lookup rather than
    being special-cased here.
    """
    columns = download.comparison_columns(comparison)
    headers = [header for header, *_rest in columns]
    letter = get_column_letter(headers.index(views.KIND_HEADER) + 1)
    start = header_row(comparison_sheet, columns) + 1

    seen, baselines = set(), 0
    for offset, kind in enumerate(comparison[views.KIND_HEADER]):
        cell = comparison_sheet[f"{letter}{start + offset}"]
        assert cell.value == kind
        tint = export.KIND_TINT_COLOURS.get(kind.lower())
        if tint is None:
            assert kind == views.BASELINE_KIND_LABEL, kind
            baselines += 1
            continue
        assert cell.fill.fgColor.rgb.endswith(tint), kind
        seen.add(kind.lower())
    assert seen == set(scenarios.KINDS)
    assert baselines == 1, "exactly one row is the baseline"


def test_no_column_is_narrower_than_what_it_holds(
    plan_sheet, view, comparison_sheet, comparison
):
    """The #### test, which is the one a reader notices first.

    Excel replaces a number too wide for its column with hashes - a
    figure deleted from a file whose job is to carry figures. Measured
    against what each cell actually PRINTS under its own number format,
    including the total row, which is the widest thing on most of these
    blocks and the one a hand-set width always forgets.
    """
    checks = (
        (plan_sheet, download.PLAN_COLUMNS, len(view)),
        (comparison_sheet, download.comparison_columns(comparison), len(comparison)),
    )
    for sheet, columns, count in checks:
        start = header_row(sheet, columns) + 1
        for index, (header, _source, number_format, _width, _total) in enumerate(
            columns, start=1
        ):
            letter = get_column_letter(index)
            width = sheet.column_dimensions[letter].width
            assert width, (sheet.title, header)
            for row in range(start, start + count):
                needs = export.shown_width(sheet[f"{letter}{row}"].value, number_format)
                if needs is None:
                    continue
                assert needs <= width, (sheet.title, header, row, needs, width)

    # And the TOTAL row, which no cell can be measured for: it holds a
    # =SUM() and openpyxl stores the text of a formula, not its result.
    # This is the one the hand-set widths always missed - a plan-year
    # total is two digits longer than any week inside it.
    totalled = 0
    for header, source, number_format, _width, total in download.PLAN_COLUMNS:
        if not total or source is None:
            continue
        totalled += 1
        letter = get_column_letter(
            [name for name, *_r in download.PLAN_COLUMNS].index(header) + 1
        )
        needs = export.shown_width(float(view[source].sum()), number_format)
        assert needs <= plan_sheet.column_dimensions[letter].width, (
            header, needs, plan_sheet.column_dimensions[letter].width,
        )
    assert totalled == 6


def test_a_narrow_column_comes_out_narrow(plan_sheet):
    """Fitted means fitted, not "at least Excel's default".

    This is the half of auto-fit that is easy to miss. openpyxl reports
    13.0 for a column nobody has set and CREATES it in the act of being
    asked, so code that read the current width before widening made every
    column at least 13 - and a column of plan weeks, 1 to 52, sat at 13
    looking deliberate. The floor is the declared width and the header's
    own three lines, and nothing else.
    """
    headers = [header for header, *_rest in download.PLAN_COLUMNS]
    letter = get_column_letter(headers.index("Plan week") + 1)
    width = plan_sheet.column_dimensions[letter].width
    declared = next(
        spec[3] for spec in download.PLAN_COLUMNS if spec[0] == "Plan week"
    )
    assert width == max(declared, export.header_floor("Plan week"))
    assert width < 13


def test_the_header_fits_in_the_three_lines_it_is_given(plan_sheet, comparison):
    """A clipped header is the same fault as #### one row up.

    The header row is 42 points tall and the headers wrap, so three lines
    is all there is. A column fitted to narrow data would cut its own
    header in half without ever showing a hash.
    """
    for columns in (download.PLAN_COLUMNS, download.comparison_columns(comparison)):
        for header, *_rest in columns:
            floor = export.header_floor(header)
            longest = max(len(word) for word in str(header).split())
            assert floor >= longest
            assert floor * export.HEADER_LINES >= len(str(header))


# ---------------------------------------------------------------------------
# Nothing a reader sees is written in the model's own words
# ---------------------------------------------------------------------------


def _strings(sheet):
    return [
        cell.value
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    ]


def test_no_role_code_reaches_a_reader(plan_sheet, comparison_sheet):
    """"tier_1" is the model talking to itself in front of a leader.

    Checked over every string in the file rather than over the Role
    column, because a code can arrive in a sentence as easily as in a
    cell - and the file is forwarded to people who have no glossary.
    """
    codes = set(export.role_names())
    for sheet in (plan_sheet, comparison_sheet):
        for text in _strings(sheet):
            for code in codes:
                assert code not in text, (sheet.title, code, text[:60])


def test_nothing_a_reader_sees_says_labour(plan_sheet, comparison_sheet):
    """US spelling, everywhere a reader looks (Siri, 11/23).

    The stored measure is still `labour_cost`, and deliberately: it is a
    key in the saved results file, so renaming it would be a migration
    rather than a spelling fix. Nothing a reader sees carries it.
    """
    for sheet in (plan_sheet, comparison_sheet):
        for text in _strings(sheet):
            assert "labour" not in text.lower(), (sheet.title, text[:60])
    assert views.LABOUR_HEADER == "Labor cost"
    assert "Labor cost" in [header for header, *_r in download.PLAN_COLUMNS]


def test_no_cell_holds_an_error_literal(plan_sheet, comparison_sheet):
    """Half of "open and recalculate", which is as far as no-Excel goes."""
    for sheet in (plan_sheet, comparison_sheet):
        for text in _strings(sheet):
            for error in ERRORS:
                assert error not in text, (sheet.title, text[:60])


# ---------------------------------------------------------------------------
# The CSV, which is deliberately not touched
# ---------------------------------------------------------------------------


def test_the_csv_still_carries_the_codes_and_the_full_precision(view):
    """The other button's job did not change (Siri, 11/23).

    A downloaded row is matched or joined against a workbook row, and a
    code is what survives that. Plain names are a thing done to a SCREEN
    and to the formatted file; the raw extract keeps the model's own
    words.
    """
    text = views.csv_bytes(view).decode("utf-8")
    first = text.splitlines()[0]
    assert first.split(",")[2] == export.ROLE_HEADER
    assert "tier_1" in text
    assert "Tier 1" not in text
    # The header still follows the workbook, which is where "Labor cost"
    # came from in the first place.
    assert "Labor cost" in first


def test_the_two_downloads_are_named_for_what_they_are():
    assert download.EXCEL_LABEL == "Download Excel (formatted)"
    assert download.CSV_LABEL == "Download CSV (raw data)"
    assert download.plan_name("baseline", "north").endswith(".xlsx")
    assert download.comparison_csv_name().endswith(".csv")
    assert download.EXCEL_MIME.endswith("spreadsheetml.sheet")


def test_the_file_is_bytes_a_browser_can_be_handed(view, filters):
    """st.download_button takes bytes, and an empty file is still a file."""
    book = download.plan_book(view, run_name="AI targeted", filters=filters)
    data = download.excel_bytes(book)
    assert isinstance(data, bytes)
    # A zip container, which is what an .xlsx is.
    assert data[:2] == b"PK"
    assert len(data) > 10_000
