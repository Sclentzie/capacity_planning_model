"""The colours on the page, measured in a real browser (Siri, 11/20).

WHY THIS FILE EXISTS, AND WHY IT IS NOT IN requirements.txt.

Every other test in this project asks Python what colour something is.
That is the wrong question for the bug 11/20 fixed, because the bug was
not in what Python decided - it was that Python's decision stopped being
true. Switching Light/Dark in Streamlit's menu re-themes the browser
WITHOUT rerunning the script, so a shade this app chose stayed on the
page after the palette under it had changed, and no test that asks
Python could ever see it.

So these tests drive the real menu in a real browser and read back what
is actually painted. They are the only tests here that can fail for the
original bug.

They need playwright and a downloaded browser, which are a large
dependency for one file, so they are NOT in requirements.txt and this
module skips itself when they are absent. To run them:

    python -m venv .browser && .browser\\Scripts\\pip install playwright
    .browser\\Scripts\\playwright install chromium
    .browser\\Scripts\\pip install -r requirements.txt
    .browser\\Scripts\\python -m pytest tests/test_live_colours.py

What they measured when they were written, in both palettes, switching
with no refresh - and every one of these matches views.RENDERED_COLOURS:

                     light      dark
    better (green)   #246b27    #7ed5b0
    worse (red)      #9e303b    #f2919a
    no change (blue) #244fb4    #85b8f8
    Choice (violet)  #382456    #e4d0f5
    Risk (orange)    #422405    #f2cda6
    Sensitivity      #343432    #bad8d9

The light "better" green was #0c5a44 until 11/23, when greenColor was set
in the theme file: Streamlit's own green derived to a teal on cream that
read as dark gray. The light and dark greens are no longer the same hue,
on purpose - the two palettes are never on screen together.

The dark Sensitivity shade was #dadde1 until 11/20, when the theme file's
dark grayColor was re-hued to pull Sensitivity off Choice: measured as
painted, those two were 19.3 apart, which is one pale colour in a sidebar
that writes them one above the other. These tests are what measured it.
"""
import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from capacity_model import views

sync_playwright = pytest.importorskip(
    "playwright.sync_api",
    reason="playwright is deliberately not in requirements.txt",
).sync_playwright

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "dashboard.py"

# A run that moves four measures the right way and one the wrong way, so
# green and red are both on the page at once, and one that rounds away,
# which is the only way to get the no-change blue.
BETTER_RUN = "Centralized phones"
WORSE_RUN = "Volume +15%"
NO_CHANGE_RUN = "Wider lunch window"


def free_port():
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def served():
    """The real app, served, for the length of this module.

    Started once rather than per test: a Streamlit boot plus a browser
    launch is most of the runtime here, and nothing these tests do
    changes the server's state in a way the next test could see.
    """
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            with contextlib.closing(socket.socket()) as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            if process.poll() is not None:
                pytest.fail("the app exited before it served anything")
            time.sleep(0.5)
        else:
            pytest.fail("the app never came up")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        with contextlib.suppress(Exception):
            process.wait(timeout=20)


@pytest.fixture(scope="module")
def page(served):
    with sync_playwright() as play:
        browser = play.chromium.launch()
        tab = browser.new_page(viewport={"width": 1600, "height": 1400})
        tab.goto(served, wait_until="networkidle")
        tab.wait_for_timeout(5000)
        yield tab
        browser.close()


def set_theme(tab, label):
    """Switch the palette through the menu a reader actually uses.

    NOT a reload, and that is the entire point: this is the one gesture
    that re-themes the page without rerunning the script. The Escape
    presses are because the menu's popover stays open over the button
    and swallows the next click on it.
    """
    tab.keyboard.press("Escape")
    tab.wait_for_timeout(400)
    tab.click('[data-testid="stMainMenuButton"]')
    item = f'[data-testid="stMainMenuItem-theme-{label}"]'
    tab.wait_for_selector(item, timeout=20000)
    tab.click(item)
    tab.wait_for_timeout(1500)
    tab.keyboard.press("Escape")
    tab.wait_for_timeout(800)


def pick(tab, text):
    for label in tab.query_selector_all('[data-testid="stRadio"] label'):
        if text in label.inner_text():
            label.click()
            tab.wait_for_timeout(2500)
            return
    pytest.fail(f"no radio option matching {text!r}")


def as_hex(css):
    parts = (
        css.replace("rgba(", "").replace("rgb(", "").rstrip(")").split(",")
    )
    return "#%02x%02x%02x" % tuple(int(part) for part in parts[:3])


def painted(tab):
    """Every coloured word on the page: its text, and its real colour.

    getComputedStyle, so this is the colour the browser resolved and not
    the string the app sent. That difference is the whole measurement:
    the app sends ":violet[Choice]" and Streamlit decides what violet is.
    """
    found = tab.evaluate("""
      () => Array.from(document.querySelectorAll('span'))
        .filter(s => s.style && s.style.color)
        .map(s => [s.textContent.trim(), getComputedStyle(s).color])
    """)
    return {text: as_hex(colour) for text, colour in found if text}


def background(tab):
    return as_hex(tab.evaluate("getComputedStyle(document.body).backgroundColor"))


def shades_of(tab, names):
    """What the page painted each of a set of words, keyed by the word."""
    seen = painted(tab)
    return {name: seen[name] for name in names if name in seen}


@pytest.mark.parametrize("mode,label", [
    (views.LIGHT_MODE, "Light"), (views.DARK_MODE, "Dark"),
])
def test_the_kinds_are_painted_the_shades_we_recorded(page, mode, label):
    """The three kind labels, measured rather than assumed.

    These reach the page as ":violet[Choice]" and friends, so the hex is
    Streamlit's choice. views.RENDERED_COLOURS says what that choice is;
    this is what checks that it still is.
    """
    set_theme(page, label)
    assert background(page) == views.page_colour(mode)

    for kind, name in views.KIND_MARKDOWN.items():
        word = views.kind_label(kind)
        shades = shades_of(page, [word])
        assert word in shades, f"{word} was never painted"
        expected = views.rendered_colour(name, mode)
        assert shades[word] == expected, (
            f"{mode}: {word} is painted {shades[word]}, and "
            f"views.RENDERED_COLOURS says {expected}"
        )

        # And readable where it is written, which is the thing the hex
        # is for. Measured against the real background, not a guess.
        ratio = views.contrast_ratio(shades[word], background(page))
        assert ratio >= views.AA_TEXT, f"{mode}: {word} is {ratio:.2f}:1"


@pytest.mark.parametrize("run,colour", [
    (BETTER_RUN, views.BETTER_COLOUR),
    (WORSE_RUN, views.WORSE_COLOUR),
    (NO_CHANGE_RUN, views.SAME_COLOUR),
])
def test_a_card_delta_is_painted_the_shade_we_recorded(page, run, colour):
    """Each of the three card colours, on a run that actually draws it.

    The baseline draws none at all - it has nothing to compare against -
    so a test that only ever loaded the landing page would measure an
    empty set and pass.
    """
    for mode, label in ((views.LIGHT_MODE, "Light"), (views.DARK_MODE, "Dark")):
        set_theme(page, label)
        pick(page, run)
        expected = views.rendered_colour(colour, mode)
        painted_now = set(painted(page).values())
        assert expected in painted_now, (
            f"{mode}: {run} never painted {colour} ({expected}); "
            f"the page had {sorted(painted_now)}"
        )
        ratio = views.contrast_ratio(expected, views.page_colour(mode))
        assert ratio >= views.AA_TEXT, f"{mode}: {colour} is {ratio:.2f}:1"


def test_a_theme_switch_recolours_the_page_with_no_refresh(page):
    """THE BUG, stated as a test. Switch, do not reload, and look.

    This is the one test in the project that could have caught 11/20's
    bug on 11/19. Everything else asks Python what it decided; this asks
    the browser what is on the screen after a reader has done the one
    thing that changes the screen without asking Python again.

    Both directions, because the bug had both: a light shade stranded on
    a charcoal page, and a dark shade stranded on a cream one.
    """
    words = [views.kind_label(kind) for kind in views.KIND_MARKDOWN]

    set_theme(page, "Light")
    pick(page, BETTER_RUN)
    light = shades_of(page, words)
    light_page = background(page)
    assert len(light) == len(words), f"only painted {sorted(light)}"

    # No reload between these three lines. That is the test.
    set_theme(page, "Dark")
    dark = shades_of(page, words)
    dark_page = background(page)

    set_theme(page, "Light")
    back = shades_of(page, words)

    assert light_page != dark_page, "the page background never changed"
    for word in words:
        assert dark[word] != light[word], (
            f"{word} is still {light[word]} on the dark page - this is "
            "the stale-colour bug, and it is back"
        )
        assert back[word] == light[word], (
            f"{word} came back as {back[word]}, not {light[word]}"
        )
        # Readable at both ends, not merely different.
        for shade, under in ((light[word], light_page), (dark[word], dark_page)):
            ratio = views.contrast_ratio(shade, under)
            assert ratio >= views.AA_TEXT, f"{word} {shade} on {under}"


def test_every_figure_in_the_table_follows_the_theme(page):
    """The other half of the rule: what the page does NOT colour.

    The scope and kind words carry a colour on purpose. Every other cell
    sets none at all - "color: inherit" - because the text colour is the
    one thing Streamlit keeps live by itself, and a cell that sets its
    own would be a shade chosen for whichever palette happened to be
    showing. That is the 11/20 bug, one layer in.

    So: the figures have to be painted the page's own text colour in
    both palettes, and have to CHANGE when the palette does, while the
    figures themselves stay the same numbers.

    (The translucent washes this file used to measure here are no longer
    on the page. They are the workbook's now - the scope and kind fills
    in the Excel file are mixed down from them - and a test in
    test_download composites each one and fails if the two drift.)
    """
    def cells():
        return page.evaluate("""
          () => Array.from(document.querySelectorAll('.cm-table tbody td'))
            .slice(0, 40)
            .filter(td => !td.querySelector('.cm-word'))
            .map(td => [td.textContent.trim(), getComputedStyle(td).color])
        """)

    set_theme(page, "Light")
    pick(page, views.ONE_RUN_VIEW)
    page.wait_for_selector(".cm-table tbody td", timeout=20000)
    light = cells()
    assert light, "no figures in the table to measure"
    for _text, colour in light:
        assert as_hex(colour) == views.text_colour(views.LIGHT_MODE), colour

    set_theme(page, "Dark")
    dark = cells()
    assert [text for text, _c in dark] == [text for text, _c in light], (
        "the figures changed when the palette did"
    )
    for _text, colour in dark:
        assert as_hex(colour) == views.text_colour(views.DARK_MODE), colour

    set_theme(page, "Light")


@pytest.fixture(scope="module")
def expected_cards():
    """What every card on every run MEANS, taken from the model.

    THE PAGE IS NOT ASKED. The point of the test below is to compare two
    independent answers - what the model says a measure did, and what
    colour the browser painted it - so the expectation has to come from
    the model and not from anything the page rendered.
    """
    from capacity_model import api

    results = api.load_results()
    return {
        entry["name"]: {
            card["label"]: views.card_colour(card)
            for card in views.headline_cards(results, entry["key"])
        }
        for entry in views.menu_items(results)
    }


def pick_run(tab, name):
    """Click a run in the sidebar by the name the picker shows.

    Clicked through the DOM rather than with a real mouse click, because
    the spacing test below runs with the sidebar COLLAPSED, and a
    collapsed sidebar is nought pixels wide - Playwright refuses to click
    what it cannot see, while the element is still there and still wired
    up. Verified both ways: the run changes and the sidebar stays shut.
    """
    return tab.evaluate("""
      (name) => {
        for (const option of document.querySelectorAll(
               '[data-testid="stRadio"] label')) {
          if (option.innerText.trim().startsWith(name)) {
            option.click();
            return true;
          }
        }
        return false;
      }
    """, name)


def cards_on(tab):
    """Each card now: its label, and the colour its delta line is painted.

    Keyed off the container each card is drawn in, so a card is read as a
    whole rather than by position - the baseline draws no delta at all,
    and reading by position would shift every card after it by one.
    """
    found = tab.evaluate("""
      () => Array.from(document.querySelectorAll('[class*="st-key-cmcard"]'))
        .map(card => {
          const label = card.querySelector('.cm-card-label');
          const delta = card.querySelector('span.stMarkdownColoredText');
          return [label ? label.textContent.trim() : null,
                  delta ? getComputedStyle(delta).color : null];
        })
    """)
    return [(label, as_hex(colour) if colour else None)
            for label, colour in found]


def running(tab):
    """Is Streamlit still rerunning? Its status widget says so."""
    return bool(tab.query_selector('[data-testid="stStatusWidget"]'))


def settled_cards(tab, tries=60, stable=3):
    """The cards once the rerun has finished AND they have stopped moving.

    TWO CONDITIONS, AND BOTH ARE NEEDED. Streamlit's status widget says
    when the script is still running, which rules out reading a page
    that has not been sent its new numbers yet. But the widget going
    away is not the end of it: the six delta lines are six elements, and
    the browser applies them in quick succession, so the row can still
    be half-repainted for a moment after the server is done.

    So this also waits for the reading to repeat unchanged three times.
    Reading once - or once the status widget clears - makes this test
    flaky in the one way a colour test must not be: it would sometimes
    read a row that really IS all one colour, because it caught the
    previous run's, and sometimes pass anyway. That is the exact bug
    this file is here to catch, and a test that can see it by accident
    can also miss it by accident.
    """
    last, repeats = None, 0
    for _ in range(tries):
        now = None if running(tab) else cards_on(tab)
        if now and now == last:
            repeats += 1
            if repeats >= stable:
                return now
        else:
            repeats = 0
        last = now
        tab.wait_for_timeout(120)
    return last


@pytest.mark.parametrize("mode,label", [
    (views.LIGHT_MODE, "Light"), (views.DARK_MODE, "Dark"),
])
def test_every_card_wears_its_own_colour(page, expected_cards, mode, label):
    """EVERY card, on EVERY run, in both palettes (Siri, 11/21).

    WHY THIS EXISTS, AND WHAT IT CATCHES THAT THE TESTS ABOVE DO NOT.
    test_a_card_delta_is_painted_the_shade_we_recorded asks whether a
    colour appears ON THE PAGE for a run that should produce it. That is
    a weaker question than it looks: a page whose six cards were ALL
    green would answer it yes for a run that moved four measures the
    right way, and nothing here would have noticed. Siri reported seeing
    exactly that shape - a row of cards wearing one colour instead of
    each its own - so this asks the question properly: card by card,
    against what the model says that card did, with nothing shared
    between the two sides of the comparison.

    It pins what a reader is actually promised. Not "green is somewhere
    on the page", but "the overtime card is green BECAUSE overtime fell".
    """
    set_theme(page, label)
    # The hex Streamlit paints each colour name as, in this palette, so a
    # painted colour can be read back as the meaning it stands for.
    meaning_of = {views.rendered_colour(name, mode): name
                  for name in views.CARD_COLOURS}

    wrong, checked = [], 0
    for run, wanted in expected_cards.items():
        assert pick_run(page, run), f"no radio option for {run!r}"
        for card, painted in settled_cards(page):
            checked += 1
            want = wanted.get(card)
            got = meaning_of.get(painted) if painted else None
            if want != got:
                wrong.append(
                    f"{mode}/{run}/{card}: the model says {want}, the page "
                    f"painted {got} ({painted})"
                )

    assert checked >= len(expected_cards), "no cards were read at all"
    assert not wrong, (
        "%d of %d cards are not wearing their own colour:\n  %s"
        % (len(wrong), checked, "\n  ".join(wrong[:25]))
    )


# The sidebar, driven the same way and for the same reason. Two
# different controls: Streamlit swaps the collapse button for an expand
# button once the sidebar is shut, and the shut state SURVIVES A RELOAD,
# so a test that collapses it has to open it again itself.
TOGGLE_SIDEBAR = """
  (wanted) => {
    const selector = wanted
      ? '[data-testid="stExpandSidebarButton"] button'
      : '[data-testid="stSidebarCollapseButton"] button';
    const button = document.querySelector(selector)
      || document.querySelector(selector.replace(' button', ''));
    if (!button) return false;
    button.click();
    return true;
  }
"""
SIDEBAR_WIDTH = """
  () => { const bar = document.querySelector('[data-testid="stSidebar"]');
          return bar ? Math.round(bar.getBoundingClientRect().width) : 0; }
"""


def set_sidebar(tab, open_it):
    """Open or close the sidebar, and prove it took."""
    for _ in range(3):
        width = tab.evaluate(SIDEBAR_WIDTH)
        if (width > 0) == open_it:
            return True
        if not tab.evaluate(TOGGLE_SIDEBAR, open_it):
            return False
        tab.wait_for_timeout(1200)
    return (tab.evaluate(SIDEBAR_WIDTH) > 0) == open_it

# Each card's two boxes: the big number's, and the change line's.
CARD_BOXES = """
  () => Array.from(document.querySelectorAll('[class*="st-key-cmcard"]'))
    .map(card => {
      const value = card.querySelector('.cm-card-value');
      const delta = card.querySelector(
        '[data-testid="stMarkdownContainer"] p');
      if (!value || !delta) return null;
      const vb = value.getBoundingClientRect();
      const db = delta.getBoundingClientRect();
      return {label: (card.querySelector('.cm-card-label') || {}).textContent,
              value: value.textContent.trim(),
              delta: delta.textContent.trim(),
              gap: db.top - vb.bottom,
              weight: getComputedStyle(delta).fontWeight};
    }).filter(Boolean)
"""

# A card's number and its change line must not touch, let alone overlap.
# One clear pixel is the test; the design leaves eight.
CLEAR_PIXELS = 1.0

WINDOW_WIDTHS = (1280, 1440, 1920)


def settled_boxes(tab, tries=40, stable=2):
    """The card boxes once the rerun has finished and the layout is still."""
    last, repeats = None, 0
    for _ in range(tries):
        now = None if running(tab) else tab.evaluate(CARD_BOXES)
        if now and now == last:
            repeats += 1
            if repeats >= stable:
                return now
        else:
            repeats = 0
        last = now
        tab.wait_for_timeout(120)
    return last or []


@pytest.mark.parametrize("width", WINDOW_WIDTHS)
def test_a_number_never_covers_its_own_change_line(page, expected_cards, width):
    """THE CARD'S OWN NUMBER SAT ON TOP OF ITS CHANGE LINE (Siri, 11/23).

    Measured before the fix, at every width and on every card: the two
    boxes overlapped by TWELVE pixels. Streamlit puts a negative margin
    under a markdown block and the card's container sets gap:0, so the
    change line was pulled up under the figure - worst where the figure
    was widest, a dollar value with commas, which is exactly where a
    reader most needs to read both.

    Checked at three window widths with the sidebar open and collapsed,
    because the column a card sits in is narrower with the sidebar out,
    and a figure that wraps to two lines is the case most likely to
    close the gap again. Every run, because the widest value in the
    model is the one that will find this.
    """
    page.set_viewport_size({"width": width, "height": 1200})
    page.wait_for_timeout(500)
    tight, narrowest, checked = [], None, 0
    try:
        for collapsed in (False, True):
            assert set_sidebar(page, not collapsed), (
                "could not %s the sidebar" % ("close" if collapsed else "open")
            )
            assert (page.evaluate(SIDEBAR_WIDTH) == 0) == collapsed, (
                "the sidebar is not %s, so this pass measured the same "
                "layout twice" % ("collapsed" if collapsed else "open")
            )
            where = "collapsed" if collapsed else "open"
            for run in expected_cards:
                assert pick_run(page, run), f"no radio option for {run!r}"
                boxes = settled_boxes(page)
                assert boxes, f"{width}/{where}/{run}: no cards on the page"
                for box in boxes:
                    checked += 1
                    if narrowest is None or box["gap"] < narrowest:
                        narrowest = box["gap"]
                    if box["gap"] < CLEAR_PIXELS:
                        tight.append(
                            f"{width}px {where} / {run} / {box['label']}: "
                            f"{box['value']!r} and {box['delta']!r} are "
                            f"{box['gap']:.1f}px apart"
                        )
    finally:
        # Put the page back the way the other tests expect to find it. A
        # reload will NOT do it: Streamlit remembers a collapsed sidebar
        # across one, so it has to be opened through its own control.
        set_sidebar(page, True)
        page.set_viewport_size({"width": 1600, "height": 1400})
        page.wait_for_timeout(1200)

    assert checked, "nothing was measured"
    assert not tight, (
        "%d of %d card readings have the number touching its own change "
        "line (closest %.1fpx):\n  %s"
        % (len(tight), checked, narrowest, "\n  ".join(tight[:20]))
    )


@pytest.mark.parametrize("width", WINDOW_WIDTHS)
def test_the_change_line_is_bold_at_every_width(page, expected_cards, width):
    """Bold on every card (Siri, 11/23).

    Read off the rendered element rather than off the stylesheet, since
    the stylesheet has to win against Streamlit's own paragraph styling
    to take effect at all - which is the part that could quietly stop
    being true.
    """
    page.set_viewport_size({"width": width, "height": 1200})
    page.wait_for_timeout(500)
    try:
        pick_run(page, "Volume +15%")
        boxes = settled_boxes(page)
        assert boxes, "no cards on the page"
        for box in boxes:
            assert int(box["weight"]) >= 600, (
                f"{width}px / {box['label']}: the change line is weight "
                f"{box['weight']}"
            )
    finally:
        page.set_viewport_size({"width": 1600, "height": 1400})
        page.wait_for_timeout(500)


# ---------------------------------------------------------------------------
# The outside reviewers' fixes, measured in a real browser (11/24)
#
# EVERY ONE OF THESE IS A LIGHT-DARK() PAIR, and the whole question is
# whether the browser resolves the pair against the palette that is
# showing AT THE MOMENT IT PAINTS. Nothing in Python can answer that: the
# script sends one string and never runs again when the reader switches.
# So these drive the real menu and read back what was painted.
# ---------------------------------------------------------------------------

# Every coloured word on the page, by the class that colours it.
WORDS = """
  () => {
    const out = {};
    for (const span of document.querySelectorAll('span.cm-word')) {
      for (const name of span.classList) {
        if (name === 'cm-word') continue;
        const style = getComputedStyle(span);
        (out[name] = out[name] || []).push({
          text: span.textContent.trim(),
          colour: style.color,
          weight: style.fontWeight,
        });
      }
    }
    return out;
  }
"""


def words_by_class(tab):
    return tab.evaluate(WORDS)


def expected_shade(pair, mode):
    """Which end of a recorded pair the browser should have picked."""
    return pair[0] if mode == views.LIGHT_MODE else pair[1]


@pytest.mark.parametrize("mode,label", [
    (views.LIGHT_MODE, "Light"), (views.DARK_MODE, "Dark"),
])
def test_every_scope_word_is_bold_and_in_its_own_colour(page, mode, label):
    """The reviewers' first fix, measured where it has to be true.

    One text colour cannot serve both pages - readable text needs 4.5:1,
    and the two pages leave no shade in between - so each scope word goes
    out as a light-dark() pair. This reads back what the browser resolved
    and holds it to the recorded pair, to AA against the real page, and
    to the bold the reviewers asked for.
    """
    set_theme(page, label)
    pick(page, views.ONE_RUN_VIEW)
    page.wait_for_selector("span.cm-word", timeout=20000)

    found = words_by_class(page)
    for region, pair in views.SCOPE_WORD_COLOURS.items():
        name = views.scope_word_class(region)
        assert name in found, f"{mode}: no {region} scope word on the page"
        for word in found[name]:
            painted = as_hex(word["colour"])
            assert painted == expected_shade(pair, mode), (
                f"{mode}: {word['text']} is painted {painted}, and "
                f"views.SCOPE_WORD_COLOURS says {expected_shade(pair, mode)}"
            )
            assert int(word["weight"]) >= 700, word
            ratio = views.contrast_ratio(painted, background(page))
            assert ratio >= views.AA_TEXT, (
                f"{mode}: {word['text']} is only {ratio:.2f}:1"
            )


@pytest.mark.parametrize("mode,label", [
    (views.LIGHT_MODE, "Light"), (views.DARK_MODE, "Dark"),
])
def test_every_kind_word_is_bold_and_in_its_own_colour(page, mode, label):
    """The same fix on the comparison table, where the Kind column is.

    The shades are the ones the theme file declares, so this also checks
    that the word in the table and the word in the sidebar caption - two
    different mechanisms, CSS and Streamlit's markdown - land on one
    colour rather than two that nearly match.
    """
    set_theme(page, label)
    try:
        pick(page, views.COMPARE_VIEW)
        page.wait_for_selector("span.cm-word", timeout=20000)
        found = words_by_class(page)
        for kind, pair in views.KIND_WORD_COLOURS.items():
            name = views.kind_word_class(kind)
            assert name in found, f"{mode}: no {kind} on the page"
            painted = as_hex(found[name][0]["colour"])
            assert painted == expected_shade(pair, mode), (kind, painted)
            assert int(found[name][0]["weight"]) >= 700
            ratio = views.contrast_ratio(painted, background(page))
            assert ratio >= views.AA_TEXT, f"{mode}: {kind} {ratio:.2f}:1"

            # The sidebar says the same word through Streamlit's own
            # markdown. One colour, two mechanisms.
            caption = shades_of(page, [views.kind_label(kind)])
            if views.kind_label(kind) in caption:
                assert caption[views.kind_label(kind)] == painted, kind
    finally:
        pick(page, views.ONE_RUN_VIEW)


def test_switching_the_palette_recolours_every_word_with_no_refresh(page):
    """THE TEST THIS FILE EXISTS FOR, for the words (11/24).

    Not a reload: the menu, which re-themes the browser without rerunning
    the script. Every coloured word has to move to the other end of its
    pair, and nothing may stay where it was - a word that did not move is
    a shade this module picked, which is the bug of 11/20 come back in a
    new place.
    """
    set_theme(page, "Light")
    pick(page, views.ONE_RUN_VIEW)
    page.wait_for_selector("span.cm-word", timeout=20000)
    light = {
        name: as_hex(words[0]["colour"])
        for name, words in words_by_class(page).items()
    }
    assert light, "no coloured words to measure"

    set_theme(page, "Dark")
    dark = {
        name: as_hex(words[0]["colour"])
        for name, words in words_by_class(page).items()
    }

    assert set(light) == set(dark), "a word disappeared on the switch"
    for name in light:
        assert light[name] != dark[name], (
            f"{name} stayed {light[name]} through a palette switch"
        )
    for region, pair in views.SCOPE_WORD_COLOURS.items():
        name = views.scope_word_class(region)
        assert (light[name], dark[name]) == pair, (region, light[name],
                                                   dark[name])
    set_theme(page, "Light")


# What each download button was actually painted: its fill, its edge and
# the colour of the label on it.
BUTTONS = """
  () => {
    const out = {};
    for (const key of ['%s', '%s']) {
      const host = document.querySelector('.st-key-' + key + ' button');
      if (!host) continue;
      const style = getComputedStyle(host);
      out[key] = {fill: style.backgroundColor,
                  edge: style.borderTopColor,
                  text: style.color};
    }
    return out;
  }
""" % (views.EXCEL_BUTTON_KEY, views.CSV_BUTTON_KEY)


@pytest.mark.parametrize("mode,label", [
    (views.LIGHT_MODE, "Light"), (views.DARK_MODE, "Dark"),
])
def test_the_two_downloads_wear_different_colours(page, mode, label):
    """The reviewers' second fix: two buttons, two hues, both palettes.

    A wash and an edge in the hue, and the LABEL left alone - text that
    follows the theme is readable in both palettes without anybody
    choosing a shade for it. So the label's colour has to be the page's
    own text colour, and the two edges have to be the two recorded pairs
    and different from each other.
    """
    set_theme(page, label)
    pick(page, views.ONE_RUN_VIEW)
    page.wait_for_selector(f".st-key-{views.EXCEL_BUTTON_KEY} button",
                           timeout=20000)
    buttons = page.evaluate(BUTTONS)
    assert set(buttons) == {views.EXCEL_BUTTON_KEY, views.CSV_BUTTON_KEY}

    excel = buttons[views.EXCEL_BUTTON_KEY]
    csv = buttons[views.CSV_BUTTON_KEY]
    assert as_hex(excel["edge"]) == expected_shade(views.EXCEL_COLOUR, mode)
    assert as_hex(csv["edge"]) == expected_shade(views.CSV_COLOUR, mode)
    assert as_hex(excel["edge"]) != as_hex(csv["edge"])
    assert as_hex(excel["fill"]) != as_hex(csv["fill"])

    # Translucent, so the page shows through and neither button is a
    # block of colour competing with the table above it.
    for button in (excel, csv):
        assert button["fill"].startswith("rgba("), button["fill"]
        alpha = float(button["fill"].rstrip(")").split(",")[-1])
        assert 0 < alpha < 0.5, button["fill"]

    # The label follows the theme, and reads on the fill it sits on.
    for button in (excel, csv):
        text = as_hex(button["text"])
        assert text == views.text_colour(mode), (mode, text)
        over = views.composite_over(button["fill"], background(page))
        ratio = views.contrast_ratio(text, over)
        assert ratio >= views.AA_TEXT, f"{mode}: label is {ratio:.2f}:1"


# The two banners, as the browser drew them.
BANNERS = """
  () => {
    const out = {};
    for (const [key, selector] of [['synthetic', '.cm-synthetic'],
                                   ['note', '.cm-note']]) {
      const host = document.querySelector(selector);
      if (!host) continue;
      const style = getComputedStyle(host);
      const lead = host.querySelector('.cm-banner-lead');
      out[key] = {fill: style.backgroundColor, text: style.color,
                  edge: style.borderLeftColor,
                  lead: lead ? getComputedStyle(lead).color : null};
    }
    return out;
  }
"""


@pytest.mark.parametrize("mode,label", [
    (views.LIGHT_MODE, "Light"), (views.DARK_MODE, "Dark"),
])
def test_the_synthetic_banner_is_soft_and_still_readable(page, mode, label):
    """The reviewers' third fix: softer than the warning block it replaced.

    Softer is measurable: the fill is translucent rather than solid, so
    the page shows through it. Prominent is measurable too: the lead word
    is in the hue and the edge is the hue. And the sentence still has to
    read, which is what AA against the composited fill says.
    """
    set_theme(page, label)
    pick(page, views.ONE_RUN_VIEW)
    page.wait_for_selector(".cm-synthetic", timeout=20000)
    banner = page.evaluate(BANNERS)["synthetic"]

    assert banner["fill"].startswith("rgba("), banner["fill"]
    alpha = float(banner["fill"].rstrip(")").split(",")[-1])
    assert 0 < alpha <= 0.2, f"the synthetic banner is not soft: {alpha}"

    hue = expected_shade(views.SYNTHETIC_COLOUR, mode)
    assert as_hex(banner["lead"]) == hue
    assert as_hex(banner["edge"]) == hue

    over = views.composite_over(banner["fill"], background(page))
    for colour in (banner["text"], banner["lead"]):
        ratio = views.contrast_ratio(as_hex(colour), over)
        assert ratio >= views.AA_TEXT, f"{mode}: {ratio:.2f}:1 on {over}"


def test_the_note_banner_is_the_dark_design_reversed(page):
    """The reviewers' fourth fix, stated as the relationship they asked for.

    They liked the dark one - a wash of the hue behind bright text of the
    same hue - and asked for light to be that design REVERSED. So the two
    ends have to be opposite: on charcoal the panel is a wash and the
    text is the hue; on cream the hue fills the panel and the text is
    pale. Both have to clear AA, which is the part a reversal could
    quietly cost.
    """
    results = {}
    for mode, label in ((views.LIGHT_MODE, "Light"), (views.DARK_MODE, "Dark")):
        set_theme(page, label)
        pick(page, views.ONE_RUN_VIEW)
        pick(page, BETTER_RUN)
        page.wait_for_selector(".cm-note", timeout=20000)
        banner = page.evaluate(BANNERS)["note"]
        results[mode] = banner

        # The light panel is SOLID - that is what the reversal means -
        # and a solid fill is what the text sits on. The dark one is a
        # wash, so the page underneath has to be mixed in first.
        fill = banner["fill"]
        over = (
            views.composite_over(fill, background(page))
            if fill.startswith("rgba(") else as_hex(fill)
        )
        ratio = views.contrast_ratio(as_hex(banner["text"]), over)
        assert ratio >= views.AA_TEXT, f"{mode}: {ratio:.2f}:1 on {over}"

    light, dark = results[views.LIGHT_MODE], results[views.DARK_MODE]
    # Light: the hue fills the panel and the text is pale.
    assert as_hex(light["fill"]) == views.NOTE_COLOUR[0], light["fill"]
    assert views.relative_luminance(as_hex(light["text"])) > 0.7, light["text"]
    # Dark: the panel is a wash and the text is the hue.
    assert dark["fill"].startswith("rgba("), dark["fill"]
    assert as_hex(dark["text"]) == views.NOTE_COLOUR[1], dark["text"]
    # Which is the reversal, stated once: the two fills are at opposite
    # ends of the lightness range, not two versions of the same panel.
    assert views.relative_luminance(
        as_hex(light["fill"])
    ) < views.relative_luminance(as_hex(light["text"]))
    assert views.relative_luminance(
        as_hex(dark["text"])
    ) > views.relative_luminance(
        views.composite_over(dark["fill"], views.page_colour(views.DARK_MODE))
    )


def test_picking_a_run_puts_the_reader_back_at_the_top(page):
    """The reviewers' fifth fix, measured as a scroll position.

    A run is a whole page: picking one rewrites everything in the main
    column, but the column keeps the scroll position the last run was
    being read at - so a reader looking at week 40 of the table picks a
    new run and lands in the middle of a table with the name of the run
    they chose off the top of the screen.

    The main element rather than the window, because Streamlit scrolls
    the main block and the window never moves.
    """
    set_theme(page, "Light")
    pick(page, views.ONE_RUN_VIEW)
    # A KNOWN STARTING RUN. This module shares one page across its tests,
    # so whichever run the last one left selected is still selected - and
    # picking the run that is already showing is not a change, which is
    # exactly when the page is meant NOT to scroll.
    pick(page, BETTER_RUN)
    page.wait_for_timeout(1000)

    scroller = '[data-testid="stMain"]'
    page.eval_on_selector(scroller, "el => el.scrollTo(0, 1200)")
    page.wait_for_timeout(600)
    moved = page.eval_on_selector(scroller, "el => el.scrollTop")
    assert moved > 400, f"the page did not scroll at all: {moved}"

    pick(page, WORSE_RUN)
    page.wait_for_timeout(2500)
    after = page.eval_on_selector(scroller, "el => el.scrollTop")
    assert after < 50, (
        f"picking a run left the reader at {after}px down the page"
    )


def test_the_table_header_sticks_and_the_body_scrolls(page):
    """What the grid gave and CSS had to give back (11/24).

    The HTML table replaced st.dataframe so a word in it could carry a
    colour. Two of the three things the grid did for free are CSS - a
    sticky header and a scrolling body - and this is what says they are
    really there rather than a table running 780 rows down the page.
    """
    pick(page, views.ONE_RUN_VIEW)
    page.wait_for_selector(".cm-table-wrap", timeout=20000)
    measured = page.evaluate("""
      () => {
        const wrap = document.querySelector('.cm-table-wrap');
        const head = document.querySelector('.cm-table thead th');
        return {
          scrolls: wrap.scrollHeight > wrap.clientHeight + 40,
          height: wrap.clientHeight,
          sticky: getComputedStyle(head).position,
          rows: document.querySelectorAll('.cm-table tbody tr').length,
          tooltip: head.getAttribute('title'),
        };
      }
    """)
    assert measured["sticky"] == "sticky", measured
    assert measured["scrolls"], measured
    assert 200 < measured["height"] < 700, measured
    assert measured["rows"] > 100, measured
    # And the header still says what the column is, which is the third
    # thing the grid did.
    assert measured["tooltip"], "a column header lost its tooltip"
