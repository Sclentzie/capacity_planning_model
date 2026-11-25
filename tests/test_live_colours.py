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


def test_the_washes_are_the_same_string_in_both_palettes(page):
    """The other half of the fix: what does NOT change.

    A scope tint and a kind's cell are translucent, so the same string
    is sent to both readers and the browser mixes it against whichever
    page is showing. Here that is checked the other way round from the
    tests above - the value must be IDENTICAL after a switch, because
    nothing about it was ever chosen from the mode.
    """
    def swatches():
        return page.evaluate("""
          () => Array.from(document.querySelectorAll('span.cm-swatch'))
            .map(s => getComputedStyle(s).backgroundColor)
        """)

    set_theme(page, "Light")
    pick(page, BETTER_RUN)
    light = swatches()
    assert light, "no key swatches on the page"

    set_theme(page, "Dark")
    assert swatches() == light, (
        "a key swatch changed when the palette did, so it is being "
        "chosen from the mode rather than composited by the browser"
    )


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
# The kind key's swatch (Siri, 11/23)
# ---------------------------------------------------------------------------

# Every swatch in the kind key: how big the browser actually drew it, and
# what it drew it in.
SWATCH_BOXES = """
  () => Array.from(document.querySelectorAll('.cm-key-table .cm-swatch'))
    .map(swatch => {
      const box = swatch.getBoundingClientRect();
      const style = getComputedStyle(swatch);
      return {width: box.width, height: box.height,
              display: style.display,
              background: style.backgroundColor};
    })
"""

# A swatch has to read as a block of colour rather than as a rule under
# the word beside it. Ten pixels each way is the floor; the stylesheet
# asks for about fifteen.
MIN_SWATCH = 10.0


def test_the_kind_key_swatch_is_a_block_and_not_a_line(page):
    """IT WAS A THIN LINE, AND ONLY HERE (Siri's walkthrough, 11/23).

    The same .cm-swatch class is used by both keys and looked right in
    one of them, which is what made this worth measuring rather than
    reading. An inline element has no width or height - the browser
    ignores both - and in this key the swatch is a span inside a table
    cell, so 0.85rem square was silently dropped and all that rendered
    was a background on a zero-height box. In the region key the same
    class is a flex item, and a flex item is blockified, so it drew
    correctly by luck.

    A stylesheet cannot be read to rule that out: the rule was there and
    correct, and the browser was entitled to ignore it. Only the box the
    browser drew says which happened.
    """
    page.set_viewport_size({"width": 1600, "height": 1400})
    page.wait_for_timeout(400)
    try:
        pick(page, views.COMPARE_VIEW)
        page.wait_for_selector(".cm-key-table .cm-swatch", timeout=20000)
        swatches = page.evaluate(SWATCH_BOXES)

        assert len(swatches) == len(views.kind_key_rows()) == 3, swatches
        for swatch in swatches:
            assert swatch["display"] != "inline", (
                "an inline swatch has no width or height: " + str(swatch)
            )
            assert swatch["width"] >= MIN_SWATCH, swatch
            assert swatch["height"] >= MIN_SWATCH, swatch
            # Square-ish, so it reads as a chip rather than as a bar.
            assert abs(swatch["width"] - swatch["height"]) <= 2.0, swatch
            # And actually carrying a colour, not an empty box.
            assert swatch["background"].startswith("rgba("), swatch
            assert not swatch["background"].endswith(", 0)"), swatch
    finally:
        pick(page, views.ONE_RUN_VIEW)
