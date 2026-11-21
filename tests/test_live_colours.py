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
    better (green)   #0c5a44    #7ed5b0
    worse (red)      #9e303b    #f2919a
    no change (blue) #244fb4    #85b8f8
    Choice (violet)  #382456    #e4d0f5
    Risk (orange)    #422405    #f2cda6
    Sensitivity      #343432    #dadde1
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
