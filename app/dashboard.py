"""The capacity model dashboard (spec Section 8, Session 7).

Task 7a: the minimum that proves the pieces connect. It loads the saved
results and shows the Summary comparison table, and that is all - design,
filters and the build-your-own panel are 7b and 7c.

WHY IT READS A FILE. Running the 21 scenarios takes about nine minutes.
A hosted app that did that on every page view would be unusable, so the
runs are saved by `python -m capacity_model.api` and committed, and this
reads them. The numbers are the workbook's own: both come through
capacity_model.api, which is the single doorway into the model.

Run it:
    .venv\\Scripts\\python -m streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

import streamlit as st

# The model lives under src/ and is not installed as a package, so the
# path is added here rather than relying on PYTHONPATH - a hosted app has
# no shell to set it in.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from capacity_model import api  # noqa: E402  (after the path is set)

TITLE = "Northbrook Records Services - capacity model (SYNTHETIC)"


@st.cache_data
def saved_results():
    """The saved runs, read once per session rather than per rerun."""
    return api.load_results()


def main():
    st.set_page_config(page_title="Capacity model (synthetic)", layout="wide")
    st.title(TITLE)

    results = saved_results()
    meta = results["meta"]

    # The synthetic statement is on the page, not in a footnote. A
    # screenshot of a dashboard travels further than its repository.
    st.warning(meta["label"], icon=":material/science:")

    st.subheader("Every scenario, side by side")
    st.caption(
        f"{meta['scenario_count']} runs over {meta['plan_weeks']} plan weeks. "
        "All-in cost is labour plus the automation's own invoice. No scenario "
        "re-tunes its roster, so each one reports what the change costs the "
        "operation it actually has."
    )

    # No number formatting here on purpose: 7a is the plumbing, and spec
    # Section 8 gives design and usability their own session. The figures
    # are shown as the model holds them.
    st.dataframe(
        api.comparison_view(results["comparison"]),
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Detail per run - the weekly plan, turnaround, cost lines and the "
        "hiring plan - is in the saved results and gets its own views in "
        "7b. A build-your-own panel that runs the model on demand is 7c."
    )


main()
