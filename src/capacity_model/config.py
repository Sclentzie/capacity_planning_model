"""Load the YAML input files and find the repository root.

Session 2. (Not in the spec Section 2 file list - a small shared utility, so
that generate_data, forecast, and every later module read the inputs the same
way instead of each re-implementing the loading.)

CLAUDE.md rule this exists to enforce: every input lives in the YAML files and
no numbers are hard-coded in the logic. If a module needs a number, it asks
this loader for it.
"""

from pathlib import Path

import yaml

# config.py lives at <repo>/src/capacity_model/config.py, so the repo root is
# three parents up. Resolved from the file rather than the working directory,
# so a run works from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]

ASSUMPTIONS_FILE = "assumptions.yaml"
SITES_FILE = "sites.yaml"
SKILLS_FILE = "skills.yaml"


def load_yaml(name):
    """Read one YAML input file from the repository root."""
    path = REPO_ROOT / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_assumptions():
    return load_yaml(ASSUMPTIONS_FILE)


def load_sites():
    return load_yaml(SITES_FILE)


def load_skills():
    return load_yaml(SKILLS_FILE)


def load_all():
    """All three input files at once: the full set of inputs for a run."""
    return load_assumptions(), load_sites(), load_skills()


def resolve_path(*parts):
    """Turn repo-relative path pieces from assumptions.yaml into a real path."""
    return REPO_ROOT.joinpath(*parts)


# ---------------------------------------------------------------------------
# Scenario settings (spec Section 7)
#
# A scenario is assumptions.yaml with a few keys replaced, so these read the
# merged config and nothing here knows or cares whether it is looking at the
# baseline. Kept in config.py because both work_volume.py and supply.py need
# them and neither reads the other.
# ---------------------------------------------------------------------------


def scenario(assumptions=None):
    """The scenario block: its name, its starts week, and what it changes."""
    assumptions = assumptions or load_assumptions()
    return assumptions["scenario"]


def scenario_starts_week(assumptions=None):
    """The absolute week index a scenario's changes begin (spec 7).

    The YAML holds a PLAN week, because "week 20" is how the spec table
    says it; everything downstream counts in absolute week indexes. Plan
    week 1 is the first plan week, so the two differ by one.
    """
    assumptions = assumptions or load_assumptions()
    return (
        assumptions["calendar"]["history_weeks"]
        + int(scenario(assumptions)["starts_week"])
        - 1
    )


def scenario_multiplier(name, assumptions=None):
    """One of the scenario multipliers, 1.0 on the baseline."""
    assumptions = assumptions or load_assumptions()
    return float(scenario(assumptions)[f"{name}_multiplier"])
