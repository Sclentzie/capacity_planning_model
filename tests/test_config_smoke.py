"""Smoke test: the input files load and their most basic promises hold.

This is not yet a spec Section 9 validation check - those arrive with the
logic they check. It is the seatbelt for Session 2: if a share is mistyped or
a date is moved off a Monday, every number downstream is quietly wrong, so we
catch it here before any model code reads the file.
"""

import datetime
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name):
    with open(REPO_ROOT / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def assumptions():
    return load_yaml("assumptions.yaml")


@pytest.fixture(scope="module")
def sites():
    return load_yaml("sites.yaml")


@pytest.fixture(scope="module")
def skills():
    return load_yaml("skills.yaml")


def test_all_three_input_files_load(assumptions, sites, skills):
    """Every input lives in YAML (CLAUDE.md), so all three must parse."""
    assert assumptions and sites and skills


def test_request_type_shares_sum_to_one(assumptions):
    """Spec Section 1: the 11 request-type shares are a mix of all requests.

    If they do not sum to 100%, the model is either inventing or losing
    requests the moment it splits demand by type.
    """
    shares = [rt["share"] for rt in assumptions["request_types"]]
    assert len(shares) == 11, "spec Section 1 defines 11 request types"
    assert sum(shares) == pytest.approx(1.0, abs=1e-9)


def test_plan_and_history_start_on_a_monday(assumptions):
    """Spec Section 2: the grain is weekly and weeks start Monday.

    A week labelled by a date that is not a Monday would shift every
    week-of-year season and holiday lookup by a day or more.
    """
    calendar = assumptions["calendar"]
    for key in ("history_start", "plan_start"):
        start = calendar[key]
        assert isinstance(start, datetime.date), f"{key} must parse as a date"
        assert start.weekday() == 0, (
            f"{key} is {start:%Y-%m-%d}, a {start:%A}; spec Section 2 requires a Monday"
        )
