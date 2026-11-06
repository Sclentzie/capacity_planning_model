"""The headline figures, pinned.

Added 10/26 alongside the plan.py split (spec Section 2) into plan.py,
coverage.py and hiring.py. That change was a reorganization with no
behaviour in it, and the only way to say so with a straight face is to
have something that fails if behaviour moves.

Every other test in this suite checks a RULE: that overtime never exceeds
its cap, that a borrow never costs the lender more than its surplus, that
each shrinkage line averages its annual percentage. Those are the tests
worth having, because they keep holding when the numbers legitimately
change. This file is the opposite kind, and it is here on purpose: four
numbers from the whole model, so a refactor that quietly moved one of them
cannot pass unnoticed.

It is therefore also the test that SHOULD fail whenever an assumption
changes. When it does, check the change was intended and update the
numbers here with the decision date, the way the figures below carry
10/26. A pinned number with no story behind it is worse than no pin at all.
"""

import pytest

from capacity_model import automation, config, plan, work_volume

# Baseline as of 10/26, after task 4e (hiring) and the plan.py split.
EXPECTED_CLASSES = 3
EXPECTED_OVERTIME_HOURS = 2767.6
EXPECTED_UNCOVERED_HOURS = 420.0
EXPECTED_TOTAL_COST = 7_456_636


@pytest.fixture(scope="module")
def built():
    assumptions, sites, _skills = config.load_all()
    work = work_volume.build(assumptions, sites)["work"]
    return plan.build_all(assumptions, sites, work)


def test_the_hiring_plan_still_places_three_classes(built):
    """Spec 6.5. Two Tier 1 classes and one Tier 2, all in the remote pools.

    Nothing triggers at the counter or on the Records bench: N1's worst
    site-week is 0.88 FTE short, just under the 1.0 threshold, and that is
    the right answer - its gap is a heavy Monday against a flat roster, so
    another body would sit idle on Friday.
    """
    classes = built["hiring"]
    assert len(classes) == EXPECTED_CLASSES
    assert list(classes["role"]) == ["tier_2", "tier_1", "tier_1"]
    assert list(classes["class_size"]) == [2, 3, 3]
    assert set(classes["scope"]) == {"NETWORK"}
    assert (classes["flag"] == "").all(), "no hire-by date is in the past"


def test_the_overtime_bill_is_unchanged(built):
    """Spec 6.2, 6.4: 2,767.6 hours across all five roles, after hiring.

    Down from 5,597.3 on the opening roster. Hiring does not add
    throughput where overtime was already covering the work - hours
    applied to a queue are capped at required - it buys the same hours at
    1.0x instead of 1.5x, and buys them reliably.
    """
    hours = built["role_cost"]["overtime_hours"].sum()
    assert hours == pytest.approx(EXPECTED_OVERTIME_HOURS, abs=0.05)
    assert built["before_hiring"]["role_cost"]["overtime_hours"].sum() == pytest.approx(
        5597.3, abs=0.05
    )


def test_the_uncovered_hours_are_unchanged(built):
    """Spec 6.2: 420.0 hours nobody worked, all of them at the counter.

    The back office ends the plan year with none: hiring closed Tier 1's
    13.7 hours. What is left is live work - a walk-in who arrived on a
    Monday when the counter was short, which no amount of Friday slack and
    no back-office catch-up can repay.
    """
    queued = built["cost"]["uncovered_hours"].sum()
    live = built["onsite"]["uncovered_hours"].sum()
    assert queued + live == pytest.approx(EXPECTED_UNCOVERED_HOURS, abs=0.05)
    assert queued == pytest.approx(0.0, abs=1e-6)
    assert live == pytest.approx(EXPECTED_UNCOVERED_HOURS, abs=0.05)


def test_the_total_cost_is_unchanged(built):
    """Spec 6.4: $7,456,636 for the plan year, regular plus overtime.

    Up $197,259 on the opening roster. Regular cost pays everybody on the
    payroll including the people still in compliance training, which is
    precisely why hiring late is expensive.
    """
    total = built["role_cost"]["total_cost"].sum()
    assert total == pytest.approx(EXPECTED_TOTAL_COST, abs=1.0)

    before = built["before_hiring"]["role_cost"]["total_cost"].sum()
    assert before == pytest.approx(7_259_378, abs=1.0)
    assert total - before == pytest.approx(197_259, abs=2.0)


def test_the_split_modules_are_wired_to_the_same_answers(built):
    """plan.py, coverage.py and hiring.py, all reading one build.

    The split (spec Section 2, 10/26) moved code between modules and
    nothing else. If build_all had been left reaching into a stale copy of
    anything, these three would stop agreeing with each other.
    """
    from capacity_model import coverage, hiring

    assert len(built["events"]) == 1284
    assert set(built["events"]["event_type"]) == {"E1", "E2", "E3"}
    assert set(built["events"].columns) == set(
        ["event_id"] + [c for c in coverage.EVENT_COLUMNS if c != "event_id"]
    )

    # The gap table the hiring trigger reads is the one on the result.
    assert len(built["gap_fte"]) == 52 * (6 + 6 + 3)  # onsite, records, pools
    assert hiring.find_candidates(
        built["gap_fte"], built["prepared"], config.load_assumptions()
    ) == []

    # And the turnaround the requester experiences.
    turnaround = built["turnaround"]
    assert len(turnaround) == 52
    assert turnaround["turnaround_days"].mean() == pytest.approx(4.671, abs=0.005)
    assert turnaround["turnaround_days"].max() == pytest.approx(6.418, abs=0.005)


def test_automation_is_off_so_every_pin_above_still_describes_the_baseline(built):
    """Task 4f, 10/26: the automated supply line changes nothing at 0%.

    Spec 5.4 and 5.5 added a machine beside every human role and a path for
    automated work to come back. The baseline keeps all four 7.1 rates at
    zero, so none of it runs - and the four numbers pinned above are the
    proof, because they were measured before the automation code existed.

    This test is here so the claim is made explicitly rather than inferred
    from the others passing: if a future change let automation leak into
    the baseline, the reader should be told which rule was broken, not
    just that a total moved.
    """
    assumptions = config.load_assumptions()
    assert automation.enabled(assumptions) is False
    assert automation.target_share(assumptions) == 0.0

    # Nothing automated, nothing came back, and no column appeared.
    required = built["prepared"]["required"]
    assert "work_kind" not in required.columns
    assert automation.build(assumptions, config.load_sites(), built["prepared"]["work"]).empty
