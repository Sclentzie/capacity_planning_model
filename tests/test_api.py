"""Spec Section 9, Sessions 6 and 7: the doorway, and the saved results.

Task 7a. Section 8 asks for a dashboard showing "the same numbers as the
workbook", and the 11/09 decision puts the model behind one inputs-in,
results-out function so the dashboard and any later tool call it the same
way. These tests cover the two ways that promise can break.

WHAT THE DOORWAY CAN GET WRONG

    IT CAN ACCEPT A KEY THE MODEL DOES NOT HAVE. This is the worst failure
      in the whole design and it is silent: a control named
      `atrition_multiplier` would merge in happily, change nothing, and
      hand back a run that looks exactly like the baseline. A reader would
      conclude that doubling attrition costs nothing. So every route in -
      a top-level key, an assumptions override, a sites override, the
      identity block - is checked to REFUSE a name the baseline does not
      carry.

    IT CAN MISLABEL A RUN. `what_it_changes` is the sentence a reader uses
      to know what they are looking at. It reported "Nothing" for a run
      with demand at 1.15 until the identity filter was narrowed, because
      the model's own scenario block lives at overrides.scenario.* and the
      filter was dropping every key with that prefix. Tested here on a run
      that really does change something.

    IT CAN DISAGREE WITH THE WORKBOOK. The detail tables are the
      workbook's own builders, so this file ties the saved results back to
      them. Those builders are already checked cell by cell against the
      written workbook in test_export_baseline_tabs.py, so the chain runs
      saved file -> builder -> workbook cell.

WHAT THE SAVED FILE CAN GET WRONG

    IT CAN GO STALE. It is the one generated file this repository commits,
      because a hosted app cannot spend nine minutes before it shows
      anything. A committed number that no longer matches the model is
      worse than no number: the workbook would say one thing and the
      dashboard another, with nothing to say which was right. So the big
      test here re-runs every scenario and compares EVERY CELL of every
      saved table. It is meant to fail the day an assumption or a formula
      changes, and the fix is to rebuild the file.

    IT CAN LOSE PRECISION. Figures that tie to the cent have to still tie
      after a save and a load, so the round trip is checked exactly rather
      than approximately.
"""

import math

import pandas as pd
import pytest
import yaml

from capacity_model import api, config, export, scenarios


CENT = 0.005


@pytest.fixture(scope="module")
def baseline_assumptions():
    return config.load_assumptions()


@pytest.fixture(scope="module")
def sites():
    return config.load_sites()


@pytest.fixture(scope="module")
def result(baseline_assumptions, sites):
    """Every scenario, run once - the pass the workbook is written from."""
    return scenarios.run(None, baseline_assumptions, sites)


@pytest.fixture(scope="module")
def fresh(result):
    """The same runs shaped through the doorway, with no file involved."""
    return {
        "comparison": result["comparison"],
        "scenarios": {
            key: api.results(built, key=key)
            for key, built in result["runs"].items()
        },
    }


@pytest.fixture(scope="module")
def saved(baseline_assumptions):
    """What is actually committed in app/data."""
    path = api.results_path(baseline_assumptions)
    if not path.exists():
        pytest.fail(
            f"{path.name} is missing; rebuild it with "
            "`python -m capacity_model.api` (about nine minutes)"
        )
    return api.load_results(path, baseline_assumptions)


@pytest.fixture(scope="module")
def baseline_run(result):
    return result["runs"][export.BASELINE_KEY]


def same_number(left, right):
    """Equal, counting two NaNs as equal - a blank is a blank."""
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return left == right


# ---------------------------------------------------------------------------
# The doorway refuses what it cannot run
# ---------------------------------------------------------------------------


def test_an_unknown_top_level_key_is_refused(baseline_assumptions, sites):
    """`overides` for `overrides` would silently run the baseline."""
    with pytest.raises(ValueError, match="unexpected top-level keys"):
        api.run({"overides": {}}, baseline=baseline_assumptions, sites=sites)


def test_a_typo_in_an_assumptions_override_is_refused(baseline_assumptions, sites):
    """The failure this whole doorway exists to prevent.

    `atrition_multiplier` is not an input. Merged in silence it would
    change nothing, and the run would come back looking entirely
    reasonable - the baseline, reported as a doubled-attrition scenario.
    """
    with pytest.raises(KeyError, match="not a baseline input"):
        api.run(
            {"overrides": {"scenario": {"atrition_multiplier": 2.0}}},
            baseline=baseline_assumptions,
            sites=sites,
        )


def test_a_typo_in_a_sites_override_is_refused(baseline_assumptions, sites):
    """sites.yaml is reachable too, and refuses on the same terms."""
    with pytest.raises(KeyError, match="not a baseline input"):
        api.run(
            {"site_overrides": {"clusterz": {}}},
            baseline=baseline_assumptions,
            sites=sites,
        )


def test_a_typo_in_the_identity_block_is_refused(baseline_assumptions, sites):
    """`start_week` for `starts_week` would quietly start in week 1."""
    with pytest.raises(ValueError, match="unexpected keys in the scenario block"):
        api.run(
            {
                "scenario": {
                    "name": "Typo run",
                    "starts_week": 10,
                    "what_it_changes": "nothing",
                    "start_week": 10,
                }
            },
            baseline=baseline_assumptions,
            sites=sites,
        )


def test_a_file_and_a_dashboard_call_are_refused_in_the_same_words(
    tmp_path, baseline_assumptions, sites
):
    """One validator, so the laxer of two cannot be the way in.

    The same typo is sent down both routes - a YAML file on disk and a
    mapping handed over by a caller - and the errors are compared as
    STRINGS. If these ever became separate code paths, the one that
    allowed more would be the way a wrong number arrived, and nobody
    would be looking at it.
    """
    typo = {"overrides": {"scenario": {"atrition_multiplier": 2.0}}}

    path = tmp_path / "typo.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "scenario": {
                    "name": "Typo run",
                    "starts_week": 1,
                    "what_it_changes": "a misspelled key",
                },
                **typo,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(KeyError) as from_file:
        scenarios.run_scenario(
            scenarios.load_scenario(path), baseline_assumptions, sites
        )
    with pytest.raises(KeyError) as from_caller:
        api.run(typo, baseline=baseline_assumptions, sites=sites)
    assert str(from_file.value) == str(from_caller.value)

    # And a good one arrives identically by either route.
    file_scenario = scenarios.load_scenario(
        scenarios.scenario_dir() / "baseline.yaml"
    )
    typed = api.scenario_from_overrides(
        {
            "scenario": {
                "name": file_scenario["name"],
                "starts_week": file_scenario["starts_week"],
                "what_it_changes": file_scenario["what_it_changes"],
            },
            "overrides": file_scenario["overrides"],
        },
        key="baseline",
    )
    for field in ("key", "name", "starts_week", "what_it_changes", "overrides"):
        assert typed[field] == file_scenario[field]


# ---------------------------------------------------------------------------
# The doorway labels and defaults
# ---------------------------------------------------------------------------


def test_the_identity_defaults_come_from_the_yaml_not_this_module(
    baseline_assumptions,
):
    """A missing starts week falls back to assumptions.yaml's own value.

    CLAUDE.md: no numbers in the logic. The default has to be the
    baseline's input, not a 1 typed into api.py.
    """
    scenario = api.scenario_from_overrides({}, baseline_assumptions)
    assert scenario["starts_week"] == int(
        baseline_assumptions["scenario"]["starts_week"]
    )
    assert scenario["name"] == api.CUSTOM_RUN_NAME
    assert scenario["what_it_changes"] == api.NOTHING_CHANGED


def test_the_label_names_the_input_that_actually_changed():
    """It said "Nothing" on a demand-plus-15 run until this was fixed.

    declared_keys appends the three identity keys as `scenario.name` and
    friends, and the model's own block is `overrides.scenario.*` - so
    filtering the whole prefix threw away the real change. A label that
    under-claims is as misleading as one that over-claims.
    """
    changed = api.describe_overrides(
        {"overrides": {"scenario": {"demand_multiplier": 1.15}}}
    )
    assert "scenario.demand_multiplier" in changed
    assert changed != api.NOTHING_CHANGED

    # And the identity keys themselves are still not reported as changes.
    named = api.describe_overrides(
        {"scenario": {"name": "x", "starts_week": 2, "what_it_changes": "y"}}
    )
    assert named == api.NOTHING_CHANGED

    both = api.describe_overrides(
        {"site_overrides": {"coverage": {"borrow_minimum_hours": 2}}}
    )
    assert "coverage.borrow_minimum_hours" in both


def test_the_doorway_actually_runs_the_model(baseline_assumptions, sites, fresh):
    """A guard against a doorway that cannot move the answer.

    Every refusal test above passes just as well on a function that
    ignores its overrides entirely, so one real custom run has to come
    back DIFFERENT from the baseline - and in the direction spec 9's
    Session 5 direction check names: more volume never lowers cost.
    """
    custom = api.run(
        {"overrides": {"scenario": {"demand_multiplier": 1.15}}},
        baseline=baseline_assumptions,
        sites=sites,
    )
    baseline = fresh["scenarios"][export.BASELINE_KEY]

    assert custom["summary"]["total_cost"] > baseline["summary"]["total_cost"]
    assert (
        custom["summary"]["plan_year_requests"]
        > baseline["summary"]["plan_year_requests"]
    )
    # Same shape as any other run, so the dashboard renders it identically.
    assert sorted(custom) == sorted(baseline)
    for name in api.DETAIL_TABLES:
        assert list(custom[name].columns) == list(baseline[name].columns)


# ---------------------------------------------------------------------------
# The doorway's numbers are the workbook's numbers
# ---------------------------------------------------------------------------


def test_the_detail_tables_are_the_workbook_s_own_builders(
    baseline_run, baseline_assumptions, saved
):
    """Saved file -> builder -> workbook cell, one chain.

    These builders are already checked cell by cell against the written
    workbook in test_export_baseline_tabs.py, so tying the saved results
    to them ties the dashboard to the tab.
    """
    workbook_plan = export.capacity_plan_table(baseline_run, baseline_assumptions)
    workbook_hiring = export.hiring_plan_table(baseline_run, baseline_assumptions)
    workbook_surplus = export.surplus_table(baseline_run, baseline_assumptions)

    saved_baseline = saved["scenarios"][export.BASELINE_KEY]
    pd.testing.assert_frame_equal(
        saved_baseline["plan"], workbook_plan, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        saved_baseline["hiring"], workbook_hiring, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        saved_baseline["surplus"], workbook_surplus, check_dtype=False
    )
    assert len(workbook_plan) == 780      # 15 (role, scope) pairs x 52 weeks


def test_the_comparison_columns_carry_the_workbook_s_headers():
    """The same figure is called the same thing in both places.

    A dashboard column headed differently from the tab is how two people
    reading the same number come away with different figures.
    """
    workbook_headers = {
        header for header, _source, _fmt, _width in export.SCENARIO_COLUMNS
    }
    for header, _key in api.COMPARISON_MEASURES:
        assert header in workbook_headers, header

    sources = {
        header: source
        for header, source, _fmt, _width in export.SCENARIO_COLUMNS
    }
    for header, key in api.COMPARISON_MEASURES:
        if sources[header] is not None:
            assert sources[header] == key, header
        else:
            # The workbook builds this one as a live Excel formula, so it
            # has no source column; all-in cost is labour plus machine and
            # summarise already holds that sum.
            assert header == "All-in cost" and key == "total_cost"


def test_the_cost_lines_add_up_to_the_all_in_cost(fresh):
    """A cost view that nearly reconciles is worse than none.

    Every line of the bill, labour and machine, against the all-in figure
    on the comparison row - to the cent, for all 21 runs.
    """
    for key, run in fresh["scenarios"].items():
        lines = run["cost_lines"]
        assert abs(lines["amount"].sum() - run["summary"]["total_cost"]) < CENT, key

        labour = lines[lines["basis"] != api.MACHINE_BASIS]["amount"].sum()
        machine = lines[lines["basis"] == api.MACHINE_BASIS]["amount"].sum()
        assert abs(labour - run["summary"]["labour_cost"]) < CENT, key
        assert abs(machine - run["summary"]["machine_cost"]) < CENT, key


def test_turnaround_is_labelled_in_plan_weeks_like_every_tab(fresh, baseline_assumptions):
    """Nobody plans in week 171 (export.plan_week's reason)."""
    goal = baseline_assumptions["backlog"]["overall_turnaround_goal_days"]
    plan_weeks = baseline_assumptions["calendar"]["plan_weeks"]
    for key, run in fresh["scenarios"].items():
        table = run["turnaround"]
        assert list(table["plan_week"]) == list(range(1, plan_weeks + 1)), key
        # The flag is the comparison row's own count, not a second opinion.
        assert int(table["over_goal"].sum()) == run["summary"][
            "weeks_over_turnaround_goal"
        ], key
        assert (table["turnaround_days"] > goal).equals(table["over_goal"]), key


# ---------------------------------------------------------------------------
# The saved file cannot go stale in silence
# ---------------------------------------------------------------------------


def test_the_saved_results_match_a_fresh_run_of_the_model(saved, fresh):
    """The expensive one, and the point of the whole file.

    app/data holds the only generated file this repository commits. If it
    drifts from the model, the workbook and the dashboard disagree and
    nothing says which is right - so every cell of every saved table is
    compared against a fresh run of all 21 scenarios.

    This test is MEANT to fail when an assumption or a formula changes.
    The fix is to rebuild the file (`python -m capacity_model.api`, about
    nine minutes), not to loosen the comparison.
    """
    assert set(saved["scenarios"]) == set(fresh["scenarios"])
    pd.testing.assert_frame_equal(
        api.comparison_view(saved["comparison"]),
        api.comparison_view(fresh["comparison"]),
        check_dtype=False,
    )

    for key, fresh_run in fresh["scenarios"].items():
        saved_run = saved["scenarios"][key]
        assert saved_run["name"] == fresh_run["name"], key
        assert saved_run["starts_week"] == fresh_run["starts_week"], key
        assert saved_run["what_it_changes"] == fresh_run["what_it_changes"], key

        assert set(saved_run["summary"]) == set(fresh_run["summary"]), key
        for measure, value in fresh_run["summary"].items():
            assert same_number(
                saved_run["summary"][measure], api._plain(value)
            ), f"{key}.{measure}"

        for measure, value in fresh_run["machine_spend"].items():
            assert same_number(
                saved_run["machine_spend"][measure], api._plain(value)
            ), f"{key}.machine.{measure}"

        for name in api.DETAIL_TABLES:
            pd.testing.assert_frame_equal(
                saved_run[name],
                fresh_run[name],
                check_dtype=False,
                obj=f"{key}.{name}",
            )


def test_the_saved_file_records_the_inputs_it_was_built_from(saved):
    """The cheap half of the staleness check.

    A hash per input file catches the ordinary case - somebody changed an
    assumption and did not rebuild - without running the model. It cannot
    catch a changed FORMULA, which is why the test above exists too.
    """
    assert saved["meta"]["inputs"] == api.inputs_fingerprint()


def test_the_saved_file_says_it_is_synthetic(saved, baseline_assumptions):
    """CLAUDE.md: generated data is labelled wherever it goes."""
    meta = saved["meta"]
    assert meta["synthetic"] is True
    assert "SYNTHETIC" in meta["label"]
    assert "fictional" in meta["label"].lower()
    assert "SYNTHETIC" in api.results_path(baseline_assumptions).name
    assert meta["scenario_count"] == len(
        scenarios.list_scenarios(baseline_assumptions)
    )


def test_the_round_trip_keeps_every_figure_to_the_cent(fresh, baseline_assumptions,
                                                       tmp_path):
    """Saved and loaded has to be the same number, not nearly.

    Written through json, so the floats go out as Python's shortest
    round-trip repr. Checked rather than assumed, because "ties to the
    cent" is the claim the whole workbook rests on.
    """
    path = tmp_path / api.results_path(baseline_assumptions).name
    api.save_results(fresh, baseline_assumptions, path=path)
    reloaded = api.load_results(path, baseline_assumptions)

    for key, run in fresh["scenarios"].items():
        for measure, value in run["summary"].items():
            if isinstance(value, float) and not math.isnan(value):
                assert reloaded["scenarios"][key]["summary"][measure] == value, (
                    f"{key}.{measure}"
                )
        pd.testing.assert_frame_equal(
            reloaded["scenarios"][key]["cost_lines"],
            run["cost_lines"],
            check_dtype=False,
            obj=key,
        )


def test_rebuilding_the_file_twice_gives_the_same_bytes(fresh, baseline_assumptions,
                                                        tmp_path):
    """A fixed seed should mean a fixed file.

    The gzip timestamp is zeroed for this reason: a wall-clock stamp would
    make every rebuild show up as a change in git even when not one number
    moved.
    """
    first = api.save_results(
        fresh, baseline_assumptions, path=tmp_path / "first.json.gz"
    ).read_bytes()
    second = api.save_results(
        fresh, baseline_assumptions, path=tmp_path / "second.json.gz"
    ).read_bytes()
    assert first == second


def test_the_dashboard_opens_the_saved_results_file(saved):
    """Task 7a's whole purpose: prove the pieces connect.

    Streamlit's own AppTest runs the app script headlessly, so this is the
    real app loading the real committed file rather than a mock of either.
    It reads a file and runs no model, so it costs a second.

    What it would catch: a results file the app cannot open, a run missing
    from it, a path shim that works from the repository root and nowhere
    else. All three would leave the model's tests green and the dashboard
    blank.

    It asserts on the FILE reaching the page, not on the page's layout.
    The first version of this test compared the first table on screen
    against the comparison table, and 7b broke it by putting a different
    table there first - a test that failed for a change that was correct.
    What the saved file turns into on screen is tests/test_views.py's
    job.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(config.REPO_ROOT / "app" / "dashboard.py"), default_timeout=180
    ).run()
    assert not app.exception, app.exception

    offered = app.sidebar.radio[0].options
    assert len(offered) == len(saved["scenarios"])
    names = {run["name"] for run in saved["scenarios"].values()}
    assert set(offered) == names

    # The synthetic statement is ON the page. A screenshot of a dashboard
    # travels further than the repository it came from.
    assert any("SYNTHETIC" in warning.value for warning in app.warning)


def test_all_results_runs_the_scenarios_it_is_given(baseline_assumptions, sites):
    """The saver and the test fixture are the same composition.

    Run on one scenario file rather than 21, because what is being checked
    is the shape of what comes back, not the arithmetic inside it.
    """
    only_baseline = [scenarios.scenario_dir(baseline_assumptions) / "baseline.yaml"]
    bundle = api.all_results(baseline_assumptions, sites, only_baseline)

    assert list(bundle["scenarios"]) == [export.BASELINE_KEY]
    assert len(bundle["comparison"]) == 1
    run = bundle["scenarios"][export.BASELINE_KEY]
    for name in api.DETAIL_TABLES:
        assert name in run and len(run[name]) >= 0
    assert run["name"] == "Baseline"
