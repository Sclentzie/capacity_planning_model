"""The build-your-own panel: the levers, the validator, the cache (7c).

THE TEST THAT MATTERS MOST IS THE REPRODUCTION ONE. A panel that runs
the model on inputs a reader assembled is only worth having if it is the
same model: set the controls to what a scenario file says and the figures
have to come back as that scenario's, to the cent. Anything less and the
panel is a second, slightly different model wearing the same name.
"""

import sys
from datetime import time
from pathlib import Path

import pytest

from capacity_model import api, config, controls, export, scenarios, views

A_CENT = 0.005

APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"

# The dashboard's own session-state key for the Advanced box's queued
# changes. Named here rather than imported, because importing the app
# module would run the app.
ADVANCED_STATE = "custom_advanced"


@pytest.fixture(scope="module")
def catalogue():
    return controls.catalogue()


@pytest.fixture(scope="module")
def saved():
    return api.load_results()


# ---------------------------------------------------------------------------
# Every lever the files pull is on the panel
# ---------------------------------------------------------------------------


def _leaves(node, path=()):
    """Every value in an overrides tree, by its dotted path."""
    if isinstance(node, dict):
        for name, value in node.items():
            yield from _leaves(value, path + (str(name),))
    else:
        yield path, node


def scenario_file_levers():
    """What the 21 files actually change, as (block, path) pairs.

    A list of rows - the service goals, the break windows - collapses to
    the path of the list itself, because that is the grain at which the
    model's merge finds a row and the grain a control speaks for.
    """
    levers = set()
    for path in sorted(scenarios.scenario_dir().glob("*.yaml")):
        scenario = scenarios.load_scenario(path)
        for keypath, _value in _leaves(scenario["overrides"]):
            levers.add((controls.ASSUMPTIONS, keypath))
        for keypath, _value in _leaves(scenario.get("site_overrides") or {}):
            levers.add((controls.SITES, keypath))
    # Every file declares one, and it is a lever a reader has to be able
    # to move: when a change starts decides how much of the year it hits.
    levers.add((controls.HEADER, ("starts_week",)))
    return levers


def panel_levers(catalogue):
    """What the panel can reach, at the same grain."""
    return {(control["block"], control["path"]) for control in catalogue}


def test_every_lever_the_scenario_files_pull_is_on_the_panel(catalogue):
    """Spec decisions, 11/16: "every lever the 21 scenarios change".

    Checked against the files themselves rather than against a list
    retyped here, so a lever added to a scenario file later cannot go
    missing from the panel without this failing.
    """
    missing = set()
    reachable = panel_levers(catalogue)
    for block, path in sorted(scenario_file_levers()):
        if (block, path) in reachable:
            continue
        # A control on one row of a list covers the list it edits.
        if any(
            block == other_block and path == other_path[: len(path)]
            for other_block, other_path in reachable
        ):
            continue
        missing.add((block, ".".join(path)))
    assert not missing, f"levers with no control: {sorted(missing)}"


def test_the_panel_does_not_offer_a_key_the_model_has_not_got(catalogue):
    """A control pointing at a key that does not exist would be refused
    by the doorway the first time anybody moved it - which is too late
    to find out. Every control's own path is resolved against the YAML
    here instead.
    """
    assumptions = config.load_assumptions()
    sites = config.load_sites()
    for control in catalogue:
        # Raises KeyError if the path or the row code is not there.
        value = controls.baseline_value(control, assumptions, sites)
        assert value is not None or control["baseline"] is None, control["key"]


def test_the_baseline_beside_a_control_is_the_yaml_s_own_value(catalogue):
    """Read, never restated, so it cannot drift from the model's default."""
    assumptions = config.load_assumptions()
    spot_checks = {
        "demand_multiplier": assumptions["scenario"]["demand_multiplier"],
        "attrition_multiplier": assumptions["scenario"]["attrition_multiplier"],
        "realization": assumptions["automation"]["rates"]["realization"],
        "extra_uplift": assumptions["automation"]["extra_uplift"],
        "starts_week": assumptions["scenario"]["starts_week"],
    }
    by_key = {control["key"]: control for control in catalogue}
    for key, expected in spot_checks.items():
        assert by_key[key]["baseline"] == expected, key


def test_every_range_contains_the_baseline_it_starts_at(catalogue):
    """A slider whose range excluded its own starting value would open
    somewhere nobody chose, and the first thing a reader did with it
    would be to move the run off the baseline by accident.
    """
    for control in catalogue:
        baseline = control["baseline"]
        if control["minimum"] is not None:
            assert control["minimum"] <= baseline <= control["maximum"], (
                f"{control['key']}: baseline {baseline} outside "
                f"{control['minimum']}-{control['maximum']}"
            )
        if control["options"] is not None:
            if control["kind"] == controls.MULTISELECT:
                assert set(baseline) <= set(control["options"]), control["key"]
            else:
                assert baseline in control["options"], control["key"]


def test_every_control_says_what_it_is_for(catalogue):
    """A lever with no explanation is a lever nobody will touch.

    Checked as a written sentence rather than a character count: "The
    latest an unpaid lunch can start." is a complete explanation at 37
    characters, and a length rule would have failed it while passing
    forty characters of restated label.
    """
    for control in catalogue:
        assert control["label"] and control["label"][0].isupper(), control["key"]
        assert control["group"] in controls.GROUP_ORDER, control["key"]

        note = control["help"].strip()
        assert note.endswith((".", "%")), control["key"]
        assert note[0].isupper(), control["key"]
        assert note.lower() != control["label"].lower(), control["key"]


# ---------------------------------------------------------------------------
# Assembling what the panel says
# ---------------------------------------------------------------------------


def test_only_what_moved_is_sent(catalogue):
    """A panel that declared all thirty-six controls on every run would
    make the run's own "what it changes" line meaningless.
    """
    nothing = controls.overrides_from_values(catalogue, {})
    assert nothing == {}

    one = controls.overrides_from_values(catalogue, {"demand_multiplier": 1.15})
    assert one == {"overrides": {"scenario": {"demand_multiplier": 1.15}}}


def test_a_control_moved_and_moved_back_is_not_a_change(catalogue):
    """Compared against the YAML's value, not against where it started
    the session, so putting a slider back is really putting it back.
    """
    baseline = {control["key"]: control["baseline"] for control in catalogue}
    assert controls.overrides_from_values(catalogue, baseline) == {}
    assert controls.describe_changes(catalogue, baseline) == []


def test_two_controls_on_one_row_arrive_as_one_row(catalogue):
    """The break windows and the service goals are rows in a list, found
    by their code. Two controls on the same row have to be merged into
    one row here, or the model would be handed the same code twice.
    """
    mapping = controls.overrides_from_values(
        catalogue, {"lunch_from": "11:30", "lunch_to": "14:30"}
    )
    rows = mapping["overrides"]["breaks_and_lunch"]["reliefs"]
    assert len(rows) == 1
    assert rows[0] == {
        "code": "lunch",
        "window": {"from": "11:30", "to": "14:30"},
    }


def test_the_one_lever_in_sites_yaml_goes_to_the_sites_block(catalogue):
    """The borrow minimum is in sites.yaml, and the doorway takes that
    file's overrides in a block of its own.
    """
    mapping = controls.overrides_from_values(
        catalogue, {"minimum_borrow_hours": 2}
    )
    assert mapping == {"site_overrides": {"travel": {"minimum_borrow_hours": 2}}}


def test_the_summary_reads_like_the_panel_not_like_the_yaml(catalogue):
    """What was changed, above the results, in the control's own words."""
    lines = controls.describe_changes(
        catalogue, {"demand_multiplier": 1.15, "central_phones": True}
    )
    assert any("Volume multiplier: 1.15 (baseline 1)" == line for line in lines)
    assert any("Centralized phones: on (baseline off)" == line for line in lines)


# ---------------------------------------------------------------------------
# The same model, or the panel is not worth having
# ---------------------------------------------------------------------------

# The inputs of five saved scenarios, as a reader would set them in the
# panel. Hand-written on purpose: deriving them from the files would test
# the derivation rather than the panel.
MIRRORED = {
    "volume_plus_15": {"demand_multiplier": 1.15},
    "ai_targeted": {
        "simple_task_share": 0.40,
        "ai_handling": 0.10,
        "automation_displacement": 0.10,
        "only_work_types": ["W5", "W8"],
        "starts_week": 13,
    },
    "wider_lunch_window": {"lunch_to": "14:30"},
    "central_phones": {"central_phones": True, "starts_week": 13},
    "site_shortfall_n1_borrow_2h": {
        "site_shortfall_site": "N1",
        "site_shortfall_share": 0.20,
        "site_shortfall_weeks": 4,
        "minimum_borrow_hours": 2,
        "starts_week": 20,
    },
}


@pytest.mark.parametrize("key", sorted(MIRRORED))
def test_the_panel_reproduces_a_saved_scenario_to_the_cent(key, catalogue, saved):
    """Set the controls to a scenario file's inputs; get its numbers.

    Five runs, covering every shape of control the panel has: a plain
    multiplier, an AI run with a work-type list, a time window inside a
    row of a list, a toggle, and the one lever that lives in sites.yaml.
    Every figure in the summary is checked, not just the headline, and
    the tolerance is a cent.
    """
    mapping = controls.overrides_from_values(catalogue, MIRRORED[key])
    run = controls.run_signature(controls.signature(mapping))

    expected = saved["scenarios"][key]["summary"]
    for measure, value in expected.items():
        mine = run["summary"].get(measure)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        assert isinstance(mine, (int, float)), f"{key}.{measure} missing"
        assert float(mine) == pytest.approx(float(value), abs=A_CENT), (
            f"{key}.{measure}"
        )


def test_a_reproduced_run_is_labelled_a_custom_run(catalogue):
    """It reproduces the figures; it does not claim to be the file.

    The run is named "Custom run (synthetic)" however faithfully it
    copies a scenario, because a screenshot of a card travels further
    than the panel that made it.
    """
    mapping = controls.overrides_from_values(catalogue, {"demand_multiplier": 1.15})
    run = controls.run_signature(controls.signature(mapping))
    assert run["name"] == api.CUSTOM_RUN_NAME
    assert "synthetic" in run["name"].lower()
    assert run["key"] == controls.CUSTOM_KEY
    # And it says what it changed, built from the keys declared.
    assert "demand_multiplier" in run["what_it_changes"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_input_that_does_not_exist_is_refused():
    """The doorway's own validator, reached through the panel's assembly.

    Caught before the model runs, so a typo costs a message rather than
    twenty-five seconds. It has to be check_overrides and not
    scenario_from_overrides: the second builds the run record and does
    not touch the YAML, so it accepts a key that does not exist and the
    refusal arrives at the end of the run instead of the start.
    """
    with pytest.raises(KeyError, match="already exist"):
        api.check_overrides(
            {"overrides": {"automation": {"rates": {"ai_handlingg": 0.1}}}}
        )

    with pytest.raises(KeyError, match="already exist"):
        api.check_overrides({"overrides": {"invented_block": {"anything": 1}}})

    # sites.yaml is validated the same way, through its own block.
    with pytest.raises(KeyError, match="already exist"):
        api.check_overrides(
            {"site_overrides": {"travel": {"minimum_borrow_hourz": 2}}}
        )

    # And a real one is accepted, so the check is not refusing everything.
    api.check_overrides({"overrides": {"scenario": {"demand_multiplier": 1.15}}})


def test_a_combination_the_model_refuses_is_refused(catalogue):
    """Spec 7.1: rates in a group add, and a total above 100% is a typo.

    The panel cannot prevent this - 80% and 80% are each inside their own
    slider's range - so it has to carry the model's refusal through, with
    the model's own sentence.
    """
    mapping = controls.overrides_from_values(
        catalogue, {"ai_handling": 0.8, "automation_displacement": 0.8}
    )
    with pytest.raises(ValueError, match="over 100%"):
        controls.run_signature(controls.signature(mapping))


def test_a_value_of_the_wrong_type_is_refused_before_it_is_run():
    """The Advanced box hands back text, and the coercion says no."""
    assert controls.coerce("0.25", 0.5) == 0.25
    assert controls.coerce("7", 4) == 7
    assert controls.coerce("yes", False) is True
    with pytest.raises(ValueError):
        controls.coerce("soon", 4)
    with pytest.raises(ValueError):
        controls.coerce("maybe", False)


# ---------------------------------------------------------------------------
# Running the same inputs twice
# ---------------------------------------------------------------------------


def test_the_same_inputs_are_not_run_twice(catalogue):
    """A reader comparing two sets of inputs goes back and forth between
    them. Doing the arithmetic again for inputs already run would be the
    slowest possible way to answer a question already answered.
    """
    mapping = controls.overrides_from_values(catalogue, {"demand_multiplier": 1.2})
    signature = controls.signature(mapping)

    first = controls.run_signature(signature)
    before = controls.run_signature.cache_info()
    second = controls.run_signature(signature)
    after = controls.run_signature.cache_info()

    assert after.hits == before.hits + 1
    # The same object, not an equal one: nothing was recomputed.
    assert second is first


def test_the_same_changes_in_a_different_order_are_one_run(catalogue):
    """The signature is sorted, so the panel cannot run the same inputs
    twice just because two controls were moved in the other order.
    """
    one = controls.overrides_from_values(
        catalogue, {"demand_multiplier": 1.3, "central_phones": True}
    )
    other = controls.overrides_from_values(
        catalogue, {"central_phones": True, "demand_multiplier": 1.3}
    )
    assert controls.signature(one) == controls.signature(other)


def test_resetting_forgets_the_runs():
    """Reset to baseline means it, including the remembered results."""
    controls.run_signature(controls.signature({}))
    assert controls.run_signature.cache_info().currsize > 0
    controls.forget_runs()
    assert controls.run_signature.cache_info().currsize == 0


# ---------------------------------------------------------------------------
# The Advanced box
# ---------------------------------------------------------------------------


def test_advanced_search_finds_an_input_by_its_key():
    found = controls.search_inputs("platform_per_week")
    assert found
    assert found[0]["path"] == "automation.cost.platform_per_week"
    assert found[0]["baseline"] == 500


def test_advanced_search_finds_an_input_by_what_it_is_for():
    """The description is the comment assumptions.yaml carries above the
    input, so searching in plain words finds something.
    """
    found = controls.search_inputs("shrinkage")
    assert found, "nothing found for a word the file uses throughout"
    assert any(row["description"] for row in found), (
        "no description came through from the YAML comments"
    )


def test_the_advanced_box_does_not_offer_housekeeping_or_lists():
    """Paths, the random seed and the document's own metadata are not
    modelling inputs, and a list cannot be changed safely from a text
    box - it is a row keyed by code.
    """
    paths = {row["path"] for row in controls.searchable_inputs()}
    assert not any(path.startswith("paths.") for path in paths)
    assert not any(path.startswith("random.") for path in paths)
    assert not any(path.startswith("meta.") for path in paths)
    # A real input is there, so the exclusions have not emptied it.
    assert "automation.uptime" in paths


def test_an_advanced_change_joins_the_overrides(catalogue):
    """Typed by key, it lands where the doorway expects it."""
    mapping = controls.overrides_from_values(
        catalogue, {}, {"automation.cost.platform_per_week": 750}
    )
    assert mapping == {
        "overrides": {"automation": {"cost": {"platform_per_week": 750}}}
    }
    # And the doorway takes it, because the key is a real one.
    api.scenario_from_overrides(mapping)


# ---------------------------------------------------------------------------
# What the panel hands the rest of the page
# ---------------------------------------------------------------------------


def test_a_custom_run_is_shown_through_the_browse_view_s_own_code(catalogue, saved):
    """Spliced beside the baseline, so the cards, the charts and the
    table are the same functions the One run view calls. A custom run
    and a saved run cannot be reported differently.
    """
    mapping = controls.overrides_from_values(catalogue, {"demand_multiplier": 1.15})
    run = controls.run_signature(controls.signature(mapping))
    bundle = views.custom_bundle(saved, run)

    assert set(bundle["scenarios"]) == {views.BASELINE_KEY, controls.CUSTOM_KEY}
    assert bundle["meta"] is saved["meta"]

    cards = views.headline_cards(bundle, controls.CUSTOM_KEY)
    assert len(cards) == len(views.CARD_MEASURES)
    cost = next(card for card in cards if card["measure"] == "total_cost")
    # +15% volume costs more, and the card has to say so in red.
    assert cost["difference"] > 0
    assert views.card_colour(cost) == views.WORSE_COLOUR

    # The charts and the table take the same bundle.
    assert not views.weekly_gap(bundle, controls.CUSTOM_KEY).empty
    assert not views.plan_view(bundle["scenarios"][controls.CUSTOM_KEY]["plan"]).empty


def test_the_download_carries_the_inputs_that_were_changed(catalogue, saved):
    """A custom run's inputs exist nowhere else. A file without them
    would be unreadable a week later.
    """
    mapping = controls.overrides_from_values(catalogue, {"demand_multiplier": 1.15})
    run = controls.run_signature(controls.signature(mapping))
    view = views.plan_view(run["plan"])
    changes = controls.describe_changes(catalogue, {"demand_multiplier": 1.15})

    text = views.custom_csv_bytes(view, changes).decode("utf-8")
    head, _, body = text.partition("Plan week,")
    assert "Volume multiplier: 1.15" in head
    assert "synthetic" in head.lower()
    # Every provenance line is a comment, so the data rows still match
    # the workbook's and pandas skips them with comment="#".
    assert all(
        line.startswith(views.CSV_COMMENT)
        for line in head.strip().splitlines()
    )
    assert body, "the rows themselves are still there"
    assert "CUSTOM" in views.custom_csv_name("NETWORK")
    assert "SYNTHETIC" in views.custom_csv_name("NETWORK")


def test_a_custom_run_with_nothing_changed_is_the_baseline(catalogue, saved):
    """The honest edge case: a panel nobody touched is the baseline, and
    every card should say no change rather than a rounding artefact.
    """
    run = controls.run_signature(controls.signature({}))
    bundle = views.custom_bundle(saved, run)
    for card in views.headline_cards(bundle, controls.CUSTOM_KEY):
        assert card["no_change"] is True, card["measure"]
        assert card["difference_text"] == views.NO_CHANGE


# ---------------------------------------------------------------------------
# Spec 7.1's defaults, on the panel (11/17)
# ---------------------------------------------------------------------------

# Spec 7.1's table, retyped here on purpose. This is the one place in the
# project where restating the numbers is the right thing to do: the panel
# reads them from the YAML and from the 7.1 scenario file, and a test that
# read them the same way would only prove that one line of code agrees
# with itself. These are what the spec says, in a reader's hand.
SPEC_71 = {
    "deflection_enabled": True,
    "simple_contact_share": 0.40,
    "ai_deflection": 0.15,
    "self_service": 0.05,
    "easy_contact_ratio": 0.50,
    "escalation_uplift": 0.0,
    "simple_task_share": 0.40,
    "ai_handling": 0.10,
    "automation_displacement": 0.10,
    "easy_task_ratio": 0.50,
    "extra_uplift": 0.0,
    "realization": 1.0,
    "only_work_types": [],
}


def test_the_7_1_defaults_on_the_panel_are_the_ones_the_spec_prints(catalogue):
    """Every field of 7.1's table, and nothing it does not have."""
    assert controls.spec_defaults(catalogue) == SPEC_71


def test_a_7_1_default_is_claimed_only_where_7_1_gives_one(catalogue):
    """The proof-of-concept pair sits in the same group on screen but is
    a Section 7 addition from 11/02. Reading its value out of the 7.1 run
    would have reported the baseline as a "7.1 default", which is a claim
    the spec does not make.
    """
    by_key = {control["key"]: control for control in catalogue}
    for key in ("poc_enabled", "poc_true_error_rate"):
        assert by_key[key]["group"] == controls.AUTOMATION
        assert by_key[key]["spec"] is False
        assert by_key[key]["spec_default"] is None

    # And no control outside that group claims one, because 7.1 is about
    # the AI and automation inputs and nothing else.
    for control in catalogue:
        if control["group"] != controls.AUTOMATION:
            assert control["spec"] is False, control["key"]


def test_the_7_1_defaults_reproduce_the_saved_ai_run_to_the_cent(
    catalogue, saved
):
    """The button is only worth having if it loads the real programme.

    Pressed, plus the one control outside the group that run moves - it
    starts in plan week 13 - the panel has to reproduce the saved "AI and
    automation" run. If it does not, the button is loading a programme of
    its own that happens to be named after the spec.
    """
    values = dict(SPEC_71)
    values["starts_week"] = 13
    mapping = controls.overrides_from_values(catalogue, values)
    run = controls.run_signature(controls.signature(mapping))

    expected = saved["scenarios"]["ai_and_automation"]["summary"]
    for measure, value in expected.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        mine = run["summary"].get(measure)
        assert isinstance(mine, (int, float)), f"{measure} missing"
        assert float(mine) == pytest.approx(float(value), abs=A_CENT), measure


def test_the_7_1_defaults_come_from_the_spec_s_own_run():
    """Read from scenarios/ai_and_automation.yaml merged onto the YAML,
    so the numbers cannot drift from the run the spec describes. Checked
    here because a second copy of 0.40, 0.10 and 0.10 in this module
    would have been the thing that drifted.
    """
    merged = controls.spec_config()
    rates = merged["automation"]["rates"]
    assert (rates["simple_task_share"], rates["ai_handling"]) == (0.40, 0.10)
    assert merged["scenario"]["deflection"]["enabled"] is True
    # The baseline itself has the supply-side rates switched off, which is
    # why the button exists at all.
    assert config.load_assumptions()["automation"]["rates"]["ai_handling"] == 0.0


def test_the_7_1_button_sets_that_group_and_leaves_the_rest_alone():
    """On the page: press it, read the widgets back."""
    app = open_panel()
    app.button(key="load_spec_defaults").click().run()
    assert not app.exception

    by_key = {
        element.key: element
        for group in ("slider", "toggle", "multiselect")
        for element in getattr(app, group)
        if element.key
    }
    assert by_key["ctl_simple_task_share"].value == pytest.approx(0.40)
    assert by_key["ctl_ai_handling"].value == pytest.approx(0.10)
    assert by_key["ctl_automation_displacement"].value == pytest.approx(0.10)
    assert by_key["ctl_deflection_enabled"].value is True
    # Not a "run everything" button: the volume and the start week are
    # decisions of their own and it does not touch them.
    assert by_key["ctl_demand_multiplier"].value == pytest.approx(1.0)
    assert by_key["ctl_starts_week"].value == 1


def test_the_7_1_default_is_shown_beside_the_baseline(catalogue):
    """On the control's own label, not only in its tooltip: a reader
    looking at an AI rate is usually asking how far from 7.1 it is.
    """
    app = open_panel()
    labels = [
        element.label
        for group in ("slider", "toggle", "multiselect")
        for element in getattr(app, group)
        if element.key and element.key.startswith("ctl_")
    ]
    rate = next(label for label in labels if label.startswith("AI handling"))
    assert "baseline 0%" in rate
    assert f"{controls.SPEC_LABEL} 10%" in rate
    # And a control 7.1 says nothing about does not pretend otherwise.
    volume = next(label for label in labels if label.startswith("Volume"))
    assert controls.SPEC_LABEL not in volume


# ---------------------------------------------------------------------------
# Work types by their plain names (11/17)
# ---------------------------------------------------------------------------


def test_the_work_type_picker_offers_plain_names(catalogue):
    """Same rule as the roles: nobody should have to learn the codes to
    use the dashboard. The code stays available for anybody tracing a row
    back to the spec - on the control's tooltip, because Streamlit cannot
    give one option of a multiselect a tooltip of its own.
    """
    control = next(c for c in catalogue if c["key"] == "only_work_types")
    names = control["option_names"]
    assert set(control["options"]) <= set(names)
    assert names["W8"] == "Digital retrieval"

    app = open_panel()
    picker = next(
        element for element in app.multiselect
        if element.key == "ctl_only_work_types"
    )
    assert "Digital retrieval" in picker.options
    assert "W8" not in picker.options
    assert "W8" in picker.help


def test_a_chosen_work_type_is_written_in_words(catalogue):
    """The summary above the results, and the CSV's own provenance line,
    name the rows in words - with the code, which is what the Next Dollar
    ranking and the spec call them.
    """
    lines = controls.describe_changes(
        catalogue, {"only_work_types": ["W5", "W8"]}
    )
    assert any(
        "Authorization validation (W5), Digital retrieval (W8)" in line
        for line in lines
    )
    # Empty is not "nothing automated": it is every eligible row.
    control = next(c for c in catalogue if c["key"] == "only_work_types")
    assert controls.write(control, []) == "all eligible"


# ---------------------------------------------------------------------------
# Break and lunch windows (11/17)
# ---------------------------------------------------------------------------


def test_a_window_that_closes_before_it_opens_is_refused(catalogue):
    problems = controls.window_problems(catalogue, {"lunch_to": "10:30"})
    assert len(problems) == 1
    assert "close after it opens" in problems[0]
    # Equal ends are refused too: a window of no minutes has nowhere to
    # put a thirty-minute lunch.
    assert controls.window_problems(catalogue, {"lunch_to": "11:00"})


def test_a_window_outside_the_opening_hours_is_refused(catalogue):
    """Stricter than the model's own check on purpose. The model asks
    whether any interval falls inside the window, so a lunch window from
    07:00 passes it - and the sites open at 08:00.
    """
    early = controls.window_problems(catalogue, {"lunch_from": "07:00"})
    assert len(early) == 1
    assert "while the site is open" in early[0]

    late = controls.window_problems(
        catalogue, {"afternoon_break_to": "17:30"}
    )
    assert len(late) == 1
    assert "while the site is open" in late[0]


def test_a_window_the_model_could_not_place_is_refused(catalogue):
    """Half-hour intervals: a window of 165 minutes is not a whole number
    of them, which is the third thing the model refuses.
    """
    problems = controls.window_problems(catalogue, {"lunch_from": "11:15"})
    assert len(problems) == 1
    assert "30-minute intervals" in problems[0]


def test_the_lunch_windows_the_saved_runs_use_are_not_refused(catalogue):
    """The check has to be exactly as strict as the model, and no more.
    Both lunch runs are legal windows and have to stay legal.
    """
    assert controls.window_problems(catalogue, {}) == []
    assert controls.window_problems(catalogue, {"lunch_to": "14:30"}) == []
    assert controls.window_problems(
        catalogue,
        {"lunch_to": "14:30", "afternoon_break_from": "14:30"},
    ) == []


def test_the_window_is_checked_against_this_run_s_opening_hours(catalogue):
    """The opening hours are an input too, and the Advanced box reaches
    them. Checked against the merged config rather than against the file,
    so a window legal at 08:00 is refused in a run that opens at 09:00.
    """
    mapping = controls.overrides_from_values(
        catalogue, {"lunch_from": "08:00"}, {"intraday.site_hours.opens": "09:00"}
    )
    merged = controls.merged_config(mapping)
    assert merged["intraday"]["site_hours"]["opens"] == "09:00"

    problems = controls.window_problems(
        catalogue, {"lunch_from": "08:00"}, merged
    )
    assert len(problems) == 1
    assert "open 09:00 to 17:00" in problems[0]
    # The same window is fine in a run that opens at eight.
    assert controls.window_problems(catalogue, {"lunch_from": "08:00"}) == []


def test_an_impossible_window_costs_a_message_not_a_run():
    """On the page, before the model runs: the refusal is the point.

    The model would refuse it too, as a sentence about interval masks,
    twenty-five seconds in.
    """
    app = open_panel()
    opens = next(
        element for element in app.time_input if element.key == "ctl_lunch_from"
    )
    # Past the window's own closing time of 14:00.
    opens.set_value(time(15, 0)).run()
    before = controls.run_signature.cache_info()

    app.button(key="run_custom").click().run()
    assert not app.exception
    assert any("close after it opens" in note.value for note in app.error)

    after = controls.run_signature.cache_info()
    assert (after.hits, after.misses) == (before.hits, before.misses)


# ---------------------------------------------------------------------------
# The Advanced box's own guards (11/17)
# ---------------------------------------------------------------------------


def test_a_share_outside_nought_to_a_hundred_is_refused():
    """The hole this closes: a slider has ends and a text box has none,
    so automation.uptime at 500% - a machine up five times as often as
    there is time - was accepted, because no formula complains about it.
    """
    with pytest.raises(ValueError, match="between 0% and 100%"):
        controls.guard("automation.uptime", 5.0)
    with pytest.raises(ValueError, match="between 0% and 100%"):
        controls.guard("automation.work_types.W1.qa_sample_rate", -0.2)
    # And the value it already has is fine, or the box would be useless.
    assert controls.guard("automation.uptime", 0.9) == 0.9


def test_an_uplift_keeps_its_hundred_per_cent_cap():
    """Siri 11/17: the uplifts stay shares. An escalation uplift above
    100% would mean a contact that tried AI first takes more than twice
    as long as an average one, which is not a planning assumption - and
    the slider that offers it on the panel stops at 25%.
    """
    assert controls.input_rule("scenario.deflection.escalation_uplift") == (
        controls.SHARE_RULE
    )
    with pytest.raises(ValueError, match="between 0% and 100%"):
        controls.guard("automation.extra_uplift", 1.5)
    assert controls.guard("automation.extra_uplift", 0.05) == 0.05


def test_an_amount_cannot_be_negative():
    for path, value in (
        ("supply.roles.tier_1.hourly_cost", -28),
        ("automation.cost.platform_per_week", -1),
        ("fte.paid_hours_per_week", -40),
        ("supply.roles.tier_1.training_weeks", -3),
    ):
        with pytest.raises(ValueError, match="cannot be negative"):
            controls.guard(path, value)
    assert controls.guard("supply.roles.tier_1.hourly_cost", 0) == 0


def test_a_multiplier_cannot_be_negative_but_may_be_zero():
    """Siri 11/17. Zero was refused in the first build, on the reasoning
    that a multiplier of nothing is meaningless. It is not: attrition at
    zero is "nobody leaves", and the Supply slider has offered it since
    7c - a text box that refused what the slider beside it allows would
    have been the page contradicting itself.
    """
    with pytest.raises(ValueError, match="cannot be negative"):
        controls.guard("demand.holiday.factor", -0.5)
    assert controls.guard("scenario.attrition_multiplier", 0) == 0
    assert controls.guard("demand.holiday.factor", 0.6) == 0.6

    # And the slider really does offer it, which is what this agrees with.
    catalogue = controls.catalogue()
    attrition = next(
        c for c in catalogue if c["key"] == "attrition_multiplier"
    )
    assert attrition["minimum"] == 0.0


def test_a_value_with_no_range_to_be_outside_of_is_left_alone():
    """A name, a date, a true or false, and a growth rate that can
    legitimately be negative: no rule is invented for these.
    """
    assert controls.guard("onsite_team.primary_goal", "walk_ins") == "walk_ins"
    assert controls.guard("shrinkage.normalize_to_annual_rate", True) is True
    assert controls.input_rule("demand.annual_growth") is None
    assert controls.guard("demand.annual_growth", -0.02) == -0.02


def test_every_input_passes_the_rule_its_own_name_earns_it():
    """THE TEST THAT KEEPS THIS HONEST. The kind of number an input is is
    read off its name, which is a guess - a good one, and a guess. This
    puts every input in assumptions.yaml through the rule its name earns
    it, at the value the file actually holds. A name the classification
    reads wrongly - the random seed under `shrinkage`, a cost that
    happens to be called a rate - shows up here as its own baseline being
    refused, which is the loudest way for it to show up.
    """
    for row in controls.searchable_inputs():
        controls.guard(row["path"], row["baseline"])


def test_a_value_far_from_its_baseline_warns_but_still_runs():
    """Yellow, not red (11/17). The model is not wrong at ten times the
    platform fee; nobody has ever run it there.
    """
    warning = controls.tested_range_warning(
        "automation.cost.platform_per_week", 5000, 500
    )
    assert warning and controls.TESTED_RANGE_NOTE in warning
    # It is a warning and not a refusal, so the guard stays quiet.
    controls.guard("automation.cost.platform_per_week", 5000)


def test_the_ordinary_sensitivity_runs_do_not_warn():
    """Half the realization and twice the attrition are the sensitivity
    runs this project is built on. A warning that fired on those would be
    a warning nobody read.
    """
    assert controls.tested_range_warning(
        "automation.rates.realization", 0.5, 1.0
    ) is None
    assert controls.tested_range_warning(
        "scenario.attrition_multiplier", 2.0, 1.0
    ) is None
    # Three times is inside; past it is not.
    assert controls.tested_range_warning("automation.uptime", 0.3, 0.1) is None
    assert controls.tested_range_warning("automation.uptime", 0.35, 0.1)


def test_nothing_is_claimed_about_a_baseline_of_zero():
    """The rule is a ratio and there is no multiple of zero. The four
    automation rates sit at zero in the baseline, so a ratio rule would
    have warned about every AI run there is.
    """
    assert controls.tested_range_warning(
        "automation.rates.ai_handling", 0.10, 0.0
    ) is None
    assert controls.tested_range_warning("automation.extra_uplift", 0.2, 0.0) is None


def test_the_advanced_box_refuses_and_warns_on_the_page():
    """Through the real widgets: type a value, press Add.

    Refused values never join the run; allowed-but-far ones do, and say
    so in yellow beside the change.
    """
    app = open_panel()
    app.text_input(key="advanced_search").set_value("uptime").run()
    app.text_input(key="advanced_value").set_value("5.0").run()
    app.button(key="advanced_add").click().run()
    assert not app.exception
    assert any("between 0% and 100%" in note.value for note in app.error)
    assert not app.session_state[ADVANCED_STATE]

    # A value inside the rule but a long way from the baseline: queued,
    # and warned about.
    app.text_input(key="advanced_value").set_value("0.2").run()
    app.button(key="advanced_add").click().run()
    assert not app.exception
    assert app.session_state[ADVANCED_STATE] == {"automation.uptime": 0.2}
    assert any(
        controls.TESTED_RANGE_NOTE in note.value for note in app.warning
    )


# ---------------------------------------------------------------------------
# The panel on the page
# ---------------------------------------------------------------------------


def app_test():
    """The real dashboard, run headlessly."""
    from streamlit.testing.v1 import AppTest

    source = str(APP.parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return AppTest.from_file(str(APP), default_timeout=600)


def open_panel():
    """The dashboard with the build-your-own view showing."""
    app = app_test()
    app.run()
    picker = next(element for element in app.sidebar.radio if element.key == "view")
    picker.set_value("Build your own").run()
    assert not app.exception
    return app


def test_every_control_reaches_the_page(catalogue):
    """One widget per catalogue record, found by its own key.

    By key rather than by count: a panel that drew thirty-six widgets
    but drew one of them twice would pass a count.
    """
    app = open_panel()
    drawn = {
        element.key
        for group in ("slider", "toggle", "selectbox", "multiselect",
                      "number_input", "time_input")
        for element in getattr(app, group)
        if element.key and element.key.startswith("ctl_")
    }
    expected = {"ctl_" + control["key"] for control in catalogue}
    assert drawn == expected


def test_nothing_runs_until_the_button_is_pressed(catalogue):
    """Streamlit reruns on every widget change. A panel that ran the
    model on each of those would spend twenty-five seconds answering a
    question nobody had finished asking.
    """
    app = open_panel()
    before = controls.run_signature.cache_info()

    slider = next(element for element in app.slider
                  if element.key == "ctl_demand_multiplier")
    slider.set_value(1.45).run()
    assert not app.exception

    after = controls.run_signature.cache_info()
    assert (after.hits, after.misses) == (before.hits, before.misses)
    # And it says so, rather than leaving a blank page.
    assert any("press Run" in note.value for note in app.info)


def test_pressing_run_shows_the_result_in_the_browse_view_s_cards():
    """End to end, on the page: set a lever, press Run, read the cards.

    The figure checked here is the one the +15% volume risk run saves,
    which is the point of the whole panel: the same inputs give the same
    numbers, whether they come from a file or from a reader.
    """
    app = open_panel()
    slider = next(element for element in app.slider
                  if element.key == "ctl_demand_multiplier")
    slider.set_value(1.15).run()
    app.button(key="run_custom").click().run()
    assert not app.exception

    assert api.CUSTOM_RUN_NAME in [head.value for head in app.subheader]
    assert any(
        "Volume multiplier: 1.15" in str(block.value) for block in app.markdown
    )

    saved = api.load_results()
    expected = views.format_value(
        float(saved["scenarios"]["volume_plus_15"]["summary"]["total_cost"]),
        export.MONEY,
    )
    cards = [
        str(block.value) for block in app.markdown
        if "cm-card-label" in str(block.value)
    ]
    assert cards, "no cards reached the page"
    assert any("All-in cost" in card for card in cards)
    # The run really is the +15% one: its all-in cost is on a card.
    assert any(expected in card for card in cards), expected

    # And the rows and their download came with it.
    assert app.dataframe
    assert any(
        type(element).__name__ == "DownloadButton" for element in app.main
    )
