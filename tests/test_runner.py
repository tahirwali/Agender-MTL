"""Offline checks for the step runner. Run with:  python -m pytest tests -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agender_mtl.runner import StepRunner


def test_failed_step_does_not_stop_the_run(tmp_path):
    runner = StepRunner(tmp_path, time_budget_minutes=5)
    assert runner.run("works", lambda: 3) == 3
    assert runner.run("breaks", lambda: 1 / 0) is None
    assert runner.failed() == ["breaks"]
    saved = json.loads((tmp_path / "steps.json").read_text())
    assert [s["status"] for s in saved] == ["ok", "error"]


def test_steps_are_skipped_after_the_budget(tmp_path):
    runner = StepRunner(tmp_path, time_budget_minutes=0)
    assert runner.run("late", lambda: 1) is None
    assert runner.steps[0]["status"] == "skipped"
