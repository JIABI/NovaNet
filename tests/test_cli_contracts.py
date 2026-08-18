from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _cli_scripts() -> list[Path]:
    candidates = [
        *sorted((REPOSITORY_ROOT / "experiments").glob("*.py")),
        *sorted((REPOSITORY_ROOT / "scripts").glob("*.py")),
        *sorted(REPOSITORY_ROOT.glob("train*.py")),
    ]
    return [
        path
        for path in candidates
        if 'if __name__ == "__main__"' in path.read_text(encoding="utf-8")
    ]


@pytest.mark.parametrize(
    "script",
    _cli_scripts(),
    ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)),
)
def test_experiment_and_training_clis_expose_help_without_starting_work(script):
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
