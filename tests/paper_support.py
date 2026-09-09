"""Keep optional manuscript integration tests out of code-only checkouts."""

from pathlib import Path
import unittest


def require_local_paper() -> None:
    """Skip only when the private manuscript directory is absent entirely."""
    paper = Path(__file__).resolve().parents[1] / "paper" / "ivc_draft_20260821"
    if not paper.is_dir():
        raise unittest.SkipTest(
            "Optional manuscript integration test: paper/ is not distributed "
            "with this repository"
        )
