"""Worker smoke test — Step 1.

Imports the entry point so a syntax/import regression fails CI fast. The
full dispatch loop is exercised in later steps.
"""

from __future__ import annotations


def test_main_imports() -> None:
    """Importing the entry point must not raise."""
    import app.main  # noqa: F401
