"""In-process smoke test of the Streamlit app via ``streamlit.testing``.

Runs the whole ``app/streamlit_app.py`` script (both tabs) without a browser and
asserts it raises no exception. This exercises the real Streamlit API surface the
app uses — notably ``st.pydeck_chart(on_select=...)`` for Page 2 — so a floor
that is too low (e.g. streamlit 1.38, which lacks ``on_select``) fails CI here
rather than only in a live browser. No API and no data files are required: the
health call is refused (→ unreachable) and the residual parquet is absent (→ the
Page 1 missing-data branch), both handled gracefully.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_script_runs_without_exception() -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("pydeck")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.run()

    assert not at.exception, f"app script raised: {list(at.exception)}"
