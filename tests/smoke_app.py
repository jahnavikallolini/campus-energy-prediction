"""Headless smoke test of the Streamlit dashboard.  Run: python -m tests.smoke_app
Runs the app with Streamlit's AppTest for several selections and fails on any exception."""
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def main():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5 and len(at.metric) == 5
    for b in ("Lecture", "Boys_main", "Mess"):
        at.sidebar.selectbox[0].set_value(b).run()
        assert not at.exception, (b, [e.value for e in at.exception])
    at.sidebar.selectbox[1].set_value("Linear Regression").run()
    at.sidebar.selectbox[2].set_value("E_matched").run()
    assert not at.exception
    at.sidebar.checkbox[0].check().run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    print("dashboard smoke test passed (5 tabs, buildings/model/feature-set/post-break/saved-model check)")


if __name__ == "__main__":
    main()
