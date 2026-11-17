from streamlit.testing.v1 import AppTest


def test_app_starts_without_errors():
    at = AppTest.from_file("../app/main.py").run(timeout=30)
    assert not at.exception
    assert "Auto-EDA Agent" in at.title[0].value