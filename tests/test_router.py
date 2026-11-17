import pandas as pd

from core.loaders import LoadedFile
from core.router import choose_path


def make(tables=0, text=""):
    return LoadedFile(name="f", file_type="pdf", tables=[pd.DataFrame({"a": [1]})] * tables, text=text)


def test_routes():
    assert choose_path(make(tables=1)) == "tabular"
    assert choose_path(make(text="word " * 100)) == "text"
    assert choose_path(make(tables=1, text="word " * 100)) == "both"
    assert choose_path(make()) == "empty"