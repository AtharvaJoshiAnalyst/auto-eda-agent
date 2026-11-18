import pandas as pd
import pytest

from core.loaders import UnsupportedFileError, load_file


def test_csv_loads_one_table(csv_bytes):
    loaded = load_file("people.csv", csv_bytes)
    assert loaded.file_type == "csv"
    assert len(loaded.tables) == 1
    assert loaded.tables[0].shape == (6, 3)


def test_excel_skips_empty_sheets(xlsx_bytes):
    loaded = load_file("people.xlsx", xlsx_bytes)
    assert len(loaded.tables) == 1
    assert any("empty" in w for w in loaded.warnings)


def test_docx_extracts_text_and_numeric_table(docx_bytes):
    loaded = load_file("report.docx", docx_bytes)
    assert "Quarterly report" in loaded.text
    df = loaded.tables[0]
    assert list(df.columns) == ["Region", "Sales"]
    assert df["Sales"].sum() == 2150  # "1,200" was converted to a number


def test_pdf_extracts_text_and_table(pdf_bytes):
    loaded = load_file("report.pdf", pdf_bytes)
    assert "Quarterly report" in loaded.text
    assert len(loaded.tables) == 1
    assert loaded.tables[0]["Sales"].sum() == 2150


def test_unsupported_type_raises():
    with pytest.raises(UnsupportedFileError):
        load_file("notes.txt", b"hello")
        

def test_text_dates_become_datetime():
    csv = b"order_date,city\n2026-01-05,Leeds\n2026-01-08,Bath\n2026-01-11,York\n"
    df = load_file("orders.csv", csv).tables[0]
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert not pd.api.types.is_datetime64_any_dtype(df["city"])
    
