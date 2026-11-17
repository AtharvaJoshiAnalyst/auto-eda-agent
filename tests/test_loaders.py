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