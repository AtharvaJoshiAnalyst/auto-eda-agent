"""Shared test fixtures: small sample files built in memory."""

import io

import pandas as pd
import pytest
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [22, 35, 41, 29, 300, None],          # 300 = outlier, one missing
        "city": ["London", "Leeds", "London", "Bath", "Leeds", "London"],
        "constant": ["x"] * 6,
    })


@pytest.fixture
def csv_bytes(sample_df):
    return sample_df.to_csv(index=False).encode()


@pytest.fixture
def xlsx_bytes(sample_df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        sample_df.to_excel(writer, sheet_name="people", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="empty", index=False)
    return buf.getvalue()


@pytest.fixture
def docx_bytes():
    doc = Document()
    doc.add_paragraph("Quarterly report. " * 30)
    table = doc.add_table(rows=3, cols=2)
    for r, row in enumerate([["Region", "Sales"], ["North", "1,200"], ["South", "950"]]):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def pdf_bytes():
    buf = io.BytesIO()
    table = Table([["Region", "Sales"], ["North", "1200"], ["South", "950"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, "black")]))
    text = Paragraph("Quarterly report. " * 30, getSampleStyleSheet()["Normal"])
    SimpleDocTemplate(buf, pagesize=A4).build([text, table])
    return buf.getvalue()