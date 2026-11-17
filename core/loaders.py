""" Read uploaded files for handling different types of data sources (CSV, Excel, pdf etc.) into one common structure. """

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pdfplumber
from docx import Document

SUPPORTED_TYPES = ["csv", "xlsx", "pdf", "docx"]

@dataclass
class LoadedFile:
    
    """ Everything we extracted from one uploaded file. """
    
    name: str
    file_type: str
    tables: list[pd.DataFrame] = field(default_factory=list)
    text: str = ""
    warnings: list[str] = field(default_factory=list)
    
    
class UnsupportedFileError(ValueError):
    """ Raised when the file type is not supported. """
    
    
def load_file(name: str, data: bytes) -> LoadedFile:
    """ Pick the right loader for the file type and load it. """
    
    file_type = Path(name).suffix.lower().lstrip(".")
    loaders = {
        "csv": _load_csv,
        "xlsx": _load_excel,
        "pdf": _load_pdf,
        "docx": _load_docx,
    }
    
    if file_type not in loaders:
        raise UnsupportedFileError(f"'. {file_type}'  files are not supported file type. Use one of: {', '.join(SUPPORTED_TYPES)}"
        )
        
    result = loaders[file_type](data)
    result.name = name
    result.file_type = file_type
    return result

# ------------ individual loaders for each file type -----------------

def _load_csv(data: bytes) -> LoadedFile:
    """ UTF-8 first, then a common windows/excel encoding. """  
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding, sep=None, engine="python")
            return LoadedFile(name="", file_type="csv", tables=[df])
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV file.")
    
def _load_excel(data: bytes) -> LoadedFile:
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None) #every sheet 
    tables, warnings = [], []
    for sheet_name, df in sheets.items():
        if df.empty:
            warnings.append(f"Sheet '{sheet_name}' is empty and was skipped.")
            continue
        df.attrs["source"] = f"Sheet '{sheet_name}'"
        tables.append(df)
    return LoadedFile(name="", file_type="xlsx", tables=tables, warnings=warnings)


def _load_pdf(data: bytes) -> LoadedFile:
    text_parts, tables =[], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text_parts.append(page.extract_text() or "")
            for raw in page.extract_tables():
                df = _rows_to_dataframe(raw)
                if df is not None:
                    df.attrs["source"] = f"Page {page_no}"
                    tables.append(df)
                
    text = "\n".join(text_parts)
    warnings = []
    if not text and not tables:
        warnings.append("No text found. This may be a scanned PDF, which needs OCR (not supported in v1).")
    return LoadedFile(name="", file_type="pdf", tables=tables, text=text, warnings=warnings)
        

def _load_docx(data: bytes) -> LoadedFile:
    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    tables = []
    for i, table in enumerate(doc.tables, start=1):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        df = _rows_to_dataframe(rows)
        if df is not None:
            df.attrs["source"] = f"Table {i}"
            tables.append(df)
            
    return LoadedFile(name="", file_type="docx", tables=tables, text=text)

#----------------- helper functions -----------------

def _rows_to_dataframe(rows: list[list]) -> pd.DataFrame | None:
    """ Turn a list of rows (first row is header) into a DataFrame, or None if the table is empty. """
    rows = [r for r in rows if r and any(c not in (None, "") for c in r)]  # remove empty rows
    if len(rows) < 2: # need header plus at least one row of data
        return None
    header = _clean_headers(rows[0])
    df = pd.DataFrame(rows[1:], columns=header)
    df = df.replace("", pd.NA)  
    # Text extracted from documents is all strings; convert numeric columns to numbers if possible
    for col in df.columns:
        converted = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        if converted.notna().sum() >= 0.8 * df[col].notna().sum() and df[col].notna().any():
            df[col] = converted
            
    return df
    

def _clean_headers(header: list) -> list[str]:
    """Fill blank column names and make duplicates unique."""
    seen: dict[str, int] = {}
    cleaned = []
    for i, h in enumerate(header):
        name = str(h).strip() if h not in (None, "") else f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cleaned.append(name)
    return cleaned