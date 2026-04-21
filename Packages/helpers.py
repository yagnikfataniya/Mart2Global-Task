import math
import re

import pandas as pd
from fastapi import HTTPException, UploadFile
from pypdf import PdfReader


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _stringify_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def parse_pdf(file_obj) -> str:
    reader = PdfReader(file_obj)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return _normalize_text("\n\n".join(page for page in pages if page))


def _parse_dataframe(df: pd.DataFrame) -> str:
    cleaned = df.fillna("")
    rows = []
    for _, row in cleaned.iterrows():
        pairs = []
        for col in cleaned.columns:
            value = _stringify_cell(row[col])
            if value:
                pairs.append(f"{col}: {value}")
        if pairs:
            rows.append(" | ".join(pairs))
    return _normalize_text("\n".join(rows))


def parse_excel(file_obj) -> str:
    sheets = pd.read_excel(file_obj, sheet_name=None)
    parts = []
    for sheet_name, df in sheets.items():
        sheet_text = _parse_dataframe(df)
        if sheet_text:
            parts.append(f"Sheet: {sheet_name}\n{sheet_text}")
    return _normalize_text("\n\n".join(parts))


def parse_csv(file_obj) -> str:
    df = pd.read_csv(file_obj)
    return _parse_dataframe(df)


def parse_text(file_obj) -> str:
    content = file_obj.read()
    if isinstance(content, bytes):
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return _normalize_text(content.decode(encoding))
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=400, detail="Unable to decode text file.")
    return _normalize_text(content)


def parse_file(file: UploadFile) -> str:
    filename = (file.filename or "").lower()
    file.file.seek(0)

    if filename.endswith(".pdf"):
        text = parse_pdf(file.file)
    elif filename.endswith((".xlsx", ".xls")):
        text = parse_excel(file.file)
    elif filename.endswith(".csv"):
        text = parse_csv(file.file)
    elif filename.endswith((".txt", ".md")):
        text = parse_text(file.file)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PDF, Excel, CSV, TXT, or Markdown.",
        )

    if not text:
        raise HTTPException(status_code=400, detail="No readable text found in the uploaded file.")

    return text

