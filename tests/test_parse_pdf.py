import importlib.util
import re
import sys
from pathlib import Path

import fitz
import pytest


def _load_parse_module():
    module_path = Path(__file__).resolve().parents[1] / "arxiv-rag" / "parse.py"
    spec = importlib.util.spec_from_file_location("parse", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["parse"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)


def test_parse_pdf_cleans_text(tmp_path):
    parse = _load_parse_module()
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(
        pdf_path,
        ["Header\nThis is hyphen-\nated word.\n\nNext   line  with   spaces."],
    )

    result = parse.parse_pdf(str(pdf_path))

    assert result["num_pages"] == 1
    assert len(result["pages"]) == 1
    assert re.fullmatch(r"[0-9a-f]{40}", result["doc_id"])
    assert result["pages"][0]["page"] == 1
    cleaned = result["pages"][0]["text"]
    assert "hyphen-\n" not in cleaned
    assert "hyphenated" in cleaned
    assert "  " not in cleaned


def test_parse_pdf_removes_headers_footers(tmp_path):
    parse = _load_parse_module()
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(
        pdf_path,
        [
            "Header Title\nPage 1 body\nFooter 2024",
            "Header Title\nPage 2 body\nFooter 2024",
            "Header Title\nPage 3 body\nFooter 2024",
        ],
    )

    result = parse.parse_pdf(str(pdf_path), remove_headers_footers=True)

    for index, page in enumerate(result["pages"], start=1):
        text = page["text"]
        assert "Header Title" not in text
        assert "Footer 2024" not in text
        assert f"Page {index} body" in text


def test_parse_pdf_missing_file(tmp_path):
    parse = _load_parse_module()
    with pytest.raises(FileNotFoundError):
        parse.parse_pdf(str(tmp_path / "missing.pdf"))


def test_parse_pdf_unreadable(tmp_path):
    parse = _load_parse_module()
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a pdf")
    with pytest.raises(ValueError):
        parse.parse_pdf(str(bad_pdf))
