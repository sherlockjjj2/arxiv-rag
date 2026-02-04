import importlib.util
import os
import re
import sys
from pathlib import Path

import fitz
import pytest


def _load_parse_module():
    module_path = Path(__file__).resolve().parents[1] / "arxiv_rag" / "parse.py"
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


def test_parse_pdf_mutes_mupdf_diagnostics(tmp_path, monkeypatch):
    parse = _load_parse_module()
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(pdf_path, ["Page body"])

    error_calls: list[bool | None] = []
    warning_calls: list[bool | None] = []

    original_error_toggle = parse.fitz.TOOLS.mupdf_display_errors
    original_warning_toggle = parse.fitz.TOOLS.mupdf_display_warnings
    initial_error_state = bool(original_error_toggle())
    initial_warning_state = bool(original_warning_toggle())

    def _record_error_toggle(on: bool | None = None):
        error_calls.append(on)
        return original_error_toggle(on)

    def _record_warning_toggle(on: bool | None = None):
        warning_calls.append(on)
        return original_warning_toggle(on)

    monkeypatch.setattr(
        parse.fitz.TOOLS,
        "mupdf_display_errors",
        _record_error_toggle,
    )
    monkeypatch.setattr(
        parse.fitz.TOOLS,
        "mupdf_display_warnings",
        _record_warning_toggle,
    )

    parse.parse_pdf(str(pdf_path))

    explicit_error_calls = [value for value in error_calls if value is not None]
    explicit_warning_calls = [value for value in warning_calls if value is not None]
    assert explicit_error_calls[0] is False
    assert explicit_warning_calls[0] is False
    assert explicit_error_calls[-1] is initial_error_state
    assert explicit_warning_calls[-1] is initial_warning_state


def test_mute_mupdf_diagnostics_redirects_fd_stderr(capfd):
    parse = _load_parse_module()

    with parse._mute_mupdf_diagnostics():
        os.write(2, b"fd-stderr-noise\n")

    captured = capfd.readouterr()
    assert "fd-stderr-noise" not in captured.err


def test_summarize_mupdf_warnings_suppresses_low_risk_lines(monkeypatch):
    parse = _load_parse_module()
    monkeypatch.setattr(
        parse.fitz.TOOLS,
        "mupdf_warnings",
        lambda: (
            "bogus font ascent/descent values (0 / 0)\n"
            "invalid marked content and clip nesting\n"
        ),
    )

    assert parse._summarize_mupdf_warnings() is None


def test_summarize_mupdf_warnings_keeps_high_risk_lines(monkeypatch):
    parse = _load_parse_module()
    monkeypatch.setattr(
        parse.fitz.TOOLS,
        "mupdf_warnings",
        lambda: (
            "bogus font ascent/descent values (0 / 0)\n"
            "Actualtext with no position. Text may be lost or mispositioned.\n"
            "syntax error: unknown keyword: 'pagesize'\n"
        ),
    )

    summary = parse._summarize_mupdf_warnings()
    assert summary is not None
    assert "MuPDF warnings:" in summary
    assert "Actualtext with no position" in summary
    assert "syntax error: unknown keyword: 'pagesize'" in summary
    assert "bogus font ascent/descent values" not in summary
