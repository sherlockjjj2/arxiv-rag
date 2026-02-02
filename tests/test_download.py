import importlib.util
import sys
from pathlib import Path


def _load_download_module():
    module_path = Path(__file__).resolve().parents[1] / "arxiv-rag" / "download.py"
    spec = importlib.util.spec_from_file_location("download", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["download"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_base_id_validation():
    download = _load_download_module()
    assert download._is_valid_base_id("2301.12345")
    assert download._is_valid_base_id("0704.0001")
    assert download._is_valid_base_id("cs/9901001")
    assert download._is_valid_base_id("cs.DS/0101001")
    assert not download._is_valid_base_id("2301.1234v2")
    assert not download._is_valid_base_id("not-an-id")


def test_base_id_from_versioned():
    download = _load_download_module()
    assert download._base_id_from_versioned("2301.12345v2") == "2301.12345"
    assert download._base_id_from_versioned("cs.DS/0101001v3") == "cs.DS/0101001"
    assert download._base_id_from_versioned("randomv2") == "randomv2"


def test_download_by_id_invalid_ids(tmp_path, capsys, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv_papers"
    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(download, "ID_INDEX", pdf_dir / "arxiv_ids.txt")

    exit_code = download._download_by_id(["bad-id"], retries=1, timeout=1)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "invalid arXiv ID format" in captured.err


def test_download_by_id_skips_existing(tmp_path, capsys, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv_papers"
    pdf_dir.mkdir(parents=True)
    existing_pdf = pdf_dir / "2301.12345v1.pdf"
    existing_pdf.write_bytes(b"pdf")

    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(download, "ID_INDEX", pdf_dir / "arxiv_ids.txt")

    def fake_download_pdf(*_args, **_kwargs):
        raise AssertionError("download should be skipped")

    monkeypatch.setattr(download, "_download_pdf", fake_download_pdf)
    monkeypatch.setattr(download, "_fetch_results", lambda _ids: {})

    exit_code = download._download_by_id(["2301.12345"], retries=1, timeout=1)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Skipped 2301.12345" in captured.out


def test_download_by_id_downloads_and_records(tmp_path, capsys, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv_papers"
    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(download, "ID_INDEX", pdf_dir / "arxiv_ids.txt")

    class FakeResult:
        pdf_url = "http://example.com/test.pdf"

        @staticmethod
        def get_short_id():
            return "2301.12345v2"

    def fake_fetch_results(_ids):
        return {"2301.12345": FakeResult()}

    def fake_download_pdf(_result, dest_path, _timeout, delay_seconds=None):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"pdf")
        return None

    monkeypatch.setattr(download, "_fetch_results", fake_fetch_results)
    monkeypatch.setattr(download, "_download_pdf", fake_download_pdf)

    exit_code = download._download_by_id(["2301.12345"], retries=1, timeout=1)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert (pdf_dir / "2301.12345v2.pdf").exists()
    assert "Downloaded 2301.12345" in captured.out
    assert download.ID_INDEX.read_text(encoding="utf-8").strip() == "2301.12345"
