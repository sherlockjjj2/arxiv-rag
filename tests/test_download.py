import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
import sqlite3
from datetime import datetime


def _load_download_module():
    module_path = Path(__file__).resolve().parents[1] / "arxiv_rag" / "download.py"
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("download", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["download"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_arxiv_ids_module():
    module_path = Path(__file__).resolve().parents[1] / "arxiv_rag" / "arxiv_ids.py"
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("arxiv_ids", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["arxiv_ids"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_base_id_validation():
    arxiv_ids = _load_arxiv_ids_module()
    assert arxiv_ids.is_valid_base_id("2311.12022")
    assert arxiv_ids.is_valid_base_id("0704.0001")
    assert arxiv_ids.is_valid_base_id("cs/9901001")
    assert arxiv_ids.is_valid_base_id("cs.DS/0101001")
    assert not arxiv_ids.is_valid_base_id("2301.1234v2")
    assert not arxiv_ids.is_valid_base_id("not-an-id")


def test_base_id_from_versioned():
    arxiv_ids = _load_arxiv_ids_module()
    assert arxiv_ids.base_id_from_versioned("2311.12022v2") == "2311.12022"
    assert arxiv_ids.base_id_from_versioned("cs.DS/0101001v3") == "cs.DS/0101001"
    assert arxiv_ids.base_id_from_versioned("randomv2") == "randomv2"


def test_download_by_id_invalid_ids(tmp_path, capsys, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv-papers"
    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(download, "ID_INDEX", pdf_dir / "arxiv_ids.txt")

    exit_code = download._download_by_id(["bad-id"], retries=1, timeout=1)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "invalid arXiv ID format" in captured.err


def test_download_by_id_skips_existing(tmp_path, capsys, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv-papers"
    pdf_dir.mkdir(parents=True)
    existing_pdf = pdf_dir / "2311.12022v1.pdf"
    existing_pdf.write_bytes(b"pdf")

    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(download, "ID_INDEX", pdf_dir / "arxiv_ids.txt")

    def fake_download_pdf(*_args, **_kwargs):
        raise AssertionError("download should be skipped")

    monkeypatch.setattr(download, "_download_pdf", fake_download_pdf)
    monkeypatch.setattr(download, "_fetch_results", lambda _ids: {})

    exit_code = download._download_by_id(["2311.12022"], retries=1, timeout=1)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Skipped 2311.12022" in captured.out
    assert download.ID_INDEX.read_text(encoding="utf-8").strip() == "2311.12022"


def test_download_by_id_downloads_and_records(tmp_path, capsys, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv-papers"
    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(download, "ID_INDEX", pdf_dir / "arxiv_ids.txt")

    class FakeResult:
        pdf_url = "http://example.com/test.pdf"

        @staticmethod
        def get_short_id():
            return "2311.12022v2"

    def fake_fetch_results(_ids, **_kwargs):
        return {"2311.12022": FakeResult()}

    def fake_download_pdf(_result, dest_path, _timeout, delay_seconds=None, **_kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"pdf")
        return None

    monkeypatch.setattr(download, "_fetch_results", fake_fetch_results)
    monkeypatch.setattr(download, "_download_pdf", fake_download_pdf)

    exit_code = download._download_by_id(["2311.12022"], retries=1, timeout=1)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert (pdf_dir / "2311.12022v2.pdf").exists()
    assert "Downloaded 2311.12022" in captured.out
    assert download.ID_INDEX.read_text(encoding="utf-8").strip() == "2311.12022"


def test_sync_rebuilds_index(tmp_path, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv-papers"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "2311.12022v1.pdf").write_bytes(b"pdf")

    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(download, "ID_INDEX", pdf_dir / "arxiv_ids.txt")

    download.ID_INDEX.write_text("2311.12022\n", encoding="utf-8")
    existing = download._load_id_index()

    download._sync_id_index(existing)

    assert existing == {"2311.12022"}
    assert download.ID_INDEX.read_text(encoding="utf-8").strip() == "2311.12022"


def test_download_pdf_closes_response(tmp_path, monkeypatch):
    download = _load_download_module()
    dest_path = tmp_path / "paper.pdf"

    class FakeResponse:
        status_code = 200
        headers = {}

        def __init__(self):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            self.close()

        def close(self):
            self.closed = True

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=1024):
            yield b"pdf"

    response = FakeResponse()

    def fake_get(*_args, **_kwargs):
        return response

    monkeypatch.setattr(download.requests, "get", fake_get)

    class FakeResult:
        pdf_url = "http://example.com/paper.pdf"

    assert download._download_pdf(FakeResult(), dest_path, timeout=1) is None
    assert response.closed is True
    assert dest_path.exists()


def test_download_by_id_inserts_metadata(tmp_path, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv-papers"
    db_path = tmp_path / "arxiv_rag.db"

    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(download, "ID_INDEX", pdf_dir / "arxiv_ids.txt")

    class FakeAuthor:
        def __init__(self, name):
            self.name = name

    class FakeResult:
        pdf_url = "http://example.com/test.pdf"
        title = "Test Paper"
        summary = "Abstract."
        categories = ["cs.CL", "cs.AI"]
        published = datetime(2023, 12, 18)
        updated = datetime(2024, 1, 4)
        authors = [FakeAuthor("Ada Lovelace"), FakeAuthor("Alan Turing")]
        primary_category = "cs.CL"

        @staticmethod
        def get_short_id():
            return "2311.12022v2"

    def fake_fetch_results(_ids, **_kwargs):
        return {"2311.12022": FakeResult()}

    def fake_download_pdf(_result, dest_path, _timeout, delay_seconds=None, **_kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"pdf")
        return None

    monkeypatch.setattr(download, "_fetch_results", fake_fetch_results)
    monkeypatch.setattr(download, "_download_pdf", fake_download_pdf)

    exit_code = download._download_by_id(
        ["2311.12022"],
        retries=1,
        timeout=1,
        db_path=db_path,
    )

    assert exit_code == 0
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT paper_id, title, authors, abstract, categories, published_date, pdf_path, source_type "
            "FROM papers WHERE paper_id = ?",
            ("2311.12022",),
        ).fetchone()

    assert row is not None
    assert row[0] == "2311.12022"
    assert row[1] == "Test Paper"
    assert "Ada Lovelace" in row[2]
    assert row[3] == "Abstract."
    assert "cs.CL" in row[4]
    assert row[5] == "2023-12-18"
    assert row[6].endswith("2311.12022v2.pdf")
    assert row[7] == "arxiv"


def test_download_by_id_no_db_skips_insert(tmp_path, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv-papers"

    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(download, "ID_INDEX", pdf_dir / "arxiv_ids.txt")

    class FakeResult:
        pdf_url = "http://example.com/test.pdf"
        title = "Test Paper"
        summary = "Abstract."
        categories = ["cs.CL"]
        published = datetime(2023, 12, 18)
        updated = datetime(2024, 1, 4)
        authors = []
        primary_category = "cs.CL"

        @staticmethod
        def get_short_id():
            return "2311.12022v2"

    def fake_fetch_results(_ids, **_kwargs):
        return {"2311.12022": FakeResult()}

    def fake_download_pdf(_result, dest_path, _timeout, delay_seconds=None, **_kwargs):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"pdf")
        return None

    def fail_bulk_insert(*_args, **_kwargs):
        raise AssertionError("metadata insert should be skipped when db is disabled")

    monkeypatch.setattr(download, "_fetch_results", fake_fetch_results)
    monkeypatch.setattr(download, "_download_pdf", fake_download_pdf)
    monkeypatch.setattr(download, "_bulk_insert_metadata", fail_bulk_insert)

    exit_code = download._download_by_id(
        ["2311.12022"],
        retries=1,
        timeout=1,
        db_path=None,
    )

    assert exit_code == 0


def test_backfill_inserts_metadata_for_existing_pdfs(tmp_path, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv-papers"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "2311.12022v1.pdf").write_bytes(b"pdf")
    db_path = tmp_path / "arxiv_rag.db"

    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)

    class FakeAuthor:
        def __init__(self, name):
            self.name = name

    class FakeResult:
        pdf_url = "http://example.com/test.pdf"
        title = "Test Paper"
        summary = "Abstract."
        categories = ["cs.CL", "cs.AI"]
        published = datetime(2023, 12, 18)
        updated = datetime(2024, 1, 4)
        authors = [FakeAuthor("Ada Lovelace")]
        primary_category = "cs.CL"

        @staticmethod
        def get_short_id():
            return "2311.12022v1"

    def fake_fetch_results(_ids, **_kwargs):
        return {"2311.12022": FakeResult()}

    monkeypatch.setattr(download, "_fetch_results", fake_fetch_results)

    exit_code = download._backfill_metadata(db_path)
    assert exit_code == 0

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT paper_id, title FROM papers WHERE paper_id = ?",
            ("2311.12022",),
        ).fetchone()

    assert row == ("2311.12022", "Test Paper")


def test_backfill_no_pdfs(tmp_path, monkeypatch):
    download = _load_download_module()
    pdf_dir = tmp_path / "arxiv-papers"
    db_path = tmp_path / "arxiv_rag.db"

    monkeypatch.setattr(download, "PDF_DIR", pdf_dir)

    def fail_fetch_results(_ids):
        raise AssertionError("no API calls expected when there are no PDFs")

    monkeypatch.setattr(download, "_fetch_results", fail_fetch_results)

    assert download._backfill_metadata(db_path) == 0


def test_create_arxiv_client_uses_timeout_session():
    download = _load_download_module()
    config = download.default_config()
    config = download.replace(
        config,
        client_timeout_seconds=7.5,
    )

    client = download.create_arxiv_client(config)

    assert isinstance(client.client._session, download._ArxivTimeoutSession)
    assert client.client._session.timeout_seconds == 7.5


def test_main_handles_arxiv_api_request_error(monkeypatch, capsys):
    download = _load_download_module()
    args = Namespace(
        query="RAG",
        ids=None,
        backfill_db=False,
        no_db=False,
        db=Path("data/arxiv_rag.db"),
        max_results=5,
        sort="relevance",
        retries=3,
        timeout=30,
        api_retries=5,
        api_timeout=20.0,
    )
    monkeypatch.setattr(download, "_parse_args", lambda _config=None: args)

    def fake_search(*_args, **_kwargs):
        raise download.requests.exceptions.ProxyError("proxy unavailable")

    monkeypatch.setattr(download, "_search", fake_search)

    exit_code = download.main()
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "failed to reach the arXiv API" in captured.err
