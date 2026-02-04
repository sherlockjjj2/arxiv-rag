import importlib.util
import sys
from pathlib import Path

from typer.testing import CliRunner


def _load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "arxiv_rag" / "cli.py"
    spec = importlib.util.spec_from_file_location("arxiv_rag.cli", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["arxiv_rag.cli"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_base_ids_from_txt(tmp_path: Path) -> None:
    cli = _load_cli_module()
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text(
        "# comment\n2312.10997\n\n2401.00001v2\n",
        encoding="utf-8",
    )

    assert cli._load_base_ids_from_file(ids_path) == ["2312.10997", "2401.00001v2"]


def test_load_base_ids_from_json_variants(tmp_path: Path) -> None:
    cli = _load_cli_module()
    list_path = tmp_path / "ids.json"
    list_path.write_text('["2312.10997", "2401.00001v2"]', encoding="utf-8")
    object_path = tmp_path / "ids_obj.json"
    object_path.write_text(
        '{"ids": ["2312.10997", "2401.00001v2"]}',
        encoding="utf-8",
    )

    assert cli._load_base_ids_from_file(list_path) == ["2312.10997", "2401.00001v2"]
    assert cli._load_base_ids_from_file(object_path) == [
        "2312.10997",
        "2401.00001v2",
    ]


def test_load_base_ids_rejects_unknown_extension(tmp_path: Path) -> None:
    cli = _load_cli_module()
    ids_path = tmp_path / "ids.csv"
    ids_path.write_text("2312.10997\n", encoding="utf-8")

    try:
        cli._load_base_ids_from_file(ids_path)
    except ValueError as exc:
        assert ".txt or .json" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for unsupported extension")


def test_normalize_base_ids_dedupes_and_validates() -> None:
    cli = _load_cli_module()
    normalized, invalid = cli._normalize_base_ids(
        [
            "2312.10997v2",
            "2312.10997",
            "bad-id",
            "2401.00001",
        ]
    )

    assert normalized == ["2312.10997", "2401.00001"]
    assert invalid == ["bad-id"]


def test_index_command_uses_add_ids_ingest_flow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_cli_module()
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("2312.10997\n", encoding="utf-8")

    calls: dict[str, object] = {}

    def fake_run_ingest_and_index(**kwargs):
        calls.update(kwargs)
        return cli.IngestSummary(
            requested_count=1,
            parsed_count=1,
            chunked_count=1,
            indexed_chunks=10,
            failures=[],
            warnings=[],
        )

    monkeypatch.setattr(cli, "_run_ingest_and_index", fake_run_ingest_and_index)
    monkeypatch.setattr(cli, "_print_ingest_summary", lambda _summary: None)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "index",
            "--add-ids",
            str(ids_path),
            "--db",
            str(tmp_path / "arxiv_rag.db"),
            "--pdf-dir",
            str(tmp_path / "arxiv-papers"),
            "--parsed-dir",
            str(tmp_path / "parsed"),
            "--chroma-dir",
            str(tmp_path / "chroma"),
            "--no-progress",
        ],
    )

    assert result.exit_code == 0
    assert calls["add_ids"] == ids_path


def test_index_command_suppresses_arxiv_info_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_cli_module()
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("2312.10997\n", encoding="utf-8")

    arxiv_logger = cli.logging.getLogger("arxiv")
    original_level = arxiv_logger.level
    arxiv_logger.setLevel(cli.logging.INFO)

    def fake_run_ingest_and_index(**_kwargs):
        return cli.IngestSummary(
            requested_count=1,
            parsed_count=1,
            chunked_count=1,
            indexed_chunks=10,
            failures=[],
            warnings=[],
        )

    monkeypatch.setattr(cli, "_run_ingest_and_index", fake_run_ingest_and_index)
    monkeypatch.setattr(cli, "_print_ingest_summary", lambda _summary: None)

    runner = CliRunner()
    try:
        result = runner.invoke(
            cli.app,
            [
                "index",
                "--add-ids",
                str(ids_path),
                "--db",
                str(tmp_path / "arxiv_rag.db"),
                "--pdf-dir",
                str(tmp_path / "arxiv-papers"),
                "--parsed-dir",
                str(tmp_path / "parsed"),
                "--chroma-dir",
                str(tmp_path / "chroma"),
                "--no-progress",
            ],
        )
        assert result.exit_code == 0
        assert arxiv_logger.level == cli.logging.WARNING
    finally:
        arxiv_logger.setLevel(original_level)
