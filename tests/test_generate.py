from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _load_generate():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from arxiv_rag import generate
    from arxiv_rag.generate import Chunk, GenerationConfig

    return generate, Chunk, GenerationConfig


def test_load_prompt_template_reads_file(tmp_path: Path) -> None:
    generate, _, _ = _load_generate()
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Prompt with {chunks}.", encoding="utf-8")

    template = generate.load_prompt_template(prompt_path)

    assert template == "Prompt with {chunks}."


def test_load_prompt_template_raises_on_missing(tmp_path: Path) -> None:
    generate, _, _ = _load_generate()
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        generate.load_prompt_template(missing_path)


def test_load_prompt_template_raises_on_empty(tmp_path: Path) -> None:
    generate, _, _ = _load_generate()
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        generate.load_prompt_template(empty_path)


def test_load_prompt_template_uses_resources_when_default_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generate, _, _ = _load_generate()
    missing_path = tmp_path / "missing.txt"

    monkeypatch.setattr(generate, "_DEFAULT_PROMPT_PATH", missing_path)

    template = generate.load_prompt_template()

    assert "{chunks}" in template


def test_render_prompt_requires_placeholder() -> None:
    generate, _, _ = _load_generate()

    with pytest.raises(ValueError, match=r"\{chunks\}"):
        generate.render_prompt("no placeholder", "chunk")


def test_validate_chunks_rejects_invalid() -> None:
    generate, Chunk, _ = _load_generate()

    chunk = Chunk(paper_id="", page_number=1, text="text", chunk_uid="uid")
    with pytest.raises(ValueError, match="paper_id"):
        generate.validate_chunks([chunk])

    chunk = Chunk(paper_id="1234.5678", page_number=0, text="text", chunk_uid="uid")
    with pytest.raises(ValueError, match="page_number"):
        generate.validate_chunks([chunk])

    chunk = Chunk(paper_id="1234.5678", page_number=1, text=" ", chunk_uid="uid")
    with pytest.raises(ValueError, match="text"):
        generate.validate_chunks([chunk])


def test_format_chunks_includes_metadata() -> None:
    generate, Chunk, _ = _load_generate()

    chunk = Chunk(
        paper_id="1234.5678",
        page_number=2,
        text="chunk text",
        chunk_uid="uid-1",
        title="Paper Title",
    )

    rendered = generate.format_chunks([chunk])

    assert "Chunk 1" in rendered
    assert "[arXiv:1234.5678 p.2]" in rendered
    assert "chunk_uid=uid-1" in rendered
    assert "Paper Title" in rendered
    assert rendered.endswith("chunk text")


def test_generate_answer_uses_prompt_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    generate, Chunk, GenerationConfig = _load_generate()
    captured: dict[str, object] = {}

    class DummyClient:
        def __init__(self, config: GenerationConfig) -> None:
            captured["config"] = config

        def generate(self, *, prompt: str, query: str) -> str:
            captured["prompt"] = prompt
            captured["query"] = query
            return "final answer"

    monkeypatch.setattr(generate, "GenerationClient", DummyClient)
    monkeypatch.setattr(generate, "load_prompt_template", lambda path=None: "P\n{chunks}")

    chunks = [
        Chunk(
            paper_id="1234.5678",
            page_number=1,
            text="chunk text",
            chunk_uid="uid-1",
        )
    ]

    answer = generate.generate_answer("What is this?", chunks, model="gpt-4o-mini")

    assert answer == "final answer"
    assert captured["query"] == "What is this?"
    assert "[arXiv:1234.5678 p.1]" in str(captured["prompt"])
    assert isinstance(captured.get("config"), GenerationConfig)
    assert getattr(captured["config"], "model") == "gpt-4o-mini"
