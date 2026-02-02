"""Typer-based CLI for arxiv-rag."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import typer

from arxiv_rag.retrieve import format_snippet, search_fts

app = typer.Typer(help="CLI for arXiv RAG utilities.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Show help when no subcommand is provided."""

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def query(
    question: str,
    top_k: int = typer.Option(5, help="Number of chunks to return."),
    db: Path = typer.Option(
        Path("data/arxiv_rag.db"),
        help="SQLite database path.",
        exists=False,
        dir_okay=False,
    ),
    snippet_chars: int = typer.Option(
        240,
        help="Maximum characters to display per snippet.",
    ),
    show_score: bool = typer.Option(
        False,
        help="Include BM25 score in the output.",
    ),
) -> None:
    """Query SQLite FTS and print matching chunks."""

    if snippet_chars <= 0:
        typer.echo("snippet_chars must be > 0", err=True)
        raise typer.Exit(code=1)

    try:
        results = search_fts(question, top_k=top_k, db_path=db)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except sqlite3.Error as exc:
        typer.echo(f"SQLite error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if not results:
        typer.echo("No results found.")
        raise typer.Exit(code=0)

    for index, result in enumerate(results):
        snippet = format_snippet(result.text, snippet_chars)
        line = f"[arXiv:{result.paper_id} p.{result.page_number}] {snippet}"
        if show_score and result.score is not None:
            line = f"{line} (score={result.score:.3f})"
        typer.echo(line)
        if index < len(results) - 1:
            typer.echo()


if __name__ == "__main__":
    app()
