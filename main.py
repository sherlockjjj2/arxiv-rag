"""Compatibility entrypoint for running the CLI via python main.py."""

from arxiv_rag.cli import app


def main() -> None:
    """Invoke the Typer CLI app."""
    app()


if __name__ == "__main__":
    main()
