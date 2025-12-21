from __future__ import annotations

from rich.console import Console

from .config import AppConfig
from .pipeline import run_mvp

console = Console()


def main() -> None:
    config = AppConfig()
    out = run_mvp(config)
    console.print(f"[bold]Output:[/bold] {out}")


if __name__ == "__main__":
    main()

