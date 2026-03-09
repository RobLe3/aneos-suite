"""
ANEOSMenuBase — Pure UI helper base class for the aNEOS menu system.

Contains only stateless display/formatting helpers with no domain logic,
no API calls, and no external dependencies beyond Rich (optional).

Extracted from aneos_menu.py (Phase 15E) as the first step of the God-Class
refactor. Full domain split is deferred to Phase 16+.
"""

from typing import Generator, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class ANEOSMenuBase:
    """Pure UI helpers shared by all menu modules."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self):
        self.console: Optional["Console"] = Console() if HAS_RICH else None

    # ------------------------------------------------------------------
    # Basic messaging
    # ------------------------------------------------------------------

    def show_error(self, message: str) -> None:
        """Display an error message."""
        if self.console:
            self.console.print(f"[bold red]❌ Error:[/bold red] {message}")
        else:
            print(f"❌ Error: {message}")

    def show_info(self, message: str) -> None:
        """Display an informational message."""
        if self.console:
            self.console.print(f"[bold blue]ℹ️  Info:[/bold blue] {message}")
        else:
            print(f"ℹ️  Info: {message}")

    def show_success(self, message: str) -> None:
        """Display a success message."""
        if self.console:
            self.console.print(f"[bold green]✅ Success:[/bold green] {message}")
        else:
            print(f"✅ Success: {message}")

    def wait_for_input(self, prompt: str = "Press Enter to continue") -> None:
        """Pause until the user presses Enter."""
        if self.console and HAS_RICH:
            Prompt.ask(prompt, default="")
        else:
            input(f"{prompt}...")

    # ------------------------------------------------------------------
    # Structured display
    # ------------------------------------------------------------------

    def display_table(self, headers: List[str], rows: List[List], title: str = "") -> None:
        """Render a Rich table or fall back to plain text."""
        if self.console and HAS_RICH:
            table = Table(title=title)
            for h in headers:
                table.add_column(h, style="cyan")
            for row in rows:
                table.add_row(*[str(c) for c in row])
            self.console.print(table)
        else:
            if title:
                print(f"\n{title}")
            print("  ".join(str(h) for h in headers))
            for row in rows:
                print("  ".join(str(c) for c in row))

    def display_panel(self, content: str, title: str = "", style: str = "blue") -> None:
        """Render a Rich panel or fall back to plain text."""
        if self.console and HAS_RICH:
            self.console.print(Panel(content, title=title, style=style))
        else:
            if title:
                print(f"\n{'='*40} {title} {'='*40}")
            print(content)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def format_probability(self, p: float) -> str:
        """Return a colour-coded probability string."""
        pct = p * 100
        if pct >= 50:
            colour = "red"
        elif pct >= 10:
            colour = "yellow"
        else:
            colour = "green"
        formatted = f"{pct:.2f}%"
        if self.console and HAS_RICH:
            return f"[{colour}]{formatted}[/{colour}]"
        return formatted

    def format_sigma(self, sigma: float) -> str:
        """Return sigma tier label."""
        if sigma >= 5.0:
            return "EXCEPTIONAL (σ≥5)"
        if sigma >= 4.0:
            return "ANOMALOUS (σ≥4)"
        if sigma >= 3.0:
            return "SIGNIFICANT (σ≥3)"
        if sigma >= 2.0:
            return "INTERESTING (σ≥2)"
        return "INCONCLUSIVE (σ<2)"

    def format_designation(self, designation: str) -> str:
        """Return Rich markup for a NEO designation."""
        if self.console and HAS_RICH:
            return f"[bold cyan]{designation}[/bold cyan]"
        return designation

    # ------------------------------------------------------------------
    # Terminal utilities
    # ------------------------------------------------------------------

    def _get_terminal_width(self) -> int:
        """Return terminal width (default 80)."""
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except Exception:
            return 80

    def _truncate(self, text: str, width: int) -> str:
        """Truncate text to width characters, adding '…' if needed."""
        if len(text) <= width:
            return text
        return text[: width - 1] + "…"

    def _color_for_score(self, score: float) -> str:
        """Return a Rich colour name for a 0–1 anomaly score."""
        if score >= 0.8:
            return "red"
        if score >= 0.5:
            return "yellow"
        if score >= 0.2:
            return "blue"
        return "green"

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def track_progress(
        self, items: list, description: str = "Processing…"
    ) -> Generator:
        """Yield items from *items* with a Rich progress bar.

        Falls back to plain numbered print lines when Rich is unavailable.
        Usage::

            for item in self.track_progress(my_list, "Analysing"):
                process(item)
        """
        if self.console and HAS_RICH:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
            ) as prog:
                task = prog.add_task(description, total=len(items))
                for item in items:
                    yield item
                    prog.advance(task)
        else:
            total = len(items)
            for i, item in enumerate(items, 1):
                print(f"  [{i}/{total}] {description}")
                yield item

    # ------------------------------------------------------------------
    # File browser
    # ------------------------------------------------------------------

    def browse_files(
        self,
        search_dirs: Optional[List[str]] = None,
        extensions: Optional[List[str]] = None,
        prompt: str = "Select file",
    ) -> Optional[str]:
        """Display a numbered list of matching files and return the chosen path.

        Searches *search_dirs* (defaults to current directory) for files with any
        of *extensions* (defaults to ``[".txt"]``).  The user may pick by number
        or type a literal path.  Returns ``None`` if the user cancels.
        """
        from pathlib import Path as _Path

        dirs = search_dirs or ["."]
        exts = set(extensions or [".txt"])
        found: List[_Path] = []
        seen: set = set()
        for d in dirs:
            p = _Path(d)
            if p.is_dir():
                for f in sorted(p.iterdir()):
                    if f.is_file() and f.suffix in exts and f not in seen:
                        seen.add(f)
                        found.append(f)

        if found:
            rows = [
                [
                    str(i + 1),
                    f.name,
                    str(f.parent) if str(f.parent) != "." else "(here)",
                    f"{max(1, f.stat().st_size // 1024)} KB",
                ]
                for i, f in enumerate(found)
            ]
            self.display_table(
                headers=["#", "File", "Directory", "Size"],
                rows=rows,
                title="Available files",
            )
        else:
            self.show_info("No matching files found in common locations.")

        raw = self._ask(f"{prompt} — enter number or full path (blank to cancel)").strip()
        if not raw:
            return None
        try:
            idx = int(raw) - 1
            if found and 0 <= idx < len(found):
                return str(found[idx])
        except ValueError:
            pass
        return raw  # treat as a literal path
