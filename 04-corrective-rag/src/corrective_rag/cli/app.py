"""CLI application for Corrective RAG."""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from corrective_rag.config import Settings
from corrective_rag.exceptions import CorrectiveRAGError
from corrective_rag.pipelines.corrective import CorrectivePipeline

console = Console()


def display_banner():
    """Display application banner."""
    console.print(
        Panel.fit(
            "[bold green]Corrective RAG[/bold green]\n"
            "Self-Correcting Retrieval with Validation",
            border_style="green",
        )
    )


def initialize_pipeline() -> CorrectivePipeline:
    """Initialize the corrective RAG pipeline."""
    settings = Settings()
    pipeline = CorrectivePipeline(settings)

    console.print("[yellow]Loading documents...[/yellow]")
    try:
        chunk_count = pipeline.ingest_documents()
        console.print(f"[green]Loaded {chunk_count} chunks into vector store[/green]")
    except CorrectiveRAGError as e:
        console.print(f"[yellow]Warning: {e}[/yellow]")

    return pipeline


def interactive_loop(pipeline: CorrectivePipeline):
    """Run interactive query loop."""
    console.print("\n[bold]Enter your questions (type 'quit' to exit):[/bold]\n")

    while True:
        try:
            question = console.input("[bold cyan]Question: [/bold cyan]")

            if question.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not question.strip():
                continue

            answer = pipeline.query(question)

            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(answer))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except CorrectiveRAGError as e:
            console.print(f"[red]Error: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")


def main():
    """Main entry point."""
    try:
        display_banner()
        pipeline = initialize_pipeline()
        interactive_loop(pipeline)
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
