"""CLI application for HyDE RAG."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from hyde_rag.config import get_settings
from hyde_rag.pipelines.hyde import HyDEPipeline

app = typer.Typer(help="HyDE RAG - Hypothetical Document Embeddings")
console = Console()


@app.command()
def ingest(source: str = typer.Argument(..., help="File or directory to ingest")):
    """Ingest documents into the vector store."""
    settings = get_settings()
    pipeline = HyDEPipeline(settings)

    with console.status("[bold green]Ingesting documents..."):
        try:
            chunk_count = pipeline.ingest_documents(source)
            console.print(
                f"[green]Successfully ingested {chunk_count} chunks from {source}[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(4, "--top-k", "-k", help="Number of documents to retrieve"),
):
    """Query the HyDE RAG system."""
    settings = get_settings()
    pipeline = HyDEPipeline(settings)

    console.print(
        Panel.fit(
            "[bold green]HyDE RAG[/bold green]\nHypothetical Document Embeddings",
            border_style="green",
        )
    )

    console.print(f"\n[bold blue]Question:[/bold blue] {question}\n")

    with console.status("[yellow]Processing query...[/yellow]"):
        try:
            result = pipeline.query(question, top_k=top_k)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    # Display hypothetical documents
    console.print("[yellow]Hypothetical Documents Generated:[/yellow]")
    for i, hyp_doc in enumerate(result.hypothetical_docs, 1):
        console.print(f"\n[cyan]Doc {i} (confidence: {hyp_doc.confidence:.2f}):[/cyan]")
        console.print(f"{hyp_doc.content[:200]}...")

    # Display retrieved context
    console.print("\n[yellow]Retrieved Context:[/yellow]")
    for i, chunk in enumerate(result.context, 1):
        console.print(f"\n[cyan]Chunk {i} (from {chunk.source}):[/cyan]")
        console.print(f"{chunk.content[:200]}...")

    # Display answer
    console.print("\n[bold green]Answer:[/bold green]")
    console.print(Markdown(result.answer))


@app.command()
def interactive():
    """Start interactive query mode."""
    settings = get_settings()
    pipeline = HyDEPipeline(settings)

    console.print(
        Panel.fit(
            "[bold green]HyDE RAG - Interactive Mode[/bold green]\n"
            "Type 'quit' to exit",
            border_style="green",
        )
    )

    # Ingest documents from data directory
    data_dir = Path(settings.data_dir)
    if data_dir.exists():
        console.print(f"\n[yellow]Ingesting documents from {data_dir}...[/yellow]")
        try:
            chunk_count = pipeline.ingest_documents(str(data_dir))
            console.print(f"[green]Ingested {chunk_count} chunks[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")

    while True:
        try:
            question = console.input("\n[bold cyan]Question: [/bold cyan]")

            if question.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not question.strip():
                continue

            result = pipeline.query(question)

            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(result.answer))

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    app()
