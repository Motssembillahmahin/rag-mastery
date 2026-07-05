"""CLI application for Multimodal RAG."""

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from multimodal_rag.config import MultimodalRAGConfig
from multimodal_rag.pipelines.multimodal import MultimodalRAGPipeline
from multimodal_rag.models.schemas import QueryRequest

console = Console()


def display_results(response):
    """Display query results."""
    console.print("\n[bold green]Answer:[/bold green]")
    console.print(Markdown(response.answer))

    if response.sources:
        console.print("\n[bold]Sources:[/bold]")
        for i, source in enumerate(response.sources, 1):
            source_type = source.source_type.upper()
            console.print(
                f"  [cyan]{i}.[/cyan] [{source_type}] {source.source} "
                f"(score: {source.weighted_score:.3f})"
            )

    console.print(
        f"\n[dim]Retrieved {response.num_text_chunks} text chunks, "
        f"{response.num_images} images[/dim]"
    )


def main():
    """Main entry point for the CLI."""
    console.print(
        Panel.fit(
            "[bold green]Multimodal RAG[/bold green]\n"
            "Handle Text, Images, and Tables",
            border_style="green",
        )
    )

    config = MultimodalRAGConfig()
    pipeline = MultimodalRAGPipeline(config)

    # Ingest documents
    console.print("\n[bold]Ingesting documents...[/bold]")
    try:
        doc_stats = pipeline.ingest_documents()
        console.print(
            f"[green]Loaded {doc_stats['documents']} documents "
            f"({doc_stats['chunks']} chunks)[/green]"
        )
    except Exception as e:
        console.print(f"[yellow]Warning: Document ingestion failed: {e}[/yellow]")

    # Ingest images
    console.print("\n[bold]Ingesting images...[/bold]")
    try:
        img_stats = pipeline.ingest_images()
        console.print(f"[green]Processed {img_stats['images']} images[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Image ingestion failed: {e}[/yellow]")

    # Display stats
    stats = pipeline.get_stats()
    console.print(f"\n[green]Vector stores: {stats['text_count']} text, {stats['image_count']} images[/green]")

    # Interactive query loop
    console.print("\n[bold]Enter your questions (type 'quit' to exit):[/bold]\n")

    while True:
        try:
            question = console.input("[bold cyan]Question: [/bold cyan]")

            if question.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not question.strip():
                continue

            request = QueryRequest(question=question)
            response = pipeline.query(request)
            display_results(response)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
