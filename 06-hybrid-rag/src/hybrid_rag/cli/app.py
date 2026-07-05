"""CLI application for Hybrid RAG."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from hybrid_rag.config import HybridRAGConfig
from hybrid_rag.pipelines.hybrid import HybridPipeline
from hybrid_rag.models.schemas import QueryRequest

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
        f"\n[dim]Retrieved {response.num_results} results "
        f"(fusion: {response.fusion_method})[/dim]"
    )


def main():
    """Main entry point for the CLI."""
    console.print(
        Panel.fit(
            "[bold green]Hybrid RAG[/bold green]\n"
            "Combined Keyword + Semantic Search",
            border_style="green",
        )
    )

    config = HybridRAGConfig()
    pipeline = HybridPipeline(config)

    # Ingest documents
    console.print("\n[bold]Ingesting documents...[/bold]")
    try:
        stats = pipeline.ingest_documents()
        console.print(
            f"[green]Loaded {stats['documents']} documents "
            f"({stats['chunks']} chunks)[/green]"
        )
    except Exception as e:
        console.print(f"[yellow]Warning: Document ingestion failed: {e}[/yellow]")

    # Display stats
    stats = pipeline.get_stats()
    console.print(
        f"\n[green]Vector store: {stats['vector_count']} vectors, "
        f"BM25: {stats['bm25_count']} documents[/green]"
    )

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
