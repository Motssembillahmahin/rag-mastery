"""CLI application for Graph RAG."""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from graph_rag.config import Settings
from graph_rag.exceptions import GraphRAGError
from graph_rag.pipelines.graph import GraphPipeline

console = Console()


def display_banner():
    """Display application banner."""
    console.print(
        Panel.fit(
            "[bold green]Graph RAG[/bold green]\n"
            "Knowledge Graph-Based Retrieval",
            border_style="green",
        )
    )


def initialize_pipeline() -> GraphPipeline:
    """Initialize the graph RAG pipeline."""
    settings = Settings()
    pipeline = GraphPipeline(settings)

    console.print("[yellow]Loading documents and building knowledge graph...[/yellow]")
    try:
        chunk_count = pipeline.ingest_documents()
        console.print(f"[green]Loaded {chunk_count} chunks into vector store[/green]")
        console.print(
            f"[green]Graph: {pipeline.knowledge_graph.entity_count} entities, "
            f"{pipeline.knowledge_graph.relationship_count} relationships[/green]"
        )
    except GraphRAGError as e:
        console.print(f"[yellow]Warning: {e}[/yellow]")

    return pipeline


def interactive_loop(pipeline: GraphPipeline):
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
        except GraphRAGError as e:
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
