"""Naive RAG - Basic Retrieval-Augmented Generation pipeline."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from config import RAGConfig

console = Console()


class NaiveRAG:
    """Basic RAG pipeline implementation."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.get_ollama_url(),
        )
        self.llm = ChatOllama(
            model=config.llm_model,
            base_url=config.get_ollama_url(),
        )
        self.vector_store = None

    def load_documents(self, file_path: str) -> list[Document]:
        """Load documents from a file."""
        path = Path(file_path)

        if not path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            return []

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path))
        else:
            console.print(f"[red]Error: Unsupported file type: {path.suffix}[/red]")
            return []

        return loader.load()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents: list[Document]) -> Chroma:
        """Create ChromaDB vector store from documents."""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_dir,
        )
        return self.vector_store

    def retrieve(self, query: str, top_k: int = None) -> list[Document]:
        """Retrieve relevant documents for a query."""
        if self.vector_store is None:
            console.print("[red]Error: Vector store not initialized[/red]")
            return []

        k = top_k or self.config.top_k
        return self.vector_store.similarity_search(query, k=k)

    def generate(self, query: str, context: list[Document]) -> str:
        """Generate answer using retrieved context."""
        context_text = "\n\n".join([doc.page_content for doc in context])

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""
        )

        chain = prompt | self.llm
        response = chain.invoke({"context": context_text, "question": query})

        return response.content

    def query(self, question: str) -> str:
        """Full RAG pipeline: retrieve and generate."""
        console.print(f"\n[bold blue]Question:[/bold blue] {question}\n")

        # Retrieve
        console.print("[yellow]Retrieving relevant documents...[/yellow]")
        docs = self.retrieve(question)

        if not docs:
            return "No relevant documents found."

        console.print(f"[green]Found {len(docs)} relevant chunks[/green]\n")

        # Display retrieved chunks
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            console.print(f"[cyan]Chunk {i} (from {source}):[/cyan]")
            console.print(f"{doc.page_content[:200]}...\n")

        # Generate
        console.print("[yellow]Generating answer...[/yellow]")
        answer = self.generate(question, docs)

        return answer


def main():
    """Main entry point."""
    console.print(
        Panel.fit(
            "[bold green]Naive RAG[/bold green]\n"
            "Basic Retrieval-Augmented Generation Pipeline",
            border_style="green",
        )
    )

    config = RAGConfig()
    rag = NaiveRAG(config)

    # Example: Load and index a document
    data_dir = Path(config.data_dir)

    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        console.print(f"[yellow]Created data directory: {data_dir}[/yellow]")
        console.print("[yellow]Add PDF or TXT files to the data/ directory[/yellow]")

    # Load all documents from data directory
    documents = []
    for file_path in data_dir.glob("*"):
        if file_path.suffix.lower() in [".pdf", ".txt"]:
            docs = rag.load_documents(str(file_path))
            documents.extend(docs)

    if documents:
        # Split and index
        chunks = rag.split_documents(documents)
        console.print(f"[green]Split {len(documents)} documents into {len(chunks)} chunks[/green]")

        rag.create_vector_store(chunks)
        console.print("[green]Vector store created successfully![/green]")
    else:
        console.print("[yellow]No documents found in data/ directory[/yellow]")

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

            answer = rag.query(question)

            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(answer))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()