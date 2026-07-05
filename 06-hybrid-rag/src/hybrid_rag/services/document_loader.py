"""Document loader service for Hybrid RAG."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from hybrid_rag.config import HybridRAGConfig
from hybrid_rag.exceptions import DocumentLoadError


class DocumentLoader:
    """Loads and processes text documents."""

    def __init__(self, config: HybridRAGConfig):
        """Initialize the document loader.

        Args:
            config: Configuration settings.
        """
        self.config = config

    def load_document(self, file_path: str) -> list[Document]:
        """Load a single document.

        Args:
            file_path: Path to the document file.

        Returns:
            List of loaded documents.

        Raises:
            DocumentLoadError: If loading fails.
        """
        path = Path(file_path)

        if not path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        supported_extensions = self.config.get_supported_document_extensions()

        if suffix not in supported_extensions:
            raise DocumentLoadError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(supported_extensions)}"
            )

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path))

            return loader.load()
        except Exception as e:
            raise DocumentLoadError(f"Failed to load {file_path}: {e}") from e

    def load_documents_from_directory(self, directory: str | None = None) -> list[Document]:
        """Load all documents from a directory.

        Args:
            directory: Directory path. Uses config.data_dir if None.

        Returns:
            List of loaded documents.

        Raises:
            DocumentLoadError: If directory doesn't exist or loading fails.
        """
        dir_path = Path(directory or self.config.data_dir)

        if not dir_path.exists():
            raise DocumentLoadError(f"Directory not found: {dir_path}")

        all_documents = []
        supported_extensions = self.config.get_supported_document_extensions()

        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    docs = self.load_document(str(file_path))
                    all_documents.extend(docs)
                except DocumentLoadError as e:
                    print(f"Warning: {e}")

        return all_documents

    def load_documents(self, paths: list[str]) -> list[Document]:
        """Load multiple documents.

        Args:
            paths: List of file paths.

        Returns:
            List of loaded documents.
        """
        all_documents = []
        for path in paths:
            try:
                docs = self.load_document(path)
                all_documents.extend(docs)
            except DocumentLoadError as e:
                print(f"Warning: {e}")

        return all_documents
