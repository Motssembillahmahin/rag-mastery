"""Document loading service."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from adaptive_rag.config import Settings
from adaptive_rag.exceptions import DocumentLoadError


class DocumentLoader:
    """Handles document loading and splitting."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, settings: Settings):
        self.settings = settings

    def load_file(self, file_path: str) -> list[Document]:
        """Load a single file."""
        path = Path(file_path)

        if not path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Unsupported file type: {path.suffix}. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path))

        try:
            return loader.load()
        except Exception as e:
            raise DocumentLoadError(f"Failed to load {file_path}: {e}") from e

    def load_directory(self, dir_path: str) -> list[Document]:
        """Load all supported files from a directory."""
        path = Path(dir_path)

        if not path.exists():
            raise DocumentLoadError(f"Directory not found: {dir_path}")

        documents = []
        for file_path in path.glob("*"):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                docs = self.load_file(str(file_path))
                documents.extend(docs)

        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        return text_splitter.split_documents(documents)
