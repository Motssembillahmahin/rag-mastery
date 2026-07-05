"""Document loading service."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from corrective_rag.config import Settings
from corrective_rag.exceptions import DocumentLoadError


class DocumentLoader:
    """Handles document loading and splitting."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, settings: Settings):
        self.settings = settings

    def load_file(self, file_path: str) -> list[Document]:
        """Load documents from a file."""
        path = Path(file_path)

        if not path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(f"Unsupported file type: {path.suffix}")

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path))

        return loader.load()

    def load_directory(self, directory: str | Path) -> list[Document]:
        """Load all supported documents from a directory."""
        dir_path = Path(directory)

        if not dir_path.exists():
            raise DocumentLoadError(f"Directory not found: {directory}")

        documents = []
        for file_path in dir_path.glob("*"):
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
