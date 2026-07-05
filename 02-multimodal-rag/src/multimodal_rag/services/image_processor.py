"""Image processor service for Multimodal RAG."""

from pathlib import Path

from langchain_core.documents import Document
from multimodal_rag.config import MultimodalRAGConfig
from multimodal_rag.exceptions import ImageProcessingError, OCRError
from multimodal_rag.models.schemas import ImageMetadata, ProcessedImage


class ImageProcessor:
    """Processes images: OCR extraction and description generation."""

    def __init__(self, config: MultimodalRAGConfig):
        """Initialize the image processor.

        Args:
            config: Configuration settings.
        """
        self.config = config

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted text from the image.

        Raises:
            OCRError: If OCR extraction fails.
        """
        if not self.config.ocr_enabled:
            return ""

        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except ImportError:
            raise OCRError(
                "pytesseract not installed. "
                "Install with: pip install pytesseract"
            )
        except Exception as e:
            raise OCRError(f"OCR failed for {image_path}: {e}") from e

    def get_image_info(self, image_path: str) -> tuple[int, int, str]:
        """Get image dimensions and format.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (width, height, format).
        """
        try:
            from PIL import Image

            image = Image.open(image_path)
            return image.width, image.height, image.format or ""
        except Exception:
            return 0, 0, ""

    def describe_image(self, image_path: str) -> str:
        """Generate a description of the image using vision model.

        Args:
            image_path: Path to the image file.

        Returns:
            Description of the image.
        """
        try:
            # For now, return a basic description
            # In production, this would call a vision model like LLaVA
            return f"Image located at {image_path}"
        except Exception:
            return ""

    def process_image(self, image_path: str) -> ProcessedImage:
        """Process an image and extract all information.

        Args:
            image_path: Path to the image file.

        Returns:
            ProcessedImage with metadata and content.

        Raises:
            ImageProcessingError: If processing fails.
        """
        path = Path(image_path)

        if not path.exists():
            raise ImageProcessingError(f"Image not found: {image_path}")

        try:
            # Extract OCR text
            ocr_text = self.extract_text_from_image(image_path)

            # Get image info
            width, height, format_str = self.get_image_info(image_path)

            # Generate description
            description = self.describe_image(image_path)

            # Create metadata
            metadata = ImageMetadata(
                source=str(path.absolute()),
                ocr_text=ocr_text,
                description=description,
                width=width,
                height=height,
                format=format_str,
                has_text=bool(ocr_text),
            )

            # Combine text for embedding
            content_parts = []
            if ocr_text:
                content_parts.append(f"OCR Text: {ocr_text}")
            if description:
                content_parts.append(f"Description: {description}")

            content = "\n".join(content_parts) if content_parts else str(path.name)

            return ProcessedImage(metadata=metadata, content=content)
        except OCRError:
            raise
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to process image {image_path}: {e}"
            ) from e

    def load_and_process_images(self, directory: str | None = None) -> list[ProcessedImage]:
        """Load and process all images from a directory.

        Args:
            directory: Directory path. Uses config.images_dir if None.

        Returns:
            List of processed images.
        """
        dir_path = Path(directory or self.config.images_dir)

        if not dir_path.exists():
            return []

        processed_images = []
        supported_extensions = self.config.get_supported_image_extensions()

        for img_path in dir_path.rglob("*"):
            if img_path.suffix.lower() in supported_extensions:
                try:
                    processed = self.process_image(str(img_path))
                    processed_images.append(processed)
                except ImageProcessingError as e:
                    print(f"Warning: {e}")

        return processed_images

    def to_documents(self, processed_images: list[ProcessedImage]) -> list[Document]:
        """Convert processed images to LangChain documents.

        Args:
            processed_images: List of processed images.

        Returns:
            List of LangChain Document objects.
        """
        documents = []
        for img in processed_images:
            doc = Document(
                page_content=img.content,
                metadata={
                    "source": img.source,
                    "type": "image",
                    "has_text": img.has_text,
                    "ocr_text": img.metadata.ocr_text,
                    "description": img.metadata.description,
                },
            )
            documents.append(doc)

        return documents
