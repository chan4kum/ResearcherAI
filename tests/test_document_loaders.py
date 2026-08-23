from pathlib import Path

from app.services.document.loaders.base import compute_checksum
from app.services.document.loaders.factory import DocumentLoaderFactory
from app.services.document.loaders.pdf import PDFDocumentLoader
from app.services.document.loaders.text import TextDocumentLoader


def test_compute_checksum_deterministic() -> None:
    """Verify SHA-256 checksum is consistent and deterministic."""
    data = b"Hello, Enterprise Agentic AI"
    checksum1 = compute_checksum(data)
    checksum2 = compute_checksum(data)
    assert checksum1 == checksum2
    assert len(checksum1) == 64
    assert checksum1 != compute_checksum(b"Different data")


def test_text_document_loader_bytes() -> None:
    """Verify TextDocumentLoader extracts text and calculates metadata accurately."""
    loader = TextDocumentLoader()
    raw_bytes = b"Line 1: Introduction\nLine 2: Methodology\nLine 3: Conclusion"
    doc = loader.load_bytes(
        content_bytes=raw_bytes,
        source_name="intro.txt",
        custom_metadata={"category": "research"},
    )

    assert doc.doc_id is not None
    assert "Line 1: Introduction" in doc.content
    assert doc.metadata.source == "intro.txt"
    assert doc.metadata.file_type == "txt"
    assert doc.metadata.file_size_bytes == len(raw_bytes)
    assert doc.metadata.character_count == len(doc.content)
    assert doc.metadata.word_count == 9
    assert doc.metadata.page_count == 1
    assert doc.metadata.custom_metadata == {"category": "research"}
    assert doc.metadata.checksum == compute_checksum(raw_bytes)


def test_text_document_loader_file() -> None:
    """Verify TextDocumentLoader loads from a physical disk file."""
    loader = TextDocumentLoader()
    sample_path = Path("tests/fixtures/sample.txt")
    doc = loader.load_file(sample_path)

    assert "Semiconductor Manufacturing Process" in doc.content
    assert doc.metadata.source == "sample.txt"
    assert doc.metadata.file_type == "txt"
    assert doc.metadata.word_count > 0


def test_pdf_document_loader_real_pdf() -> None:
    """Verify PDFDocumentLoader parses and extracts text from a real PDF file."""
    loader = PDFDocumentLoader()
    pdf_path = Path("tests/fixtures/sample.pdf")
    doc = loader.load_file(pdf_path)

    assert doc.doc_id is not None
    expected_header = "Enterprise Agentic Research Platform: Semiconductor Manufacturing"
    assert expected_header in doc.content
    assert doc.metadata.source == "sample.pdf"
    assert doc.metadata.file_type == "pdf"
    assert doc.metadata.page_count == 1
    assert doc.metadata.character_count > 0
    assert doc.metadata.word_count > 0


def test_document_loader_factory_resolution() -> None:
    """Verify DocumentLoaderFactory selects PDF vs Text loader based on extension."""
    pdf_loader = DocumentLoaderFactory.get_loader("report.pdf")
    assert isinstance(pdf_loader, PDFDocumentLoader)

    txt_loader = DocumentLoaderFactory.get_loader("notes.txt")
    assert isinstance(txt_loader, TextDocumentLoader)

    md_loader = DocumentLoaderFactory.get_loader("README.md")
    assert isinstance(md_loader, TextDocumentLoader)

    csv_loader = DocumentLoaderFactory.get_loader("data.csv")
    assert isinstance(csv_loader, TextDocumentLoader)
