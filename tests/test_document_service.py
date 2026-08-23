from pathlib import Path

from app.services.document.models import IngestStatus
from app.services.document.service import DocumentService
from app.services.document.store import DocumentStore


def test_document_service_ingest_bytes() -> None:
    """Verify DocumentService ingests raw bytes and extracts metadata."""
    store = DocumentStore(storage_dir=None)
    service = DocumentService(store=store)

    result = service.ingest_bytes(
        content_bytes=b"Autonomous AI agents leverage LLMs for reasoning.",
        source_name="agents.txt",
    )

    assert result.status == IngestStatus.INGESTED
    assert result.doc_id != ""
    assert result.metadata is not None
    assert result.metadata.word_count == 7
    assert store.count() == 1


def test_document_service_duplicate_deduplication() -> None:
    """Verify duplicate document with identical content is not re-ingested."""
    store = DocumentStore(storage_dir=None)
    service = DocumentService(store=store)
    content = b"Exact duplicate document content"

    # First ingestion -> INGESTED
    res1 = service.ingest_bytes(content_bytes=content, source_name="doc_v1.txt")
    assert res1.status == IngestStatus.INGESTED
    assert store.count() == 1

    # Second ingestion of same bytes -> SKIPPED_DUPLICATE
    res2 = service.ingest_bytes(content_bytes=content, source_name="doc_copy.txt")
    assert res2.status == IngestStatus.SKIPPED_DUPLICATE
    assert res2.doc_id == res1.doc_id
    assert res2.checksum == res1.checksum
    assert store.count() == 1  # No duplicate stored


def test_document_service_ingest_real_pdf() -> None:
    """Verify DocumentService ingests real sample PDF file."""
    store = DocumentStore(storage_dir=None)
    service = DocumentService(store=store)

    result = service.ingest_file("tests/fixtures/sample.pdf")
    assert result.status == IngestStatus.INGESTED
    assert result.metadata is not None
    assert result.metadata.file_type == "pdf"
    assert result.metadata.page_count == 1

    doc = service.get_document(result.doc_id)
    assert doc is not None
    assert "Semiconductor Manufacturing" in doc.content


def test_document_service_incremental_directory_ingestion(tmp_path: Path) -> None:
    """Verify incremental directory loading: existing files skipped, only new files ingested."""
    store = DocumentStore(storage_dir=None)
    service = DocumentService(store=store)

    # 1. Create first document in temp directory
    file1 = tmp_path / "file1.txt"
    file1.write_text("First document content", encoding="utf-8")

    # Initial batch ingestion
    batch1 = service.ingest_directory(tmp_path)
    assert len(batch1) == 1
    assert batch1[0].status == IngestStatus.INGESTED
    assert store.count() == 1

    # 2. Add a second document to the same directory
    file2 = tmp_path / "file2.txt"
    file2.write_text("Second document content", encoding="utf-8")

    # Second batch ingestion -> file1 is skipped, file2 is ingested
    batch2 = service.ingest_directory(tmp_path)
    assert len(batch2) == 2

    # Map results by source filename
    results_by_source = {r.source: r.status for r in batch2}
    assert results_by_source["file1.txt"] == IngestStatus.SKIPPED_DUPLICATE
    assert results_by_source["file2.txt"] == IngestStatus.INGESTED
    assert store.count() == 2
