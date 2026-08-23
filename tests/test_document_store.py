from pathlib import Path

from app.services.document.loaders.factory import DocumentLoaderFactory
from app.services.document.store import DocumentStore


def test_document_store_in_memory_crud() -> None:
    """Verify in-memory storage, retrieval, lookup, and deletion."""
    store = DocumentStore(storage_dir=None)
    doc = DocumentLoaderFactory.load_bytes(
        content_bytes=b"Quantum computing notes",
        source_name="quantum.txt",
    )

    assert store.count() == 0
    assert not store.has_checksum(doc.metadata.checksum)

    # Add document
    store.add(doc)
    assert store.count() == 1
    assert store.has_checksum(doc.metadata.checksum)

    # Get by ID
    retrieved = store.get(doc.doc_id)
    assert retrieved is not None
    assert retrieved.content == "Quantum computing notes"

    # Get by checksum
    by_checksum = store.get_by_checksum(doc.metadata.checksum)
    assert by_checksum is not None
    assert by_checksum.doc_id == doc.doc_id

    # List documents
    docs_list = store.list_documents()
    assert len(docs_list) == 1
    assert docs_list[0].source == "quantum.txt"

    # Delete document
    assert store.delete(doc.doc_id) is True
    assert store.count() == 0
    assert not store.has_checksum(doc.metadata.checksum)
    assert store.get(doc.doc_id) is None


def test_document_store_disk_persistence(tmp_path: Path) -> None:
    """Verify document persistence across store re-instantiations."""
    store1 = DocumentStore(storage_dir=tmp_path)
    doc1 = DocumentLoaderFactory.load_bytes(
        content_bytes=b"Persisted document content",
        source_name="persist.txt",
    )
    store1.add(doc1)

    # Verify JSON file written to disk
    json_file = tmp_path / f"{doc1.doc_id}.json"
    assert json_file.exists()

    # Create new store instance pointing to same directory (simulates restart)
    store2 = DocumentStore(storage_dir=tmp_path)
    assert store2.count() == 1
    assert store2.has_checksum(doc1.metadata.checksum)
    retrieved = store2.get(doc1.doc_id)
    assert retrieved is not None
    assert retrieved.content == "Persisted document content"

    # Delete removes file from disk
    store2.delete(doc1.doc_id)
    assert not json_file.exists()
