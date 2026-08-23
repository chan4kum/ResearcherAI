"""Debugging utility demonstrating deterministic document -> chunks visualization."""

import sys
from pathlib import Path

from app.services.document.chunker import DocumentChunker, display_document_chunks
from app.services.document.loaders.factory import DocumentLoaderFactory
from app.services.document.models import ChunkingConfig


def run_chunk_demo(
    file_path: str = "tests/fixtures/sample.txt",
    chunk_size: int = 150,
    chunk_overlap: int = 30,
) -> None:
    """Load a sample document, chunk it, and print the visual diagram."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: Sample file '{file_path}' does not exist.")
        sys.exit(1)

    print(f"\n--- Loading Document: {file_path} ---")
    doc = DocumentLoaderFactory.load_file(path)

    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunker = DocumentChunker(default_config=config)
    chunks = chunker.chunk_document(doc, config=config)

    diagram = display_document_chunks(doc, chunks)
    print(diagram)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample.txt"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    run_chunk_demo(target, chunk_size=size, chunk_overlap=overlap)
