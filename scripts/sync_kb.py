#!/usr/bin/env python3
"""CLI utility to scan, incrementally ingest, chunk, embed, and index KB directory."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db.repository import create_vector_repository
from app.db.session import DatabaseManager
from app.services.document.service import DocumentService
from app.services.embedding.service import EmbeddingService


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync knowledge base directory with Enterprise Agentic Knowledge Platform."
    )
    parser.add_argument(
        "--kb-dir",
        type=str,
        default=None,
        help="Path to knowledge base directory (defaults to configured settings, e.g., 'KB')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in characters (e.g. 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap in characters (e.g. 50)",
    )

    args = parser.parse_args()
    settings = get_settings()
    target_dir = args.kb_dir or settings.knowledge_base_dir

    print("\n============================================================")
    print(" Knowledge Base Synchronization Pipeline")
    print("============================================================")
    print(f" Target Directory:   {target_dir}")
    print(f" Embedding Provider: {settings.embedding_provider} ({settings.embedding_model})")
    print(f" Vector Repository:  {settings.vector_repository_type}")
    print("============================================================\n")

    db_manager = DatabaseManager(settings=settings)
    vector_repo = create_vector_repository(settings=settings, db_manager=db_manager)
    embedding_service = EmbeddingService(settings=settings)
    doc_service = DocumentService(
        embedding_service=embedding_service,
        vector_repository=vector_repo,
        settings=settings,
    )

    result = await doc_service.sync_knowledge_base(
        kb_dir=target_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Status:            {result['status'].upper()}")
    print(f"Total Files Scanned: {result['total_files']}")
    print(f"New Ingested:        {result['ingested']}")
    print(f"Skipped Duplicates:  {result['skipped']}")
    print(f"Indexed Documents:   {result['indexed_documents']}")
    print(f"Total Indexed Chunks:{result['total_indexed_chunks']}")
    print("============================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
