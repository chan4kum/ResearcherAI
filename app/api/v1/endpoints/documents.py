from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.core.logging import get_logger
from app.models.schemas import (
    ChunkDocumentRequest,
    ChunkResponse,
    ChunkTextRequest,
    DocumentIngestResponse,
    DocumentListResponse,
    EmbedDocumentResponse,
    SyncKnowledgeBaseRequest,
    SyncKnowledgeBaseResponse,
    TextIngestRequest,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResultItem,
)
from app.services.document.chunker import display_document_chunks
from app.services.document.models import Document, DocumentMetadata
from app.services.document.service import DocumentService

logger = get_logger("app.api.v1.documents")
router = APIRouter()


def get_document_service(request: Request) -> DocumentService:
    """Dependency resolver for DocumentService instance attached to FastAPI state."""
    service: DocumentService | None = getattr(request.app.state, "document_service", None)
    if not service:
        service = DocumentService()
        request.app.state.document_service = service
    return service


@router.post(
    "/upload",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and ingest a local file (TXT, PDF, MD, etc.)",
)
async def upload_document(
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentIngestResponse:
    """Accept a multipart file upload, extract text, preserve metadata, and deduplicate."""
    filename = file.filename or "uploaded_file"
    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty",
            )

        result = doc_service.ingest_bytes(
            content_bytes=content_bytes,
            source_name=filename,
        )

        char_count = result.metadata.character_count if result.metadata else 0
        word_count = result.metadata.word_count if result.metadata else 0
        page_count = result.metadata.page_count if result.metadata else None

        return DocumentIngestResponse(
            doc_id=result.doc_id,
            status=result.status.value,
            source=result.source,
            checksum=result.checksum,
            message=result.message,
            character_count=char_count,
            word_count=word_count,
            page_count=page_count,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("document_upload_error", filename=filename, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded file: {exc}",
        ) from exc


@router.post(
    "/ingest-text",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest raw text content directly",
)
async def ingest_text(
    request: TextIngestRequest,
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentIngestResponse:
    """Directly ingest a raw string payload as a Document."""
    content_bytes = request.content.encode("utf-8")
    meta = request.custom_metadata.copy()
    if request.document_type:
        meta["document_type"] = request.document_type
    if request.department:
        meta["department"] = request.department
    if request.date:
        meta["date"] = request.date
    if request.author:
        meta["author"] = request.author
    if request.tags:
        meta["tags"] = request.tags

    result = doc_service.ingest_bytes(
        content_bytes=content_bytes,
        source_name=request.source_name,
        custom_metadata=meta,
    )

    char_count = result.metadata.character_count if result.metadata else 0
    word_count = result.metadata.word_count if result.metadata else 0

    return DocumentIngestResponse(
        doc_id=result.doc_id,
        status=result.status.value,
        source=result.source,
        checksum=result.checksum,
        message=result.message,
        character_count=char_count,
        word_count=word_count,
        page_count=1,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all ingested documents and their metadata",
)
async def list_documents(
    doc_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """Retrieve metadata for all ingested documents in the store."""
    docs = doc_service.list_documents()
    return DocumentListResponse(
        total=len(docs),
        documents=[doc.model_dump() for doc in docs],
    )


@router.get(
    "/{doc_id}",
    status_code=status.HTTP_200_OK,
    summary="Get extracted content and metadata for a specific document",
)
async def get_document(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> dict[str, Any]:
    """Retrieve full text and metadata for a specific document."""
    doc = doc_service.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found",
        )
    return doc.model_dump()


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an ingested document",
)
async def delete_document(
    doc_id: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> dict[str, Any]:
    """Delete a document from memory and disk store."""
    success = doc_service.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found",
        )
    return {"status": "deleted", "doc_id": doc_id}


@router.post(
    "/{doc_id}/chunk",
    response_model=ChunkResponse,
    status_code=status.HTTP_200_OK,
    summary="Partition an ingested document into deterministic chunks",
)
async def chunk_document_endpoint(
    doc_id: str,
    body: ChunkDocumentRequest | None = None,
    doc_service: DocumentService = Depends(get_document_service),
) -> ChunkResponse:
    """Partition an existing document into chunks with preserved metadata."""
    doc = doc_service.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found",
        )

    chunk_size = body.chunk_size if body else None
    chunk_overlap = body.chunk_overlap if body else None

    try:
        chunks = doc_service.chunk_document(
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        diagram = display_document_chunks(doc, chunks)
        return ChunkResponse(
            doc_id=doc_id,
            total_chunks=len(chunks),
            chunks=[c.model_dump() for c in chunks],
            preview_diagram=diagram,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.post(
    "/chunk-text",
    response_model=ChunkResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk raw text directly without persisting parent document",
)
async def chunk_text_endpoint(
    request: ChunkTextRequest,
    doc_service: DocumentService = Depends(get_document_service),
) -> ChunkResponse:
    """Partition raw text string into chunks on the fly."""
    try:
        chunks = doc_service.chunk_text(
            text=request.text,
            source_name=request.source_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            custom_metadata=request.custom_metadata,
        )
        adhoc_meta = DocumentMetadata(
            doc_id="adhoc_doc",
            source=request.source_name,
            file_type="txt",
            checksum="adhoc",
            character_count=len(request.text),
            word_count=len(request.text.split()),
            custom_metadata=request.custom_metadata,
        )
        doc = Document(doc_id="adhoc_doc", content=request.text, metadata=adhoc_meta)
        diagram = display_document_chunks(doc, chunks)
        return ChunkResponse(
            doc_id="adhoc_text",
            total_chunks=len(chunks),
            chunks=[c.model_dump() for c in chunks],
            preview_diagram=diagram,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.post(
    "/{doc_id}/embed",
    response_model=EmbedDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk an ingested document, generate vector embeddings, and store in vector database",
)
async def embed_document_endpoint(
    doc_id: str,
    body: ChunkDocumentRequest | None = None,
    doc_service: DocumentService = Depends(get_document_service),
) -> EmbedDocumentResponse:
    """Execute Document -> Text Extraction -> Chunking -> Embedding -> Vector DB pipeline."""
    doc = doc_service.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found",
        )

    chunk_size = body.chunk_size if body else None
    chunk_overlap = body.chunk_overlap if body else None

    try:
        embedded_chunks = await doc_service.embed_and_index_document(
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        dimensions = len(embedded_chunks[0].embedding) if embedded_chunks else 0
        return EmbedDocumentResponse(
            doc_id=doc_id,
            total_chunks=len(embedded_chunks),
            dimensions=dimensions,
            chunks=[ec.model_dump() for ec in embedded_chunks],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.post(
    "/search",
    response_model=VectorSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic vector similarity search across ingested document chunks in vector database",
)
async def search_documents_endpoint(
    request: VectorSearchRequest,
    doc_service: DocumentService = Depends(get_document_service),
) -> VectorSearchResponse:
    """Execute semantic similarity search using query vector embeddings."""
    results = await doc_service.search_similar_chunks(
        query=request.query,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
        filters=request.filters,
    )
    items = [
        VectorSearchResultItem(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            content=chunk.content,
            similarity=sim,
            document_type=chunk.metadata.document_type,
            department=chunk.metadata.department,
            date=chunk.metadata.date,
            author=chunk.metadata.author,
            tags=chunk.metadata.tags,
            metadata=chunk.metadata.model_dump(),
        )
        for chunk, sim in results
    ]
    return VectorSearchResponse(
        query=request.query,
        mode=request.mode or "semantic",
        total_results=len(items),
        results=items,
    )


@router.post(
    "/sync-kb",
    response_model=SyncKnowledgeBaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Synchronize and index knowledge base directory into the vector database",
)
async def sync_knowledge_base_endpoint(
    request: SyncKnowledgeBaseRequest | None = None,
    doc_service: DocumentService = Depends(get_document_service),
) -> SyncKnowledgeBaseResponse:
    """Scan KB directory, ingest documents, chunk, embed, and index into vector database."""
    kb_dir = request.kb_dir if request else None
    chunk_size = request.chunk_size if request else None
    chunk_overlap = request.chunk_overlap if request else None

    result = await doc_service.sync_knowledge_base(
        kb_dir=kb_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return SyncKnowledgeBaseResponse(
        kb_dir=result["kb_dir"],
        status=result["status"],
        total_files=result["total_files"],
        ingested=result["ingested"],
        skipped=result["skipped"],
        indexed_documents=result["indexed_documents"],
        total_indexed_chunks=result["total_indexed_chunks"],
    )



