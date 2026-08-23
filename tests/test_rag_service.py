import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.services.document.models import ChunkMetadata, DocumentChunk
from app.services.document.service import DocumentService
from app.services.document.store import DocumentStore
from app.services.embedding.service import EmbeddingService
from app.services.llm.mock import MockLLMProvider
from app.services.llm.service import LLMService
from app.services.rag.models import Citation
from app.services.rag.retriever import VectorRetriever
from app.services.rag.service import DEFAULT_RAG_SYSTEM_PROMPT, RAGService


@pytest.fixture
def mock_embedding_service(settings: Settings) -> EmbeddingService:
    return EmbeddingService(settings=settings)


@pytest.fixture
def in_memory_repo() -> InMemoryVectorRepository:
    return InMemoryVectorRepository()


@pytest.fixture
def vector_retriever(
    mock_embedding_service: EmbeddingService,
    in_memory_repo: InMemoryVectorRepository,
) -> VectorRetriever:
    return VectorRetriever(
        embedding_service=mock_embedding_service,
        vector_repository=in_memory_repo,
    )


@pytest.fixture
def mock_llm_service(settings: Settings) -> LLMService:
    return LLMService(
        provider=MockLLMProvider(model_name="mock-rag-model"),
        settings=settings,
    )


@pytest.fixture
def rag_service(
    vector_retriever: VectorRetriever,
    mock_llm_service: LLMService,
    settings: Settings,
) -> RAGService:
    return RAGService(
        retriever=vector_retriever,
        llm_service=mock_llm_service,
        settings=settings,
    )


@pytest.mark.asyncio
async def test_vector_retriever_empty_query(vector_retriever: VectorRetriever) -> None:
    results = await vector_retriever.retrieve(query="   ", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_vector_retriever_returns_citations(
    vector_retriever: VectorRetriever,
    in_memory_repo: InMemoryVectorRepository,
    mock_embedding_service: EmbeddingService,
) -> None:
    # Seed chunks into vector repository
    chunk1 = DocumentChunk(
        chunk_id="chunk_001",
        doc_id="doc_123",
        content="Semiconductor manufacturing uses photolithography to transfer circuit patterns.",
        metadata=ChunkMetadata(
            chunk_id="chunk_001",
            doc_id="doc_123",
            source="semiconductors.txt",
            checksum="a1b2c3d4e5",
            file_type="txt",
            index=0,
            start_char=0,
            end_char=75,
            character_count=75,
            word_count=8,
            custom_metadata={"topic": "hardware"},
        ),
    )
    embedded_chunk = (await mock_embedding_service.embed_chunks([chunk1]))[0]
    await in_memory_repo.store_chunks([embedded_chunk])

    citations = await vector_retriever.retrieve(query="photolithography circuits", top_k=3)
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk_001"
    assert citations[0].doc_id == "doc_123"
    assert citations[0].source == "semiconductors.txt"
    assert citations[0].file_type == "txt"
    assert citations[0].chunk_index == 0
    assert "photolithography" in citations[0].content
    assert citations[0].metadata == {"topic": "hardware"}


def test_rag_service_format_context(rag_service: RAGService) -> None:
    # Test empty citations
    empty_str = rag_service.format_context([])
    assert "No relevant context found" in empty_str

    # Test populated citations
    citations = [
        Citation(
            chunk_id="c1",
            doc_id="d1",
            source="doc1.txt",
            file_type="txt",
            chunk_index=0,
            content="First chunk text.",
            similarity=0.95,
        ),
        Citation(
            chunk_id="c2",
            doc_id="d1",
            source="doc1.txt",
            file_type="txt",
            chunk_index=1,
            content="Second chunk text.",
            similarity=0.88,
        ),
    ]
    formatted = rag_service.format_context(citations)
    assert "[Citation 1] (Source: doc1.txt, Chunk: 0, Similarity: 0.9500)" in formatted
    assert "First chunk text." in formatted
    assert "[Citation 2] (Source: doc1.txt, Chunk: 1, Similarity: 0.8800)" in formatted
    assert "Second chunk text." in formatted


def test_rag_service_build_prompts(rag_service: RAGService) -> None:
    citations = [
        Citation(
            chunk_id="c1",
            doc_id="d1",
            source="doc1.txt",
            file_type="txt",
            chunk_index=0,
            content="Lithography pattern transfer.",
            similarity=0.91,
        )
    ]
    sys_prompt, user_prompt = rag_service.build_prompts(
        question="How does lithography work?",
        citations=citations,
    )
    assert sys_prompt == DEFAULT_RAG_SYSTEM_PROMPT
    assert "Context Information:" in user_prompt
    assert "[Citation 1]" in user_prompt
    assert "Lithography pattern transfer." in user_prompt
    assert "Question: How does lithography work?" in user_prompt


@pytest.mark.asyncio
async def test_rag_service_answer_empty_question(rag_service: RAGService) -> None:
    with pytest.raises(ValueError, match="Question cannot be empty"):
        await rag_service.answer(question="   ")


@pytest.mark.asyncio
async def test_rag_service_answer_no_context_found(rag_service: RAGService) -> None:
    # No documents in repository
    response = await rag_service.answer(question="What is quantum gravity?")
    assert response.question == "What is quantum gravity?"
    assert response.retrieved_chunks_count == 0
    assert response.citations == []
    assert "not contain information" in response.answer.lower()


@pytest.mark.asyncio
async def test_rag_service_evaluation_known_document(
    rag_service: RAGService,
    in_memory_repo: InMemoryVectorRepository,
    mock_embedding_service: EmbeddingService,
    settings: Settings,
) -> None:
    """Evaluation Acceptance Test: Ingest sample document and verify grounded answer & citations."""
    # 1. Ingest sample.txt
    doc_store = DocumentStore()
    doc_service = DocumentService(
        store=doc_store,
        embedding_service=mock_embedding_service,
        vector_repository=in_memory_repo,
        settings=settings,
    )

    ingest_result = doc_service.ingest_file("tests/fixtures/sample.txt")
    await doc_service.embed_and_index_document(
        ingest_result.doc_id, chunk_size=100, chunk_overlap=20
    )

    # 2. Query RAG service about the ingested document
    response = await rag_service.answer(
        question="What are the key stages of semiconductor manufacturing?",
        top_k=3,
    )

    # 3. Verify evaluation acceptance criteria
    assert response.question == "What are the key stages of semiconductor manufacturing?"
    assert response.retrieved_chunks_count > 0
    assert len(response.citations) > 0
    assert response.citations[0].source == "sample.txt"
    assert response.citations[0].file_type == "txt"
    assert "Based on [Citation 1]" in response.answer
    assert response.model == "mock-rag-model"
    assert response.provider == "mock"
