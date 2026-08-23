"""RAG (Retrieval-Augmented Generation) package with vector retrieval and citation tracking."""

from app.services.rag.adaptive import (
    AdaptiveRetrievalResult,
    AdaptiveRetriever,
    EvidenceEvaluation,
    EvidenceEvaluator,
    EvidenceSufficiencyStatus,
)
from app.services.rag.agentic_retrieval import (
    AgenticRetrievalEngine,
    AgenticRetrievalResult,
    AgenticRetrievalTrace,
    RetrievalPlan,
    RetrievalPlanner,
    RetrievalStepType,
    RetrievalTraceStep,
    RetrievalTraceStore,
)
from app.services.rag.analyzer import QueryAnalyzer
from app.services.rag.bm25 import BM25Index
from app.services.rag.critic import (
    CriticAgent,
    CriticEvaluation,
    CriticIssue,
    CriticIssueSeverity,
    CriticIssueType,
    SelfCorrectionAttempt,
    SelfCorrectionEngine,
    SelfCorrectionResult,
)
from app.services.rag.evaluator import (
    EvaluationReason,
    RetrievalEvaluation,
    RetrievalEvaluator,
)
from app.services.rag.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.services.rag.hyde import HyDEGenerator, HyDEResult
from app.services.rag.loop import (
    AgenticResearchLoopConfig,
    AgenticResearchLoopResult,
    AgenticResearchOrchestrator,
)
from app.services.rag.models import Citation, RAGResponse
from app.services.rag.query_analysis import (
    ExtractedEntity,
    QueryAnalysis,
    QueryIntent,
)
from app.services.rag.reranker import (
    BaseReranker,
    MockReranker,
    RerankMeasurement,
    RerankSummary,
    create_reranker,
)
from app.services.rag.research import (
    MultiStepResearchExecutor,
    MultiStepResearchPlanner,
    ParallelResearchConfig,
    ResearchEvidenceStore,
    ResearchExecutionResult,
    ResearchPlan,
    ResearchSubquestion,
    ResearchSubquestionType,
    SubquestionExecutionResult,
    SubquestionExecutionStatus,
)
from app.services.rag.retriever import (
    BaseRetriever,
    BM25Retriever,
    HybridRetriever,
    HyDERetriever,
    VectorRetriever,
    create_retriever,
)
from app.services.rag.rewriter import (
    IterativeRetrievalResult,
    QueryRewriteAttempt,
    QueryRewriter,
)
from app.services.rag.router import RetrievalRouter
from app.services.rag.routing import RoutingDecision, SourceDestination
from app.services.rag.service import DEFAULT_RAG_SYSTEM_PROMPT, RAGService
from app.services.rag.sources import (
    BaseRetrievalSource,
    KeywordSearchSource,
    RetrievalSourceRegistry,
    SourceResult,
    SourceType,
    StructuredDatabasePlaceholderSource,
    VectorDatabaseSource,
    WebSearchPlaceholderSource,
)
from app.services.rag.verification import (
    AnswerVerifier,
    ClaimSupportStatus,
    FactualClaim,
    VerificationReport,
)

__all__ = [
    "AdaptiveRetrievalResult",
    "AdaptiveRetriever",
    "AgenticResearchLoopConfig",
    "AgenticResearchLoopResult",
    "AgenticResearchOrchestrator",
    "AgenticRetrievalEngine",
    "AgenticRetrievalResult",
    "AgenticRetrievalTrace",
    "AnswerVerifier",
    "BM25Index",
    "BM25Retriever",
    "BaseReranker",
    "BaseRetrievalSource",
    "BaseRetriever",
    "Citation",
    "ClaimSupportStatus",
    "CriticAgent",
    "CriticEvaluation",
    "CriticIssue",
    "CriticIssueSeverity",
    "CriticIssueType",
    "DEFAULT_RAG_SYSTEM_PROMPT",
    "EvaluationReason",
    "EvidenceEvaluation",
    "EvidenceEvaluator",
    "EvidenceSufficiencyStatus",
    "ExtractedEntity",
    "FactualClaim",
    "HyDEGenerator",
    "HyDEResult",
    "HyDERetriever",
    "HybridRetriever",
    "IterativeRetrievalResult",
    "KeywordSearchSource",
    "MockReranker",
    "MultiStepResearchExecutor",
    "MultiStepResearchPlanner",
    "ParallelResearchConfig",
    "QueryAnalysis",
    "QueryAnalyzer",
    "QueryIntent",
    "QueryRewriteAttempt",
    "QueryRewriter",
    "RAGResponse",
    "RAGService",
    "RerankMeasurement",
    "RerankSummary",
    "ResearchEvidenceStore",
    "ResearchExecutionResult",
    "ResearchPlan",
    "ResearchSubquestion",
    "ResearchSubquestionType",
    "RetrievalEvaluation",
    "RetrievalEvaluator",
    "RetrievalPlan",
    "RetrievalPlanner",
    "RetrievalRouter",
    "RetrievalSourceRegistry",
    "RetrievalStepType",
    "RetrievalTraceStep",
    "RetrievalTraceStore",
    "RoutingDecision",
    "SelfCorrectionAttempt",
    "SelfCorrectionEngine",
    "SelfCorrectionResult",
    "SourceDestination",
    "SourceResult",
    "SourceType",
    "StructuredDatabasePlaceholderSource",
    "SubquestionExecutionResult",
    "SubquestionExecutionStatus",
    "VectorDatabaseSource",
    "VectorRetriever",
    "VerificationReport",
    "WebSearchPlaceholderSource",
    "create_reranker",
    "create_retriever",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]
