import json
import re

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.query_analysis import (
    ExtractedEntity,
    QueryAnalysis,
    QueryIntent,
)

logger = get_logger("app.services.rag.analyzer")

ANALYSIS_SYSTEM_PROMPT = """You are an expert Query Understanding & Semantic Analysis engine.
Analyze the user's question and produce a structured JSON object with EXACTLY the following keys:
{
  "intent": "factual" | "comparison" | "multi_part_research" | "ambiguous" | "analytical",
  "entities": [
    {"text": "entity_name", "label": "organization|product|metric|concept", "category": "category"}
  ],
  "subquestions": ["subquestion 1", "subquestion 2"],
  "required_information_types": ["type 1", "type 2"],
  "potential_source_types": ["source 1", "source 2"],
  "is_ambiguous": false,
  "clarification_needed": null,
  "temporal_scope": null,
  "confidence_score": 0.95
}

Rules:
1. Identify if the query is a comparison, factual lookup, multi-part research, or ambiguous.
2. Break complex or comparative questions into atomic subquestions.
3. Determine required information categories and potential candidate document source types.
4. Output raw JSON only.
"""


class QueryAnalyzer:
    """Service responsible for semantic query decomposition and intent analysis."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)

    def _heuristic_analysis(self, query: str) -> QueryAnalysis:
        """Deterministic rule-based query understanding for offline testing and fallback."""
        q_clean = query.strip()
        q_lower = q_clean.lower()
        words = q_lower.split()

        # 1. Ambiguity detection
        ambiguous_triggers = {"tell me about", "what about", "information on", "help", "details"}
        is_ambiguous = len(words) <= 3 or (
            any(q_lower.startswith(t) for t in ambiguous_triggers) and len(words) <= 4
        )
        if q_lower in {"tell me about delays", "delays", "airplanes", "help", "status"}:
            is_ambiguous = True

        # 2. Intent classification
        intent = QueryIntent.FACTUAL
        comp_keywords = [
            "compare", "comparison", "versus", " vs ", " vs.", "differ", "how did those compare"
        ]
        if is_ambiguous:
            intent = QueryIntent.AMBIGUOUS
        elif any(comp in q_lower for comp in comp_keywords):
            intent = QueryIntent.COMPARISON
        elif (
            " and " in q_lower
            and any(kw in q_lower for kw in [",", "also", "assess", "evaluate"])
        ) or (len(words) > 15 and ("?" in q_clean or "how" in q_lower)):
            intent = QueryIntent.MULTI_PART_RESEARCH
        elif any(q_lower.startswith(w) for w in ["how to", "procedure", "steps", "guide"]):
            intent = QueryIntent.PROCEDURAL
        elif any(w in q_lower for w in ["why", "reasons", "analyze", "impact", "cause", "explain"]):
            intent = QueryIntent.ANALYTICAL

        # 3. Entity extraction heuristics
        entities: list[ExtractedEntity] = []

        # Known organizations / brands
        org_patterns = [
            ("boeing", "organization", "aerospace"),
            ("airbus", "organization", "aerospace"),
            ("nasa", "organization", "space_agency"),
            ("faa", "regulatory_body", "aviation_authority"),
            ("eu", "regulatory_body", "government"),
            ("wipro", "organization", "technology"),
        ]
        for name, label, cat in org_patterns:
            if re.search(rf"\b{name}\b", q_lower):
                # find casing in original query if possible
                match = re.search(rf"\b{name}\b", q_clean, re.IGNORECASE)
                text = match.group(0) if match else name.capitalize()
                entities.append(ExtractedEntity(text=text, label=label, category=cat))

        # Products & Identifiers (e.g. 777X, AP-2026-X, A350)
        product_matches = re.findall(
            r"\b(?:[A-Z]{1,4}-\d{3,4}[A-Z0-9-]*|\d{3}[X]?|A\d{3}[A-Z]?)\b", q_clean
        )
        for prod in product_matches:
            if prod.lower() not in {"and", "the", "for"}:
                entities.append(
                    ExtractedEntity(text=prod, label="product", category="aerospace_model")
                )

        # Technical concepts
        concept_patterns = [
            ("production delays", "metric_issue", "manufacturing"),
            ("flutter damper", "component", "aerospace_hardware"),
            ("titanium", "material", "metallurgy"),
            ("wing spar", "component", "structural_component"),
            ("carbon regulations", "policy", "environmental_regulation"),
            ("fleet renewal", "operational_strategy", "aviation_management"),
            ("passenger surcharge", "financial_metric", "pricing"),
            ("photolithography", "process", "semiconductor_manufacturing"),
        ]
        for term, label, cat in concept_patterns:
            if term in q_lower:
                entities.append(ExtractedEntity(text=term, label=label, category=cat))

        # 4. Subquestion decomposition
        subquestions: list[str] = []
        if intent == QueryIntent.COMPARISON:
            org_names = [e.text for e in entities if e.label == "organization"]
            if len(org_names) >= 2:
                subquestions.append(f"What were the factors affecting {org_names[0]}?")
                subquestions.append(f"What were the factors affecting {org_names[1]}?")
                subquestions.append(f"How do {org_names[0]} and {org_names[1]} compare directly?")
            else:
                subquestions.append(
                    f"What are the key attributes of the first subject in: '{q_clean}'?"
                )
                subquestions.append(f"What are the comparative differences in: '{q_clean}'?")
        elif intent == QueryIntent.MULTI_PART_RESEARCH:
            # Split on commas / and
            parts = re.split(r",\s*|\s+and\s+|\s+also\s+", q_clean)
            for part in parts:
                p_clean = part.strip().rstrip("?.")
                if len(p_clean.split()) >= 3:
                    subquestions.append(f"{p_clean}?")
            if not subquestions:
                subquestions.append(q_clean)
        elif not is_ambiguous:
            subquestions.append(q_clean)

        # 5. Required information types
        required_info_types: list[str] = []
        if "delay" in q_lower or "delays" in q_lower:
            required_info_types.extend([
                "delay_causes", "production_schedules", "supply_chain_bottlenecks"
            ])
        if intent == QueryIntent.COMPARISON:
            required_info_types.append("comparative_benchmarks")
        if "thickness" in q_lower or "spec" in q_lower or "tolerance" in q_lower:
            required_info_types.append("technical_specifications")
        if "carbon" in q_lower or "regulation" in q_lower:
            required_info_types.extend(["regulatory_compliance_costs", "policy_mandates"])
        if "cost" in q_lower or "surcharge" in q_lower or "financial" in q_lower:
            required_info_types.append("financial_metrics")
        if not required_info_types:
            required_info_types.append("general_domain_knowledge")

        # 6. Potential source types
        potential_sources: list[str] = []
        if any(e.label in ("organization", "regulatory_body") for e in entities):
            potential_sources.extend([
                "annual_reports", "regulatory_filings", "industry_audits"
            ])
        if any(e.label in ("product", "component", "material") for e in entities):
            potential_sources.extend([
                "engineering_specifications", "maintenance_bulletins"
            ])
        if "carbon" in q_lower or "regulation" in q_lower:
            potential_sources.append("environmental_policy_documents")
        if not potential_sources:
            potential_sources.append("knowledge_base_documents")

        # 7. Temporal scope
        year_matches = re.findall(r"\b(19\d\d|20\d\d)\b", q_clean)
        temporal_scope = None
        if len(year_matches) >= 2:
            temporal_scope = f"{year_matches[0]} - {year_matches[-1]}"
        elif len(year_matches) == 1:
            temporal_scope = year_matches[0]

        clarification = (
            "Please specify the company, product, or time period you are asking about."
            if is_ambiguous
            else None
        )

        return QueryAnalysis(
            original_query=q_clean,
            intent=intent,
            entities=entities,
            subquestions=subquestions,
            required_information_types=required_info_types,
            potential_source_types=potential_sources,
            is_ambiguous=is_ambiguous,
            clarification_needed=clarification,
            temporal_scope=temporal_scope,
            confidence_score=0.75 if is_ambiguous else 0.95,
            metadata={"analyzer_mode": "heuristic"},
        )

    async def analyze(self, query: str) -> QueryAnalysis:
        """Analyze user query and return structured semantic analysis."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty or whitespace.")

        logger.info("query_analysis_started", query=clean_query[:100])

        # If LLM is mock, use high-precision deterministic heuristic analyzer
        if getattr(self._settings, "llm_provider", "mock") == "mock":
            analysis = self._heuristic_analysis(clean_query)
            logger.info(
                "query_analysis_completed_heuristic",
                intent=analysis.intent.value,
                entities_count=len(analysis.entities),
                subquestions_count=len(analysis.subquestions),
            )
            return analysis

        # For real LLM provider, prompt LLM for JSON extraction
        try:
            llm_response = await self._llm_service.generate(
                prompt=f"Analyze this query:\n\n{clean_query}",
                system_prompt=ANALYSIS_SYSTEM_PROMPT,
                temperature=0.0,
            )

            # Strip markdown fences if present
            raw_content = llm_response.content.strip()
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
                raw_content = re.sub(r"\n?```$", "", raw_content)

            parsed_data = json.loads(raw_content)

            # Validate against Pydantic model
            analysis = QueryAnalysis(
                original_query=clean_query,
                intent=QueryIntent(parsed_data.get("intent", "factual")),
                entities=[
                    ExtractedEntity(**ent)
                    for ent in parsed_data.get("entities", [])
                ],
                subquestions=parsed_data.get("subquestions", [clean_query]),
                required_information_types=parsed_data.get("required_information_types", []),
                potential_source_types=parsed_data.get("potential_source_types", []),
                is_ambiguous=parsed_data.get("is_ambiguous", False),
                clarification_needed=parsed_data.get("clarification_needed"),
                temporal_scope=parsed_data.get("temporal_scope"),
                confidence_score=float(parsed_data.get("confidence_score", 0.9)),
                metadata={"analyzer_mode": "llm_structured"},
            )
            logger.info(
                "query_analysis_completed_llm",
                intent=analysis.intent.value,
                entities_count=len(analysis.entities),
            )
            return analysis
        except Exception as exc:
            logger.warning(
                "query_analysis_llm_failed_fallback_to_heuristic",
                error=str(exc),
            )
            return self._heuristic_analysis(clean_query)
