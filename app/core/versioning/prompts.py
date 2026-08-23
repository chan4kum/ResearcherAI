"""
app/core/versioning/prompts.py — Prompt Version Registry

Maintains versioned prompt templates for:
- Planner agent node
- Tool decision agent node
- Answer synthesis agent node
- RAG context synthesis
- Query rewriter
- HyDE hypothetical passage generator
"""

from __future__ import annotations

from typing import Any

from app.core.versioning.models import PromptVersion


class PromptNotFoundError(KeyError):
    """Raised when a requested prompt or version is not found in the registry."""
    pass


class PromptRegistry:
    """Registry managing versioned prompt templates across the platform."""

    def __init__(self) -> None:
        self._prompts: dict[str, dict[str, PromptVersion]] = {}
        self._defaults: dict[str, str] = {}
        self._register_canonical_prompts()

    def register(self, prompt: PromptVersion, set_as_default: bool = True) -> None:
        """Register a new PromptVersion under its logical name."""
        if prompt.name not in self._prompts:
            self._prompts[prompt.name] = {}
        self._prompts[prompt.name][prompt.version] = prompt
        if set_as_default or prompt.name not in self._defaults:
            self._defaults[prompt.name] = prompt.version

    def get(self, name: str, version: str | None = None) -> PromptVersion:
        """Retrieve a PromptVersion by name and optional version (defaults to latest default)."""
        if name not in self._prompts:
            raise PromptNotFoundError(f"Prompt '{name}' is not registered.")

        target_version = version or self._defaults.get(name)
        if not target_version or target_version not in self._prompts[name]:
            available = list(self._prompts[name].keys())
            raise PromptNotFoundError(
                f"Version '{target_version}' for prompt '{name}' not found. Available versions: {available}"
            )
        return self._prompts[name][target_version]

    def set_default_version(self, name: str, version: str) -> None:
        """Change the active default version for a prompt."""
        if name not in self._prompts or version not in self._prompts[name]:
            raise PromptNotFoundError(f"Prompt '{name}' with version '{version}' does not exist.")
        self._defaults[name] = version

    def get_active_versions(self) -> dict[str, str]:
        """Return a mapping of all registered prompt names to their active default versions."""
        return dict(self._defaults)

    def list_prompts(self, name: str | None = None) -> list[PromptVersion]:
        """List all prompt versions, optionally filtered by prompt name."""
        if name:
            if name not in self._prompts:
                return []
            return list(self._prompts[name].values())
        all_prompts: list[PromptVersion] = []
        for versions in self._prompts.values():
            all_prompts.extend(versions.values())
        return all_prompts

    def _register_canonical_prompts(self) -> None:
        """Register initial built-in canonical prompt versions."""

        # ── 1. Planner Node ───────────────────────────────────────────────
        self.register(
            PromptVersion(
                name="planner",
                version="1.0.0",
                description="Canonical LangGraph reasoning planner prompt formulating 2 to 4 steps",
                variables=["guidance"],
                template=(
                    "ROLE: PLANNER. You are a strategic reasoning agent. "
                    "Analyze the given user task and formulate 2 to 4 concise, actionable steps.\n"
                    "Output each step on a new line numbered 1., 2., 3., etc."
                ),
            )
        )
        self.register(
            PromptVersion(
                name="planner",
                version="1.1.0",
                description="Enhanced planner prompt with explicit dependency ordering and safety constraints",
                variables=["guidance"],
                template=(
                    "ROLE: STRATEGIC PLANNER. You are an enterprise reasoning agent.\n"
                    "Analyze the user task carefully. Break it down into 2 to 4 sequential, unambiguous steps.\n"
                    "Rules:\n"
                    "1. Order steps by logical dependency.\n"
                    "2. Avoid redundant subtasks.\n"
                    "3. Format strictly as numbered list: 1., 2., 3."
                ),
            ),
            set_as_default=False,
        )

        # ── 2. Tool Decider Node ──────────────────────────────────────────
        self.register(
            PromptVersion(
                name="tool_decider",
                version="1.0.0",
                description="Canonical JSON tool decider analyzing task against available registry tools",
                variables=["tools_desc"],
                template=(
                    "ROLE: TOOL_DECIDER. You are a tool selection and routing component.\n"
                    "Analyze the user task and execution plan against the available tools below:\n"
                    "{tools_desc}\n\n"
                    "Determine if one of the registered tools is required to accurately solve the task.\n"
                    "Respond ONLY with a valid JSON object matching this schema:\n"
                    "{{\n"
                    '  "tool_name": "calculator" | "app_info" | null,\n'
                    '  "tool_args": {{ "param_name": value }}\n'
                    "}}\n"
                    "If no tool is needed or appropriate, set tool_name to null."
                ),
            )
        )

        # ── 3. Executor / Answer Node ─────────────────────────────────────
        self.register(
            PromptVersion(
                name="executor",
                version="1.0.0",
                description="Canonical execution prompt synthesizing final answer from plan and tool observations",
                variables=["constraints"],
                template=(
                    "ROLE: EXECUTOR. You are an enterprise intelligence agent. "
                    "Complete the user task thoroughly and accurately following the provided plan "
                    "and incorporating tool observations if available."
                ),
            )
        )

        # ── 4. RAG Synthesizer ────────────────────────────────────────────
        self.register(
            PromptVersion(
                name="rag_synthesizer",
                version="1.0.0",
                description="Canonical strict grounded RAG synthesizer with citation tags",
                variables=[],
                template=(
                    "You are an expert research and knowledge assistant.\n"
                    "Answer the user's question accurately, concisely, and strictly using the provided context.\n"
                    "Rules:\n"
                    "1. Ground your answer entirely in the provided retrieved context chunks.\n"
                    "2. Explicitly cite the supporting source using its citation tag (e.g., [Citation 1]).\n"
                    "3. If the context is insufficient, state that the information is not available.\n"
                    "4. Do NOT hallucinate facts, figures, or external details not present in the context."
                ),
            )
        )

        # ── 5. Query Rewriter ─────────────────────────────────────────────
        self.register(
            PromptVersion(
                name="query_rewriter",
                version="1.0.0",
                description="Canonical query reformulation prompt expanding acronyms and synonyms",
                variables=[],
                template=(
                    "You are an expert information retrieval query reformulation assistant.\n"
                    "Given a user query that returned low-relevance results, rewrite it to improve vector search recall.\n"
                    "Rules:\n"
                    "1. Expand acronyms and replace informal phrases with standard technical terminology.\n"
                    "2. Preserve the core semantic intent of the original question.\n"
                    "3. Output ONLY the rewritten search query with no conversational preamble."
                ),
            )
        )

        # ── 6. HyDE Generator ─────────────────────────────────────────────
        self.register(
            PromptVersion(
                name="hyde_generator",
                version="1.0.0",
                description="Hypothetical Document Embeddings synthetic passage generator",
                variables=[],
                template=(
                    "You are an expert technical and scientific document synthesizer.\n"
                    "Given a user query, write a plausible, highly-detailed excerpt or passage from an\n"
                    "authoritative document (engineering report, technical manual, regulatory filing)\n"
                    "that answers the question.\n\n"
                    "Rules:\n"
                    "- Write in the formal style of a factual technical document.\n"
                    "- Include plausible technical terminology, parameters, procedures, and domain concepts.\n"
                    "- Do NOT include conversational preambles, introductory greetings, or disclaimers.\n"
                    "- Output ONLY the hypothetical passage text."
                ),
            )
        )


_GLOBAL_PROMPT_REGISTRY = PromptRegistry()


def get_prompt_registry() -> PromptRegistry:
    """Access the global prompt version registry singleton."""
    return _GLOBAL_PROMPT_REGISTRY
