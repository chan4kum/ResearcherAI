from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# 1. HTTP Layer Metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total count of HTTP requests processed by the platform.",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Latency distribution of HTTP requests in seconds.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

APPLICATION_ERRORS_TOTAL = Counter(
    "application_errors_total",
    "Total count of unhandled or platform errors.",
    ["error_code", "component"],
)

# 2. Agent Execution Metrics
AGENT_EXECUTIONS_TOTAL = Counter(
    "agent_executions_total",
    "Total count of agent task executions.",
    ["status", "model", "provider"],
)

AGENT_EXECUTION_DURATION_SECONDS = Histogram(
    "agent_execution_duration_seconds",
    "Latency distribution of end-to-end agent task executions in seconds.",
    ["status"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

AGENT_TOOL_CALLS_TOTAL = Counter(
    "agent_tool_calls_total",
    "Total count of tool invocations performed by agents.",
    ["tool_name", "status"],
)

AGENT_TOOL_DURATION_SECONDS = Histogram(
    "agent_tool_duration_seconds",
    "Latency distribution of individual tool invocations in seconds.",
    ["tool_name"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

# 3. Retrieval & Search Metrics
RETRIEVAL_EXECUTIONS_TOTAL = Counter(
    "retrieval_executions_total",
    "Total count of RAG retrieval executions.",
    ["strategy", "status"],
)

RETRIEVAL_DURATION_SECONDS = Histogram(
    "retrieval_duration_seconds",
    "Latency distribution of retrieval and reranking pipelines in seconds.",
    ["strategy"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

RETRIEVAL_ITERATIONS_TOTAL = Counter(
    "retrieval_iterations_total",
    "Total count of retrieval rewrite / refinement loop iterations.",
    ["strategy", "iteration"],
)

# 4. LLM & Cost Metrics
LLM_CALLS_TOTAL = Counter(
    "llm_calls_total",
    "Total count of upstream LLM invocations.",
    ["model", "provider", "status"],
)

LLM_CALL_DURATION_SECONDS = Histogram(
    "llm_call_duration_seconds",
    "Latency distribution of upstream LLM invocations in seconds.",
    ["model", "provider"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total count of tokens consumed by LLM operations.",
    ["model", "provider", "token_type"],  # token_type: prompt, completion, total
)

LLM_ESTIMATED_COST_DOLLARS_TOTAL = Counter(
    "llm_estimated_cost_dollars_total",
    "Total estimated cost in USD for LLM token usage.",
    ["model", "provider"],
)

# Token Pricing Table (USD per 1 Million Tokens)
MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
    "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
    "text-embedding-3-small": {"prompt": 0.02, "completion": 0.0},
    "mock": {"prompt": 0.05, "completion": 0.10},
}


def calculate_estimated_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost in USD based on model pricing."""
    pricing = MODEL_PRICING_PER_1M.get(
        model,
        MODEL_PRICING_PER_1M.get("mock", {"prompt": 0.05, "completion": 0.10}),
    )
    prompt_cost = (prompt_tokens / 1_000_000.0) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000.0) * pricing["completion"]
    return round(prompt_cost + completion_cost, 8)


def record_llm_metrics(
    model: str,
    provider: str,
    status: str,
    duration_sec: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Record LLM call, latency, token consumption, and estimated cost."""
    clean_model = model or "unknown"
    clean_provider = provider or "unknown"

    LLM_CALLS_TOTAL.labels(model=clean_model, provider=clean_provider, status=status).inc()
    LLM_CALL_DURATION_SECONDS.labels(model=clean_model, provider=clean_provider).observe(
        duration_sec
    )

    total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens > 0:
        LLM_TOKENS_TOTAL.labels(
            model=clean_model, provider=clean_provider, token_type="prompt"
        ).inc(prompt_tokens)
    if completion_tokens > 0:
        LLM_TOKENS_TOTAL.labels(
            model=clean_model, provider=clean_provider, token_type="completion"
        ).inc(completion_tokens)
    if total_tokens > 0:
        LLM_TOKENS_TOTAL.labels(
            model=clean_model, provider=clean_provider, token_type="total"
        ).inc(total_tokens)

    cost = calculate_estimated_cost(clean_model, prompt_tokens, completion_tokens)
    if cost > 0:
        LLM_ESTIMATED_COST_DOLLARS_TOTAL.labels(
            model=clean_model, provider=clean_provider
        ).inc(cost)


def get_prometheus_metrics() -> tuple[bytes, str]:
    """Generate latest Prometheus metrics formatted text and content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
