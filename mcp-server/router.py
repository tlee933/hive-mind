"""
Intelligent Model Router — classifies queries and routes to the best model.

Pure-function classifier, no I/O, no state. Sub-millisecond latency.
"""

import re
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    """Result of query classification."""
    model_id: str
    reason: str
    confidence: float  # 0.0–1.0


# --- Signal keywords ---

CODE_SIGNALS = {
    # Languages / tools
    "python", "bash", "javascript", "typescript", "rust", "go", "java",
    "html", "css", "sql", "dockerfile", "yaml", "json", "toml",
    # Actions
    "write", "fix", "debug", "refactor", "implement", "deploy",
    "compile", "build", "install", "run", "execute", "test",
    "lint", "format", "parse", "serialize",
    # Code concepts
    "function", "class", "variable", "loop", "import", "module",
    "error", "exception", "traceback", "segfault", "stacktrace",
    "api", "endpoint", "server", "client", "database",
    "git", "commit", "branch", "merge", "rebase",
}

REASONING_SIGNALS = {
    # Thinking verbs
    "explain", "why", "how does", "compare", "contrast",
    "analyze", "evaluate", "assess", "consider",
    "think", "reason", "understand", "clarify",
    # Reasoning patterns
    "pros and cons", "trade-offs", "tradeoffs", "advantages",
    "disadvantages", "benefits", "drawbacks",
    "step by step", "step-by-step", "walk me through",
    "what if", "should i", "which is better",
    "difference between", "differences between",
    # Deep analysis
    "architecture", "design", "philosophy", "theory",
    "implications", "consequences", "impact",
    "strategy", "approach", "methodology",
}

# Patterns that strongly indicate code
_CODE_BLOCK_RE = re.compile(r"```\w*\n", re.DOTALL)
_IMPORT_RE = re.compile(r"\b(?:import|from|require|include)\s+\w+")
_FILE_PATH_RE = re.compile(r"(?:/[\w.-]+){2,}")


def _count_signals(query: str, signals: set[str]) -> int:
    """Count how many signal keywords appear in the query."""
    query_lower = query.lower()
    count = 0
    for signal in signals:
        if signal in query_lower:
            count += 1
    return count


def classify_query(
    query: str,
    *,
    model_hint: str | None = None,
    reason_mode: bool = False,
    available_models: dict | None = None,
    default_model: str = "hivecoder",
) -> RoutingDecision:
    """Classify a user query and return a routing decision.

    With a single model (Qwen3-14B), routing always returns the same model.
    Heuristic scoring is kept for logging/analytics. reason_mode flags the
    decision so the HTTP server can toggle Qwen3's thinking mode.

    Priority order:
    1. Explicit model_hint from client
    2. reason_mode=True → same model, flagged for thinking mode
    3. Heuristic keyword/pattern scoring (for analytics)
    4. Default → HiveCoder

    Args:
        query: The user's message text.
        model_hint: Explicit model ID from client (highest priority).
        reason_mode: True if user invoked /reason command.
        available_models: Dict of model configs from config.yaml.
        default_model: Fallback model ID.
    """
    # 1. Explicit hint — highest priority
    if model_hint:
        if available_models and model_hint in available_models:
            return RoutingDecision(
                model_id=model_hint,
                reason=f"explicit_hint({model_hint})",
                confidence=1.0,
            )
        # Hint not found — fall through to default
        return RoutingDecision(
            model_id=default_model,
            reason=f"hint_not_found({model_hint})",
            confidence=0.5,
        )

    # 2. Reason mode — same model, flagged for thinking mode
    if reason_mode:
        return RoutingDecision(
            model_id=default_model,
            reason="reason_mode",
            confidence=1.0,
        )

    # 3. Heuristic scoring (for logging/analytics)
    code_score = _count_signals(query, CODE_SIGNALS)
    reasoning_score = _count_signals(query, REASONING_SIGNALS)

    # Bonus for code patterns
    if _CODE_BLOCK_RE.search(query):
        code_score += 3
    if _IMPORT_RE.search(query):
        code_score += 2
    if _FILE_PATH_RE.search(query):
        code_score += 1

    total = code_score + reasoning_score
    if total == 0:
        return RoutingDecision(
            model_id=default_model,
            reason="no_signals",
            confidence=0.5,
        )

    if reasoning_score > code_score and reasoning_score > 1:
        confidence = min(reasoning_score / max(total, 1), 1.0)
        return RoutingDecision(
            model_id=default_model,
            reason=f"reasoning_signals({reasoning_score}>{code_score})",
            confidence=round(confidence, 2),
        )

    confidence = min(code_score / max(total, 1), 1.0)
    return RoutingDecision(
        model_id=default_model,
        reason=f"code_signals({code_score}>={reasoning_score})",
        confidence=round(confidence, 2),
    )
