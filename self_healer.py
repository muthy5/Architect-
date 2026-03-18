"""
self_healer.py -- Self-healing error recovery with learning from previous runs.

Tracks operation outcomes, learns failure patterns, and auto-applies recovery
strategies that worked in the past. Persists run history to a local JSON file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HISTORY_FILE = Path(os.environ.get(
    "ARCHITECT_HEAL_LOG",
    Path.home() / ".architect_run_history.json",
))
MAX_HISTORY_ENTRIES = 500          # rolling window
MAX_RETRIES_DEFAULT = 3
BACKOFF_BASE = 2                   # seconds

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Single operation attempt record."""

    timestamp: str
    operation: str               # e.g. "llm_call", "json_parse", "file_read"
    context: dict[str, Any]      # provider, model, filename, etc.
    success: bool
    error_type: str | None = None
    error_message: str | None = None
    error_fingerprint: str | None = None
    recovery_strategy: str | None = None
    recovery_succeeded: bool = False
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RecoveryStrategy:
    """A learned recovery strategy for a specific error pattern."""

    error_fingerprint: str
    strategy_name: str
    params: dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    fail_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RecoveryStrategy:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Error fingerprinting
# ---------------------------------------------------------------------------

def _fingerprint(operation: str, error_type: str, error_msg: str) -> str:
    """Create a stable fingerprint for an error pattern.

    Groups similar errors together (e.g. all rate-limit errors, all JSON parse
    errors) so recovery strategies can be reused.
    """
    # Normalize the message -- strip variable parts like timestamps, IDs
    msg_lower = error_msg.lower()

    # Classify into known buckets
    if "rate" in msg_lower and "limit" in msg_lower:
        bucket = "rate_limit"
    elif "timeout" in msg_lower or "timed out" in msg_lower:
        bucket = "timeout"
    elif "connection" in msg_lower or "network" in msg_lower:
        bucket = "network"
    elif "auth" in msg_lower or "api key" in msg_lower or "unauthorized" in msg_lower:
        bucket = "auth"
    elif "json" in msg_lower or "decode" in msg_lower or "parse" in msg_lower:
        bucket = "json_parse"
    elif "encoding" in msg_lower or "unicode" in msg_lower or "codec" in msg_lower:
        bucket = "encoding"
    elif "pdf" in msg_lower:
        bucket = "pdf"
    elif "token" in msg_lower and ("limit" in msg_lower or "max" in msg_lower):
        bucket = "token_limit"
    elif "overloaded" in msg_lower or "529" in msg_lower or "503" in msg_lower:
        bucket = "server_overloaded"
    elif "400" in msg_lower or "bad request" in msg_lower or "invalid_request" in msg_lower:
        bucket = "bad_request"
    elif error_type == "TypeError":
        bucket = "type_error"
    else:
        bucket = "unknown"

    raw = f"{operation}:{error_type}:{bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Built-in recovery strategies
# ---------------------------------------------------------------------------

# Registry of strategy names -> descriptions
STRATEGY_REGISTRY: dict[str, str] = {
    "retry_with_backoff":       "Retry after exponential backoff delay",
    "retry_with_longer_timeout": "Retry with increased timeout",
    "reduce_input_size":        "Truncate input text to fit token limits",
    "switch_model":             "Fall back to a different model",
    "add_json_instruction":     "Retry with stronger JSON formatting instructions",
    "force_json_cleanup":       "Aggressively clean and re-parse JSON response",
    "try_alternate_encoding":   "Try a different file encoding",
    "fallback_pdf_to_markdown": "Fall back from PDF to Markdown export",
    "chunk_input":              "Split large input into smaller chunks",
    "skip_and_continue":        "Skip the failing item and continue with others",
    "coerce_types":             "Coerce field values to expected types before construction",
}

# Default strategy mapping: fingerprint bucket -> ordered strategies to try
_DEFAULT_STRATEGIES: dict[str, list[str]] = {
    "rate_limit":       ["retry_with_backoff"],
    "timeout":          ["retry_with_longer_timeout", "reduce_input_size"],
    "network":          ["retry_with_backoff"],
    "auth":             [],  # Can't auto-fix auth; surface to user
    "json_parse":       ["force_json_cleanup", "add_json_instruction"],
    "encoding":         ["try_alternate_encoding"],
    "pdf":              ["fallback_pdf_to_markdown"],
    "token_limit":      ["reduce_input_size", "chunk_input"],
    "server_overloaded": ["retry_with_backoff", "switch_model"],
    "bad_request":      ["reduce_input_size", "switch_model"],
    "type_error":       ["coerce_types", "skip_and_continue"],
    "unknown":          ["retry_with_backoff", "skip_and_continue"],
}


def _bucket_from_fingerprint(operation: str, error_type: str, error_msg: str) -> str:
    """Extract the error bucket without hashing (for strategy lookup)."""
    msg_lower = error_msg.lower()
    if "rate" in msg_lower and "limit" in msg_lower:
        return "rate_limit"
    if "timeout" in msg_lower or "timed out" in msg_lower:
        return "timeout"
    if "connection" in msg_lower or "network" in msg_lower:
        return "network"
    if "auth" in msg_lower or "api key" in msg_lower or "unauthorized" in msg_lower:
        return "auth"
    if "json" in msg_lower or "decode" in msg_lower or "parse" in msg_lower:
        return "json_parse"
    if "encoding" in msg_lower or "unicode" in msg_lower or "codec" in msg_lower:
        return "encoding"
    if "pdf" in msg_lower:
        return "pdf"
    if "token" in msg_lower and ("limit" in msg_lower or "max" in msg_lower):
        return "token_limit"
    if "overloaded" in msg_lower or "529" in msg_lower or "503" in msg_lower:
        return "server_overloaded"
    if "400" in msg_lower or "bad request" in msg_lower or "invalid_request" in msg_lower:
        return "bad_request"
    if error_type == "TypeError":
        return "type_error"
    return "unknown"


# ---------------------------------------------------------------------------
# SelfHealer — main class
# ---------------------------------------------------------------------------

class SelfHealer:
    """Tracks run history, learns from failures, and applies recovery strategies."""

    def __init__(self, history_file: Path | str | None = None):
        self.history_file = Path(history_file) if history_file else HISTORY_FILE
        self.history: list[RunRecord] = []
        self.strategies: dict[str, RecoveryStrategy] = {}  # fingerprint -> best strategy
        self._load_history()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_history(self) -> None:
        """Load run history and learned strategies from disk."""
        if not self.history_file.exists():
            return
        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)
            self.history = [RunRecord.from_dict(r) for r in data.get("history", [])]
            self.strategies = {
                k: RecoveryStrategy.from_dict(v)
                for k, v in data.get("strategies", {}).items()
            }
        except Exception as e:
            log.warning(
                "Corrupted heal history file '%s', resetting learned "
                "strategies: %s", self.history_file, e,
            )
            self.history = []
            self.strategies = {}

    def _save_history(self) -> None:
        """Persist run history and strategies to disk."""
        # Trim to rolling window
        if len(self.history) > MAX_HISTORY_ENTRIES:
            self.history = self.history[-MAX_HISTORY_ENTRIES:]
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "history": [r.to_dict() for r in self.history],
                "strategies": {k: v.to_dict() for k, v in self.strategies.items()},
            }
            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            log.warning(
                "Failed to write heal history to '%s': %s",
                self.history_file, e,
            )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_success(
        self,
        operation: str,
        context: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
        recovery_strategy: str | None = None,
    ) -> None:
        """Record a successful operation."""
        record = RunRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            context=context or {},
            success=True,
            duration_ms=duration_ms,
            recovery_strategy=recovery_strategy,
            recovery_succeeded=bool(recovery_strategy),
        )
        self.history.append(record)

        # Update strategy success count
        if recovery_strategy:
            for strat in self.strategies.values():
                if strat.strategy_name == recovery_strategy:
                    strat.success_count += 1

        self._save_history()

    def record_failure(
        self,
        operation: str,
        error: Exception,
        context: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
        recovery_strategy: str | None = None,
    ) -> str:
        """Record a failed operation. Returns the error fingerprint."""
        error_type = type(error).__name__
        error_msg = str(error)
        fp = _fingerprint(operation, error_type, error_msg)

        record = RunRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            context=context or {},
            success=False,
            error_type=error_type,
            error_message=error_msg[:500],
            error_fingerprint=fp,
            duration_ms=duration_ms,
            recovery_strategy=recovery_strategy,
            recovery_succeeded=False,
        )
        self.history.append(record)

        # Update strategy fail count
        if recovery_strategy:
            for strat in self.strategies.values():
                if strat.strategy_name == recovery_strategy:
                    strat.fail_count += 1

        self._save_history()
        return fp

    # ------------------------------------------------------------------
    # Learning & Strategy Selection
    # ------------------------------------------------------------------

    def get_recovery_strategies(
        self,
        operation: str,
        error: Exception,
    ) -> list[str]:
        """Get ordered list of recovery strategies for this error.

        Prioritizes strategies that worked before for similar errors, then
        falls back to default strategies for the error bucket.
        """
        error_type = type(error).__name__
        error_msg = str(error)
        fp = _fingerprint(operation, error_type, error_msg)
        bucket = _bucket_from_fingerprint(operation, error_type, error_msg)

        strategies: list[str] = []

        # 1. Check if we have a learned strategy for this exact fingerprint
        if fp in self.strategies:
            learned = self.strategies[fp]
            if learned.success_rate > 0.3:  # Only use if >30% success rate
                strategies.append(learned.strategy_name)

        # 2. Check history for strategies that worked for similar operations
        similar_successes: dict[str, int] = defaultdict(int)
        for record in reversed(self.history[-100:]):
            if (
                record.operation == operation
                and record.recovery_succeeded
                and record.recovery_strategy
            ):
                similar_successes[record.recovery_strategy] += 1

        # Sort by frequency and add
        for strat_name, _ in sorted(
            similar_successes.items(), key=lambda x: x[1], reverse=True
        ):
            if strat_name not in strategies:
                strategies.append(strat_name)

        # 3. Fall back to defaults for this bucket
        for strat_name in _DEFAULT_STRATEGIES.get(bucket, []):
            if strat_name not in strategies:
                strategies.append(strat_name)

        return strategies

    def learn_strategy(
        self,
        operation: str,
        error: Exception,
        strategy_name: str,
        succeeded: bool,
    ) -> None:
        """Record whether a recovery strategy worked, updating learned weights."""
        error_type = type(error).__name__
        error_msg = str(error)
        fp = _fingerprint(operation, error_type, error_msg)

        if fp not in self.strategies:
            self.strategies[fp] = RecoveryStrategy(
                error_fingerprint=fp,
                strategy_name=strategy_name,
            )

        strat = self.strategies[fp]
        if succeeded:
            strat.success_count += 1
            strat.strategy_name = strategy_name  # Prefer the one that just worked
        else:
            strat.fail_count += 1

        self._save_history()

    # ------------------------------------------------------------------
    # Auto-healing execution
    # ------------------------------------------------------------------

    def execute_with_healing(
        self,
        operation: str,
        func: Callable[..., Any],
        *args: Any,
        context: dict[str, Any] | None = None,
        max_retries: int = MAX_RETRIES_DEFAULT,
        recovery_handlers: dict[str, Callable] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with automatic error recovery.

        On failure, consults the learned strategy history, applies the best
        recovery action, and retries. Each outcome is recorded so future runs
        improve.

        Args:
            operation: Name of the operation (e.g. "llm_call").
            func: The callable to execute.
            context: Metadata for logging (provider, model, etc.).
            max_retries: Maximum retry attempts.
            recovery_handlers: Map of strategy_name -> callable that modifies
                args/kwargs before retry. Signature: (args, kwargs, error) -> (args, kwargs).
        """
        context = context or {}
        recovery_handlers = recovery_handlers or {}
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            try:
                result = func(*args, **kwargs)
                duration = (time.monotonic() - t0) * 1000
                strategy = context.get("_current_strategy")
                self.record_success(
                    operation, context, duration, recovery_strategy=strategy
                )
                if strategy and last_error:
                    self.learn_strategy(operation, last_error, strategy, succeeded=True)
                return result

            except Exception as e:
                duration = (time.monotonic() - t0) * 1000
                strategy_used = context.get("_current_strategy")
                fp = self.record_failure(
                    operation, e, context, duration,
                    recovery_strategy=strategy_used,
                )

                if strategy_used and last_error:
                    self.learn_strategy(
                        operation, last_error, strategy_used, succeeded=False
                    )

                last_error = e

                if attempt >= max_retries:
                    raise

                # Get recovery strategies
                strategies = self.get_recovery_strategies(operation, e)

                applied = False
                for strat_name in strategies:
                    if strat_name in recovery_handlers:
                        try:
                            args, kwargs = recovery_handlers[strat_name](
                                args, kwargs, e
                            )
                            context["_current_strategy"] = strat_name
                            applied = True
                            break
                        except Exception as handler_err:
                            log.warning(
                                "Recovery handler '%s' failed for operation "
                                "'%s': %s", strat_name, operation, handler_err,
                            )
                            continue

                    elif strat_name == "retry_with_backoff":
                        delay = BACKOFF_BASE ** (attempt + 1)
                        time.sleep(delay)
                        context["_current_strategy"] = strat_name
                        applied = True
                        break

                if not applied:
                    # Default: simple backoff retry
                    delay = BACKOFF_BASE ** (attempt + 1)
                    time.sleep(delay)
                    context["_current_strategy"] = "retry_with_backoff"

    # ------------------------------------------------------------------
    # Stats & diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics from run history."""
        if not self.history:
            return {
                "total_runs": 0,
                "success_rate": 0.0,
                "total_recoveries": 0,
                "top_errors": [],
                "top_strategies": [],
            }

        total = len(self.history)
        successes = sum(1 for r in self.history if r.success)
        recoveries = sum(1 for r in self.history if r.recovery_succeeded)

        # Top error types
        error_counts: dict[str, int] = defaultdict(int)
        for r in self.history:
            if not r.success and r.error_type:
                error_counts[r.error_type] += 1
        top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Top strategies
        strategy_stats = []
        for strat in self.strategies.values():
            strategy_stats.append({
                "name": strat.strategy_name,
                "success_rate": f"{strat.success_rate:.0%}",
                "uses": strat.success_count + strat.fail_count,
            })
        strategy_stats.sort(key=lambda x: x["uses"], reverse=True)

        return {
            "total_runs": total,
            "success_rate": f"{successes / total:.1%}" if total else "0%",
            "total_recoveries": recoveries,
            "top_errors": top_errors,
            "top_strategies": strategy_stats[:5],
        }

    def get_recent_failures(self, limit: int = 10) -> list[RunRecord]:
        """Get most recent failures for diagnostics."""
        return [r for r in reversed(self.history) if not r.success][:limit]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_healer: SelfHealer | None = None


def get_healer() -> SelfHealer:
    """Get or create the module-level SelfHealer singleton."""
    global _healer
    if _healer is None:
        _healer = SelfHealer()
    return _healer
