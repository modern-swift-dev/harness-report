#!/usr/bin/env python3
"""Build an offline coding-agent usage report. Python 3.10+; no dependencies.

    python3 harness_metrics.py [directory] --output report.html

Supports Codex, Claude Code, Copilot CLI, OpenCode, and T3 Code local storage.
With no directory, discovers installed harnesses; use --harness to select them.
Codex discovery includes ~/.codex/sessions and ~/.codex/archived_sessions.
An explicit directory reads only that Codex archive plus supplied source paths.
Calendar windows use --timezone (Toronto by default, or UTC without timezone data).
Pricing uses embedded rates and the bundled openrouter_prices.json by default.
Use --live-prices to fetch OpenRouter rates. Reports embed prices and work offline.
Local storage is read only; costs are API estimates, not subscription bills.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import tempfile
from typing import Any, Callable, Iterable, Iterator
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


TIMEZONE: tzinfo
try:
    TIMEZONE = ZoneInfo("America/Toronto")
except ZoneInfoNotFoundError:
    TIMEZONE = timezone.utc


def report_timezone(value: str) -> tzinfo:
    if value == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise argparse.ArgumentTypeError(f"unknown or unavailable timezone: {value}; UTC is always available") from error


PRICING_DATE = "2026-09-30"
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
MILLION = Decimal(1_000_000)
FAST_COST_MULTIPLIER = Decimal("1.5")
PRICE_PROXIES = {"gpt-5.3-codex-spark": "gpt-5.4-mini", "codex-auto-review": "gpt-5.6-luna"}
ANTHROPIC_PRICING_DATE = "2026-10-01"
ANTHROPIC_PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
OPENROUTER_SOURCE = "https://openrouter.ai/api/v1/models"
BUNDLED_OPENROUTER_PRICES = Path(__file__).resolve().with_name("openrouter_prices.json")


class Harness(str, Enum):
    CODEX = "codex"
    CLAUDE = "claude"
    COPILOT = "copilot"
    OPENCODE = "opencode"
    T3 = "t3"


class Category(str, Enum):
    INPUT = "Uncached input"
    CACHE_READ = "Cache read"
    OUTPUT = "Non-reasoning output"
    REASONING = "Reasoning"
    CACHE_WRITE = "Cache write"


class SpeedMode(str, Enum):
    NORMAL = "Normal"
    FAST = "Fast"
    OTHER = "Other tier"
    MIXED = "Mixed modes (timing)"


class ModelTier(str, Enum):
    BUDGET = "Budget"
    MEDIUM = "Medium"
    HIGH = "High"
    UNCLASSIFIED = "Unclassified"
    MIXED = "Mixed tiers (timing)"


def model_tier(model: str | None) -> ModelTier:
    name = pricing_model(model)
    if name.startswith(("anthropic/", "openai/")):
        name = pricing_model(name.split("/", 1)[1])
    if name.startswith("claude-"):
        return (ModelTier.BUDGET if "haiku" in name else ModelTier.MEDIUM if "sonnet" in name
                else ModelTier.HIGH if any(family in name for family in ("opus", "fable", "mythos"))
                else ModelTier.UNCLASSIFIED)
    if name.startswith("gpt-") and name.endswith(("-mini", "-nano")):
        return ModelTier.BUDGET
    if name in ("gpt-5.4", "gpt-5.5"):
        return ModelTier.MEDIUM
    if name.startswith("gpt-"):
        return {"luna": ModelTier.BUDGET, "terra": ModelTier.BUDGET,
                "sol": ModelTier.MEDIUM, "astra": ModelTier.HIGH}.get(
                    name.rsplit("-", 1)[-1], ModelTier.UNCLASSIFIED)
    return ModelTier.UNCLASSIFIED


def recorded_mode(settings: dict[str, Any], fallback: SpeedMode = SpeedMode.NORMAL) -> SpeedMode:
    """Missing fields preserve known settings; otherwise unknown is assumed Normal."""
    if "service_tier" not in settings:
        return fallback
    tier = settings["service_tier"]
    if tier in ("priority", "fast"):
        return SpeedMode.FAST
    if tier == "default":
        return SpeedMode.NORMAL
    if not isinstance(tier, str) or tier in ("", "auto"):
        return SpeedMode.NORMAL
    return SpeedMode.OTHER


def report_model(model: str | None, mode: SpeedMode, long_context: bool = False) -> str:
    name = model or "Unknown model"
    if name in {"Mixed models (timing)", "Mixed modes (timing)",
                "Mixed contexts (timing)", "Mixed contexts (tools)"}:
        return name
    if mode == SpeedMode.FAST:
        name += "-fast"
    if long_context:
        name += "-long"
    return name


@dataclass(frozen=True)
class Rates:
    input: Decimal
    cached: Decimal | None
    output: Decimal
    write: Decimal | None = None
    write_hour: Decimal | None = None


@dataclass(frozen=True)
class Price:
    short: Rates
    long: Rates | None = None
    threshold: int = 272_000
    fast_multiplier: Decimal | None = FAST_COST_MULTIPLIER
    context_rates: tuple[tuple[int, Rates], ...] = ()

    def is_long_context(self, input_tokens: int) -> bool:
        return (self.long is not None and input_tokens > self.threshold or
                any(input_tokens >= minimum for minimum, _ in self.context_rates))


def rates(input: str, cached: str | None, output: str, write: str | None = None) -> Rates:
    return Rates(Decimal(input), Decimal(cached) if cached else None,
                 Decimal(output), Decimal(write) if write else None)


# USD per million tokens. Exact model IDs only. Verified against the Standard
# pricing data at PRICING_SOURCE on PRICING_DATE. Long context: >272K input.
PRICES: dict[str, Price] = {
    "gpt-6-astra": Price(rates("10", "1", "50", "12.5"), rates("20", "2", "75", "25")),
    "gpt-6.1-sol": Price(rates("2", ".1", "10", "2.5"), rates("4", ".2", "15", "5")),
    "gpt-6-sol": Price(rates("2", ".2", "10", "2.5"), rates("4", ".4", "15", "5")),
    "gpt-6-luna": Price(rates(".1", ".01", ".5", ".125"), rates(".2", ".02", ".75", ".25")),
    "gpt-5.6-sol": Price(rates("4", ".4", "20", "5"), rates("8", ".8", "30", "10")),
    "gpt-5.6-terra": Price(rates("2", ".2", "12", "2.5"), rates("4", ".4", "18", "5")),
    "gpt-5.6-luna": Price(rates(".2", ".02", "1.2", ".25"), rates(".4", ".04", "1.8", ".5")),
    "gpt-5.5": Price(rates("5", ".5", "30"), rates("10", "1", "45")),
    "gpt-5.5-pro": Price(rates("30", None, "180"), rates("60", None, "270")),
    "gpt-5.4": Price(rates("2.5", ".25", "15"), rates("5", ".5", "22.5")),
    "gpt-5.4-pro": Price(rates("30", None, "180"), rates("60", None, "270")),
    "gpt-5.4-mini": Price(rates(".75", ".075", "4.5")),
    "gpt-5.4-nano": Price(rates(".2", ".02", "1.25")),
    "gpt-5.3-codex": Price(rates("1.75", ".175", "14")),
    "gpt-5.2": Price(rates("1.75", ".175", "14")),
    "gpt-5.1": Price(rates("1.25", ".125", "10")),
    "gpt-5": Price(rates("1.25", ".125", "10")),
    "gpt-5-mini": Price(rates(".25", ".025", "2")),
    "gpt-5-nano": Price(rates(".05", ".005", ".4")),
}

# Anthropic standard global API rates, verified separately on 2026-10-01.
# 4.6+ has no long-context surcharge. Only published fast-mode rates are used.
for names, base, cached, output, fast in [
    (("claude-fable-5-1", "claude-mythos-5-1"), "10", ".25", "50", None),
    (("claude-fable-5", "claude-mythos-5"), "10", "1", "50", None),
    (("claude-opus-5-5",), "4", ".2", "20", Decimal(2)),
    (("claude-sonnet-5-5", "claude-sonnet-5"), "2", ".2", "10", None),
    (("claude-opus-5", "claude-opus-4-8"), "5", ".5", "25", Decimal(2)),
    (("claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5"), "5", ".5", "25", None),
    (("claude-opus-4-1", "claude-opus-4"), "15", "1.5", "75", None),
    (("claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4"), "3", ".3", "15", None),
    (("claude-haiku-4-5",), "1", ".1", "5", None),
    (("claude-3-5-haiku",), ".8", ".08", "4", None),
]:
    short = Rates(Decimal(base), Decimal(cached), Decimal(output),
                  Decimal(base) * Decimal("1.25"), Decimal(base) * 2)
    for name in names:
        long = (Rates(Decimal(base) * 2, Decimal(cached) * 2, Decimal(output) * Decimal("1.5"),
                      Decimal(base) * Decimal("2.5"), Decimal(base) * 4)
                if name in {"claude-sonnet-4-5", "claude-sonnet-4"} else None)
        PRICES[name] = Price(short, long, 200_000, fast)


def pricing_model(model: str | None) -> str:
    name = PRICE_PROXIES.get(model or "", model or "")
    if name.startswith(("Claude ", "GPT-")):
        name = re.sub(r"\s*\([^)]*\)$", "", name).lower().replace(" ", "-")
    # Claude API snapshot IDs and Copilot's dotted version aliases.
    if name.startswith("claude-"):
        name = name.replace(".", "-")
        base, _, date = name.rpartition("-")
        if len(date) == 8 and date.isdigit():
            name = base
    return name


def decimal_amount(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        amount = Decimal(str(value))
        return amount if amount.is_finite() and amount >= 0 else None
    except InvalidOperation:
        return None


def openrouter_prices(value: Any) -> dict[str, Price]:
    """The public model catalog quotes USD per token, not per million."""
    result: dict[str, Price] = {}
    rows = value.get("data", []) if isinstance(value, dict) else []
    if not isinstance(rows, list):
        raise ValueError("OpenRouter catalog must contain a data array")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        pricing = row.get("pricing")
        if not isinstance(pricing, dict):
            continue
        prompt, completion = (decimal_amount(pricing.get(k)) for k in ("prompt", "completion"))
        if prompt is None or completion is None:
            continue
        def catalog_rates(table: dict[str, Any]) -> Rates:
            input_rate, output_rate = (decimal_amount(table.get(k)) for k in ("prompt", "completion"))
            if input_rate is None or output_rate is None:
                raise ValueError("invalid OpenRouter input/output rates")
            cached, write, hour = (decimal_amount(table.get(k)) for k in
                                   ("input_cache_read", "input_cache_write", "input_cache_write_1h"))
            return Rates(input_rate * MILLION, cached * MILLION if cached is not None else None,
                         output_rate * MILLION, write * MILLION if write is not None else None,
                         hour * MILLION if hour is not None else None)
        overrides = pricing.get("overrides", [])
        tiers: list[tuple[int, Rates]] = []
        if isinstance(overrides, list):
            for override in overrides:
                if not isinstance(override, dict) or type(override.get("min_prompt_tokens")) is not int:
                    continue
                merged = {**pricing, **override}
                if decimal_amount(merged.get("prompt")) is not None and decimal_amount(merged.get("completion")) is not None:
                    tiers.append((override["min_prompt_tokens"], catalog_rates(merged)))
        result[row["id"]] = Price(catalog_rates(pricing), fast_multiplier=None,
                                   context_rates=tuple(sorted(tiers, key=lambda item: item[0])))
    if not result:
        raise ValueError("OpenRouter catalog contains no valid model prices")
    return result


def router_model(model: str | None, prices: dict[str, Price]) -> str | None:
    """Exact IDs first; translate known model aliases without fuzzy matching."""
    if not model:
        return None
    if model in prices:
        return model
    name = re.sub(r"\s*\([^)]*\)$", "", model).strip().lower().replace(" ", "-")
    name = pricing_model(name)
    vendor = next((vendor for prefix, vendor in [("claude-", "anthropic"), ("gpt-", "openai"),
                  ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
                  ("gemini-", "google"), ("deepseek-", "deepseek"), ("glm-", "z-ai"),
                  ("kimi-", "moonshotai"), ("minimax-", "minimax"), ("qwen-", "qwen")]
                   if name.startswith(prefix)), None)
    if vendor == "minimax":
        name = name.replace("minimax-", "minimax-m", 1)
    elif vendor == "qwen":
        name = name.replace("qwen-", "qwen", 1)
    candidate = f"{vendor}/{name}" if vendor else name
    if candidate in prices:
        return candidate
    matches = [identifier for identifier in prices
               if identifier.partition("/")[0] == vendor and pricing_model(identifier.partition("/")[2]) == name]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class Usage:
    input: int
    output: int
    cached: int = 0
    reasoning: int = 0
    write: int = 0
    write_hour: int = 0  # Subset of write, not an additional token count.

    @property
    def total(self) -> int:
        return self.input + self.output

    def categories(self) -> dict[Category, int]:
        return {Category.INPUT: self.input - self.cached - self.write,
                Category.CACHE_READ: self.cached,
                Category.OUTPUT: self.output - self.reasoning,
                Category.REASONING: self.reasoning,
                Category.CACHE_WRITE: self.write}


def parse_usage(value: Any) -> Usage | None:
    if not isinstance(value, dict):
        return None
    names = ("input_tokens", "output_tokens", "cached_input_tokens",
             "reasoning_output_tokens", "cache_write_input_tokens")
    values = [value.get(name, 0) for name in names]
    if any(type(v) is not int or v < 0 for v in values):
        return None
    if "input_tokens" not in value or "output_tokens" not in value:
        return None
    usage = Usage(*values)
    if usage.cached + usage.write > usage.input or usage.reasoning > usage.output:
        return None
    return usage


def usage_delta(current: Usage, previous: Usage) -> tuple[Usage, bool]:
    if current.input < previous.input or current.output < previous.output:
        return current, True
    # Cache/reasoning counters can decrease independently on compaction.
    return Usage(current.input - previous.input, current.output - previous.output,
                 max(0, current.cached - previous.cached),
                 max(0, current.reasoning - previous.reasoning),
                 max(0, current.write - previous.write)), False


def model_price(model: str | None, catalog: dict[str, Price] | None = None) -> Price | None:
    price = PRICES.get(pricing_model(model))
    router_id = router_model(model, catalog or {}) if catalog and price is None else None
    if router_id:
        price = (catalog or {})[router_id]
    return price


def is_long_context(model: str | None, usage: Usage,
                    catalog: dict[str, Price] | None = None, aggregate: bool = False) -> bool:
    price = model_price(model, catalog)
    return not aggregate and price is not None and price.is_long_context(usage.input)


def price_usage(model: str | None, usage: Usage, mode: SpeedMode = SpeedMode.NORMAL,
                catalog: dict[str, Price] | None = None,
                aggregate: bool = False) -> tuple[dict[Category, Decimal], dict[Category, int]]:
    counts = usage.categories()
    price = model_price(model, catalog)
    # Unknown per-request context in session aggregates uses normal-context rates.
    multiplier = (price.fast_multiplier if price else None) if mode == SpeedMode.FAST else Decimal(1)
    selected = price.short if price else None
    if price and not aggregate:
        if price.long and usage.input > price.threshold:
            selected = price.long
        for minimum, tier in price.context_rates:
            if usage.input >= minimum:
                selected = tier
    costs: dict[Category, Decimal] = {}
    unpriced: dict[Category, int] = {}
    for category, count in counts.items():
        rate = None
        if selected:
            rate = {Category.INPUT: selected.input, Category.CACHE_READ: selected.cached,
                    Category.OUTPUT: selected.output, Category.REASONING: selected.output,
                    Category.CACHE_WRITE: selected.write}[category]
        if (rate is None or multiplier is None or
                category == Category.CACHE_WRITE and usage.write_hour and selected and selected.write_hour is None) and count:
            unpriced[category] = count
        else:
            cost = Decimal(count) * (rate or Decimal(0))
            if category == Category.CACHE_WRITE and usage.write_hour and selected and selected.write_hour is not None:
                cost += Decimal(usage.write_hour) * (selected.write_hour - (rate or Decimal(0)))
            costs[category] = cost * (multiplier or Decimal(0)) / MILLION
    return costs, unpriced


def timestamp(value: Any) -> datetime | None:
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(value, timezone.utc)
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError, OverflowError, OSError):
        pass
    return None


def milliseconds(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return value / 1000
    return None


@dataclass
class Quality:
    counts: Counter[str] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def warn(self, kind: str, path: Path, line: int, message: str) -> None:
        self.counts[kind] += 1
        if len(self.warnings) < 20:
            warning = f"{path}:{line}: {message}"
            self.warnings.append(warning)
            print(warning, file=sys.stderr)


@dataclass
class UsageEvent:
    at: datetime
    usage: Usage
    model: str | None
    key: str
    mode: SpeedMode = SpeedMode.NORMAL
    aggregate: bool = False


@dataclass(frozen=True)
class BillingEvent:
    at: datetime
    unit: str
    amount: Decimal


@dataclass
class Turn:
    id: str
    start: datetime | None = None
    end: datetime | None = None
    duration: float | None = None
    model: str | None = None
    mode: SpeedMode = SpeedMode.NORMAL
    completed: bool = False
    aborted: bool = False
    # Timing-only output count for harnesses that persist no per-response usage.
    output_tokens: int | None = None
    modern: list[UsageEvent] = field(default_factory=list)
    legacy: list[UsageEvent] = field(default_factory=list)

    @property
    def usage(self) -> list[UsageEvent]:
        return self.modern if self.modern else self.legacy


@dataclass
class ToolCall:
    at: datetime
    turn_id: str
    model: str | None
    mode: SpeedMode = SpeedMode.NORMAL


@dataclass
class Thread:
    id: str
    turns: dict[str, Turn] = field(default_factory=dict)
    calls: dict[str, ToolCall] = field(default_factory=dict)
    harness: Harness = Harness.CODEX
    billing: list[BillingEvent] = field(default_factory=list)


CALL_TYPES = {"function_call", "custom_tool_call", "web_search_call", "tool_search_call"}


def inherited_baseline(kind: str, payload: dict[str, Any], previous: Usage) -> Usage:
    """Copied parent counters advance the child's baseline without adding usage."""
    if kind != "event_msg" or payload.get("type") != "token_count":
        return previous
    cumulative = parse_usage((payload.get("info") or {}).get("total_token_usage"))
    return previous if cumulative is None else cumulative


def read_thread(thread_id: str, paths: list[Path], quality: Quality) -> Thread:
    """Keep only compact metrics, discarding message text and tool arguments."""
    thread = Thread(thread_id)
    current_turn: str | None = None
    current_model: str | None = None
    current_mode = SpeedMode.NORMAL
    previous = Usage(0, 0)
    seen_usage: set[str] = set()
    seen_legacy: set[str] = set()
    seen_events: set[tuple[str, str, str]] = set()
    for path in paths:
        history_boundary = 0
        inherited = False
        first_meta = True
        forked = False
        first_context: str | None = None
        prefix_turns: set[str] = set()
        call_turns: dict[str, str] = {}
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    quality.warn("Malformed lines", path, index + 1, "invalid JSON; skipped")
                    continue
                if not isinstance(row, dict) or not isinstance(row.get("payload"), dict):
                    quality.warn("Malformed records", path, index + 1, "expected a record with an object payload; skipped")
                    continue
                payload = row["payload"]
                kind = row.get("type")
                event = payload.get("type")
                if (not isinstance(kind, str) or
                        "type" in payload and not isinstance(event, str) or
                        any(payload.get(name) is not None and not isinstance(payload[name], str)
                            for name in ("model", "turn_id", "thread_id", "response_id", "call_id")) or
                        "thread_settings" in payload and not isinstance(payload["thread_settings"], dict) or
                        # Rate-limit-only token_count events record a null info.
                        payload.get("info") is not None and not isinstance(payload["info"], dict) or
                        payload.get("internal_chat_message_metadata_passthrough") is not None and
                        not isinstance(payload["internal_chat_message_metadata_passthrough"], dict)):
                    quality.warn("Malformed records", path, index + 1, "invalid record fields; skipped")
                    continue
                settings = payload.get("thread_settings", {})
                if settings.get("model") is not None and not isinstance(settings["model"], str):
                    quality.warn("Malformed records", path, index + 1, "invalid settings model; skipped")
                    continue
                if kind == "session_meta":
                    if first_meta:
                        boundary = payload.get("subagent_history_start_ordinal", 0)
                        history_boundary = boundary if type(boundary) is int and boundary >= 0 else 0
                        source = payload.get("source")
                        forked = bool(payload.get("forked_from_id") or payload.get("parent_thread_id") or
                                      isinstance(source, dict) and isinstance(source.get("subagent"), dict) and
                                      "thread_spawn" in source["subagent"])
                        first_meta = False
                        current_turn = None
                        current_model = None
                        current_mode = recorded_mode(payload)
                    elif payload.get("id") != thread_id:
                        inherited = True
                    continue
                # Older archives put the final file ordinal in this field even
                # when every event belongs to the child. A copied session_meta
                # record, not the ordinal alone, establishes inherited history.
                if inherited and index < history_boundary:
                    quality.counts["Inherited records excluded"] += 1
                    previous = inherited_baseline(kind, payload, previous)
                    continue
                explicit_thread = payload.get("thread_id")
                if explicit_thread == thread_id:
                    inherited = False
                elif explicit_thread and explicit_thread != thread_id:
                    quality.counts["Inherited records excluded"] += 1
                    previous = inherited_baseline(kind, payload, previous)
                    continue
                if history_boundary and index >= history_boundary:
                    inherited = False
                if inherited:
                    quality.counts["Inherited records excluded"] += 1
                    previous = inherited_baseline(kind, payload, previous)
                    continue
                at = timestamp(row.get("timestamp"))
                if kind not in {"turn_context", "event_msg", "response_item", "token_usage_record"}:
                    continue
                if kind == "event_msg" and event == "thread_settings_applied":
                    settings = payload.get("thread_settings", {})
                    current_model = settings.get("model") or current_model
                    current_mode = recorded_mode(settings, current_mode)
                    continue
                if kind == "response_item" and event not in CALL_TYPES:
                    continue
                if kind == "event_msg" and event not in {"task_started", "task_complete", "turn_aborted", "token_count"}:
                    continue
                if at is None:
                    quality.warn("Invalid timestamps", path, index + 1, "missing or invalid timestamp; skipped")
                    continue
                metadata = payload.get("internal_chat_message_metadata_passthrough") or {}
                turn_id = payload.get("turn_id") or metadata.get("turn_id") or current_turn
                if kind == "event_msg" and event == "task_started":
                    turn_id = payload.get("turn_id") or f"anonymous:{at.isoformat()}"
                    current_turn = turn_id
                elif kind == "turn_context" and payload.get("turn_id"):
                    current_turn = payload["turn_id"]
                    turn_id = current_turn
                if not isinstance(turn_id, str):
                    turn_id = "unattributed"
                turn = thread.turns.setdefault(turn_id, Turn(turn_id))
                if first_context is None:
                    prefix_turns.add(turn_id)
                if kind == "turn_context":
                    if first_context is None:
                        first_context = turn_id
                    current_model = payload.get("model") or current_model
                    current_mode = recorded_mode(payload, current_mode)
                    turn.model = current_model
                    turn.mode = current_mode
                elif kind == "response_item":
                    call_id = payload.get("call_id") or payload.get("id")
                    key = str(call_id) if call_id else hashlib.sha256(line.encode()).hexdigest()
                    if key in thread.calls:
                        quality.counts["Duplicate tool calls excluded"] += 1
                    else:
                        thread.calls[key] = ToolCall(at, turn_id, current_model, recorded_mode(payload, current_mode))
                        call_turns[key] = turn_id
                elif kind == "token_usage_record":
                    raw_usage = payload.get("usage")
                    usage = parse_usage(raw_usage)
                    if usage is None:
                        quality.warn("Invalid usage records", path, index + 1, "invalid token usage; skipped")
                        continue
                    key = payload.get("response_id") or hashlib.sha256(line.encode()).hexdigest()
                    if key in seen_usage:
                        quality.counts["Duplicate usage records excluded"] += 1
                        continue
                    seen_usage.add(key)
                    turn.modern.append(UsageEvent(at, usage, current_model, str(key), recorded_mode(payload, current_mode)))
                elif event == "token_count":
                    info = payload.get("info") or {}
                    cumulative = parse_usage(info.get("total_token_usage"))
                    if cumulative is None:
                        if info.get("total_token_usage") is not None:
                            quality.warn("Invalid usage records", path, index + 1, "invalid cumulative token usage; skipped")
                        continue
                    delta, reset = usage_delta(cumulative, previous)
                    previous = cumulative
                    key = f"{at.isoformat()}:{cumulative}"
                    if key in seen_legacy:
                        continue
                    seen_legacy.add(key)
                    if not delta.total:
                        continue
                    if reset:
                        quality.counts["Token counter resets"] += 1
                    # The per-response fields preserve pricing context even if
                    # cumulative cache/reasoning counters were reset separately.
                    last = parse_usage(info.get("last_token_usage"))
                    if last and last.input == delta.input and last.output == delta.output:
                        delta = last
                    if min(delta.categories().values()) < 0:
                        quality.warn("Invalid usage records", path, index + 1, "inconsistent cumulative token delta; skipped")
                        continue
                    turn.legacy.append(UsageEvent(at, delta, current_model, key, recorded_mode(payload, current_mode)))
                else:
                    key = (turn_id, str(event), at.isoformat())
                    if key in seen_events:
                        continue
                    seen_events.add(key)
                    if event == "task_started":
                        # Record timestamps retain milliseconds; started_at is
                        # commonly a Unix timestamp truncated to whole seconds.
                        turn.start = at
                        turn.model = current_model
                        turn.mode = recorded_mode(payload, current_mode)
                    elif event == "task_complete":
                        turn.end = at
                        turn.completed = True
                        turn.aborted = False
                        turn.duration = milliseconds(payload.get("duration_ms"))
                        if turn.duration is None:
                            start = timestamp(payload.get("started_at")) or turn.start
                            if start and at >= start:
                                turn.duration = (at - start).total_seconds()
                    elif event == "turn_aborted":
                        turn.aborted = True
                        turn.completed = False
                        turn.end = at
        # Some older forks replay parent events with rewritten thread IDs and
        # timestamps, but omit their turn contexts. Only the first actual turn
        # context establishes local activity. Keep the imported cumulative
        # counter as a baseline so the child's increments remain accurate.
        if forked and first_context is not None:
            for turn_id in prefix_turns - {first_context}:
                removed = thread.turns.pop(turn_id, None)
                if removed:
                    quality.counts["Inherited turns excluded"] += 1
                    quality.counts["Inherited usage records excluded"] += len(removed.usage)
                for key, call_turn in call_turns.items():
                    if call_turn == turn_id:
                        thread.calls.pop(key, None)
    return thread


def json_rows(path: Path, quality: Quality) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected object")
            except (ValueError, UnicodeError):
                quality.warn("Malformed lines", path, line_number, "invalid JSON object; skipped")
                continue
            yield line_number, row


def separate_usage(value: Any, *, opencode: bool = False) -> Usage | None:
    """Claude and OpenCode store uncached input separately from cache tokens.

    OpenCode also stores reasoning separately from its ordinary output count.
    """
    if not isinstance(value, dict):
        return None
    cache = value.get("cache", {}) if opencode else value.get("cache_creation", {})
    if not isinstance(cache, dict):
        return None
    names = ("input", "output") if opencode else ("input_tokens", "output_tokens")
    if any(name not in value for name in names):
        return None
    uncached, output = (value[name] for name in names)
    read = cache.get("read", 0) if opencode else value.get("cache_read_input_tokens", 0)
    write = cache.get("write", 0) if opencode else value.get("cache_creation_input_tokens", 0)
    reasoning = value.get("reasoning", 0) if opencode else 0
    hour = 0 if opencode else cache.get("ephemeral_1h_input_tokens", 0)
    if any(type(v) is not int or v < 0 for v in (uncached, output, read, write, reasoning, hour)) or hour > write:
        return None
    return Usage(uncached + read + write, output + reasoning, read, reasoning, write, hour)


def read_claude(thread_id: str, paths: list[Path], quality: Quality) -> Thread:
    thread = Thread(thread_id, harness=Harness.CLAUDE)
    responses: dict[str, tuple[str, UsageEvent]] = {}
    seen_users: set[str] = set()
    current_turn = "unattributed"
    for path in paths:
        for line, row in json_rows(path, quality):
            kind, message = row.get("type"), row.get("message")
            if not isinstance(kind, str) or kind not in {"user", "assistant", "system"}:
                continue
            at = timestamp(row.get("timestamp"))
            if at is None:
                quality.warn("Invalid timestamps", path, line, "missing or invalid timestamp; skipped")
                continue
            if kind == "user" and isinstance(message, dict):
                content = message.get("content")
                tool_result = isinstance(content, list) and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content)
                if tool_result or row.get("isMeta"):
                    continue
                identifier = str(row.get("uuid") or f"user:{at.isoformat()}")
                current_turn = identifier
                if identifier not in seen_users:
                    seen_users.add(identifier)
                    thread.turns.setdefault(identifier, Turn(identifier, start=at))
            elif kind == "assistant" and isinstance(message, dict):
                turn = thread.turns.setdefault(current_turn, Turn(current_turn))
                model = message.get("model") if isinstance(message.get("model"), str) else None
                # Claude Code records local errors as "<synthetic>" messages with no real usage.
                synthetic = model == "<synthetic>"
                if synthetic:
                    model = None
                turn.model = model or turn.model
                raw_usage = message.get("usage")
                mode = SpeedMode.FAST if isinstance(raw_usage, dict) and raw_usage.get("speed") == "fast" else SpeedMode.NORMAL
                content = message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                            thread.calls.setdefault(block["id"], ToolCall(at, current_turn, model, mode))
                usage = separate_usage(message["usage"]) if "usage" in message else None
                if "usage" in message and usage is None:
                    quality.warn("Invalid usage records", path, line, "invalid Claude token usage; skipped")
                elif usage is not None:
                    if not synthetic or usage.total:
                        key = str(row.get("requestId") or message.get("id") or row.get("uuid") or hashlib.sha256(json.dumps(row).encode()).hexdigest())
                        old = responses.get(key)
                        # Streaming records repeat request usage. Keep the fullest
                        # valid response rather than charging once per content block.
                        if old:
                            quality.counts["Duplicate usage records excluded"] += 1
                        if old is None or usage.output >= old[1].usage.output:
                            responses[key] = (current_turn, UsageEvent(at, usage, model, key, mode))
                stop_reason = message.get("stop_reason")
                if isinstance(stop_reason, str) and stop_reason in {"end_turn", "stop_sequence"}:
                    turn.end, turn.completed = at, True
            elif kind == "system" and row.get("subtype") == "turn_duration":
                turn = thread.turns.setdefault(current_turn, Turn(current_turn))
                turn.end, turn.completed = at, True
                turn.duration = milliseconds(row.get("durationMs"))
    for turn_id, response in responses.values():
        thread.turns[turn_id].modern.append(response)
    # Claude Code rarely logs turn_duration; time completed turns from the prompt instead.
    for turn in thread.turns.values():
        if turn.duration is None and turn.completed and turn.start and turn.end and turn.end >= turn.start:
            turn.duration = (turn.end - turn.start).total_seconds()
    return thread


def copilot_usage(value: Any) -> Usage | None:
    if not isinstance(value, dict):
        return None
    return parse_usage({"input_tokens": value.get("inputTokens"),
                        "output_tokens": value.get("outputTokens"),
                        "cached_input_tokens": value.get("cacheReadTokens") if value.get("cacheReadTokens") is not None else 0,
                        "cache_write_input_tokens": value.get("cacheWriteTokens") if value.get("cacheWriteTokens") is not None else 0,
                        "reasoning_output_tokens": value.get("reasoningTokens") if value.get("reasoningTokens") is not None else 0})


def add_billing(thread: Thread, at: datetime, unit: str, value: Any) -> None:
    amount = decimal_amount(value)
    if amount is not None:
        thread.billing.append(BillingEvent(at, unit, amount))


def read_copilot(thread_id: str, paths: list[Path], quality: Quality) -> Thread:
    thread = Thread(thread_id, harness=Harness.COPILOT)
    current_turn, model = "unattributed", None
    # Loop turn IDs restart per interaction and repeat across subagents; map them to the latest start.
    active: dict[tuple[str, str], str] = {}
    message_outputs: dict[tuple[str, str], int] = {}
    seen: set[str] = set()
    seen_responses: set[str] = set()
    summaries: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for path in paths:
        for line, row in json_rows(path, quality):
            kind, data = row.get("type"), row.get("data")
            if not isinstance(data, dict):
                quality.warn("Malformed records", path, line, "expected Copilot object data; skipped")
                continue
            if (not isinstance(kind, str) or
                    any(data.get(name) is not None and not isinstance(data[name], str)
                        for name in ("selectedModel", "newModel", "model"))):
                quality.warn("Malformed records", path, line, "invalid Copilot record fields; skipped")
                continue
            at = timestamp(row.get("timestamp"))
            if at is None:
                quality.warn("Invalid timestamps", path, line, "missing or invalid timestamp; skipped")
                continue
            key = str(row.get("id") or hashlib.sha256(json.dumps(row).encode()).hexdigest())
            if key in seen:
                quality.counts["Duplicate Copilot events excluded"] += 1
                continue
            seen.add(key)
            if kind == "session.start":
                model = data.get("selectedModel") or model
                continue
            if kind == "session.model_change":
                model = data.get("newModel") or model
                continue
            if kind == "session.shutdown":
                metrics = data.get("modelMetrics", {})
                if isinstance(metrics, dict):
                    for name, metric in metrics.items():
                        if isinstance(metric, dict):
                            summaries[name] = (at, metric)
                continue
            scope, raw_turn = str(row.get("agentId") or data.get("parentToolCallId") or ""), data.get("turnId")
            if kind == "assistant.turn_start":
                current_turn = key
                if raw_turn is not None:
                    active[(scope, str(raw_turn))] = key
                model = data.get("model") or model
                turn_id = key
            elif raw_turn is not None:
                turn_id = active.get((scope, str(raw_turn)), str(raw_turn))
            else:
                turn_id = current_turn
            turn = thread.turns.setdefault(turn_id, Turn(turn_id, model=model))
            if kind == "assistant.turn_start":
                turn.start = at
            elif kind == "assistant.turn_end":
                turn.end, turn.completed = at, True
                if turn.start and at >= turn.start:
                    turn.duration = (at - turn.start).total_seconds()
            elif kind in {"abort", "agent.interrupted"}:
                turn.end, turn.aborted, turn.completed = at, True, False
            elif kind == "assistant.message":
                # assistant.usage is ephemeral; each chunk of a persisted message repeats its call's completion tokens.
                output = data.get("outputTokens")
                if isinstance(output, int) and not isinstance(output, bool) and output >= 0:
                    call = (turn_id, str(data.get("apiCallId") or data.get("requestId") or data.get("messageId") or key))
                    previous = message_outputs.get(call, 0)
                    message_outputs[call] = max(previous, output)
                    turn.output_tokens = (turn.output_tokens or 0) + message_outputs[call] - previous
                    turn.model = data.get("model") or turn.model
            elif kind == "tool.execution_start":
                call_id = str(data.get("toolCallId") or key)
                thread.calls.setdefault(call_id, ToolCall(at, turn_id, model))
            elif kind == "assistant.usage":
                usage = copilot_usage(data)
                if usage is None:
                    quality.warn("Invalid usage records", path, line, "invalid Copilot token usage; skipped")
                    continue
                response_id = str(data.get("apiCallId") or key)
                if response_id in seen_responses:
                    quality.counts["Duplicate usage records excluded"] += 1
                    continue
                seen_responses.add(response_id)
                turn.modern.append(UsageEvent(at, usage, data.get("model") or model, response_id))
                # cost is a premium-request multiplier, never USD.
                add_billing(thread, at, "Copilot premium requests", data.get("cost"))
                native = data.get("copilotUsage")
                if isinstance(native, dict):
                    add_billing(thread, at, "Copilot nano-AIU", native.get("totalNanoAiu"))
    detailed = [r for turn in thread.turns.values() for r in turn.usage]
    for name, (at, metric) in summaries.items():
        usage = copilot_usage(metric.get("usage"))
        if usage is None:
            quality.warn("Invalid usage records", paths[0], 0, "invalid Copilot shutdown usage; skipped")
            continue
        matching = [r.usage for r in detailed if pricing_model(r.model) == pricing_model(name)]
        if matching:
            usage = parse_usage({"input_tokens": usage.input - sum(u.input for u in matching),
                                 "output_tokens": usage.output - sum(u.output for u in matching),
                                 "cached_input_tokens": usage.cached - sum(u.cached for u in matching),
                                 "cache_write_input_tokens": usage.write - sum(u.write for u in matching),
                                 "reasoning_output_tokens": usage.reasoning - sum(u.reasoning for u in matching)})
            if usage is None:
                quality.counts["Copilot shutdown totals inconsistent with detailed usage"] += 1
                continue
        if usage.total:
            key = f"summary:{name}"
            turn = thread.turns.setdefault(key, Turn(key, model=name))
            turn.modern.append(UsageEvent(at, usage, name, key, aggregate=True))
            quality.counts["Copilot aggregate snapshots (dated at shutdown)"] += 1
        if not matching:
            add_billing(thread, at, "Copilot nano-AIU", metric.get("totalNanoAiu"))
            requests = metric.get("requests", {})
            if isinstance(requests, dict):
                add_billing(thread, at, "Copilot premium requests", requests.get("cost"))
    return thread


def read_opencode(paths: list[Path], quality: Quality) -> list[Thread]:
    threads: dict[str, Thread] = {}
    seen: set[str] = set()
    for path in paths:
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
                connection.execute("PRAGMA query_only=ON")
                for message_id, session_id, raw in connection.execute("SELECT id, session_id, data FROM message"):
                    if message_id in seen:
                        continue
                    try:
                        data = json.loads(raw)
                    except (ValueError, UnicodeError):
                        quality.warn("Malformed records", path, 0, "invalid OpenCode message JSON; skipped")
                        continue
                    if not isinstance(data, dict) or data.get("role") != "assistant":
                        continue
                    seen.add(message_id)
                    clock = data.get("time", {})
                    if not isinstance(clock, dict):
                        clock = {}
                    start = timestamp(clock["created"] / 1000) if type(clock.get("created")) is int else None
                    end = timestamp(clock["completed"] / 1000) if type(clock.get("completed")) is int else None
                    at = end or start
                    if at is None:
                        quality.warn("Invalid timestamps", path, 0, "invalid OpenCode message time; skipped")
                        continue
                    usage = separate_usage(data.get("tokens"), opencode=True)
                    if usage is None:
                        quality.warn("Invalid usage records", path, 0, "invalid OpenCode token usage; skipped")
                        continue
                    thread = threads.setdefault(session_id, Thread(session_id, harness=Harness.OPENCODE))
                    model = data.get("modelID") if isinstance(data.get("modelID"), str) else None
                    provider = data.get("providerID")
                    # Preserve an OpenRouter ID so its actual catalog can price
                    # models beyond the embedded OpenAI/Anthropic table.
                    if provider == "openrouter" and model:
                        model = model.removeprefix("openrouter/")
                    turn = Turn(message_id, start=start, end=end, model=model, completed=end is not None)
                    if start and end and end >= start:
                        turn.duration = (end - start).total_seconds()
                    turn.modern.append(UsageEvent(at, usage, model, message_id))
                    thread.turns[message_id] = turn
                    add_billing(thread, at, "OpenCode recorded USD", data.get("cost"))
                for part_id, message_id, session_id, raw in connection.execute("SELECT id, message_id, session_id, data FROM part"):
                    thread = threads.get(session_id)
                    if thread is None or message_id not in thread.turns:
                        continue
                    try:
                        part = json.loads(raw)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(part, dict) or part.get("type") != "tool":
                        continue
                    turn = thread.turns[message_id]
                    state = part.get("state", {})
                    clock = state.get("time", {}) if isinstance(state, dict) else {}
                    at = timestamp(clock["start"] / 1000) if isinstance(clock, dict) and type(clock.get("start")) is int else turn.start
                    if at:
                        thread.calls.setdefault(str(part.get("callID") or part_id), ToolCall(at, message_id, turn.model))
        except (OSError, sqlite3.Error) as error:
            quality.warn("Unreadable files", path, 0, str(error))
    return list(threads.values())


class Granularity(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class Window:
    label: str
    start: datetime
    end: datetime
    end_exclusive: bool = False
    granularity: Granularity | None = None
    conversations: set[str] = field(default_factory=set)
    durations: dict[str, float] = field(default_factory=dict)
    call_counts: Counter[str] = field(default_factory=Counter)
    throughput: list[float] = field(default_factory=list)
    tokens: Counter[Category] = field(default_factory=Counter)
    costs: dict[Category, Decimal] = field(default_factory=lambda: defaultdict(Decimal))
    unpriced: Counter[str] = field(default_factory=Counter)
    unpriced_categories: Counter[Category] = field(default_factory=Counter)
    models: Counter[str] = field(default_factory=Counter)
    coverage: Counter[str] = field(default_factory=Counter)
    billing: dict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))
    shared_empty: bool = False

    def contains(self, at: datetime) -> bool:
        return self.start <= at and (at < self.end if self.end_exclusive else at <= self.end)


def make_windows(now: datetime, report_zone: tzinfo = TIMEZONE) -> list[Window]:
    local = now.astimezone(report_zone)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    today = midnight.astimezone(timezone.utc)
    yesterday = (midnight - timedelta(days=1)).astimezone(timezone.utc)
    return [Window("Today", today, now),
            Window("Yesterday", yesterday, today, end_exclusive=True)] + [
        Window(f"Last {days} days", now - timedelta(days=days), now)
        for days in (7, 14, 30, 60, 90, 180, 365)]


def make_trend_windows(now: datetime, report_zone: tzinfo = TIMEZONE,
                       start: datetime | None = None,
                       granularities: Iterable[Granularity] = Granularity) -> list[Window]:
    midnight = (start or now).astimezone(report_zone).replace(hour=0, minute=0, second=0, microsecond=0)
    start = midnight.astimezone(timezone.utc)
    windows: list[Window] = []
    for granularity in granularities:
        cursor = midnight
        if granularity == Granularity.WEEKLY:
            cursor -= timedelta(days=cursor.weekday())
        elif granularity == Granularity.MONTHLY:
            cursor = cursor.replace(day=1)
        while cursor.astimezone(timezone.utc) <= now:
            if granularity == Granularity.HOURLY:
                # Step in UTC so repeated and skipped local hours remain distinct.
                following = (cursor.astimezone(timezone.utc) + timedelta(hours=1)).astimezone(report_zone)
            elif granularity == Granularity.MONTHLY:
                following = (cursor.replace(year=cursor.year + 1, month=1) if cursor.month == 12
                             else cursor.replace(month=cursor.month + 1))
            else:
                following = cursor + timedelta(days=7 if granularity == Granularity.WEEKLY else 1)
            end = min(following.astimezone(timezone.utc), now)
            label = cursor.isoformat(timespec="minutes") if granularity == Granularity.HOURLY else cursor.date().isoformat()
            windows.append(Window(label, max(cursor.astimezone(timezone.utc), start),
                                  end, end_exclusive=following.astimezone(timezone.utc) <= now,
                                  granularity=granularity, shared_empty=granularity == Granularity.HOURLY))
            cursor = following
    return windows


class WindowLookup:
    """Find the few applicable buckets instead of scanning every calendar period."""

    def __init__(self, windows: list[Window]) -> None:
        self.windows = windows
        self.reporting = [i for i, w in enumerate(windows) if w.granularity is None]
        self.indices = {g: [i for i, w in enumerate(windows) if w.granularity == g]
                        for g in Granularity}
        self.starts = {g: [windows[i].start for i in indices]
                       for g, indices in self.indices.items()}

    def matching(self, at: datetime) -> list[int]:
        result = [i for i in self.reporting if self.windows[i].contains(at)]
        for granularity, starts in self.starts.items():
            position = bisect_right(starts, at) - 1
            if position >= 0:
                index = self.indices[granularity][position]
                if self.windows[index].contains(at):
                    result.append(index)
        return result


@dataclass(frozen=True)
class MetricScope:
    model: str | None = None
    mode: SpeedMode | None = None
    tier: ModelTier | None = None

    def matches(self, model: str, mode: SpeedMode, tier: ModelTier) -> bool:
        return ((self.model is None or self.model == model)
                and (self.mode is None or self.mode == mode)
                and (self.tier is None or self.tier == tier))


@dataclass
class ModeBreakdown:
    windows: list[Window]
    by_model: dict[str, list[Window]] = field(default_factory=dict)


@dataclass
class TierBreakdown:
    windows: list[Window]
    by_model: dict[str, list[Window]] = field(default_factory=dict)
    by_mode: dict[SpeedMode, ModeBreakdown] = field(default_factory=dict)


@dataclass
class Breakdown:
    windows: list[Window]
    by_model: dict[str, list[Window]] = field(default_factory=dict)
    by_mode: dict[SpeedMode, ModeBreakdown] = field(default_factory=dict)
    by_tier: dict[ModelTier, TierBreakdown] = field(default_factory=dict)


def empty_windows(windows: list[Window]) -> list[Window]:
    # Hourly history can contain tens of thousands of empty buckets per filter.
    # Share their boundaries until a scope actually receives samples.
    return [w if w.shared_empty else Window(w.label, w.start, w.end, end_exclusive=w.end_exclusive,
                                            granularity=w.granularity,
                                            shared_empty=w.granularity == Granularity.HOURLY) for w in windows]


def writable_window(windows: list[Window], index: int) -> Window:
    window = windows[index]
    if window.shared_empty:
        window = Window(window.label, window.start, window.end,
                        end_exclusive=window.end_exclusive, granularity=window.granularity)
        windows[index] = window
    return window


def empty_breakdown(windows: list[Window]) -> Breakdown:
    return Breakdown(empty_windows(windows),
                     by_mode={mode: ModeBreakdown(empty_windows(windows))
                              for mode in (SpeedMode.NORMAL, SpeedMode.FAST)},
                     by_tier={tier: TierBreakdown(empty_windows(windows))
                              for tier in (ModelTier.BUDGET, ModelTier.MEDIUM, ModelTier.HIGH,
                                           ModelTier.UNCLASSIFIED)})


def merge_breakdown(target: Breakdown, source: Breakdown) -> None:
    # Merge raw samples and per-conversation values before computing statistics.
    def merge_windows(destination: list[Window], incoming: list[Window]) -> None:
        for index, other in enumerate(incoming):
            if other.shared_empty:
                continue
            window = writable_window(destination, index)
            window.conversations.update(other.conversations)
            for conversation, duration in other.durations.items():
                window.durations[conversation] = window.durations.get(conversation, 0) + duration
            window.call_counts.update(other.call_counts)
            window.throughput.extend(other.throughput)
            window.tokens.update(other.tokens)
            for category, cost in other.costs.items():
                window.costs[category] += cost
            window.unpriced.update(other.unpriced)
            window.unpriced_categories.update(other.unpriced_categories)
            window.models.update(other.models)
            window.coverage.update(other.coverage)
            for unit, amount in other.billing.items():
                window.billing[unit] += amount

    def merge_models(destination: dict[str, list[Window]], incoming: dict[str, list[Window]]) -> None:
        for model, model_windows in incoming.items():
            if model not in destination:
                destination[model] = empty_windows(model_windows)
            merge_windows(destination[model], model_windows)

    def merge_modes(destination: dict[SpeedMode, ModeBreakdown],
                    incoming: dict[SpeedMode, ModeBreakdown]) -> None:
        for mode, group in incoming.items():
            if mode not in destination:
                destination[mode] = ModeBreakdown(empty_windows(group.windows))
            merge_windows(destination[mode].windows, group.windows)
            merge_models(destination[mode].by_model, group.by_model)

    merge_windows(target.windows, source.windows)
    merge_models(target.by_model, source.by_model)
    merge_modes(target.by_mode, source.by_mode)
    for tier, group in source.by_tier.items():
        if tier not in target.by_tier:
            target.by_tier[tier] = TierBreakdown(empty_windows(group.windows))
        merge_windows(target.by_tier[tier].windows, group.windows)
        merge_models(target.by_tier[tier].by_model, group.by_model)
        merge_modes(target.by_tier[tier].by_mode, group.by_mode)


def add_thread(windows: list[Window], thread: Thread,
               by_model: dict[str, list[Window]] | None = None,
               by_mode: dict[SpeedMode, ModeBreakdown] | None = None,
               by_tier: dict[ModelTier, TierBreakdown] | None = None,
               catalog: dict[str, Price] | None = None,
               lookup: WindowLookup | None = None,
               scope: MetricScope | None = None) -> None:
    lookup = lookup or WindowLookup(windows)

    def targets(model: str | None, mode: SpeedMode, tier: ModelTier,
                long_context: bool = False, at_times: Iterable[datetime] = ()) -> list[Window]:
        name = report_model(model, mode, long_context)
        if scope is not None and not scope.matches(name, mode, tier):
            return []
        indices = sorted({i for at in at_times for i in lookup.matching(at)})
        result = [writable_window(windows, i) for i in indices]
        if by_model is not None:
            if name not in by_model:
                by_model[name] = empty_windows(windows)
            result += [writable_window(by_model[name], i) for i in indices]
        if by_mode is not None:
            if mode not in by_mode:
                by_mode[mode] = ModeBreakdown(empty_windows(windows))
            group = by_mode[mode]
            if name not in group.by_model:
                group.by_model[name] = empty_windows(windows)
            result += [writable_window(ws, i) for ws in (group.windows, group.by_model[name]) for i in indices]
        if by_tier is not None:
            if tier not in by_tier:
                by_tier[tier] = TierBreakdown(empty_windows(windows))
            tier_group = by_tier[tier]
            if name not in tier_group.by_model:
                tier_group.by_model[name] = empty_windows(windows)
            if mode not in tier_group.by_mode:
                tier_group.by_mode[mode] = ModeBreakdown(empty_windows(windows))
            mode_group = tier_group.by_mode[mode]
            if name not in mode_group.by_model:
                mode_group.by_model[name] = empty_windows(windows)
            result += [writable_window(ws, i) for ws in (tier_group.windows, tier_group.by_model[name],
                                        mode_group.windows, mode_group.by_model[name]) for i in indices]
        return result

    # Classify usage once; tool calls in a turn share its model/mode contexts.
    call_contexts_by_usage: dict[tuple[str, str | None, SpeedMode], set[bool]] = defaultdict(set)
    for turn in thread.turns.values():
        usage_events = turn.usage
        usage_contexts: set[bool] = set()
        for record in usage_events:
            model = record.model or turn.model
            long_context = is_long_context(model, record.usage, catalog, record.aggregate)
            usage_contexts.add(long_context)
            call_contexts_by_usage[(turn.id, model, record.mode)].add(long_context)
            counts = record.usage.categories()
            usage_total = record.usage.total
            name = report_model(model, record.mode, long_context)
            record_windows = targets(model, record.mode, model_tier(model), long_context, [record.at])
            # Trend points retain tokens, costs, and samples; category detail is for reporting windows.
            costs, unpriced = (price_usage(model, record.usage, record.mode, catalog, aggregate=record.aggregate)
                               if record_windows else ({}, {}))
            for window in record_windows:
                if usage_total:
                    window.conversations.add(thread.id)
                for category, count in counts.items():
                    window.tokens[category] += count
                for category, cost in costs.items():
                    window.costs[category] += cost
                if unpriced:
                    window.unpriced[name] += sum(unpriced.values())
                if window.granularity is not None:
                    continue
                window.models[name] += usage_total
                for category, count in unpriced.items():
                    window.unpriced_categories[category] += count
                window.coverage["Usage responses"] += 1
                if record.aggregate:
                    window.coverage["Aggregate snapshots"] += 1
        usage_models = {r.model or turn.model or "Unknown model" for r in usage_events}
        timing_model = ("Mixed models (timing)" if len(usage_models) > 1 else
                        next(iter(usage_models)) if usage_models else turn.model)
        usage_modes = {r.mode for r in usage_events}
        timing_mode = (SpeedMode.MIXED if len(usage_modes) > 1 else
                       next(iter(usage_modes)) if usage_modes else turn.mode)
        if timing_mode == SpeedMode.MIXED and len(usage_models) == 1:
            timing_model = "Mixed modes (timing)"
        if len(usage_contexts) > 1 and len(usage_models) == 1 and len(usage_modes) == 1:
            timing_model = "Mixed contexts (timing)"
        usage_tiers = {model_tier(r.model or turn.model) for r in usage_events}
        timing_tier = (ModelTier.MIXED if len(usage_tiers) > 1 else
                       next(iter(usage_tiers)) if usage_tiers else model_tier(turn.model))
        turn_windows = targets(timing_model, timing_mode, timing_tier, usage_contexts == {True},
                               [at for at in (turn.start, turn.end) if at is not None])
        for window in turn_windows:
            if turn.start and window.contains(turn.start):
                window.conversations.add(thread.id)
            if not turn.end or not window.contains(turn.end):
                continue
            window.conversations.add(thread.id)
            if not turn.completed or turn.aborted:
                window.coverage["Aborted turns"] += int(turn.aborted)
                continue
            window.coverage["Completed turns"] += 1
            if turn.duration is not None:
                window.durations[thread.id] = window.durations.get(thread.id, 0) + turn.duration
            else:
                window.coverage["Missing turn duration"] += 1
            output = sum(r.usage.output for r in usage_events) if usage_events else turn.output_tokens
            if turn.duration and output is not None:
                window.throughput.append(output / turn.duration)
            else:
                window.coverage["Missing throughput samples"] += 1
        if not turn.end and turn.start:
            for window in turn_windows:
                if window.contains(turn.start):
                    window.coverage["Unfinished turns"] += 1
    for call in thread.calls.values():
        turn = thread.turns.get(call.turn_id)
        model = call.model or (turn.model if turn else None)
        call_contexts = call_contexts_by_usage.get((turn.id, model, call.mode), set()) if turn else set()
        call_model = "Mixed contexts (tools)" if len(call_contexts) > 1 else model
        for window in targets(call_model, call.mode, model_tier(model), call_contexts == {True}, [call.at]):
            window.conversations.add(thread.id)
            window.call_counts[thread.id] += 1
    # Native billing counters are retained separately from token-rate estimates.
    # Per-model attribution is unavailable for some harness summaries.
    for event in thread.billing:
        for window in (writable_window(windows, i) for i in lookup.matching(event.at)):
            window.billing[event.unit] += event.amount


def distribution(values: Iterable[float]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "avg": None, "min": None, "median": None, "max": None,
                "p75": None, "p95": None, "p99": None}
    return {"count": len(ordered), "avg": statistics.fmean(ordered),
            "min": ordered[0], "median": statistics.median(ordered), "max": ordered[-1],
            "p75": ordered[math.ceil(.75 * len(ordered)) - 1],
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
            "p99": ordered[math.ceil(.99 * len(ordered)) - 1]}


def metric_summary(window: Window) -> dict[str, dict[str, int | float | None]]:
    return {"throughput": distribution(window.throughput),
            "length": distribution(window.durations.values()),
            "tools": distribution(window.call_counts[c] for c in window.conversations)}


def trend_summary(window: Window) -> dict[str, Any] | None:
    if not window.conversations:
        return None
    return {**metric_summary(window), "total_tokens": sum(window.tokens.values()),
            "cost": str(sum(window.costs.values(), Decimal(0))), "partial_cost": bool(window.unpriced)}


def summarize(window: Window, report_zone: tzinfo = TIMEZONE) -> dict[str, Any]:
    categories = [c for c in Category if c != Category.CACHE_WRITE or window.tokens[c]]
    return {
        "label": window.label, "start": window.start.astimezone(report_zone).isoformat(),
        "end": window.end.astimezone(report_zone).isoformat(),
        "end_exclusive": window.end_exclusive,
        "conversations": len(window.conversations), "total_tokens": sum(window.tokens.values()),
        "input_tokens": window.tokens[Category.INPUT] + window.tokens[Category.CACHE_READ] + window.tokens[Category.CACHE_WRITE],
        "output_tokens": window.tokens[Category.OUTPUT] + window.tokens[Category.REASONING],
        "cached_input_tokens": window.tokens[Category.CACHE_READ],
        "active_seconds": sum(window.durations.values()), "tool_calls": sum(window.call_counts.values()),
        "cost": str(sum(window.costs.values(), Decimal(0))),
        "partial_cost": bool(window.unpriced), "unpriced_tokens": sum(window.unpriced.values()),
        "metrics": metric_summary(window),
        "categories": [{"name": c.value, "tokens": window.tokens[c],
                        "cost": str(window.costs[c]), "unpriced_tokens": window.unpriced_categories[c]}
                       for c in categories],
        "unpriced": dict(sorted(window.unpriced.items())),
        "models": dict(sorted(window.models.items())), "coverage": dict(window.coverage),
        "recorded_billing": {unit: str(amount) for unit, amount in sorted(window.billing.items())},
    }


def first_datapoint(thread: Thread, now: datetime) -> datetime | None:
    times = [at for turn in thread.turns.values() for at in (turn.start, turn.end)
             if at is not None and at <= now]
    times.extend(record.at for turn in thread.turns.values() for record in turn.usage
                 if record.usage.total and record.at <= now)
    times.extend(call.at for call in thread.calls.values() if call.at <= now)
    return min(times, default=None)


def build_breakdown(threads: Iterable[Thread], now: datetime,
                    catalog: dict[str, Price] | None = None,
                    report_zone: tzinfo = TIMEZONE) -> dict[str, Any]:
    threads = list(threads)
    first = min((at for thread in threads if (at := first_datapoint(thread, now)) is not None), default=now)
    breakdown = empty_breakdown(make_windows(now, report_zone) + make_trend_windows(now, report_zone, first))
    lookup = WindowLookup(breakdown.windows)
    for thread in threads:
        add_thread(breakdown.windows, thread, breakdown.by_model, breakdown.by_mode,
                   breakdown.by_tier, catalog, lookup)
    return summarize_breakdown(breakdown, report_zone)


def summarize_breakdown(breakdown: Breakdown, report_zone: tzinfo = TIMEZONE) -> dict[str, Any]:
    windows = breakdown.windows
    active_models = {name: ws for name, ws in sorted(breakdown.by_model.items())
                     if any(w.conversations for w in ws)}

    def summaries(ws: list[Window]) -> list[dict[str, Any]]:
        return [summarize(w, report_zone) for w in ws if w.granularity is None]

    def trends(ws: list[Window]) -> dict[str, Any]:
        # Empty periods stay null so the chart can show gaps without repeating empty distributions.
        return {granularity.value: [trend_summary(w)
                                    for w in ws if w.granularity == granularity]
                for granularity in Granularity}

    def scope(ws: list[Window], models: dict[str, list[Window]]) -> dict[str, Any]:
        return {"windows": summaries(ws), "trends": trends(ws),
                "by_model": {name: summaries(model_windows) for name, model_windows in models.items()},
                "by_model_trends": {name: trends(model_windows) for name, model_windows in models.items()}}

    def modes(groups: dict[SpeedMode, ModeBreakdown], models: dict[str, list[Window]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for mode in breakdown.by_mode:
            group = groups.get(mode) or ModeBreakdown(empty_windows(windows))
            result[mode.value] = scope(group.windows,
                                      {name: group.by_model.get(name) or empty_windows(windows)
                                       for name in models})
        return result

    tier_summaries: dict[str, Any] = {}
    for tier, group in breakdown.by_tier.items():
        members = {name: ws for name, ws in sorted(group.by_model.items())
                   if any(w.conversations for w in ws)}
        tier_summaries[tier.value] = scope(group.windows, members)
        tier_summaries[tier.value]["by_mode"] = modes(group.by_mode, members)
    result = scope(windows, active_models)
    result.update({"by_tier": tier_summaries, "by_mode": modes(breakdown.by_mode, active_models),
                   "trend_periods": {granularity.value: [
                       {"label": w.label, "start": w.start.astimezone(report_zone).isoformat(),
                        "end": w.end.astimezone(report_zone).isoformat(), "end_exclusive": w.end_exclusive}
                       for w in windows if w.granularity == granularity]
                       for granularity in Granularity}})
    return result


def session_identity(path: Path, harness: Harness) -> str:
    identifier = str(path)
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    break
                if harness == Harness.CODEX and row.get("type") == "session_meta":
                    payload = row.get("payload", {})
                    identifier = payload.get("id") or identifier if isinstance(payload, dict) else identifier
                elif harness == Harness.CLAUDE:
                    # Subagent logs share their parent's sessionId but represent
                    # separate activity; do not merge their turn boundaries.
                    identifier = row.get("sessionId") or identifier
                    if "subagents" in path.parts:
                        identifier = f"{identifier}:{path.stem}"
                elif harness == Harness.COPILOT:
                    data = row.get("data", {})
                    identifier = data.get("sessionId") or identifier if isinstance(data, dict) else identifier
                break
    except (OSError, ValueError, UnicodeError):
        pass
    return str(identifier)


# Parsed groups committed per checkpoint when a failed report should keep imported work.
CHECKPOINT_GROUPS = 250

# Bump when the schema or parser semantics change; cached facts must match the readers.
CACHE_VERSION = 5

# Children precede parents so older caches can be dropped with foreign keys enabled.
CACHE_TABLES = ("cache_billing", "cache_calls", "cache_usage", "cache_turns", "cache_threads",
                "cache_groups", "cache_files")


def file_stamp(path: Path, ctime: bool = True) -> str:
    stat = path.stat()
    fields = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    return json.dumps(fields + [stat.st_ctime_ns] if ctime else fields)


def cache_tables(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def validate_metrics_cache(connection: sqlite3.Connection, *, allow_empty: bool = False) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = cache_tables(connection)
    if not tables and version == 0 and allow_empty:
        return
    if version == 0 or not set(CACHE_TABLES).issubset(tables):
        raise ValueError("cache path contains a different database; choose a separate --cache path")
    if version < CACHE_VERSION:
        raise ValueError("metrics cache was built by an older parser; refresh it before reading")
    if version != CACHE_VERSION:
        raise ValueError("unsupported metrics cache version; use a new --cache path")


def reset_outdated_cache(connection: sqlite3.Connection) -> None:
    """Drop facts parsed under older reader semantics so sources are parsed again."""
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = cache_tables(connection)
    # Only a cache made of this tool's tables is dropped; anything else is left untouched.
    if not 0 < version < CACHE_VERSION or not tables or not tables.issubset(CACHE_TABLES):
        return
    for table in CACHE_TABLES:
        connection.execute(f"DROP TABLE IF EXISTS {table}")
    connection.execute("PRAGMA user_version=0")


class MetricsReader:
    """Hydrate normalized facts without initializing or writing the database."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        validate_metrics_cache(connection)
        self.connection = connection

    def threads(self, quality: Quality, harness: Harness | None = None,
                span: tuple[datetime, datetime] | None = None) -> Iterator[Thread]:
        clauses: list[str] = []
        parameters: list[str] = []
        if harness is not None:
            clauses.append("cache_key IN (SELECT cache_key FROM cache_threads WHERE harness=?)")
            parameters.append(harness.value)
        if span is not None:
            # Select candidates by activity, then load whole conversations for attribution.
            candidates = []
            for table, column in (("cache_turns", "start"), ("cache_turns", "end"),
                                  ("cache_usage", "at"), ("cache_calls", "at")):
                candidates.append(f"SELECT cache_key FROM {table} WHERE {column} BETWEEN ? AND ?")
                parameters.extend(at.isoformat() for at in span)
            clauses.append("cache_key IN (" + " UNION ".join(candidates) + ")")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        groups = self.connection.execute(
            "SELECT cache_key, manifest FROM cache_groups" + where + " ORDER BY rowid", parameters)
        for key, manifest in groups:
            loaded = self.load(key, manifest, quality)
            if loaded is None:
                raise ValueError("metrics cache changed while reading; refresh and retry")
            # One OpenCode group can hold T3-attributed threads alongside OpenCode ones.
            yield from (thread for thread in loaded if harness is None or thread.harness == harness)

    def load(self, key: str, manifest: str, quality: Quality) -> list[Thread] | None:
        row = self.connection.execute("SELECT manifest, counts, warnings FROM cache_groups WHERE cache_key=?",
                                      (key,)).fetchone()
        if row is None or row[0] != manifest:
            return None
        quality.counts.update(json.loads(row[1]))
        quality.warnings.extend(json.loads(row[2])[:max(0, 20 - len(quality.warnings))])
        threads = {thread_id: Thread(thread_id, harness=Harness(harness)) for thread_id, harness in
                   self.connection.execute("SELECT thread_id, harness FROM cache_threads WHERE cache_key=? ORDER BY rowid", (key,))}
        for row in self.connection.execute(
                "SELECT thread_id, turn_id, start, end, duration, model, mode, completed, aborted, output_tokens "
                "FROM cache_turns WHERE cache_key=? ORDER BY rowid", (key,)):
            thread_id, turn_id, start, end, duration, model, mode, completed, aborted, output_tokens = row
            threads[thread_id].turns[turn_id] = Turn(
                turn_id, start=datetime.fromisoformat(start) if start else None,
                end=datetime.fromisoformat(end) if end else None, duration=duration,
                model=model, mode=SpeedMode(mode), completed=bool(completed), aborted=bool(aborted),
                output_tokens=output_tokens)
        for row in self.connection.execute(
                "SELECT thread_id, turn_id, at, model, event_key, mode, aggregate, input, output, cached, "
                "reasoning, cache_write, write_hour FROM cache_usage WHERE cache_key=? ORDER BY rowid", (key,)):
            thread_id, turn_id, at, model, event_key, mode, aggregate, *counts = row
            threads[thread_id].turns[turn_id].modern.append(UsageEvent(
                datetime.fromisoformat(at), Usage(*(int(v) for v in counts)), model, event_key,
                SpeedMode(mode), bool(aggregate)))
        for thread_id, call_id, at, turn_id, model, mode in self.connection.execute(
                "SELECT thread_id, call_id, at, turn_id, model, mode FROM cache_calls WHERE cache_key=? ORDER BY rowid", (key,)):
            threads[thread_id].calls[call_id] = ToolCall(datetime.fromisoformat(at), turn_id, model, SpeedMode(mode))
        for thread_id, at, unit, amount in self.connection.execute(
                "SELECT thread_id, at, unit, amount FROM cache_billing WHERE cache_key=? ORDER BY rowid", (key,)):
            threads[thread_id].billing.append(BillingEvent(datetime.fromisoformat(at), unit, Decimal(amount)))
        return list(threads.values())


class MetricsCache(MetricsReader):
    """SQLite stores parsed facts, never conversation text or calculated prices."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        connection.execute("PRAGMA foreign_keys=ON")
        reset_outdated_cache(connection)
        validate_metrics_cache(connection, allow_empty=True)
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS cache_files (
                harness TEXT NOT NULL, path TEXT NOT NULL, stamp TEXT NOT NULL, identity TEXT NOT NULL,
                PRIMARY KEY (harness, path));
            CREATE TABLE IF NOT EXISTS cache_groups (
                cache_key TEXT PRIMARY KEY, manifest TEXT NOT NULL, counts TEXT NOT NULL, warnings TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cache_threads (
                cache_key TEXT NOT NULL, thread_id TEXT NOT NULL, harness TEXT NOT NULL,
                PRIMARY KEY (cache_key, thread_id),
                FOREIGN KEY (cache_key) REFERENCES cache_groups ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS cache_turns (
                cache_key TEXT NOT NULL, thread_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                start TEXT, end TEXT, duration REAL, model TEXT, mode TEXT NOT NULL,
                completed INTEGER NOT NULL, aborted INTEGER NOT NULL, output_tokens INTEGER,
                PRIMARY KEY (cache_key, thread_id, turn_id),
                FOREIGN KEY (cache_key, thread_id) REFERENCES cache_threads ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS cache_usage (
                cache_key TEXT NOT NULL, thread_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                position INTEGER NOT NULL, at TEXT NOT NULL, model TEXT, event_key TEXT NOT NULL,
                mode TEXT NOT NULL, aggregate INTEGER NOT NULL,
                input TEXT NOT NULL, output TEXT NOT NULL, cached TEXT NOT NULL, reasoning TEXT NOT NULL,
                cache_write TEXT NOT NULL, write_hour TEXT NOT NULL,
                PRIMARY KEY (cache_key, thread_id, turn_id, position),
                FOREIGN KEY (cache_key, thread_id, turn_id) REFERENCES cache_turns ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS cache_calls (
                cache_key TEXT NOT NULL, thread_id TEXT NOT NULL, call_id TEXT NOT NULL,
                at TEXT NOT NULL, turn_id TEXT NOT NULL, model TEXT, mode TEXT NOT NULL,
                PRIMARY KEY (cache_key, thread_id, call_id),
                FOREIGN KEY (cache_key, thread_id) REFERENCES cache_threads ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS cache_billing (
                cache_key TEXT NOT NULL, thread_id TEXT NOT NULL, position INTEGER NOT NULL,
                at TEXT NOT NULL, unit TEXT NOT NULL, amount TEXT NOT NULL,
                PRIMARY KEY (cache_key, thread_id, position),
                FOREIGN KEY (cache_key, thread_id) REFERENCES cache_threads ON DELETE CASCADE);
        """)
        if connection.execute("PRAGMA user_version").fetchone()[0] != CACHE_VERSION:
            connection.execute(f"PRAGMA user_version={CACHE_VERSION}")

    def identity(self, path: Path, harness: Harness) -> str:
        stamp = file_stamp(path)
        row = self.connection.execute(
            "SELECT stamp, identity FROM cache_files WHERE harness=? AND path=?",
            (harness.value, str(path))).fetchone()
        if row and row[0] == stamp:
            return row[1]
        identity = session_identity(path, harness)
        # A growing file must be retried rather than cached under a stale identity.
        if file_stamp(path) == stamp:
            self.connection.execute("INSERT OR REPLACE INTO cache_files VALUES (?, ?, ?, ?)",
                                    (harness.value, str(path), stamp, identity))
        return identity

    def valid_count(self, key: str, manifest: str) -> int | None:
        row = self.connection.execute("SELECT manifest FROM cache_groups WHERE cache_key=?", (key,)).fetchone()
        if row is None or row[0] != manifest:
            return None
        return self.connection.execute("SELECT COUNT(*) FROM cache_threads WHERE cache_key=?", (key,)).fetchone()[0]

    def first_datapoint(self, now: datetime) -> datetime | None:
        # Only the selected source groups participate, including activity older than a year.
        row = self.connection.execute("""
            SELECT MIN(at) FROM (
                SELECT start AS at FROM cache_turns JOIN report_groups USING (cache_key)
                UNION ALL SELECT end AS at FROM cache_turns JOIN report_groups USING (cache_key)
                UNION ALL SELECT at FROM cache_usage JOIN report_groups USING (cache_key)
                    WHERE input != '0' OR output != '0'
                UNION ALL SELECT at FROM cache_calls JOIN report_groups USING (cache_key)
            ) WHERE at <= ?
        """, (now.isoformat(),)).fetchone()
        return datetime.fromisoformat(row[0]) if row[0] else None

    def store(self, key: str, manifest: str, threads: list[Thread], quality: Quality) -> None:
        # Replacing an affected group atomically also handles rewrites, copies and truncation.
        self.connection.execute("DELETE FROM cache_groups WHERE cache_key=?", (key,))
        self.connection.execute("INSERT INTO cache_groups VALUES (?, ?, ?, ?)",
                                (key, manifest, json.dumps(quality.counts), json.dumps(quality.warnings)))
        for thread in threads:
            self.connection.execute("INSERT INTO cache_threads VALUES (?, ?, ?)",
                                    (key, thread.id, thread.harness.value))
            self.connection.executemany("INSERT INTO cache_turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                (key, thread.id, turn.id, turn.start.isoformat() if turn.start else None,
                 turn.end.isoformat() if turn.end else None, turn.duration, turn.model,
                 turn.mode.value, turn.completed, turn.aborted, turn.output_tokens) for turn in thread.turns.values()])
            self.connection.executemany("INSERT INTO cache_usage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                (key, thread.id, turn.id, i, record.at.isoformat(), record.model, record.key,
                 record.mode.value, record.aggregate, str(record.usage.input), str(record.usage.output),
                 str(record.usage.cached), str(record.usage.reasoning), str(record.usage.write),
                 str(record.usage.write_hour))
                for turn in thread.turns.values() for i, record in enumerate(turn.usage)])
            self.connection.executemany("INSERT INTO cache_calls VALUES (?, ?, ?, ?, ?, ?, ?)", [
                (key, thread.id, call_id, call.at.isoformat(), call.turn_id, call.model, call.mode.value)
                for call_id, call in thread.calls.items()])
            self.connection.executemany("INSERT INTO cache_billing VALUES (?, ?, ?, ?, ?, ?)", [
                (key, thread.id, i, event.at.isoformat(), event.unit, str(event.amount))
                for i, event in enumerate(thread.billing)])


def t3_databases(paths: list[Path]) -> list[Path]:
    databases: set[Path] = set()
    for path in paths:
        if path.is_dir():
            # Accept either the T3 base directory or its userdata directory.
            for directory in (path, path / "userdata"):
                databases.update(p.resolve() for name in ("state.sqlite", "statev2.sqlite")
                                 if (p := directory / name).is_file())
        else:
            databases.add(path.resolve())
    return sorted(databases)


def read_t3_sessions(paths: list[Path], quality: Quality) -> dict[Harness, set[str]]:
    """Read native session references, never T3 message text or tool arguments."""
    sessions: dict[Harness, set[str]] = defaultdict(set)
    providers = {h.value: h for h in (Harness.CODEX, Harness.CLAUDE, Harness.OPENCODE)}
    for path in paths:
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("BEGIN")
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not tables.intersection({"provider_session_runtime", "orchestration_v2_projection_provider_threads"}):
                    raise ValueError("unsupported T3 Code database schema")
                if "provider_session_runtime" in tables:
                    for provider, raw in connection.execute(
                            "SELECT provider_name, resume_cursor_json FROM provider_session_runtime"):
                        if provider not in providers or raw is None:
                            continue
                        try:
                            cursor = json.loads(raw)
                        except (ValueError, TypeError):
                            quality.warn("Malformed records", path, 0, "invalid T3 resume cursor; skipped")
                            continue
                        field = "threadId" if provider == "codex" else "sessionId"
                        identifier = cursor.get(field) if isinstance(cursor, dict) else None
                        if isinstance(identifier, str) and identifier:
                            sessions[providers[provider]].add(identifier)
                        elif not isinstance(cursor, dict) or identifier is not None:
                            quality.warn("Malformed records", path, 0, "invalid T3 native session ID; skipped")
                if "orchestration_v2_projection_provider_threads" in tables:
                    for provider, raw in connection.execute(
                            "SELECT provider, payload_json FROM orchestration_v2_projection_provider_threads"):
                        if provider not in providers:
                            continue
                        try:
                            payload = json.loads(raw)
                        except (ValueError, TypeError):
                            quality.warn("Malformed records", path, 0, "invalid T3 provider thread JSON; skipped")
                            continue
                        reference = payload.get("nativeThreadRef") if isinstance(payload, dict) else None
                        identifier = reference.get("nativeId") if isinstance(reference, dict) else None
                        if isinstance(identifier, str) and identifier:
                            sessions[providers[provider]].add(identifier)
                        elif (not isinstance(payload, dict) or
                              reference is not None and not isinstance(reference, dict) or
                              identifier is not None):
                            quality.warn("Malformed records", path, 0, "invalid T3 native thread reference; skipped")
        except (OSError, sqlite3.Error, ValueError) as error:
            quality.warn("Unreadable files", path, 0, str(error))
    return dict(sessions)


def native_source_defaults() -> dict[Harness, list[Path]]:
    home = Path.home()
    return {Harness.CODEX: [home / ".codex" / name for name in ("sessions", "archived_sessions")],
            Harness.CLAUDE: [Path(os.environ.get("CLAUDE_CONFIG_DIR", str(home / ".claude"))) / "projects"],
            Harness.COPILOT: [home / ".copilot" / "session-state"],
            Harness.OPENCODE: [Path(os.environ.get("XDG_DATA_HOME", str(home / ".local" / "share"))) / "opencode"]}


def source_manifest(paths: list[Path], harness: Harness, attribution: str = "") -> str:
    stamps: list[tuple[str, str]] = []
    for path in paths:
        stamps.append((str(path), file_stamp(path) + attribution))
        if harness in (Harness.OPENCODE, Harness.T3):
            # Uncheckpointed SQLite writes live in the WAL, not the main database. SQLite opens the WAL
            # read-write even for read-only connections, and macOS then retags its provenance xattr,
            # changing ctime without a write; WAL writes always change mtime or size.
            wal = Path(str(path) + "-wal")
            stamps.append((str(wal), file_stamp(wal, ctime=False) if wal.exists() else "missing"))
    return json.dumps(stamps)


@dataclass
class PreparedGroup:
    key: str
    manifest: str
    uncached: list[Thread] | None = None
    quality: Quality | None = None
    source: str = ""


@dataclass
class PreparedSources:
    threads: Iterator[Thread]
    files: set[Path]
    first_at: datetime | None
    cache: MetricsCache
    uncached_sources: list[str] = field(default_factory=list)

    @property
    def uncached_groups(self) -> int:
        return len(self.uncached_sources)


@contextmanager
def prepare_sources(sources: dict[Harness, list[Path]], quality: Quality, progress: bool,
                    cache_path: Path | None, now: datetime,
                    checkpoint: bool = False,
                    after_commit: Callable[[sqlite3.Connection], None] | None = None) -> Iterator[PreparedSources]:
    """With checkpoint, parsed groups are committed periodically so a later failure keeps them.

    Without it, every cache update is rolled back if the caller fails (used by the live server).
    after_commit receives the writer connection once the caller's work has been committed.
    """
    t3_paths = t3_databases(sources.get(Harness.T3, []))
    if cache_path:
        opencode_paths = sources.get(Harness.OPENCODE, [])
        if Harness.T3 in sources and Harness.OPENCODE not in sources:
            opencode_paths = native_source_defaults()[Harness.OPENCODE]
        databases = t3_paths + [directory / "opencode.db" if directory.is_dir() else directory
                                for directory in opencode_paths]
        for database in databases:
            if (database.resolve() == cache_path.resolve() or
                    database.exists() and cache_path.exists() and database.samefile(cache_path)):
                raise ValueError("the metrics cache must be separate from harness storage")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    t3_quality = Quality()
    try:
        t3_manifest = source_manifest(t3_paths, Harness.T3)
    except OSError:
        t3_manifest = ""
    t3_sessions = read_t3_sessions(t3_paths, t3_quality)
    scan_sources = {h: paths for h, paths in sources.items() if h != Harness.T3}
    for harness, defaults in native_source_defaults().items():
        if harness in t3_sessions and harness not in scan_sources:
            scan_sources[harness] = [path for path in defaults if path.exists()]
    # A temporary SQLite working database also bounds memory when persistent caching is disabled.
    temporary = tempfile.TemporaryDirectory(prefix="harness-report-") if cache_path is None else nullcontext(None)
    with temporary as directory:
        working_path = cache_path if cache_path is not None else Path(directory) / "metrics.sqlite3"
        with closing(sqlite3.connect(working_path)) as connection:
            cache = MetricsCache(connection)
            with connection:
                connection.execute("BEGIN")
                connection.execute("CREATE TEMP TABLE report_groups (cache_key TEXT PRIMARY KEY)")
                prepared: list[PreparedGroup] = []
                stored = 0
                all_paths: set[Path] = set(t3_paths)
                if t3_paths:
                    key = json.dumps([Harness.T3.value, [str(path) for path in t3_paths]])
                    try:
                        stable = bool(t3_manifest) and t3_manifest == source_manifest(t3_paths, Harness.T3)
                    except OSError:
                        stable = False
                    if stable and not t3_quality.counts["Unreadable files"]:
                        cache.store(key, t3_manifest, [], t3_quality)
                        prepared.append(PreparedGroup(key, t3_manifest))
                        connection.execute("INSERT INTO report_groups VALUES (?)", (key,))
                    else:
                        prepared.append(PreparedGroup(key, t3_manifest, [], t3_quality, str(t3_paths[0])))
                readers = {Harness.CODEX: read_thread, Harness.CLAUDE: read_claude, Harness.COPILOT: read_copilot}
                for harness, directories in scan_sources.items():
                    def attributed_harness(identifier: str) -> Harness:
                        # Claude subagents use the parent's session ID plus a suffix.
                        native_id = identifier.partition(":")[0] if harness == Harness.CLAUDE else identifier
                        return Harness.T3 if native_id in t3_sessions.get(harness, set()) else harness

                    if harness in readers:
                        paths = sorted({p.resolve() for directory in directories for p in directory.rglob("*.jsonl")})
                        groups: dict[str, list[Path]] = defaultdict(list)
                        for path in paths:
                            try:
                                identity = cache.identity(path, harness)
                            except OSError:
                                identity = str(path)
                            if attributed_harness(identity) in sources:
                                groups[identity].append(path)
                        paths = sorted({path for files in groups.values() for path in files})
                    else:
                        paths = sorted({(directory / "opencode.db" if directory.is_dir() else directory).resolve()
                                        for directory in directories})
                        groups = {json.dumps([str(p) for p in paths]): paths} if paths else {}
                    hits = 0
                    parsed = 0
                    for index, (thread_id, files) in enumerate(groups.items(), 1):
                        key = json.dumps([harness.value, thread_id])
                        attribution = (json.dumps([harness in sources, sorted(t3_sessions.get(harness, set()))])
                                       if harness == Harness.OPENCODE else attributed_harness(thread_id).value)
                        try:
                            manifest = source_manifest(files, harness, attribution)
                        except OSError:
                            manifest = ""
                        count = cache.valid_count(key, manifest) if manifest else None
                        group = PreparedGroup(key, manifest)
                        if count is not None:
                            hits += count
                        else:
                            group_quality = Quality()
                            try:
                                loaded = ([readers[harness](thread_id, files, group_quality)] if harness in readers
                                          else read_opencode(files, group_quality))
                                for thread in loaded:
                                    thread.harness = attributed_harness(thread.id)
                                    if thread.harness == Harness.T3:
                                        thread.id = f"{harness.value}:{thread.id}"
                                loaded = [thread for thread in loaded if thread.harness in sources]
                            except (OSError, UnicodeError) as error:
                                group_quality.warn("Unreadable files", files[0], 0, str(error))
                                loaded = []
                            parsed += len(loaded)
                            stable = False
                            if manifest and not group_quality.counts["Unreadable files"]:
                                try:
                                    stable = manifest == source_manifest(files, harness, attribution)
                                except OSError:
                                    pass
                            if stable:
                                cache.store(key, manifest, loaded, group_quality)
                                stored += 1
                                if checkpoint and stored % CHECKPOINT_GROUPS == 0:
                                    connection.commit()
                                    connection.execute("BEGIN")
                            else:
                                group.uncached, group.quality, group.source = loaded, group_quality, str(files[0])
                        prepared.append(group)
                        if group.uncached is None:
                            connection.execute("INSERT INTO report_groups VALUES (?)", (key,))
                        if progress and (index % 250 == 0 or index == len(groups)):
                            print(f"{harness.value}: {hits:,} cached, {parsed:,} parsed; {index:,}/{len(groups):,} sources", file=sys.stderr)
                    all_paths.update(paths)
                first = cache.first_datapoint(now)
                for group in prepared:
                    for thread in group.uncached or []:
                        at = first_datapoint(thread, now)
                        if at is not None and (first is None or at < first):
                            first = at

                def threads() -> Iterator[Thread]:
                    for group in prepared:
                        if group.uncached is not None:
                            if group.quality:
                                quality.counts.update(group.quality.counts)
                                quality.warnings.extend(group.quality.warnings[:max(0, 20 - len(quality.warnings))])
                            yield from group.uncached
                        else:
                            loaded = cache.load(group.key, group.manifest, quality)
                            if loaded is None:
                                raise ValueError("metrics cache changed while generating the report; rerun the command")
                            yield from loaded

                yield PreparedSources(threads(), all_paths, first, cache,
                                      [group.source for group in prepared if group.uncached is not None])
            if after_commit is not None:
                after_commit(connection)


def collect_report(root: Path, now: datetime | None = None, progress: bool = False,
                   additional_roots: Iterable[Path] = (),
                   harness_roots: dict[Harness, list[Path]] | None = None,
                   include_codex: bool = True, catalog: dict[str, Price] | None = None,
                   catalog_metadata: dict[str, Any] | None = None,
                   report_zone: tzinfo = TIMEZONE,
                   cache_path: Path | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("report cutoff must include a timezone")
    now = now.astimezone(timezone.utc)
    quality = Quality()
    sources = {h: list(dict.fromkeys(p.resolve() for p in paths)) for h, paths in (harness_roots or {}).items()}
    if include_codex:
        sources[Harness.CODEX] = list(dict.fromkeys(p.resolve() for p in (root, *additional_roots)))
    roots = list(dict.fromkeys(p for paths in sources.values() for p in paths))
    with prepare_sources(sources, quality, progress, cache_path, now, checkpoint=True) as prepared:
        windows = make_windows(now, report_zone) + make_trend_windows(now, report_zone, prepared.first_at)
        lookup = WindowLookup(windows)
        overall = empty_breakdown(windows)
        harness_breakdowns = {harness: empty_breakdown(windows) for harness in sources}
        all_paths = prepared.files
        thread_count = 0
        models: dict[str, str | None] = {}
        # Aggregate as conversations arrive; large archives need not all reside in memory.
        if progress and all_paths:
            print("Calculating metrics over available history…", file=sys.stderr)
        for thread in prepared.threads:
            thread_count += 1
            thread.id = f"{thread.harness.value}:{thread.id}"
            breakdown = harness_breakdowns[thread.harness]
            add_thread(breakdown.windows, thread, breakdown.by_model, breakdown.by_mode,
                       breakdown.by_tier, catalog, lookup)
            for turn in thread.turns.values():
                for record in turn.usage:
                    model = record.model or turn.model
                    name = report_model(model, record.mode, is_long_context(model, record.usage, catalog, record.aggregate))
                    models[name] = model
    by_harness: dict[str, Any] = {}
    for harness, breakdown in harness_breakdowns.items():
        merge_breakdown(overall, breakdown)
        by_harness[harness.value] = summarize_breakdown(breakdown, report_zone)
    if progress and thread_count:
        print("Combining overall summary…", file=sys.stderr)
    report = summarize_breakdown(overall, report_zone)
    report.update({"generated": now.astimezone(report_zone).isoformat(), "timezone": str(report_zone),
                   "source": str(root.resolve()), "sources": [str(p) for p in roots],
                   "files": len(all_paths), "threads": thread_count,
                   "pricing_date": PRICING_DATE, "pricing_source": PRICING_SOURCE,
                   "anthropic_pricing_date": ANTHROPIC_PRICING_DATE,
                   "anthropic_pricing_source": ANTHROPIC_PRICING_SOURCE,
                   "openrouter": catalog_metadata,
                   "price_proxies": PRICE_PROXIES, "fast_cost_multiplier": str(FAST_COST_MULTIPLIER),
                   "unknown_mode_assumption": SpeedMode.NORMAL.value,
                   "by_harness": by_harness,
                   "quality": dict(quality.counts), "warnings": quality.warnings})
    # Include only relevant matched catalog rows in the offline artifact.
    matches = {name: router_model(model, catalog or {}) for name, model in models.items()
               if name in report["by_model"]}
    report["openrouter_rates"] = {model: {"id": identifier, "input": str(catalog[identifier].short.input),
                                        "cached": str(catalog[identifier].short.cached) if catalog[identifier].short.cached is not None else None,
                                        "output": str(catalog[identifier].short.output)}
                                  for model, identifier in matches.items() if identifier and catalog}
    return report


def render_report(report: dict[str, Any]) -> str:
    # Prevent paths/model names containing HTML from closing the JSON script tag.
    data = json.dumps(report, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return HTML.replace("__REPORT_DATA__", data)


def argument_parser(*, static: bool = True, description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", type=Path,
                        help="Codex archive directory; supplying one disables automatic local source discovery")
    if static:
        parser.add_argument("--output", type=Path, default=Path("harness_metrics.html"), help="HTML output (default: harness_metrics.html)")
    parser.add_argument("--timezone", type=report_timezone, default=TIMEZONE,
                        help=f"calendar window timezone (default: {TIMEZONE}; UTC works without timezone data)")
    parser.add_argument("--harness", action="append", choices=[h.value for h in Harness],
                        help="include this harness (repeatable; default: all installed harnesses)")
    parser.add_argument("--claude-dir", type=Path, help="Claude projects directory (default: ~/.claude/projects)")
    parser.add_argument("--copilot-dir", type=Path, help="Copilot session-state directory (default: ~/.copilot/session-state)")
    parser.add_argument("--opencode-dir", type=Path, help="OpenCode data directory or opencode.db path")
    parser.add_argument("--t3-dir", type=Path, help="T3 Code base/userdata directory or state database path (default: ~/.t3/userdata)")
    cache_options = parser.add_mutually_exclusive_group()
    cache_options.add_argument("--cache", type=Path, help="SQLite metrics cache path (default: user cache directory)")
    if static:
        cache_options.add_argument("--no-cache", action="store_true", help="read logs directly without a persistent cache")
    pricing_options = parser.add_mutually_exclusive_group()
    pricing_options.add_argument("--openrouter-prices", type=Path,
                                 help="saved /api/v1/models JSON catalog (default: bundled openrouter_prices.json)")
    pricing_options.add_argument("--live-prices", action="store_true",
                                 help="fetch OpenRouter pricing; fall back to bundled prices if unavailable")
    parser.add_argument("--offline", action="store_true", help="use local pricing only (the default); incompatible with --live-prices")
    return parser


def default_cache_path() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "harness-report" / "metrics.sqlite3"


def source_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[Harness, list[Path]]:
    if args.directory is not None and not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")
    selected = {Harness(name) for name in args.harness} if args.harness else set(Harness)
    home = Path.home()
    defaults = {harness: paths[0] for harness, paths in native_source_defaults().items()}
    defaults[Harness.T3] = home / ".t3" / "userdata"
    supplied = {Harness.CLAUDE: args.claude_dir, Harness.COPILOT: args.copilot_dir,
                Harness.OPENCODE: args.opencode_dir, Harness.T3: args.t3_dir}
    harness_roots: dict[Harness, list[Path]] = {}
    for harness in selected - {Harness.CODEX}:
        explicit = supplied[harness]
        if explicit and not explicit.exists():
            parser.error(f"missing {harness.value} source: {explicit}")
        if explicit:
            candidates = [explicit]
        elif args.directory is None:
            candidates = [defaults[harness]]
        else:
            candidates = []
        existing = [p for p in candidates if p.exists()]
        if harness == Harness.T3:
            databases = t3_databases(existing)
            if explicit and not databases:
                parser.error(f"missing T3 Code state database in: {explicit}")
            existing = databases
        if existing:
            harness_roots[harness] = existing
    if Harness.CODEX in selected:
        if args.directory is not None:
            codex_roots = [args.directory]
        else:
            codex_roots = [path for path in (home / ".codex" / name for name in ("sessions", "archived_sessions"))
                           if path.is_dir()]
        if codex_roots:
            harness_roots[Harness.CODEX] = codex_roots
    return harness_roots


def load_catalog(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[dict[str, Price], dict[str, Any]]:
    if args.offline and args.live_prices:
        parser.error("--offline cannot be combined with --live-prices")
    snapshot_path = args.openrouter_prices or BUNDLED_OPENROUTER_PRICES
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        catalog = openrouter_prices(snapshot)
        retrieved = snapshot.get("retrieved")
        catalog_metadata: dict[str, Any] = {"source": OPENROUTER_SOURCE,
                                           "snapshot_file": str(snapshot_path.resolve()),
                                           "bundled": args.openrouter_prices is None,
                                           "retrieved": retrieved if isinstance(retrieved, str) else None}
    except (OSError, ValueError, UnicodeError) as error:
        parser.error(f"unable to load OpenRouter catalog {snapshot_path}: {error}")
    if args.live_prices:
        try:
            print("Fetching OpenRouter model prices…", file=sys.stderr)
            request = Request(OPENROUTER_SOURCE, headers={"User-Agent": "coding-agent-metrics/1.0"})
            with urlopen(request, timeout=10) as response:
                catalog = openrouter_prices(json.load(response))
            catalog_metadata = {"source": OPENROUTER_SOURCE, "retrieved": datetime.now(timezone.utc).isoformat()}
        except (OSError, ValueError, UnicodeError) as error:
            fallback = "bundled" if args.openrouter_prices is None else "supplied snapshot"
            print(f"Live OpenRouter prices unavailable: {error}; using {fallback} prices", file=sys.stderr)
            catalog_metadata["error"] = str(error)
    return catalog, catalog_metadata


def main() -> int:
    parser = argument_parser()
    args = parser.parse_args()
    if args.offline and args.live_prices:
        parser.error("--offline cannot be combined with --live-prices")
    sources = source_paths(args, parser)
    codex_roots = sources.get(Harness.CODEX, [])
    root = codex_roots[0] if codex_roots else Path.cwd()
    if args.output.suffix.lower() != ".html":
        parser.error("output must have an .html extension")
    catalog, catalog_metadata = load_catalog(args, parser)
    try:
        report = collect_report(root, progress=True,
                                additional_roots=codex_roots[1:],
                                harness_roots={h: paths for h, paths in sources.items() if h != Harness.CODEX},
                                include_codex=Harness.CODEX in sources,
                                catalog=catalog, catalog_metadata=catalog_metadata, report_zone=args.timezone,
                                cache_path=None if args.no_cache else args.cache or default_cache_path())
        args.output.write_text(render_report(report), encoding="utf-8")
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"Unable to generate report: {error}", file=sys.stderr)
        return 1
    print(f"Report: {args.output.resolve()}")
    print(f"{report['files']:,} files; {report['threads']:,} conversations; pricing snapshot {PRICING_DATE}")
    return 0


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coding agents · Usage report</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--surface-2:#f2f1ed;--surface-3:#e9e8e3;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6f6e69;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--accent:#2a78d6;--accent-wash:rgba(42,120,214,.10);--focus:rgba(42,120,214,.55);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--other:#a9a8a1;--on-s6:#fff;--on-s7:#fff;
--q0:#f2f1ed;--q1:#cde2fb;--q2:#9ec5f4;--q3:#6da7ec;--q4:#3987e5;--q5:#256abf;--q6:#184f95;--q7:#0d366b;
--warn-bg:#fff6e0;--warn-ink:#6b4800;--warn-line:#f0d999;--tip-bg:#1d1d1b;--tip-ink:#fff;--tip-muted:#c3c2b7;--shadow:0 1px 2px rgba(11,11,11,.04),0 6px 24px rgba(11,11,11,.05)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--surface-2:#242422;--surface-3:#2f2f2c;--ink:#fff;--ink-2:#c3c2b7;--muted:#9a9891;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--accent:#3987e5;--accent-wash:rgba(57,135,229,.14);--focus:rgba(57,135,229,.7);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--other:#5c5b56;--on-s6:#fff;--on-s7:#0b0b0b;
--q0:#242422;--q1:#0d366b;--q2:#184f95;--q3:#1c5cab;--q4:#256abf;--q5:#3987e5;--q6:#6da7ec;--q7:#9ec5f4;
--warn-bg:#2b2410;--warn-ink:#f5d58a;--warn-line:#5a4a1c;--tip-bg:#f2f1ed;--tip-ink:#0b0b0b;--tip-muted:#52514e;--shadow:none}}
:root[data-theme=dark]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--surface-2:#242422;--surface-3:#2f2f2c;--ink:#fff;--ink-2:#c3c2b7;--muted:#9a9891;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--accent:#3987e5;--accent-wash:rgba(57,135,229,.14);--focus:rgba(57,135,229,.7);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--other:#5c5b56;--on-s6:#fff;--on-s7:#0b0b0b;
--q0:#242422;--q1:#0d366b;--q2:#184f95;--q3:#1c5cab;--q4:#256abf;--q5:#3987e5;--q6:#6da7ec;--q7:#9ec5f4;
--warn-bg:#2b2410;--warn-ink:#f5d58a;--warn-line:#5a4a1c;--tip-bg:#f2f1ed;--tip-ink:#0b0b0b;--tip-muted:#52514e;--shadow:none}
*{box-sizing:border-box}html{scroll-padding-top:120px}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
h1,h2,h3,p{margin:0}h1{font-size:28px;line-height:1.2;letter-spacing:-.5px;font-weight:650}h2{font-size:18px;letter-spacing:-.2px;font-weight:650}h3{font-size:14px;font-weight:600}
a{color:var(--accent)}button,select,input{font:inherit;color:inherit}button{cursor:pointer}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
main{max-width:1320px;margin:auto;padding:0 28px 48px}
.masthead{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;padding:32px 0 20px;flex-wrap:wrap}.eyebrow{font-size:12px;font-weight:600;color:var(--muted);margin-bottom:6px}.subtitle{color:var(--ink-2);margin-top:6px}
.head-actions{display:flex;align-items:center;gap:10px}.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--border);border-radius:999px;padding:5px 11px;color:var(--ink-2);font-size:12px;background:var(--surface)}.pill i{width:6px;height:6px;border-radius:50%;background:var(--s3)}
.ghost{border:1px solid var(--border);background:var(--surface);border-radius:8px;padding:6px 11px;font-size:12px;color:var(--ink-2)}.ghost:hover{background:var(--surface-2)}
.toolbar{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--page) 88%,transparent);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-bottom:1px solid var(--border);margin:0 -28px;padding:10px 28px}
.toolbar-row{display:flex;align-items:center;gap:12px 20px;flex-wrap:wrap}
.segmented{display:inline-flex;gap:2px;padding:3px;background:var(--surface-2);border:1px solid var(--border);border-radius:10px;flex-wrap:wrap}.segmented button{border:0;background:none;padding:5px 10px;border-radius:7px;color:var(--ink-2);font-size:12.5px;font-weight:550;white-space:nowrap}.segmented button:hover{color:var(--ink)}.segmented button[aria-pressed=true]{background:var(--surface);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.12)}
.filters{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.field{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}.field select,.field input{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:5px 8px;font-size:12.5px;font-weight:550;color:var(--ink);max-width:220px}
.field select[data-active=true]{border-color:var(--accent);background:var(--accent-wash)}
.scope-line{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--muted)}.chip{display:inline-flex;align-items:center;gap:4px;border-radius:999px;background:var(--accent-wash);color:var(--ink);padding:2px 4px 2px 10px;font-size:12px}.chip button{border:0;background:none;color:var(--ink-2);border-radius:50%;width:20px;height:20px;line-height:1;padding:0}.chip button:hover{background:var(--surface-3)}.link-btn{border:0;background:none;color:var(--accent);padding:0;font-size:12px}
.section{margin-top:36px}.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:12px 20px;margin-bottom:14px;flex-wrap:wrap}.section-head p{color:var(--muted);font-size:12.5px;margin-top:3px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);padding:20px;min-width:0}.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}.card-head p{color:var(--muted);font-size:12.5px;margin-top:2px}
.notice{display:flex;gap:10px;align-items:flex-start;padding:10px 14px;background:var(--warn-bg);border:1px solid var(--warn-line);border-radius:10px;color:var(--warn-ink);font-size:13px;margin-top:16px}.notice b{font-weight:650}.notice[hidden]{display:none}
.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-top:20px}.kpi{padding:16px 16px 10px;display:flex;flex-direction:column}.kpi-label{font-size:12.5px;color:var(--ink-2);font-weight:550}.kpi-value{font-size:28px;font-weight:650;letter-spacing:-.6px;line-height:1.15;margin-top:6px}.kpi-note{font-size:12px;color:var(--muted);margin-top:2px;min-height:18px}.kpi .spark{margin-top:auto;padding-top:10px}.kpi-split{margin:10px 0 0;font-size:12px;display:grid;gap:3px}.kpi-split div{display:flex;justify-content:space-between;gap:8px}.kpi-split dt{color:var(--muted)}.kpi-split dd{margin:0;font-variant-numeric:tabular-nums;font-weight:550}
.grid-2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.stack{display:grid;gap:16px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--ink-2);margin-top:12px}.legend span{display:inline-flex;align-items:center;gap:6px}.key{width:10px;height:10px;border-radius:3px;display:inline-block;flex:none}.key.line{height:2px;width:14px;border-radius:2px}
.chart{margin-top:10px;position:relative;min-height:40px}.chart svg{display:block;overflow:visible}.chart svg text{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}.chart svg:focus-visible{outline:2px solid var(--focus);outline-offset:4px;border-radius:4px}
.chart[aria-busy=true]>*{opacity:.5;transition:opacity .2s}body[aria-busy=true] main>:not(.masthead){opacity:.6;pointer-events:none;transition:opacity .2s}.empty{padding:36px 16px;text-align:center;color:var(--muted);background:var(--surface-2);border-radius:10px;font-size:13px}
.view-toggle{font-size:12px}.data-table{margin-top:12px;max-height:340px;overflow:auto;border:1px solid var(--border);border-radius:10px}
.headline{display:flex;align-items:baseline;gap:14px;margin-top:10px;flex-wrap:wrap}.headline strong{font-size:24px;font-weight:650;letter-spacing:-.4px}.headline span{font-size:12.5px;color:var(--muted)}
.dist{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px;margin-top:14px;padding-top:12px;border-top:1px solid var(--grid)}.dist div{min-width:0}.dist dt{font-size:11px;color:var(--muted)}.dist dd{margin:1px 0 0;font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.samples{font-size:12px;color:var(--muted);margin-top:8px}
.stat-chips{display:flex;gap:6px;flex-wrap:wrap}.stat-chips label{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--border);border-radius:999px;padding:4px 11px 4px 9px;font-size:12.5px;cursor:pointer;background:var(--surface);user-select:none}.stat-chips input{position:absolute;opacity:0;width:1px;height:1px}.stat-chips label:has(input:checked){background:var(--surface-2);border-color:var(--axis)}.stat-chips label:has(input:not(:checked)) .key{background:transparent!important;outline:1.5px solid var(--axis);outline-offset:-1.5px}.stat-chips label:has(input:focus-visible){outline:2px solid var(--focus);outline-offset:2px}
.composition{display:grid;gap:14px;margin-top:16px}.comp-row{display:grid;grid-template-columns:72px 1fr 90px;gap:12px;align-items:center;font-size:12.5px}.comp-row>span:first-child{color:var(--ink-2);font-weight:550}.comp-row>span:last-child{text-align:right;font-weight:650;font-variant-numeric:tabular-nums}.bar100{display:flex;gap:2px;height:30px}.bar100 div{min-width:2px;display:flex;align-items:center;justify-content:center;font-size:11.5px;font-weight:600;color:#0b0b0b;font-variant-numeric:tabular-nums}.bar100 div:first-child{border-radius:5px 0 0 5px}.bar100 div:last-child{border-radius:0 5px 5px 0}.bar100 div:only-child{border-radius:5px}
.insight{font-size:13px;color:var(--ink-2);margin-top:14px;padding:10px 12px;background:var(--surface-2);border-radius:8px}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12.5px}th{text-align:left;color:var(--muted);font-size:11.5px;font-weight:600;white-space:nowrap}th,td{padding:8px 10px;border-bottom:1px solid var(--grid)}tbody tr:last-child td{border-bottom:0}td:first-child{white-space:nowrap}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tfoot td{font-weight:650;border-top:1px solid var(--axis);border-bottom:0}.data-table th{position:sticky;top:0;background:var(--surface)}
th button{border:0;background:none;padding:0;color:inherit;font:inherit;display:inline-flex;gap:4px;align-items:center}th button:hover{color:var(--ink)}th[aria-sort] button::after{content:"";border:4px solid transparent;margin-left:2px}th[aria-sort=descending] button::after{border-top-color:currentColor;margin-top:4px}th[aria-sort=ascending] button::after{border-bottom-color:currentColor;margin-bottom:4px}
.compare tbody tr{cursor:pointer}.compare tbody tr:hover td{background:var(--surface-2)}.compare tbody tr[aria-selected=true] td{background:var(--accent-wash)}.compare td:first-child{font-weight:600;white-space:nowrap}.compare td:first-child .key{margin-right:8px;vertical-align:-1px}.databar{display:flex;align-items:center;justify-content:flex-end;gap:8px;min-width:150px}.databar i{display:block;height:8px;border-radius:0 3px 3px 0;flex:none;background:var(--s1)}.databar .track{flex:1;display:flex;justify-content:flex-start;max-width:90px}.dim{color:var(--muted)}
.heat-legend{display:flex;align-items:center;gap:4px;font-size:11.5px;color:var(--muted);margin-top:10px}.heat-legend i{width:16px;height:10px;border-radius:2px;display:inline-block}
.meters{display:grid;gap:12px;margin-top:12px}.meter-row{display:grid;grid-template-columns:1fr auto;gap:4px 12px;font-size:12.5px}.meter-row .val{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink-2)}.meter{grid-column:1/-1;height:6px;background:var(--accent-wash);border-radius:3px;overflow:hidden}.meter i{display:block;height:100%;background:var(--accent);border-radius:3px}
.counts{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:14px}.counts div{background:var(--surface-2);border-radius:8px;padding:8px 10px}.counts dt{font-size:11.5px;color:var(--muted)}.counts dd{margin:2px 0 0;font-weight:650;font-size:16px}
.defs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 24px;margin-top:8px}.defs details{border-bottom:1px solid var(--grid);padding:10px 0}.defs summary{font-weight:600;font-size:13px;cursor:pointer}.defs details p{color:var(--ink-2);font-size:12.5px;margin-top:6px;line-height:1.6}
details.more{margin-top:12px;border-top:1px solid var(--grid);padding-top:10px}details.more>summary{cursor:pointer;font-weight:600;font-size:13px}
.sources{font-size:12px;color:var(--muted);margin-top:16px;overflow-wrap:anywhere;line-height:1.6}.warnings{font:11px/1.7 ui-monospace,monospace;overflow-wrap:anywhere;padding-left:18px;color:var(--ink-2)}
.footer{display:flex;justify-content:space-between;gap:16px;margin-top:32px;color:var(--muted);font-size:12px;flex-wrap:wrap}
.range-error{color:#c03030;font-size:12.5px;margin-top:8px}
.tooltip{position:fixed;z-index:20;background:var(--tip-bg);color:var(--tip-ink);padding:8px 10px;border-radius:8px;font-size:12px;pointer-events:none;min-width:150px;max-width:300px;box-shadow:0 8px 28px rgba(0,0,0,.22)}.tooltip .tip-title{font-weight:600;margin-bottom:4px;color:var(--tip-muted)}.tooltip .tip-row{display:grid;grid-template-columns:14px auto 1fr;gap:8px;align-items:center;line-height:1.7}.tooltip .tip-row strong{font-variant-numeric:tabular-nums;font-weight:650}.tooltip .tip-row span:last-child{color:var(--tip-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tooltip .tip-row .key.line{width:12px}.tooltip .tip-foot{color:var(--tip-muted);margin-top:4px;border-top:1px solid rgba(128,128,128,.3);padding-top:4px}
noscript{display:block;padding:20px;background:var(--warn-bg)}
@media(max-width:1100px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:900px){.grid-2,.defs{grid-template-columns:1fr}main{padding:0 18px 40px}.toolbar{margin:0 -18px;padding:10px 18px}}
@media(max-width:640px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.toolbar{position:static}.dist{grid-template-columns:repeat(4,minmax(0,1fr));row-gap:8px}h1{font-size:24px}.kpi-value{font-size:24px}.comp-row{grid-template-columns:56px 1fr 72px}}
@media print{.toolbar{position:static;backdrop-filter:none}.tooltip,.view-toggle,#theme-toggle{display:none}.card{break-inside:avoid;box-shadow:none}details{display:block}}
@media (forced-colors:active){.key,.bar100 div,.databar i,.meter i{forced-color-adjust:none}}
</style>
</head>
<body><main>
<header class="masthead"><div><div class="eyebrow">Coding agents · Usage report</div><h1>Conversation metrics</h1><p class="subtitle" id="subtitle"></p></div><div class="head-actions"><span class="pill" id="mode-pill"><i></i>Offline report</span><button type="button" class="ghost" id="theme-toggle" aria-label="Color theme">Theme: System</button></div></header>
<noscript>This report requires JavaScript to display its embedded data and charts. No internet connection is needed.</noscript>
<div class="toolbar" role="region" aria-label="Report filters">
<div class="toolbar-row"><nav class="segmented" aria-label="Reporting window" id="tabs"></nav>
<div class="filters"><label class="field" for="harness-select">Harness <select id="harness-select"></select></label><label class="field" for="tier-select">Tier <select id="tier-select"></select></label><label class="field" for="model-select">Model <select id="model-select"></select></label><label class="field" for="mode-select">Mode <select id="mode-select"></select></label></div></div>
<div class="scope-line"><span id="range"></span><span id="scope-chips"></span></div>
</div>
<div class="notice" id="notice" role="status" hidden></div>
<section class="kpis" id="cards" aria-label="Selected window summary" aria-live="polite"></section>

<section class="section" id="trends-section" aria-labelledby="trends-title">
<div class="section-head"><div><h2 id="trends-title">Usage over time</h2><p id="trend-caption"></p></div>
<div class="filters"><label class="field" for="trend-start">From <input id="trend-start" type="date" required aria-describedby="trend-range-error"></label><label class="field" for="trend-end">To <input id="trend-end" type="date" required aria-describedby="trend-range-error"></label><label class="field" for="granularity-select">Interval <select id="granularity-select"><option value="hourly">Hourly</option><option value="daily" selected>Daily</option><option value="weekly">Weekly</option><option value="monthly">Monthly</option></select></label></div></div>
<p id="trend-range-error" class="range-error" role="status" hidden></p>
<div class="stack">
<article class="card" id="usage-card"><div class="card-head"><div><h3 id="usage-title">Tokens per day</h3><p id="usage-desc"></p></div><div class="filters"><div class="segmented" id="usage-metric" aria-label="Measure"></div><label class="field" for="split-select">Split by <select id="split-select"><option value="none">Nothing</option><option value="model">Model</option><option value="harness">Harness</option><option value="tier">Tier</option><option value="mode">Mode</option></select></label></div></div>
<div class="legend" id="usage-legend"></div><div class="chart" id="usage-chart"></div><div id="usage-table"></div></article>
<article class="card" id="heat-card"><div class="card-head"><div><h3>When you work</h3><p id="heat-desc"></p></div></div><div class="chart" id="heat-chart"></div><div class="heat-legend" id="heat-legend"></div><div id="heat-table"></div></article>
</div></section>

<section class="section" id="performance-section" aria-labelledby="performance-title">
<div class="section-head"><div><h2 id="performance-title">Responsiveness &amp; workload</h2><p>Lines show the selected statistics per period over the same date range · figures below each chart describe the selected window</p></div>
<fieldset class="stat-chips" id="trend-stat-options" style="border:0;padding:0;margin:0"><legend class="field" style="float:left;margin-right:6px;padding-top:5px">Show</legend></fieldset></div>
<div class="grid-2" id="performance"></div></section>

<section class="section" id="cost-section" aria-labelledby="cost-title">
<div class="section-head"><div><h2 id="cost-title">Where tokens and dollars go</h2><p>Selected window · each token is counted once · USD API-equivalent estimates</p></div></div>
<article class="card" id="composition"></article></section>

<section class="section" id="compare-section" aria-labelledby="compare-title">
<div class="section-head"><div><h2 id="compare-title">Compare</h2><p id="comparison-caption"></p></div><div class="filters"><div class="segmented" id="compare-dimension" aria-label="Compare by"></div><label class="field" for="comparison-stat-select">Statistic <select id="comparison-stat-select"></select></label></div></div>
<article class="card"><div class="table-wrap"><table class="compare" id="comparison"></table></div><p class="samples" id="comparison-note"></p></article></section>

<section class="section" id="quality-section" aria-labelledby="quality-title">
<div class="section-head"><div><h2 id="quality-title">Data quality &amp; methodology</h2><p>How complete the selected window is, and how each measurement is defined.</p></div><span class="pill" id="pricing-date"></span></div>
<div class="grid-2"><article class="card"><h3>Sample coverage</h3><p class="samples" style="margin-top:2px">Share of completed turns with each timing signal in the selected window</p><div class="meters" id="coverage-meters"></div><dl class="counts" id="coverage-counts"></dl>
<details class="more"><summary>All coverage counts</summary><div class="table-wrap"><table id="coverage"></table></div></details></article>
<article class="card"><h3>Pricing coverage</h3><div id="unpriced"></div><h3 style="margin-top:16px">Recorded costs &amp; billing units</h3><p class="samples" style="margin-top:2px">Harness totals for the selected window; tier, model, and mode filters do not apply. Recorded USD, credits, and request counters are separate from the API estimate above.</p><div class="table-wrap"><table id="recorded-billing"></table></div>
<details class="more"><summary>Matched OpenRouter prices</summary><p class="samples">Base catalog rates in USD per million tokens; context overrides are applied per request in estimated costs.</p><div class="table-wrap"><table id="router-prices"></table></div></details>
<details class="more"><summary>Parser diagnostics</summary><div class="table-wrap"><table id="diagnostics"></table></div><ul class="warnings" id="warnings"></ul></details></article></div>
<article class="card" style="margin-top:16px"><div class="card-head"><div><h3>Definitions</h3><p>Expand a topic to see exactly how it is measured.</p></div><button type="button" class="ghost" id="expand-defs">Expand all</button></div><div class="defs" id="definitions">
<details><summary>Effective throughput</summary><p>All output tokens, including reasoning, divided by full turn duration. Tool execution and waiting are included.</p></details>
<details><summary>Conversation length</summary><p>Sum of completed turn durations per thread in the selected window. Idle time between turns is excluded; subagents count separately.</p></details>
<details><summary>Tool calls</summary><p>Model-issued function, custom-tool, web-search, and tool-search calls. Outputs and mirrored completion events are excluded; nested commands inside a call are not counted separately.</p></details>
<details><summary>Token totals</summary><p>Total tokens equals input plus output. Input includes cached input and cache writes; output includes reasoning. Cached input counts cache reads and is a subset of input, not an additional token total. The composition chart separates these categories so each token is counted once.</p></details>
<details><summary>Distribution statistics</summary><p>Average is the arithmetic mean. Median is the middle sample, or the average of the two middle samples for an even count. Minimum and maximum are observed extremes. P75, P95, and P99 use nearest rank. All statistics use the same valid samples; each completed turn receives equal weight for timing and throughput.</p></details>
<details><summary>Trend points</summary><p>From and To default to the first recorded activity date and the report cutoff date. Hourly, daily, weekly (Monday start), and monthly points use periods in the report timezone across all available history. Hourly points distinguish repeated daylight saving hours by their UTC offset. The date controls show periods overlapping the chosen range; weekly and monthly statistics include the whole calendar period. The first and current periods can be partial. Statistics are calculated from each period's samples; conversation duration and calls include activity within that period. Token and cost points show recorded usage per period; periods without recorded usage show zero. Performance statistics do not affect the usage chart. Missing performance samples appear as gaps. The activity heatmap sums hourly tokens by weekday and hour of the report timezone across the chosen range. Summary sparklines use hourly points for Today and Yesterday and daily points overlapping other windows. Figures below charts describe the selected reporting window.</p></details>
<details><summary>Window boundaries</summary><p>Tokens and calls use record time; turn metrics use completion time. Today starts at midnight in the report timezone and ends at the report cutoff. Yesterday is the preceding calendar day in that timezone, excluding today's midnight. Rolling windows are exact 24-hour days. The full duration of a turn finishing in the window is assigned to that window.</p></details>
<details><summary>Model attribution</summary><p>Fast usage has a separate model entry with a “-fast” suffix. Requests crossing a published context-pricing threshold have a “-long” suffix, including cached input when selecting the threshold; combined usage has “-fast-long”. Usage below the threshold keeps the model name unless Fast. Models without context pricing and aggregate records without per-request sizes do not receive “-long”. Tokens, calls, timing, and costs are separated by recorded mode, with the Fast premium applied to the underlying model's rates. Tokens and calls use their recorded model, falling back to the turn model. Timing uses the model generating that turn. Conversation duration and tool counts include only that model's activity; a thread using multiple models or modes appears in each, so conversation counts are not additive. If several models generate output within one turn, its timing is listed under “Mixed models (timing)”. If one model uses several modes within a turn, its timing is listed under “Mixed modes (timing)” because separate durations cannot be recovered. If one model in one mode crosses context thresholds within a turn, its timing is listed under “Mixed contexts (timing)”. Tool calls follow the context class of matching model and mode usage in their turn; ambiguous calls are listed under “Mixed contexts (tools)”. Mode and tier totals retain this activity once. When the usage chart is split by model, models beyond the seven with the most tokens across the report share the “Other” color, and are combined when several appear together.</p></details>
<details><summary>Model tiers</summary><p>Budget: Luna, Terra, GPT mini and nano models, Spark, codex-auto-review, and Claude Haiku. Medium: Sol, GPT-5.4, GPT-5.5, and Claude Sonnet. High: Astra, Claude Opus, Fable, and Mythos. Models outside these groups are Unclassified. Tier metrics are calculated from underlying activity, with each conversation counted once per tier. Turns using several models in the same tier retain their timing in that tier; turns spanning tiers have timing under “Mixed tiers (timing)”. Per-model pricing and the Fast premium still apply. The model selector and comparison table show entries with recorded tokens in the selected window, harness, tier, and mode. Zero-token entries, including shared timing and tool-call buckets, remain included in aggregate totals and coverage.</p></details>
<details><summary>Mode attribution</summary><p>Logged service tier “default” is Normal; “priority” or “fast” is Fast. Settings persist until changed. Per the selected assumption, unknown mode—including missing evidence, explicit null, and “auto”—is counted as Normal in all metrics and costs. Other explicit tiers have their own bucket. Tokens and calls follow their recorded tier or the latest logged settings. This combines logged mode with the Normal assumption; a backend fallback cannot be detected without a response tier. A turn with usage in several modes has its timing under “Mixed modes (timing)” because separate durations are unavailable. Conversation durations and calls include only activity attributed to the selected mode.</p></details>
<details><summary>Harness coverage</summary><p>Claude Code reads project JSONL files, Copilot reads CLI session events, and OpenCode reads its message and tool tables. T3 Code links saved native sessions to the Codex, Claude Code, and OpenCode readers; linked sessions count once under T3 when selected. Missing native logs and other T3 providers are outside coverage. Copilot shutdown-only totals are assigned to shutdown, which does not establish when individual requests occurred. Copilot does not persist per-request usage, so its throughput uses the output token count saved with each assistant message. Timing samples require logged timing evidence.</p></details>
<details><summary>Coverage</summary><p id="coverage-def">Logs in the listed input directories include archived and active sessions. Copies sharing a conversation ID are merged; repeated usage responses, tool calls, and turn completions are counted once. Active logs are read while they may still be growing; the report cutoff limits included activity. Unfinished turns contribute recorded tokens and calls, with completion timings excluded. An older-window label does not imply a complete year of available history. Missing durations are excluded, so conversation duration can be partial.</p></details>
<details><summary>Cost estimate</summary><p>Current standard API rates are applied to every historical window, with an assumed 50% premium on OpenAI token categories recorded in Fast mode. Normal, including assumed Normal activity, uses base rates. Other explicit tiers also use base rates; their actual premiums are unknown. Codex 5.3 Spark uses GPT-5.4-mini rates and codex-auto-review uses GPT-5.6-luna rates as user-selected proxies, not published prices for those models. These are API-equivalent estimates, not subscription bills. Claude cache writes include separate 5-minute and 1-hour rates when logged; Claude Fast uses its published model-specific premium. Unpublished Fast rates remain unpriced. OpenRouter catalog rates price matched models lacking an embedded rate table. OpenCode input/cache and output/reasoning counters are normalized to avoid overlap. Recorded harness costs and billing units are shown separately. Subscription charges, tool fees, and regional uplifts are excluded. Reasoning is split out of output; cache reads/writes are split out of input. OpenAI long-context rates apply above 272,000 input tokens where published. Older Claude Sonnet rates change above 200,000; Claude 4.6+ uses standard rates throughout its context window. OpenRouter context thresholds come from the catalog. Aggregate counters without per-request sizes assume normal-context rates, including the base OpenRouter rates without context overrides. Recorded speed-mode premiums still apply where known; actual long-context costs may be higher. Blended cost per million tokens divides a category's estimated cost by its priced and unpriced tokens.</p></details>
</div>
<details class="more"><summary>Exact metrics for every window</summary><div class="table-wrap"><table id="all-metrics"></table></div></details>
<p class="sources" id="sources"></p></article>
</section>
<footer class="footer"><span id="footer"></span><span>Generated locally · No conversation content embedded</span></footer>
</main><div class="tooltip" id="tooltip" role="tooltip" hidden></div>
<script type="application/json" id="report-data">__REPORT_DATA__</script>
<script>
'use strict';
const $=id=>document.getElementById(id);
// Static reports embed their data; the live server supplies a source that loads it on demand
const source=window.reportSource||embeddedSource();
let data,firstDate,cutoffDate,modelOrder=[];
const SLOTS=7;
const metricDefs=[
    {key:'throughput',title:'Effective throughput',short:'Throughput',unit:'tok/s',desc:'Output tokens over full turn duration',sample:'turn',chart:v=>v,axis:v=>compact(v),format:v=>`${v>=1000?compact(v):number(v,v>=100?0:1)} tok/s`},
    {key:'length',title:'Conversation length',short:'Length',unit:'min',desc:'Active duration per conversation, idle time excluded',sample:'conversation',chart:v=>v/60,axis:v=>`${trim(v)}m`,format:v=>duration(v)},
    {key:'tools',title:'Tool calls per conversation',short:'Calls / conv.',unit:'calls',desc:'Model-issued calls per conversation',sample:'conversation',chart:v=>v,axis:v=>trim(v),format:v=>number(v,1)}];
const chartStats=[['avg','Average','var(--s1)'],['median','Median','var(--s2)'],['p75','P75','var(--s3)'],['p95','P95','var(--s7)'],['p99','P99','var(--s5)']];
const statDefs=[['median','Median'],['avg','Average'],['p75','P75'],['p95','P95'],['p99','P99'],['min','Minimum'],['max','Maximum']];
const distOrder=[['min','Min'],['median','Median'],['avg','Avg'],['p75','P75'],['p95','P95'],['p99','P99'],['max','Max']];
const state={window:0,harness:'',tier:'',model:'',mode:'',granularity:'daily',start:'',end:'',measure:'tokens',split:'none',stats:new Set(['median','p95']),compare:'model',statistic:'median',sort:'cost',dir:-1,usageTable:false,heatTable:false};

// Formatting
const number=(n,d=2)=>Number(n).toLocaleString('en-US',{maximumFractionDigits:d});
const integer=n=>Math.round(Number(n)).toLocaleString('en-US');
const compact=n=>Number(n).toLocaleString('en-US',{notation:'compact',maximumFractionDigits:1});
const trim=n=>Number(n).toLocaleString('en-US',{maximumFractionDigits:Math.abs(n)<10?2:1});
const money=n=>{n=Number(n);return n>0&&n<.005?'<$0.01':n.toLocaleString('en-US',{style:'currency',currency:'USD',minimumFractionDigits:2,maximumFractionDigits:2})};
const moneyAxis=n=>n>=1000?'$'+compact(n):n>=10?'$'+number(n,0):n>=1||n<=0?'$'+number(n,2):'$'+n.toLocaleString('en-US',{maximumSignificantDigits:2});
const moneyRate=n=>n.toLocaleString('en-US',{style:'currency',currency:'USD',minimumFractionDigits:2,maximumFractionDigits:n<1?3:2});
const percent=(v,total)=>total>0?(v/total*100<1&&v>0?'<1%':`${Math.round(v/total*100)}%`):'—';
function duration(s){if(s===null||s===undefined)return '—';s=Math.round(s);if(s<60)return `${s}s`;const m=Math.floor(s/60);if(m<60)return `${m}m ${String(s%60).padStart(2,'0')}s`;const h=Math.floor(m/60);return `${number(h,0)}h ${String(m%60).padStart(2,'0')}m`}
const day=(iso,opts)=>new Date(iso.slice(0,10)+'T00:00:00Z').toLocaleDateString('en-US',{...opts,timeZone:'UTC'});
const fmtDate=value=>new Date(value).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:data.timezone});
const fmtTime=value=>new Date(value).toLocaleString('en-US',{dateStyle:'medium',timeStyle:'short',timeZone:data.timezone});
function periodLabel(period,granularity,long){
    const label=period.label;
    if(granularity==='hourly'){const hour=label.slice(11,16),offset=period.repeat&&label.slice(16)?` (UTC${label.slice(16)})`:'';return long?`${day(label,{weekday:'short',month:'short',day:'numeric',year:'numeric'})} · ${hour}${offset}`:`${day(label,{month:'short',day:'numeric'})} ${hour}`}
    if(granularity==='weekly')return long?`Week of ${day(label,{month:'short',day:'numeric',year:'numeric'})}`:day(label,{month:'short',day:'numeric'});
    if(granularity==='monthly')return day(label,{month:long?'long':'short',year:'numeric'});
    return long?day(label,{weekday:'short',month:'short',day:'numeric',year:'numeric'}):day(label,{month:'short',day:'numeric'});
}

// DOM helpers
const text=(tag,value,cls)=>{const e=document.createElement(tag);if(value!==undefined&&value!==null)e.textContent=value;if(cls)e.className=cls;return e};
const svgNode=(tag,attrs={},value)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,v] of Object.entries(attrs))e.setAttribute(key,String(v));if(value!==undefined)e.textContent=value;return e};
const key=(color,line)=>{const k=text('i','','key'+(line?' line':''));k.style.background=color;return k};
function table(target,headers,rows,options={}){
    target.replaceChildren();const head=document.createElement('thead'),tr=document.createElement('tr');
    headers.forEach((h,i)=>tr.append(text('th',h,i&&!options.textColumns?.includes(i)?'num':'')));head.append(tr);target.append(head);
    const body=document.createElement('tbody');
    rows.forEach(row=>{const r=document.createElement('tr');row.forEach((v,i)=>r.append(v instanceof Node?(()=>{const td=text('td','',i&&!options.textColumns?.includes(i)?'num':'');td.append(v);return td})():text('td',v,i&&!options.textColumns?.includes(i)?'num':'')));body.append(r)});
    target.append(body);
    if(options.foot){const foot=document.createElement('tfoot'),r=document.createElement('tr');options.foot.forEach((v,i)=>r.append(text('td',v,i?'num':'')));foot.append(r);target.append(foot)}
    if(!rows.length){const r=document.createElement('tr'),td=text('td',options.empty||'Nothing recorded.','dim');td.colSpan=headers.length;r.append(td);body.append(r)}
}

// Tooltip: values lead, labels follow
const tooltip=$('tooltip');
function showTip(x,y,title,rows,foot){
    tooltip.replaceChildren(text('div',title,'tip-title'));
    for(const row of rows){const r=text('div','','tip-row');r.append(row.color?key(row.color,row.line):text('i'),text('strong',row.value),text('span',row.name));tooltip.append(r)}
    if(foot)tooltip.append(text('div',foot,'tip-foot'));
    tooltip.hidden=false;const w=tooltip.offsetWidth,h=tooltip.offsetHeight;
    tooltip.style.left=`${x+16+w>innerWidth-8?Math.max(8,x-16-w):x+16}px`;tooltip.style.top=`${Math.max(8,Math.min(y-h/2,innerHeight-h-8))}px`;
}
const hideTip=()=>{tooltip.hidden=true};

// Charts re-render on resize so text stays at its real size
const mounted=new Map();
const resizer=new ResizeObserver(entries=>{for(const entry of entries){const m=mounted.get(entry.target);const width=Math.round(entry.contentRect.width);if(m&&width&&Math.abs(width-m.width)>2){m.width=width;entry.target.replaceChildren(m.draw(width))}}});
function mount(container,draw){for(const node of mounted.keys())if(!node.isConnected&&node!==container){mounted.delete(node);resizer.unobserve(node)}const width=Math.round(container.clientWidth)||600;mounted.set(container,{draw,width});container.replaceChildren(draw(width));resizer.observe(container)}
function niceStep(raw){const power=10**Math.floor(Math.log10(raw)),f=raw/power;return (f<=1?1:f<=2?2:f<=2.5?2.5:f<=5?5:10)*power}
function ticks(max,count){if(!(max>0))max=1;const step=niceStep(max/count),top=Math.ceil(max/step-1e-9)*step,result=[];for(let i=0;i*step<=top+step/2;i++)result.push(i*step);return result}
const roundedTop=(x,y0,y1,w,r)=>r>0?`M${x},${y0}V${y1+r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1+r}V${y0}Z`:`M${x},${y0}V${y1}H${x+w}V${y0}Z`;

// Time series: stacked columns or lines, index-banded, with a crosshair readout
function timeChart(o){return width=>{
    const n=o.periods.length,spark=!!o.spark,height=o.height||(spark?44:240);
    if(!n||!o.series.length)return text('div',o.empty||'No data in this range.','empty');
    const sums=o.stacked?o.periods.map((_,i)=>o.series.reduce((t,s)=>t+(s.values[i]||0),0)):null;
    const all=o.stacked?sums:o.series.flatMap(s=>s.values).filter(v=>v!==null&&v!==undefined);
    if(!spark&&!all.some(v=>v>0)&&!o.stacked&&!all.length)return text('div',o.empty||'No valid samples in this range.','empty');
    const max=Math.max(0,...all),yt=ticks(max,spark?1:4),top=spark?max||1:yt[yt.length-1];
    const labels=yt.map(o.axis||compact),left=spark?1:Math.max(...labels.map(l=>l.length))*6.6+12,right=width-(spark?4:6),y0=spark?4:8,y1=height-(spark?2:24);
    const band=(right-left)/n,x=i=>left+band*(i+.5),y=v=>y1-(v/top)*(y1-y0);
    const svg=svgNode('svg',{width,height,viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':o.label});
    svg.append(svgNode('title',{},o.label));
    if(!spark){
        yt.forEach((t,i)=>{svg.append(svgNode('line',{x1:left,x2:right,y1:y(t),y2:y(t),style:`stroke:var(${i?'--grid':'--axis'})`,'shape-rendering':'crispEdges'}),svgNode('text',{x:left-8,y:y(t)+3.5,'text-anchor':'end'},labels[i]))});
        const count=Math.max(2,Math.min(n,Math.floor((right-left)/(o.granularity==='hourly'?110:84)))),seen=new Set();
        for(let k=0;k<count;k++){const i=n===1?0:Math.round(k*(n-1)/(count-1));if(seen.has(i))continue;seen.add(i);
            const anchor=n===1?'middle':k===0&&x(i)-left<30?'start':k===count-1&&right-x(i)<30?'end':'middle';
            svg.append(svgNode('text',{x:anchor==='start'?Math.max(left,x(i)-band/2):anchor==='end'?Math.min(right,x(i)+band/2):x(i),y:height-6,'text-anchor':anchor},periodLabel(o.periods[i],o.granularity,false)))}
    }
    const hover=svgNode('g',{visibility:'hidden','pointer-events':'none'});
    if(o.kind==='bars'){
        const gap=band>=5?Math.max(2,band*.28):0,bw=Math.max(1,Math.min(24,band-gap)),radius=bw>=8?4:0,base=new Array(n).fill(0);
        const hl=svgNode('rect',{y:y0,height:y1-y0,width:Math.max(bw+6,band),rx:4,style:'fill:var(--surface-3);opacity:.7'});hover.append(hl);svg.insertBefore(hover,svg.firstChild.nextSibling);
        const topIndex=o.periods.map((_,i)=>{let t=-1;o.series.forEach((s,j)=>{if(s.values[i]>0)t=j});return t});
        o.series.forEach((s,j)=>{let d='';s.values.forEach((v,i)=>{if(!(v>0))return;let ya=y(base[i]),yb=y(base[i]+v);const isTop=topIndex[i]===j;
            if(base[i]>0&&ya-yb>3&&bw>=4)ya-=1;if(!isTop&&ya-yb>3&&bw>=4)yb+=1;
            d+=roundedTop(x(i)-bw/2,ya,Math.min(yb,ya-(spark?.5:1)),bw,isTop?Math.min(radius,(ya-yb)/2):0);base[i]+=v});
            svg.append(svgNode('path',{d,style:`fill:${s.color}`}))});
        hover.dataset.kind='bars';hover.update=i=>{hl.setAttribute('x',x(i)-Math.max(bw+6,band)/2)};
    }else{
        const markers=[];
        o.series.forEach(s=>{let d='',run=0;const solo=[];
            s.values.forEach((v,i)=>{if(v===null||v===undefined){if(run===1)solo.push(i-1);run=0;return}d+=`${run?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`;run++});
            if(run===1)solo.push(n-1);
            if(o.area&&s.values.every(v=>v!==null))svg.append(svgNode('path',{d:`${d}L${x(n-1)},${y1}L${x(0)},${y1}Z`,style:`fill:${s.color};opacity:.1`}));
            svg.append(svgNode('path',{d,fill:'none','stroke-width':spark?1.5:2,'stroke-linejoin':'round','stroke-linecap':'round',style:`stroke:${s.color}`}));
            for(const i of solo)svg.append(svgNode('circle',{cx:x(i),cy:y(s.values[i]),r:spark?2:3,style:`fill:${s.color}`}));
            const m=svgNode('circle',{r:4,'stroke-width':2,style:`fill:${s.color};stroke:var(--surface)`});markers.push([m,s]);});
        const vl=svgNode('line',{y1:y0,y2:y1,style:'stroke:var(--axis)','shape-rendering':'crispEdges'});hover.append(vl,...markers.map(([m])=>m));svg.append(hover);
        hover.update=i=>{vl.setAttribute('x1',x(i));vl.setAttribute('x2',x(i));for(const [m,s] of markers){const v=s.values[i];m.setAttribute('visibility',v===null||v===undefined?'hidden':'visible');if(v!==null&&v!==undefined){m.setAttribute('cx',x(i));m.setAttribute('cy',y(v))}}};
    }
    const overlay=svgNode('rect',{x:left,y:0,width:right-left,height:y1,fill:'transparent'});svg.append(overlay);
    let current=-1;
    const show=(i,cx,cy)=>{current=i;hover.setAttribute('visibility','visible');hover.update(i);
        const rows=o.series.map(s=>({color:s.color,line:o.kind!=='bars',value:s.values[i]===null||s.values[i]===undefined?'—':o.format(s.values[i]),name:s.name,raw:s.values[i]}));
        const shown=o.stacked?rows.filter(r=>r.raw>0).reverse():rows;
        if(o.stacked&&o.series.length>1)shown.push({value:o.format(sums[i]),name:'Total'});
        if(!shown.length)shown.push({value:o.format(0),name:o.series.length>1?'Total':o.series[0].name});
        showTip(cx,cy,periodLabel(o.periods[i],o.granularity,true),shown,o.note?o.note(i):'')};
    const fromPointer=e=>{const r=svg.getBoundingClientRect();return Math.max(0,Math.min(n-1,Math.floor((e.clientX-r.left-left)/band)))};
    overlay.addEventListener('pointermove',e=>show(fromPointer(e),e.clientX,e.clientY));
    overlay.addEventListener('pointerleave',()=>{hover.setAttribute('visibility','hidden');hideTip()});
    if(!spark){svg.setAttribute('tabindex','0');svg.setAttribute('aria-label',`${o.label}. Use arrow keys to read values.`);
        const atKey=i=>{const r=svg.getBoundingClientRect();show(i,r.left+x(i),r.top+y0+20)};
        svg.addEventListener('focus',()=>atKey(current>=0?current:n-1));svg.addEventListener('blur',()=>{hover.setAttribute('visibility','hidden');hideTip()});
        svg.addEventListener('keydown',e=>{const step={ArrowLeft:-1,ArrowRight:1,Home:-n,End:n,PageUp:-10,PageDown:10}[e.key];if(e.key==='Escape'){hideTip();return}if(step===undefined)return;e.preventDefault();atKey(Math.max(0,Math.min(n-1,(current<0?n-1:current)+step)))});}
    return svg;
}}

// Scope helpers: every view reads from the same filter selection
const scopeGroup=s=>{let g=s.harness?data.by_harness[s.harness]:data;if(g&&s.tier)g=g.by_tier[s.tier];if(g&&s.mode)g=g.by_mode[s.mode];return g||null};
const scopeWindows=s=>{const g=scopeGroup(s);return g?(s.model?g.by_model[s.model]:g.windows)||null:null};
const scopeTrends=(s,granularity)=>{const g=scopeGroup(s);const t=g&&(s.model?g.by_model_trends[s.model]:g.trends);return t?t[granularity]:null};
const filters=()=>({harness:state.harness,tier:state.tier,mode:state.mode,model:state.model});
const harnessGroup=()=>state.harness?data.by_harness[state.harness]:data;
const tierGroup=()=>state.tier?harnessGroup().by_tier[state.tier]:harnessGroup();
const activeModels=()=>state.mode?tierGroup().by_mode[state.mode].by_model:tierGroup().by_model;
const activeWindows=()=>scopeWindows(filters());
const hasActivity=windows=>windows&&windows.some(w=>w.conversations>0||w.total_tokens>0);
const longest=windows=>windows[windows.length-1];

function embeddedSource(){return {live:false,load:async()=>JSON.parse($('report-data').textContent),series:async(scope,granularity,start,end)=>{
    const daily=data.trend_periods.daily,firstDay=daily.find(p=>p.start.slice(0,10)===start)||daily[0],rangeStart=Date.parse(firstDay.start),trends=scopeTrends(scope,granularity);
    const entries=data.trend_periods[granularity].map((period,index)=>({period,index})).filter(({period})=>period.start.slice(0,10)<=end&&(Date.parse(period.end)>rangeStart||Date.parse(period.end)===rangeStart&&!period.end_exclusive));
    return {periods:entries.map(e=>e.period),points:entries.map(({index})=>trends?.[index]??null)}}}}
// Periods overlapping the chosen dates; repeated daylight-saving hours keep their UTC offset in labels
async function series(scope,granularity,start,end){
    if(start<firstDate)start=firstDate;if(end>cutoffDate)end=cutoffDate;
    if(start>end)return {periods:[],points:[]};
    const result=await source.series(scope,granularity,start,end),counts=new Map(),hour=p=>p.label.slice(0,16);
    if(granularity==='hourly'){for(const p of result.periods)counts.set(hour(p),(counts.get(hour(p))||0)+1);for(const p of result.periods)p.repeat=counts.get(hour(p))>1}
    return result;
}
// Charts load asynchronously; only the latest request for each view may draw
const tickets={};
async function load(name,hosts,request,quiet){
    const id=tickets[name]=(tickets[name]||0)+1;
    for(const h of hosts){h.setAttribute('aria-busy','true');if(!h.firstChild&&!quiet)h.replaceChildren(text('div','Loading…','empty'))}
    try{const result=await request();return id===tickets[name]?result:null}
    catch(error){if(id===tickets[name])for(const h of hosts){mounted.delete(h);resizer.unobserve(h);if(quiet)h.remove();else h.replaceChildren(text('div',error.message||'Unable to load this chart.','empty'))}return null}
    finally{if(id===tickets[name])for(const h of hosts)h.removeAttribute('aria-busy')}
}
const measureValue=(point,measure)=>point?(measure==='cost'?Number(point.cost||0):point.total_tokens):0;

// Split entities keep their color across window and date changes
// Models are ordered once by all-history tokens, so filters never repaint a model
const entityNames={model:()=>modelOrder,harness:()=>Object.keys(data.by_harness),tier:()=>Object.keys(data.by_tier),mode:()=>Object.keys(data.by_mode)};
const entityColor=(dim,name)=>{const i=entityNames[dim]().indexOf(name);return i>=0&&i<SLOTS?`var(--s${i+1})`:'var(--other)'};
function splitEntities(split){
    const base=filters();
    if(split==='none')return [{name:'All activity',scope:base,color:'var(--s1)'}];
    const available=split==='model'?activeModels():split==='harness'?data.by_harness:split==='tier'?harnessGroup().by_tier:harnessGroup().by_mode;
    const names=[...entityNames[split]().filter(n=>n in available),...Object.keys(available).filter(n=>!entityNames[split]().includes(n))];
    return names.filter(name=>!state[split]||name===state[split]).map(name=>({name,scope:{...base,[split]:name},color:entityColor(split,name)}));
}

// Usage over time
function usageSeries(entities,results,periods,measure){
    const series=entities.map((e,k)=>{const points=results[k].points;return {name:e.name,color:e.color,values:points.map(p=>measureValue(p,measure)),partial:points.map(p=>!!p?.partial_cost)}}).filter(s=>state.split==='none'||s.values.some(v=>v>0));
    const named=series.filter(s=>s.color!=='var(--other)'),rest=series.filter(s=>s.color==='var(--other)');
    if(rest.length>1){const kept=named;kept.push({name:`Other (${rest.length})`,color:'var(--other)',values:periods.map((_,i)=>rest.reduce((t,s)=>t+s.values[i],0)),partial:periods.map((_,i)=>rest.some(s=>s.partial[i]))});return kept}
    return series;
}
async function usage(){
    const g=state.granularity,measure=state.measure,isCost=measure==='cost',entities=splitEntities(state.split),start=state.start,end=state.end;
    const per={hourly:'hour',daily:'day',weekly:'week',monthly:'month'}[g];
    $('usage-title').textContent=`${isCost?'Estimated cost':'Tokens'} per ${per}`;
    $('usage-desc').textContent=isCost?'API-equivalent USD estimate per period at current rates':'Input plus output tokens per period · includes cached input and reasoning';
    const results=await load('usage',[$('usage-chart')],()=>Promise.all(entities.map(e=>series(e.scope,g,start,end))));
    if(!results)return;
    const periods=results[0]?.periods||[],lines=usageSeries(entities,results,periods,measure),total=lines.reduce((t,s)=>t+s.values.reduce((a,b)=>a+b,0),0),fmt=isCost?money:v=>integer(v);
    $('usage-legend').replaceChildren(...(lines.length>1?lines.map(s=>{const l=text('span',s.name);l.prepend(key(s.color));return l}):[]));
    const partial=periods.map((_,i)=>lines.some(s=>s.partial[i]));
    mount($('usage-chart'),timeChart({periods,granularity:g,series:lines,kind:'bars',stacked:true,label:`${$('usage-title').textContent}, ${start} to ${end}`,format:isCost?money:v=>integer(v),axis:isCost?moneyAxis:compact,note:i=>partial[i]?'Partial estimate · some usage lacks a rate':'',empty:'No recorded usage in this range.'}));
    const peak=lines.length?periods.map((_,i)=>lines.reduce((t,s)=>t+s.values[i],0)).reduce((best,v,i,a)=>v>a[best]?i:best,0):-1;
    $('trend-caption').textContent=`${day(start,{month:'short',day:'numeric',year:'numeric'})} – ${day(end,{month:'short',day:'numeric',year:'numeric'})} · ${fmt(total)} ${isCost?'estimated':'tokens'} across ${integer(periods.length)} ${per}${periods.length===1?'':'s'}${peak>=0&&total>0?` · busiest ${per}: ${periodLabel(periods[peak],g,true)}`:''}`;
    const tableHost=$('usage-table');tableHost.replaceChildren(toggleButton('usageTable',()=>usage()));
    if(state.usageTable){const wrap=text('div','','data-table'),t=document.createElement('table');table(t,['Period',...lines.map(s=>s.name),...(lines.length>1?['Total']:[])],periods.map((p,i)=>[periodLabel(p,g,true),...lines.map(s=>fmt(s.values[i])),...(lines.length>1?[fmt(lines.reduce((a,s)=>a+s.values[i],0))]:[])]));wrap.append(t);tableHost.append(wrap)}
}
function toggleButton(flag,rerender){const b=text('button',state[flag]?'Hide data table':'Show data table','link-btn view-toggle');b.type='button';b.setAttribute('aria-expanded',state[flag]);b.style.marginTop='10px';b.addEventListener('click',()=>{state[flag]=!state[flag];rerender()});return b}

// Weekday × hour heatmap of hourly tokens in the chosen range
const weekdays=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
async function heatmap(){
    const result=await load('heatmap',[$('heat-chart')],()=>series(filters(),'hourly',state.start,state.end));
    if(!result)return;
    const grid=weekdays.map(()=>new Array(24).fill(0));
    for(const [index,period] of result.periods.entries()){const v=result.points[index]?.total_tokens||0;if(!v)continue;const d=(new Date(period.label.slice(0,10)+'T00:00:00Z').getUTCDay()+6)%7;grid[d][Number(period.label.slice(11,13))]+=v}
    const max=Math.max(0,...grid.flat()),total=grid.flat().reduce((a,b)=>a+b,0);
    let best=[0,0];grid.forEach((row,d)=>row.forEach((v,h)=>{if(v>grid[best[0]][best[1]])best=[d,h]}));
    $('heat-desc').textContent=total?`Tokens by weekday and hour (${data.timezone}) · peak ${weekdays[best[0]]} ${String(best[1]).padStart(2,'0')}:00 with ${percent(grid[best[0]][best[1]],total)} of usage`:'Tokens by weekday and hour of the report timezone';
    const level=v=>v<=0?0:Math.min(7,1+Math.floor(v/max*7*.9999));
    mount($('heat-chart'),width=>{
        if(!total)return text('div','No recorded usage in this range.','empty');
        const left=38,cw=Math.max(6,(width-left)/24),ch=Math.min(26,Math.max(14,cw*.8)),height=7*ch+22;
        const svg=svgNode('svg',{width,height,viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':'Token usage by weekday and hour'});
        grid.forEach((row,d)=>{svg.append(svgNode('text',{x:0,y:d*ch+ch/2+4},weekdays[d]));row.forEach((v,h)=>{
            const cell=svgNode('rect',{x:left+h*cw+1,y:d*ch+1,width:cw-2,height:ch-2,rx:3,style:`fill:var(--q${level(v)})`});
            const label=`${weekdays[d]} ${String(h).padStart(2,'0')}:00`;
            const show=e=>{const r=cell.getBoundingClientRect();cell.style.stroke='var(--ink)';showTip(e.clientX||r.right,e.clientY||r.top,label,[{value:integer(v),name:'tokens'},{value:percent(v,total),name:'of usage in range'}])};
            cell.addEventListener('pointerenter',show);cell.addEventListener('pointermove',show);cell.addEventListener('pointerleave',()=>{cell.style.stroke='';hideTip()});svg.append(cell)})});
        for(let h=0;h<24;h+=3)svg.append(svgNode('text',{x:left+h*cw+cw/2,y:height-4,'text-anchor':'middle'},`${String(h).padStart(2,'0')}h`));
        return svg});
    $('heat-legend').replaceChildren(...(total?[text('span','Less'),...[0,1,2,3,4,5,6,7].map(i=>{const k=text('i');k.style.background=`var(--q${i})`;return k}),text('span',`More · max ${compact(max)} tokens per cell`)]:[]));
    const host=$('heat-table');host.replaceChildren(...(total?[toggleButton('heatTable',heatmap)]:[]));
    if(state.heatTable&&total){const wrap=text('div','','data-table'),t=document.createElement('table');table(t,['Day',...Array.from({length:24},(_,h)=>String(h).padStart(2,'0'))],grid.map((row,d)=>[weekdays[d],...row.map(v=>v?compact(v):'·')]));wrap.append(t);host.append(wrap)}
}

// Performance distributions
async function performance(w){
    const granularity=state.granularity,shown=chartStats.filter(([s])=>state.stats.has(s)),charts=[];
    $('performance').replaceChildren();
    for(const def of metricDefs){
        const m=w.metrics[def.key],card=text('article','','card'),head=text('div','','card-head'),h=text('div');h.append(text('h3',def.title),text('p',def.desc));head.append(h);card.append(head);
        const headline=text('div','','headline');headline.append(text('strong',m.median===null?'—':def.format(m.median)),text('span',m.median===null?`No samples · ${data.windows[state.window].label}`:`median · P95 ${def.format(m.p95)} · ${data.windows[state.window].label}`));card.append(headline);
        if(shown.length>1){const legend=text('div','','legend');shown.forEach(([,label,color])=>{const l=text('span',label);l.prepend(key(color,true));legend.append(l)});card.append(legend)}
        const chart=text('div','','chart');card.append(chart);
        const dist=text('dl','','dist');for(const [stat,label] of distOrder){const d=text('div');d.append(text('dt',label),text('dd',m[stat]===null?'—':def.format(m[stat])));dist.append(d)}
        card.append(dist,text('p',`${integer(m.count)} valid ${def.sample} sample${m.count===1?'':'s'} in ${data.windows[state.window].label.toLowerCase()}`,'samples'));
        $('performance').append(card);charts.push([chart,def]);
    }
    if(!shown.length){for(const [chart] of charts)mount(chart,()=>text('div','Select a statistic above to draw the trend.','empty'));return}
    const result=await load('performance',charts.map(([chart])=>chart),()=>series(filters(),granularity,state.start,state.end));
    if(!result)return;
    const {periods,points}=result;
    for(const [chart,def] of charts){
        const lines=shown.map(([stat,label,color])=>({name:label,color,values:points.map(p=>{const v=p?.[def.key]?.[stat];return v===null||v===undefined?null:def.chart(v)})}));
        mount(chart,timeChart({periods,granularity,series:lines,kind:'lines',height:200,label:`${def.title} by ${granularity} period`,axis:def.axis,format:v=>def.format(def.key==='length'?v*60:v),note:i=>{const c=points[i]?.[def.key]?.count;return c?`${integer(c)} ${def.sample} sample${c===1?'':'s'}`:'No samples'},empty:'No valid samples in this range.'}));
    }
}

// Summary tiles with sparklines over the selected window
// Today and Yesterday use hourly points; longer windows use the days they overlap
function windowSpark(w){
    const hourly=['Today','Yesterday'].includes(data.windows[state.window].label),granularity=hourly?'hourly':'daily',start=w.start.slice(0,10);
    return {granularity,request:()=>series(filters(),granularity,start,hourly?start:w.end.slice(0,10))};
}
async function cards(w){
    const tools=w.metrics.tools,activeShare=w.conversations?w.tool_calls/w.conversations:0;
    const items=[
        {label:'Estimated cost',value:money(w.cost),note:w.partial_cost?'Partial · some usage unpriced':'USD · API-equivalent',spark:p=>p?Number(p.cost):0,fmt:money},
        {label:'Total tokens',value:compact(w.total_tokens),note:`${integer(w.total_tokens)} tokens`,spark:p=>p?p.total_tokens:0,fmt:integer,split:[['Input',w.input_tokens],['Output',w.output_tokens],['Cached input',w.cached_input_tokens]]},
        {label:'Conversations',value:integer(w.conversations),note:'Active threads incl. subagents',spark:p=>p?p.tools.count:0,fmt:integer},
        {label:'Active time',value:duration(w.active_seconds),note:'Completed turn durations, summed',spark:p=>p&&p.length.avg!==null?p.length.avg*p.length.count:0,fmt:duration},
        {label:'Tool calls',value:integer(w.tool_calls),note:w.conversations?`${number(activeShare,1)} per conversation`:'Model-issued calls',spark:p=>p&&p.tools.avg!==null?Math.round(p.tools.avg*p.tools.count):0,fmt:integer}];
    $('cards').replaceChildren();const sparks=[];
    for(const item of items){
        const c=text('article','','card kpi');c.append(text('div',item.label,'kpi-label'),text('div',item.value,'kpi-value'),text('div',item.note,'kpi-note'));
        if(item.split){const dl=text('dl','','kpi-split');for(const [name,count] of item.split){const row=text('div');row.append(text('dt',name),text('dd',compact(count)));row.title=`${name}: ${integer(count)} tokens`;dl.append(row)}c.append(dl)}
        const spark=text('div','','chart spark');c.append(spark);$('cards').append(c);sparks.push([spark,item]);
    }
    const s=windowSpark(w),result=await load('cards',sparks.map(([spark])=>spark),s.request,true);
    if(!result)return;
    for(const [spark,item] of sparks){
        if(result.periods.length>1)mount(spark,timeChart({periods:result.periods,granularity:s.granularity,series:[{name:item.label,color:'var(--accent)',values:result.points.map(item.spark)}],kind:'lines',area:true,spark:true,label:`${item.label} per ${s.granularity==='hourly'?'hour':'day'}`,format:item.fmt}));
        else spark.remove();
    }
}

// Token and cost composition
function composition(w){
    const host=$('composition'),cats=w.categories.map((c,i)=>({...c,cost:Number(c.cost),color:`var(--s${i+1})`,ink:i>=5?`var(--on-s${i+1})`:'#0b0b0b'})),totalCost=Number(w.cost);
    host.replaceChildren();
    const head=text('div','','card-head'),h=text('div');h.append(text('h3','Token and cost composition'),text('p',`${data.windows[state.window].label} · share of total by category${w.partial_cost?' · cost is a partial estimate':''}`));head.append(h);host.append(head);
    const legend=text('div','','legend');cats.forEach(c=>{const l=text('span',c.name);l.prepend(key(c.color));legend.append(l)});host.append(legend);
    const bars=text('div','','composition');host.append(bars);
    if(!w.total_tokens){bars.append(text('div','No recorded usage in this window.','empty'));return}
    for(const [label,field,total,fmt] of [['Tokens','tokens',w.total_tokens,compact],['Cost','cost',totalCost,money]]){
        const row=text('div','','comp-row'),bar=text('div','','bar100');bar.setAttribute('role','img');
        bar.setAttribute('aria-label',`${label}: `+cats.map(c=>`${c.name} ${percent(c[field],total)}`).join(', '));
        row.append(text('span',label),bar,text('span',fmt(total)));bars.append(row);
        mount(bar,width=>{const frag=document.createDocumentFragment();if(!(total>0)){const empty=text('div','No priced usage','dim');empty.style.flex='1';empty.style.background='var(--surface-2)';frag.append(empty);return frag}
            cats.filter(c=>c[field]>0).forEach(c=>{const seg=text('div'),share=c[field]/total,label=percent(c[field],total);seg.style.flex=`${share} 1 0`;seg.style.background=c.color;seg.style.color=c.ink;if(share*width>=label.length*7+14)seg.textContent=label;
                const show=e=>showTip(e.clientX,e.clientY,c.name,[{color:c.color,value:field==='cost'?money(c.cost):integer(c.tokens),name:field},{value:label,name:`of ${field==='cost'?'estimated cost':'tokens'}`}]);
                seg.addEventListener('pointermove',show);seg.addEventListener('pointerleave',hideTip);frag.append(seg)});return frag});
    }
    const byTokens=[...cats].sort((a,b)=>b.tokens-a.tokens)[0],byCost=[...cats].sort((a,b)=>b.cost-a.cost)[0];
    if(totalCost>0)host.append(text('p',byTokens===byCost?`${byTokens.name} is the largest share of both tokens (${percent(byTokens.tokens,w.total_tokens)}) and estimated cost (${percent(byTokens.cost,totalCost)}).`:`${byTokens.name} is ${percent(byTokens.tokens,w.total_tokens)} of tokens but only ${percent(byTokens.cost,totalCost)} of cost; ${byCost.name.toLowerCase()} drives the most spend at ${percent(byCost.cost,totalCost)}.`,'insight'));
    const wrap=text('div','','table-wrap'),t=document.createElement('table');wrap.style.marginTop='14px';
    table(t,['Category','Tokens','Share','Est. cost','Share','Blended $ / MTok'],cats.map(c=>{const name=text('span',c.name);name.prepend(key(c.color));name.firstChild.style.marginRight='8px';return [name,integer(c.tokens),percent(c.tokens,w.total_tokens),money(c.cost)+(c.unpriced_tokens?' *':''),percent(c.cost,totalCost),c.tokens?moneyRate(c.cost/c.tokens*1e6):'—']}),{foot:['Total',integer(w.total_tokens),'100%',money(w.cost),'100%',w.total_tokens?moneyRate(totalCost/w.total_tokens*1e6):'—']});
    wrap.append(t);host.append(wrap);
    if(w.partial_cost)host.append(text('p',`* ${integer(w.unpriced_tokens)} tokens lack a rate and are excluded from the estimate; see Pricing coverage below.`,'samples'));
}

// Comparison table across one dimension
const compareDims=[['model','Model'],['tier','Tier'],['harness','Harness'],['mode','Mode']];
function compareRows(){
    const dim=state.compare,s=filters();let names;
    if(dim==='model')names=Object.keys(activeModels());
    else if(dim==='tier')names=Object.keys(harnessGroup().by_tier);
    else if(dim==='harness')names=Object.keys(data.by_harness);
    else names=Object.keys(harnessGroup().by_mode);
    return names.map(name=>{const windows=scopeWindows({...s,[dim]:name});return {name,w:windows?.[state.window],color:entityColor(dim,name)}}).filter(r=>r.w&&(r.w.total_tokens>0||(dim!=='model'&&r.w.conversations>0)));
}
function comparison(){
    const dim=state.compare,label=compareDims.find(([d])=>d===dim)[1],stat=state.statistic,statLabel=statDefs.find(([s])=>s===stat)[1];
    const rows=compareRows(),maxTokens=Math.max(1,...rows.map(r=>r.w.total_tokens)),maxCost=Math.max(1e-9,...rows.map(r=>Number(r.w.cost))),totalCost=rows.reduce((t,r)=>t+Number(r.w.cost),0);
    const columns=[['name',label,r=>r.name],['conversations','Conversations',r=>r.w.conversations],['tokens','Tokens',r=>r.w.total_tokens],['cost','Est. cost',r=>Number(r.w.cost)],
        ...metricDefs.map(d=>[d.key,`${d.short} (${statLabel.toLowerCase()})`,r=>r.w.metrics[d.key][stat]])];
    const sorter=columns.find(([k])=>k===state.sort)||columns[3],get=sorter[2];
    rows.sort((a,b)=>{const x=get(a),y=get(b);if(x===null&&y===null)return 0;if(x===null)return 1;if(y===null)return -1;return (typeof x==='string'?x.localeCompare(y):x-y)*state.dir});
    const t=$('comparison');t.replaceChildren();const head=document.createElement('thead'),tr=document.createElement('tr');
    columns.forEach(([k,title],i)=>{const th=text('th','',i?'num':'');const b=text('button',title);b.type='button';b.addEventListener('click',()=>{if(state.sort===k)state.dir*=-1;else{state.sort=k;state.dir=k==='name'?1:-1}comparison();saveState()});th.append(b);if(state.sort===k)th.setAttribute('aria-sort',state.dir>0?'ascending':'descending');tr.append(th)});
    head.append(tr);t.append(head);const body=document.createElement('tbody');
    const selectedName=state[dim];
    for(const r of rows){
        const row=document.createElement('tr'),w=r.w;row.tabIndex=0;row.setAttribute('aria-selected',r.name===selectedName);row.title=r.name===selectedName?`Clear the ${label.toLowerCase()} filter`:`Filter the report to ${r.name}`;
        const name=text('td',r.name);name.prepend(key(r.color));row.append(name,text('td',integer(w.conversations),'num'));
        for(const [value,max,shown,color] of [[w.total_tokens,maxTokens,compact(w.total_tokens),'var(--accent)'],[Number(w.cost),maxCost,`${money(w.cost)}${w.partial_cost?'*':''}`,'var(--accent)']]){
            const td=text('td','','num'),bar=text('div','','databar'),track=text('span','','track'),fill=text('i');fill.style.width=`${Math.max(value>0?2:0,value/max*90)}px`;fill.style.background=color;track.append(fill);bar.append(text('span',shown),track);td.append(bar);row.append(td)}
        for(const d of metricDefs){const v=w.metrics[d.key][stat];row.append(text('td',v===null?'—':d.format(v),'num'+(v===null?' dim':'')))}
        const pick=()=>{applyFilter(dim,r.name===selectedName?'':r.name)};
        row.addEventListener('click',pick);row.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick()}});body.append(row);
    }
    if(!rows.length){const r=document.createElement('tr'),td=text('td','No recorded activity for this selection.','dim');td.colSpan=columns.length;r.append(td);body.append(r)}
    t.append(body);
    $('comparison-caption').textContent=`${data.windows[state.window].label} · ${statLabel} per timing metric · totals for conversations, tokens, and cost · ${rows.length} ${label.toLowerCase()}${rows.length===1?'':'s'} · ${money(totalCost)} total`;
    $('comparison-note').textContent=`Select a row to filter the whole report. Conversations are counted once per ${label.toLowerCase()} and can appear in several, so they are not additive.${rows.some(r=>r.w.partial_cost)?' * Partial cost estimate.':''}`;
}

// Data quality
function quality(w){
    const c=w.coverage,completed=c['Completed turns']||0;
    $('coverage-meters').replaceChildren();
    for(const [label,have] of [['Turn duration',completed-(c['Missing turn duration']||0)],['Throughput',w.metrics.throughput.count]]){
        const row=text('div','','meter-row'),meter=text('div','','meter'),fill=text('i');fill.style.width=completed?`${Math.min(100,have/completed*100)}%`:'0';meter.append(fill);
        meter.setAttribute('role','meter');meter.setAttribute('aria-label',label);meter.setAttribute('aria-valuemin','0');meter.setAttribute('aria-valuemax',String(completed));meter.setAttribute('aria-valuenow',String(have));
        row.append(text('span',label),text('span',completed?`${integer(have)} of ${integer(completed)} · ${percent(have,completed)}`:'No completed turns','val'),meter);$('coverage-meters').append(row)}
    $('coverage-counts').replaceChildren(...[['Completed turns',completed],['Aborted turns',c['Aborted turns']||0],['Unfinished turns',c['Unfinished turns']||0],['Usage responses',c['Usage responses']||0]].map(([k,v])=>{const d=text('div');d.append(text('dt',k),text('dd',integer(v)));return d}));
    const coverage=[['Completed turns',completed],['Missing turn duration',c['Missing turn duration']||0],['Throughput samples',w.metrics.throughput.count],['Missing throughput samples',c['Missing throughput samples']||0],['Conversation duration samples',w.metrics.length.count],['Aborted turns',c['Aborted turns']||0],['Unfinished turns started in window',c['Unfinished turns']||0],['Usage responses',c['Usage responses']||0],['Aggregate snapshots',c['Aggregate snapshots']||0],['Tool calls',w.tool_calls]];
    table($('coverage'),['Measurement','Count'],coverage.map(([k,v])=>[k,integer(v)]));
    $('unpriced').replaceChildren();
    if(w.partial_cost){$('unpriced').append(text('p',`⚠ ${integer(w.unpriced_tokens)} tokens have no rate and are excluded from estimated cost:`,'samples'));const t=document.createElement('table');table(t,['Model','Unpriced tokens'],Object.entries(w.unpriced).map(([k,v])=>[k,integer(v)]));$('unpriced').append(t)}
    else $('unpriced').append(text('p','✓ All recorded usage in this window has a published or assumed rate.','samples'))
    const units=harnessGroup().windows[state.window].recorded_billing;
    table($('recorded-billing'),['Recorded measurement','Amount'],Object.entries(units).map(([unit,amount])=>[unit,unit.endsWith('USD')?money(amount):number(amount)]),{empty:'No recorded billing units in this window.'});
    const visible=new Set(Object.entries(activeModels()).filter(([,ws])=>ws[state.window].total_tokens>0).map(([name])=>name));
    table($('router-prices'),['Model','OpenRouter ID','Input / MTok','Cache read / MTok','Output / MTok'],Object.entries(data.openrouter_rates).filter(([model])=>visible.has(model)).map(([model,r])=>[model,r.id,money(r.input),r.cached===null?'N/A':money(r.cached),money(r.output)]),{textColumns:[1],empty:'No models in this selection use OpenRouter catalog rates.'});
}
function exactMetrics(){const rows=[];for(const w of activeWindows())for(const def of metricDefs){const m=w.metrics[def.key];rows.push([`${w.label} · ${def.title}`,...['avg','min','median','max','p75','p95','p99'].map(s=>m[s]===null?'—':def.format(m[s])),integer(m.count)])}table($('all-metrics'),['Window / metric','Average','Minimum','Median','Maximum','P75','P95','P99','Samples'],rows)}

// Filters
function options(select,values,allLabel,current){select.replaceChildren(text('option',allLabel));select.firstChild.value='';for(const v of values){const o=text('option',v);o.value=v;select.append(o)}select.value=current;select.dataset.active=!!current}
function syncFilters(){
    const harnesses=Object.keys(data.by_harness);
    if(state.harness&&!harnesses.includes(state.harness))state.harness='';
    options($('harness-select'),harnesses,'All harnesses',state.harness);
    const tiers=Object.entries(harnessGroup().by_tier).filter(([name,g])=>hasActivity(g.windows)||name===state.tier).map(([name])=>name);
    if(state.tier&&!Object.hasOwn(harnessGroup().by_tier,state.tier))state.tier='';
    options($('tier-select'),tiers,'All tiers',state.tier);
    const modes=Object.entries(harnessGroup().by_mode).filter(([name,g])=>hasActivity(g.windows)||name===state.mode).map(([name])=>name);
    if(state.mode&&!Object.hasOwn(harnessGroup().by_mode,state.mode))state.mode='';
    options($('mode-select'),modes,'All modes',state.mode);
    // A model with no tokens in the selected window would leave every view empty.
    if(state.model&&!(Object.hasOwn(activeModels(),state.model)&&activeModels()[state.model][state.window].total_tokens>0))state.model='';
    const models=Object.entries(activeModels()).filter(([,ws])=>ws[state.window].total_tokens>0).sort((a,b)=>b[1][state.window].total_tokens-a[1][state.window].total_tokens).map(([name])=>name);
    options($('model-select'),models,`All models (${models.length})`,state.model);
}
function applyFilter(dim,value){state[dim]=value;render()}
for(const [id,dim] of [['harness-select','harness'],['tier-select','tier'],['model-select','model'],['mode-select','mode']])$(id).addEventListener('change',()=>applyFilter(dim,$(id).value));
function chips(){
    const host=$('scope-chips');host.replaceChildren();const active=[['harness','Harness'],['tier','Tier'],['model','Model'],['mode','Mode']].filter(([d])=>state[d]);
    if(!active.length){host.append(text('span','All harnesses, tiers, models, and modes'));return}
    for(const [d,label] of active){const chip=text('span',`${label}: ${state[d]}`,'chip'),b=text('button','×');b.type='button';b.setAttribute('aria-label',`Remove ${label.toLowerCase()} filter`);b.addEventListener('click',()=>applyFilter(d,''));chip.append(b);host.append(chip,document.createTextNode(' '))}
    if(active.length>1){const clear=text('button','Clear all','link-btn');clear.type='button';clear.addEventListener('click',()=>{for(const [d] of active)state[d]='';render()});host.append(clear)}
}

// State persists in the URL hash so a view can be reloaded or shared
function saveState(){const p=new URLSearchParams();const put=(k,v,d)=>{if(v!==d)p.set(k,v)};put('w',String(state.window),'0');put('h',state.harness,'');put('t',state.tier,'');put('m',state.model,'');put('md',state.mode,'');put('g',state.granularity,'daily');put('from',state.start,firstDate);put('to',state.end,cutoffDate);put('measure',state.measure,'tokens');put('split',state.split,'none');put('stats',[...state.stats].join(','),'median,p95');put('cmp',state.compare,'model');put('stat',state.statistic,'median');put('sort',`${state.sort}:${state.dir}`,'cost:-1');
    const hash=p.toString();try{history.replaceState(null,'',hash?'#'+hash:location.pathname+location.search)}catch(error){}}
function loadState(){
    let p;try{p=new URLSearchParams(location.hash.slice(1))}catch(error){return}
    const w=Number(p.get('w'));if(Number.isInteger(w)&&w>=0&&w<data.windows.length)state.window=w;
    for(const [k,f] of [['h','harness'],['t','tier'],['m','model'],['md','mode']])if(p.get(k))state[f]=p.get(k);
    if(['hourly','daily','weekly','monthly'].includes(p.get('g')))state.granularity=p.get('g');
    const valid=d=>/^\d{4}-\d{2}-\d{2}$/.test(d||'')&&d>=firstDate&&d<=cutoffDate;
    if(valid(p.get('from')))state.start=p.get('from');if(valid(p.get('to')))state.end=p.get('to');if(state.start>state.end){state.start=firstDate;state.end=cutoffDate}
    if(['tokens','cost'].includes(p.get('measure')))state.measure=p.get('measure');
    if(['none','model','harness','tier','mode'].includes(p.get('split')))state.split=p.get('split');
    if(p.has('stats'))state.stats=new Set(p.get('stats').split(',').filter(s=>chartStats.some(([c])=>c===s)));
    if(compareDims.some(([d])=>d===p.get('cmp')))state.compare=p.get('cmp');
    if(statDefs.some(([s])=>s===p.get('stat')))state.statistic=p.get('stat');
    const [sort,dir]=(p.get('sort')||'').split(':');if(sort){state.sort=sort;state.dir=dir==='1'?1:-1}
}

function render(){
    hideTip();syncFilters();chips();
    const w=activeWindows()[state.window],win=data.windows[state.window];
    [...$('tabs').children].forEach((b,i)=>b.setAttribute('aria-pressed',i===state.window));
    $('range').textContent=w.end_exclusive?`${fmtDate(w.start)} · full calendar day`:`${fmtDate(w.start)} – ${fmtTime(w.end)}`;
    const damaged=Object.keys(data.quality).some(k=>k.startsWith('Malformed')||k==='Unreadable files'||k==='Invalid usage records');
    const notice=$('notice');notice.hidden=!w.partial_cost&&!damaged;notice.replaceChildren(text('span','⚠'),text('span'));
    notice.lastChild.append(text('b',w.partial_cost?'Partial cost estimate. ':'Some records could not be read. '),document.createTextNode(w.partial_cost?`${integer(w.unpriced_tokens)} tokens lack a verified rate or the category detail needed to calculate cost. Their usage is included in token totals.`:'Review parser diagnostics under Data quality.'));
    cards(w);usage();heatmap();performance(w);composition(w);comparison();quality(w);exactMetrics();saveState();
    document.title=`${win.label} · Coding agents usage report`;
}

// Controls that do not depend on report data
for(const [stat,label,color] of chartStats){
    const option=text('label',''),input=document.createElement('input');input.type='checkbox';input.value=stat;input.checked=state.stats.has(stat);
    option.append(input,key(color,true),document.createTextNode(label));$('trend-stat-options').append(option);
    input.addEventListener('change',()=>{if(input.checked)state.stats.add(stat);else state.stats.delete(stat);if(data){performance(activeWindows()[state.window]);saveState()}});
}
for(const [value,label] of [['tokens','Tokens'],['cost','Cost']]){const b=text('button',label);b.type='button';b.dataset.value=value;b.addEventListener('click',()=>{state.measure=value;syncMeasure();if(data){usage();saveState()}});$('usage-metric').append(b)}
const syncMeasure=()=>[...$('usage-metric').children].forEach(b=>b.setAttribute('aria-pressed',b.dataset.value===state.measure));
for(const [value,label] of compareDims){const b=text('button',label);b.type='button';b.dataset.value=value;b.addEventListener('click',()=>{state.compare=value;syncCompare();if(data){comparison();saveState()}});$('compare-dimension').append(b)}
const syncCompare=()=>[...$('compare-dimension').children].forEach(b=>b.setAttribute('aria-pressed',b.dataset.value===state.compare));
for(const [stat,label] of statDefs){const option=text('option',label);option.value=stat;$('comparison-stat-select').append(option)}
$('comparison-stat-select').addEventListener('change',()=>{state.statistic=$('comparison-stat-select').value;if(data){comparison();saveState()}});
$('split-select').addEventListener('change',()=>{state.split=$('split-select').value;if(data){usage();saveState()}});
for(const id of ['trend-start','trend-end']){
    $(id).addEventListener('change',()=>{
        if(!data)return;
        const start=$('trend-start'),end=$('trend-end'),error=$('trend-range-error');
        error.hidden=start.validity.valid&&end.validity.valid&&start.value<=end.value;
        if(!error.hidden){error.textContent=`Choose dates from ${firstDate} to ${cutoffDate}, with From on or before To.`;return}
        state.start=start.value;state.end=end.value;usage();heatmap();performance(activeWindows()[state.window]);saveState();
    });
}
$('granularity-select').addEventListener('change',()=>{state.granularity=$('granularity-select').value;if(data){usage();performance(activeWindows()[state.window]);saveState()}});
$('expand-defs').addEventListener('click',()=>{const all=[...$('definitions').querySelectorAll('details')],open=!all.every(d=>d.open);all.forEach(d=>d.open=open);$('expand-defs').textContent=open?'Collapse all':'Expand all'});
const themes=['system','light','dark'];let theme='system';try{theme=localStorage.getItem('harness-report-theme')||'system'}catch(error){}
const applyTheme=()=>{if(theme==='system')delete document.documentElement.dataset.theme;else document.documentElement.dataset.theme=theme;$('theme-toggle').textContent=`Theme: ${theme[0].toUpperCase()+theme.slice(1)}`};
$('theme-toggle').addEventListener('click',()=>{theme=themes[(themes.indexOf(theme)+1)%themes.length];try{localStorage.setItem('harness-report-theme',theme)}catch(error){}applyTheme()});applyTheme();
addEventListener('scroll',hideTip,{passive:true});

async function start(){
    try{data=await source.load()}
    catch(error){
        $('subtitle').textContent='Report data is unavailable.';
        const notice=$('notice');notice.hidden=false;notice.replaceChildren(text('span','⚠'),text('span'));notice.lastChild.append(text('b','Unable to load the report. '),document.createTextNode(error.message||'Check that the local server is running.'));
        return;
    }
    firstDate=data.first_date||data.trend_periods.daily[0].start.slice(0,10);cutoffDate=data.cutoff_date||data.generated.slice(0,10);state.start=firstDate;state.end=cutoffDate;
    // Models are ordered once by all-history tokens, so filters never repaint a model
    modelOrder=Object.entries(data.by_model).map(([name,ws])=>[name,Math.max(...ws.map(w=>w.total_tokens))]).filter(([,t])=>t>0).sort((a,b)=>b[1]-a[1]).map(([name])=>name);
    data.windows.forEach((w,i)=>{const b=text('button',w.label.replace(/^Last /,''));b.type='button';b.title=w.label;b.addEventListener('click',()=>{state.window=i;render()});$('tabs').append(b)});
    for(const id of ['trend-start','trend-end']){$(id).min=firstDate;$(id).max=cutoffDate}
    $('subtitle').textContent=`${integer(data.files)} log files · ${integer(data.threads)} threads · ${data.timezone} · cutoff ${fmtTime(data.generated)}`;
    $('pricing-date').textContent=`Pricing verified ${data.pricing_date}`;
    table($('diagnostics'),['Diagnostic','Count'],Object.entries(data.quality).map(([k,v])=>[k,integer(v)]),{empty:'No parser issues recorded.'});data.warnings.forEach(w=>$('warnings').append(text('li',w)));
    $('sources').append(document.createTextNode(`Rates verified ${data.pricing_date}: `));const link=text('a','OpenAI API pricing');link.href=data.pricing_source;link.rel='noreferrer';$('sources').append(link,document.createTextNode(`. Rates are embedded in the script and are not updated automatically. ${source.live?'Cached source locations':'Input directories'}: `+data.sources.join(', ')));
    const anthropicLink=text('a','Anthropic API pricing');anthropicLink.href=data.anthropic_pricing_source;anthropicLink.rel='noreferrer';$('sources').append(document.createTextNode(` · Anthropic verified ${data.anthropic_pricing_date}: `),anthropicLink);if(data.openrouter){const routerLink=text('a','OpenRouter model catalog');routerLink.href=data.openrouter.source;routerLink.rel='noreferrer';const origin=data.openrouter.bundled?' · bundled snapshot':data.openrouter.snapshot_file?' · supplied snapshot':' · live catalog';const date=data.openrouter.retrieved?` retrieved ${fmtTime(data.openrouter.retrieved)}`:' (retrieval date unknown)';const error=data.openrouter.error?` · live fetch unavailable: ${data.openrouter.error}; using ${data.openrouter.bundled?'bundled':'supplied snapshot'} prices`:'';$('sources').append(document.createTextNode(' · '),routerLink,document.createTextNode(origin+date+error));}
    $('sources').append(document.createTextNode(' · '));const modeLink=text('a','Fast mode documentation');modeLink.href='https://developers.openai.com/api/docs/guides/fast-mode';modeLink.rel='noreferrer';$('sources').append(modeLink);
    $('footer').textContent=`Report cutoff: ${fmtTime(data.generated)} (${data.timezone})`;
    loadState();
    $('trend-start').value=state.start;$('trend-end').value=state.end;$('granularity-select').value=state.granularity;$('split-select').value=state.split;$('comparison-stat-select').value=state.statistic;
    [...$('trend-stat-options').querySelectorAll('input')].forEach(i=>{i.checked=state.stats.has(i.value)});
    syncMeasure();syncCompare();render();
}
start();
</script></body></html>
'''


if __name__ == "__main__":
    raise SystemExit(main())
