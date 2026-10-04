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
from typing import Any, Iterable, Iterator
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
    if name == "gpt-5.4-mini":
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
    ttft: float | None = None
    model: str | None = None
    mode: SpeedMode = SpeedMode.NORMAL
    completed: bool = False
    aborted: bool = False
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
        with path.open("r", encoding="utf-8") as stream:
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
                        any(name in payload and not isinstance(payload[name], dict)
                            for name in ("thread_settings", "info")) or
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
                    continue
                explicit_thread = payload.get("thread_id")
                if explicit_thread == thread_id:
                    inherited = False
                elif explicit_thread and explicit_thread != thread_id:
                    quality.counts["Inherited records excluded"] += 1
                    continue
                if history_boundary and index >= history_boundary:
                    inherited = False
                if inherited:
                    quality.counts["Inherited records excluded"] += 1
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
                        turn.ttft = milliseconds(payload.get("time_to_first_token_ms"))
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
    with path.open(encoding="utf-8") as stream:
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
            if kind not in {"user", "assistant", "system"}:
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
                turn.model = model or turn.model
                raw_usage = message.get("usage")
                mode = SpeedMode.FAST if isinstance(raw_usage, dict) and raw_usage.get("speed") == "fast" else SpeedMode.NORMAL
                content = message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                            thread.calls.setdefault(block["id"], ToolCall(at, current_turn, model, mode))
                if "usage" in message:
                    usage = separate_usage(message["usage"])
                    if usage is None:
                        quality.warn("Invalid usage records", path, line, "invalid Claude token usage; skipped")
                        continue
                    key = str(row.get("requestId") or message.get("id") or row.get("uuid") or hashlib.sha256(json.dumps(row).encode()).hexdigest())
                    old = responses.get(key)
                    # Streaming records repeat request usage. Keep the fullest
                    # valid response rather than charging once per content block.
                    if old:
                        quality.counts["Duplicate usage records excluded"] += 1
                    if old is None or usage.output >= old[1].usage.output:
                        responses[key] = (current_turn, UsageEvent(at, usage, model, key, mode))
                if message.get("stop_reason") in {"end_turn", "stop_sequence"}:
                    turn.end, turn.completed = at, True
            elif kind == "system" and row.get("subtype") == "turn_duration":
                turn = thread.turns.setdefault(current_turn, Turn(current_turn))
                turn.end, turn.completed = at, True
                turn.duration = milliseconds(row.get("durationMs"))
    for turn_id, response in responses.values():
        thread.turns[turn_id].modern.append(response)
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
    seen: set[str] = set()
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
            if kind == "assistant.turn_start":
                current_turn = str(data.get("turnId") or key)
                model = data.get("model") or model
            turn_id = str(data.get("turnId") or current_turn)
            turn = thread.turns.setdefault(turn_id, Turn(turn_id, model=model))
            if kind == "assistant.turn_start":
                turn.start = at
            elif kind == "assistant.turn_end":
                turn.end, turn.completed = at, True
                if turn.start and at >= turn.start:
                    turn.duration = (at - turn.start).total_seconds()
            elif kind in {"abort", "agent.interrupted"}:
                turn.end, turn.aborted, turn.completed = at, True, False
            elif kind == "tool.execution_start":
                call_id = str(data.get("toolCallId") or key)
                thread.calls.setdefault(call_id, ToolCall(at, turn_id, model))
            elif kind == "assistant.usage":
                usage = copilot_usage(data)
                if usage is None:
                    quality.warn("Invalid usage records", path, line, "invalid Copilot token usage; skipped")
                    continue
                response_id = str(data.get("apiCallId") or key)
                if any(r.key == response_id for t in thread.turns.values() for r in t.modern):
                    quality.counts["Duplicate usage records excluded"] += 1
                    continue
                turn.modern.append(UsageEvent(at, usage, data.get("model") or model, response_id))
                ttft = milliseconds(data.get("timeToFirstTokenMs"))
                if turn.ttft is None:
                    turn.ttft = ttft
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
        if not detailed:
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
    ttft: list[float] = field(default_factory=list)
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
            window.ttft.extend(other.ttft)
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
            # Trend points need samples and conversation membership, not token/cost breakdowns.
            costs, unpriced = (price_usage(model, record.usage, record.mode, catalog, aggregate=record.aggregate)
                               if any(w.granularity is None for w in record_windows) else ({}, {}))
            for window in record_windows:
                if usage_total:
                    window.conversations.add(thread.id)
                if window.granularity is not None:
                    continue
                for category, count in counts.items():
                    window.tokens[category] += count
                window.models[name] += usage_total
                for category, cost in costs.items():
                    window.costs[category] += cost
                for category, count in unpriced.items():
                    window.unpriced_categories[category] += count
                if unpriced:
                    window.unpriced[name] += sum(unpriced.values())
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
            if turn.ttft is not None:
                window.ttft.append(turn.ttft)
            else:
                window.coverage["Missing first-token timing"] += 1
            if turn.duration is not None:
                window.durations[thread.id] = window.durations.get(thread.id, 0) + turn.duration
            else:
                window.coverage["Missing turn duration"] += 1
            if turn.duration and usage_events:
                output = sum(r.usage.output for r in usage_events)
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
    return {"ttft": distribution(window.ttft), "throughput": distribution(window.throughput),
            "length": distribution(window.durations.values()),
            "tools": distribution(window.call_counts[c] for c in window.conversations)}


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
        return {granularity.value: [metric_summary(w) if w.conversations else None
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
        with path.open(encoding="utf-8") as stream:
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


# Bump when the schema or parser semantics change; cached facts must match the readers.
CACHE_VERSION = 1


def file_stamp(path: Path) -> str:
    stat = path.stat()
    return json.dumps([stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])


def validate_metrics_cache(connection: sqlite3.Connection, *, allow_empty: bool = False) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"cache_files", "cache_groups", "cache_threads", "cache_turns", "cache_usage",
                "cache_calls", "cache_billing"}
    if version not in (0, CACHE_VERSION):
        raise ValueError("unsupported metrics cache version; use a new --cache path")
    if not tables and version == 0 and allow_empty:
        return
    if version != CACHE_VERSION or not required.issubset(tables):
        raise ValueError("cache path contains a different database; choose a separate --cache path")


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
            yield from loaded

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
                "SELECT thread_id, turn_id, start, end, duration, ttft, model, mode, completed, aborted "
                "FROM cache_turns WHERE cache_key=? ORDER BY rowid", (key,)):
            thread_id, turn_id, start, end, duration, ttft, model, mode, completed, aborted = row
            threads[thread_id].turns[turn_id] = Turn(
                turn_id, start=datetime.fromisoformat(start) if start else None,
                end=datetime.fromisoformat(end) if end else None, duration=duration, ttft=ttft,
                model=model, mode=SpeedMode(mode), completed=bool(completed), aborted=bool(aborted))
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
                start TEXT, end TEXT, duration REAL, ttft REAL, model TEXT, mode TEXT NOT NULL,
                completed INTEGER NOT NULL, aborted INTEGER NOT NULL,
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
                 turn.end.isoformat() if turn.end else None, turn.duration, turn.ttft, turn.model,
                 turn.mode.value, turn.completed, turn.aborted) for turn in thread.turns.values()])
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
            # Uncheckpointed SQLite writes live in the WAL, not the main database.
            wal = Path(str(path) + "-wal")
            stamps.append((str(wal), file_stamp(wal) if wal.exists() else "missing"))
    return json.dumps(stamps)


@dataclass
class PreparedGroup:
    key: str
    manifest: str
    uncached: list[Thread] | None = None
    quality: Quality | None = None


@dataclass
class PreparedSources:
    threads: Iterator[Thread]
    files: set[Path]
    first_at: datetime | None
    cache: MetricsCache
    uncached_groups: int = 0


@contextmanager
def prepare_sources(sources: dict[Harness, list[Path]], quality: Quality, progress: bool,
                    cache_path: Path | None, now: datetime) -> Iterator[PreparedSources]:
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
                        prepared.append(PreparedGroup(key, t3_manifest, [], t3_quality))
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
                            else:
                                group.uncached, group.quality = loaded, group_quality
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
                                      sum(group.uncached is not None for group in prepared))


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
    with prepare_sources(sources, quality, progress, cache_path, now) as prepared:
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
    root = args.directory if args.directory is not None else Path.cwd()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
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
        codex_roots = [home / ".codex" / name for name in ("sessions", "archived_sessions")]
        harness_roots[Harness.CODEX] = [root] + ([path for path in codex_roots if path.is_dir()]
                                               if args.directory is None else [])
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
            print(f"Live OpenRouter prices unavailable: {error}; using bundled prices", file=sys.stderr)
            catalog_metadata["error"] = str(error)
    return catalog, catalog_metadata


def main() -> int:
    parser = argument_parser()
    args = parser.parse_args()
    if args.offline and args.live_prices:
        parser.error("--offline cannot be combined with --live-prices")
    sources = source_paths(args, parser)
    root = args.directory if args.directory is not None else Path.cwd()
    if args.output.suffix.lower() != ".html":
        parser.error("output must have an .html extension")
    catalog, catalog_metadata = load_catalog(args, parser)
    try:
        report = collect_report(root, progress=True,
                                additional_roots=sources.get(Harness.CODEX, [])[1:],
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
<title>Coding agents · Conversation metrics</title>
<style>
:root{color-scheme:light;--ink:#172a3c;--muted:#62768a;--paper:#f3f6fa;--line:#e1e8f0;--blue:#3975e7;--teal:#159e99;--violet:#8c67db;--orange:#e99b36;--pink:#df7896}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.6 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1280px;margin:auto;padding:44px 32px 56px}h1,h2,h3,p{margin:0}h1{font-size:38px;line-height:1.2;letter-spacing:-1.4px;font-weight:700}h2{font-size:20px;letter-spacing:-.4px}h3{font-size:16px;font-weight:650}.eyebrow{font-size:11px;letter-spacing:2px;font-weight:700;text-transform:uppercase;color:var(--teal);margin-bottom:12px}.header{display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:32px}.subtitle{color:var(--muted);margin-top:10px}.badge{display:inline-flex;align-items:center;gap:7px;border:1px solid #cfdfdb;border-radius:30px;padding:7px 12px;color:#347c6e;background:#eff9f5;font-size:12px;white-space:nowrap}.dot{width:6px;height:6px;border-radius:50%;background:#159e99}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:20px;flex-wrap:wrap}.tabs{display:flex;gap:4px;padding:5px;background:#e6ecf3;border-radius:12px;flex-wrap:wrap}button{font:inherit;cursor:pointer}.tabs button{border:0;background:none;padding:9px 14px;border-radius:8px;color:var(--muted);font-weight:600;font-size:12px}.tabs button[aria-pressed=true]{background:#fff;color:var(--ink);box-shadow:0 2px 5px #182c3c10}button:focus-visible,a:focus-visible,[tabindex]:focus-visible{outline:3px solid #3975e780;outline-offset:3px}.range{color:var(--muted);font-size:12px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:28px}.card,.panel{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 3px 14px #24406003}.card{padding:21px 23px;position:relative;overflow:hidden}.card:before{content:"";position:absolute;top:0;left:23px;width:28px;height:3px;background:var(--accent);border-radius:0 0 3px 3px}.card-label{font-size:12px;color:var(--muted);font-weight:600}.card-value{font-size:31px;letter-spacing:-1px;line-height:1.3;margin:9px 0 7px;font-variant-numeric:tabular-nums}.card-note{font-size:11px;color:var(--muted)}.section-head{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin-bottom:16px}.section-head p{color:var(--muted);font-size:12px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin-bottom:28px}.panel{padding:23px}.panel-top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}.panel p{color:var(--muted);font-size:12px;margin-top:4px}.unit{font-size:10px;background:var(--paper);padding:4px 8px;border-radius:6px;white-space:nowrap;color:var(--muted)}.legend{display:flex;gap:17px;margin-top:15px;flex-wrap:wrap;font-size:11px;color:var(--muted)}.legend span{display:inline-flex;align-items:center;gap:6px}.swatch{width:7px;height:7px;border-radius:2px;display:inline-block}.chart{margin-top:12px}.chart svg{width:100%;height:auto;display:block;overflow:visible}.stats{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:10px;border-top:1px solid var(--line);padding-top:15px;margin-top:5px}.stat{grid-column:span 3}.stat:nth-child(n+5){grid-column:span 4}.stat span{display:block;color:var(--muted);font-size:10px}.stat strong{font-size:20px;letter-spacing:-.3px;font-weight:600;font-variant-numeric:tabular-nums}.samples{font-size:11px;color:var(--muted);margin-top:10px}.breakdown-body{display:grid;grid-template-columns:190px 1fr;gap:18px;align-items:center;min-height:215px}.breakdown-body svg{width:100%;height:auto}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;color:var(--muted);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}th,td{padding:10px 5px;border-bottom:1px solid var(--line)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tfoot td{font-weight:700;border:0}td.label{white-space:nowrap}.notice{padding:12px 15px;background:#fff7e9;border:1px solid #f1dcae;border-radius:10px;color:#8a651f;font-size:12px;margin-bottom:20px}.empty{padding:44px 20px;text-align:center;color:var(--muted);background:var(--paper);border-radius:10px;margin:16px 0}.quality-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:18px}.definition{margin-bottom:12px;font-size:12px;color:var(--muted)}.definition b{color:var(--ink)}details{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;font-weight:600;font-size:12px}.sources{font-size:11px;color:var(--muted);margin-top:16px;overflow-wrap:anywhere}.sources a{color:var(--blue)}.footer{display:flex;justify-content:space-between;gap:16px;margin-top:24px;color:var(--muted);font-size:11px}.tooltip{position:fixed;background:#172a3c;color:#fff;padding:8px 12px;border-radius:8px;font-size:12px;pointer-events:none;z-index:10;max-width:280px;box-shadow:0 5px 20px #172a3c30}.warnings{font:11px/1.7 ui-monospace,monospace;overflow-wrap:anywhere;padding-left:20px}noscript{display:block;padding:20px;background:#fff7e9}
#performance{grid-template-columns:minmax(0,1fr)}
.trend-stat-controls{border:0;padding:0;margin:0 0 18px}.trend-stat-controls legend{color:var(--muted);font-size:12px;padding:0;margin-bottom:8px}.trend-stat-options{display:flex;gap:18px;flex-wrap:wrap}.trend-stat-options label{display:inline-flex;align-items:center;gap:7px;font-size:12px;cursor:pointer}.trend-stat-options input{accent-color:var(--teal);width:15px;height:15px;margin:0}.trend-stat-options input:focus-visible{outline:3px solid #3975e780;outline-offset:3px}
.token-totals{margin:14px 0 0;padding-top:10px;border-top:1px solid var(--line);font-size:11px}.token-totals div{display:flex;justify-content:space-between;gap:12px;margin-top:4px}.token-totals dt{color:var(--muted)}.token-totals dd{margin:0;font-weight:600;font-variant-numeric:tabular-nums}
.filters{display:flex;gap:16px;flex-wrap:wrap}.model-filter{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:12px}.model-filter select,.model-filter input{font:inherit;font-weight:600;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 12px;max-width:100%}.model-filter select{padding-right:34px}.model-filter select:focus-visible,.model-filter input:focus-visible{outline:3px solid #3975e780;outline-offset:3px}.range-error{color:#a33232;margin:-8px 0 16px}.model-comparison{margin-bottom:28px}.comparison-controls{align-items:center;flex-wrap:wrap}.model-comparison table{min-width:800px;margin:16px 0 10px}.model-comparison th,.model-comparison td{white-space:nowrap;padding:13px 10px}.model-comparison td:first-child{font-weight:600}.model-comparison th:first-child,.model-comparison td:first-child{position:sticky;left:0;background:#fff}.model-comparison tr[data-selected=true],.model-comparison tr[data-selected=true] td:first-child{background:#f1f5fd}.filter-scope{color:var(--muted);font-size:12px;margin:-6px 0 18px}
@media(max-width:900px){main{padding:28px 20px}.cards{grid-template-columns:repeat(2,1fr)}.breakdown-body{grid-template-columns:140px 1fr;gap:10px}.card-value{font-size:28px}}
@media(max-width:650px){h1{font-size:30px}.header{align-items:flex-start;gap:12px}.badge{font-size:10px;padding:5px 8px}.grid,.quality-grid{grid-template-columns:1fr}.cards{gap:10px}.card{padding:18px 16px}.card-value{font-size:25px}.section-head{display:block}.section-head p{margin-top:4px}.panel{padding:18px}.tabs button{padding:8px 10px;font-size:11px}.breakdown-body{grid-template-columns:130px 1fr}.footer{flex-direction:column;gap:3px}}
@media print{body{background:#fff}main{padding:0}.tabs,.tooltip{display:none}.panel,.card{break-inside:avoid;box-shadow:none}.grid{gap:10px}.panel{padding:14px}details{display:block}}
</style>
</head>
<body><main>
<header class="header"><div><div class="eyebrow">Coding agents / Usage observatory</div><h1>Conversation metrics</h1><p class="subtitle" id="subtitle"></p></div><div class="badge"><span class="dot"></span>Offline report</div></header>
<noscript>This report requires JavaScript to display its embedded data and charts. No internet connection is needed.</noscript>
<div class="toolbar"><nav class="tabs" aria-label="Reporting window" id="tabs"></nav><div class="filters"><label class="model-filter" for="harness-select">Harness <select id="harness-select"><option value="">All harnesses</option></select></label><label class="model-filter" for="tier-select">Tier <select id="tier-select"><option value="">All tiers</option></select></label><label class="model-filter" for="model-select">Model <select id="model-select"><option value="">All models</option></select></label><label class="model-filter" for="mode-select">Mode <select id="mode-select"><option value="">All modes</option></select></label></div></div>
<p class="filter-scope"><span id="model-scope">All models</span> · <span class="range" id="range"></span></p>
<div class="notice" id="notice" hidden></div>
<section class="cards" id="cards" aria-label="Selected window summary" aria-live="polite"></section>
<div class="section-head comparison-controls"><div><h2>Performance over time</h2><p id="trend-caption"></p></div><div class="filters"><label class="model-filter" for="trend-start">Start <input id="trend-start" type="date" required aria-describedby="trend-range-error"></label><label class="model-filter" for="trend-end">End <input id="trend-end" type="date" required aria-describedby="trend-range-error"></label><label class="model-filter" for="granularity-select">Data points <select id="granularity-select"><option value="hourly">Hourly</option><option value="daily" selected>Daily</option><option value="weekly">Weekly</option><option value="monthly">Monthly</option></select></label></div></div>
<p id="trend-range-error" class="range-error" role="status" hidden></p>
<fieldset class="trend-stat-controls"><legend>Displayed statistics</legend><div class="trend-stat-options" id="trend-stat-options"></div></fieldset>
<section class="grid" id="performance" aria-label="Performance charts"></section>
<div class="section-head"><h2>Tokens &amp; estimated cost</h2><p>Selected window · USD · API-equivalent token estimates</p></div>
<section class="grid" id="breakdowns" aria-label="Token and cost breakdowns"></section>
<div class="section-head comparison-controls"><div><h2>Compare tiers &amp; models</h2><p>One value per metric · conversations, tokens and costs are totals</p></div><label class="model-filter" for="comparison-stat-select">Metric statistic <select id="comparison-stat-select"></select></label></div>
<section class="panel model-comparison"><h2>By tier</h2><p id="tier-caption"></p><div class="table-wrap"><table id="tier-comparison"></table></div><p>Select a tier above to explore its charts and costs. Conversations are counted once within each tier and can appear in several tiers.</p></section>
<section class="panel model-comparison"><h2>By model</h2><p id="comparison-caption"></p><div class="table-wrap"><table id="model-comparison"></table></div><p>Select a model above to explore its charts, cost breakdown, sample counts, and all reporting windows.</p></section>
<section class="panel"><div class="panel-top"><div><h2>Definitions &amp; data quality</h2><p>Understand the measurements behind the charts.</p></div><span class="unit" id="pricing-date"></span></div>
<div class="quality-grid"><div>
<p class="definition"><b>First-token time.</b> Explicit logged time to first token, per completed turn. Missing timings are excluded.</p>
<p class="definition"><b>Effective throughput.</b> All output tokens, including reasoning, divided by full turn duration. Tool execution and waiting are included.</p>
<p class="definition"><b>Conversation length.</b> Sum of completed turn durations per thread in the selected window. Idle time between turns is excluded; subagents count separately.</p>
<p class="definition"><b>Tool calls.</b> Model-issued function, custom-tool, web-search, and tool-search calls. Outputs and mirrored completion events are excluded; nested commands inside a call are not counted separately.</p>
<p class="definition"><b>Token totals.</b> Total tokens equals input plus output. Input includes cached input and cache writes; output includes reasoning. Cached input counts cache reads and is a subset of input, not an additional token total. The composition chart separates these categories so each token is counted once.</p>
<p class="definition"><b>Distribution statistics.</b> Average is the arithmetic mean. Median is the middle sample, or the average of the two middle samples for an even count. Minimum and maximum are observed extremes. P75, P95, and P99 use nearest rank. All statistics use the same valid samples; each completed turn receives equal weight for timing and throughput.</p>
<p class="definition"><b>Trend points.</b> Start and End default to the first recorded activity date and the report cutoff date. Hourly, daily, weekly (Monday start), and monthly points use periods in the report timezone across all available history. Hourly points distinguish repeated daylight saving hours by their UTC offset, and the axis shows time within each selected date. The date controls show periods overlapping the chosen range; weekly and monthly statistics include the whole calendar period. The first and current periods can be partial. Statistics are calculated from each period's samples; conversation duration and calls include activity within that period. Missing samples appear as gaps. Figures below charts describe the selected reporting window.</p>
<p class="definition"><b>Window boundaries.</b> Tokens and calls use record time; turn metrics use completion time. Today starts at midnight in the report timezone and ends at the report cutoff. Yesterday is the preceding calendar day in that timezone, excluding today's midnight. Rolling windows are exact 24-hour days. The full duration of a turn finishing in the window is assigned to that window.</p>
<p class="definition"><b>Model attribution.</b> Fast usage has a separate model entry with a “-fast” suffix. Requests crossing a published context-pricing threshold have a “-long” suffix, including cached input when selecting the threshold; combined usage has “-fast-long”. Usage below the threshold keeps the model name unless Fast. Models without context pricing and aggregate records without per-request sizes do not receive “-long”. Tokens, calls, timing, and costs are separated by recorded mode, with the Fast premium applied to the underlying model's rates. Tokens and calls use their recorded model, falling back to the turn model. Timing uses the model generating that turn. Conversation duration and tool counts include only that model's activity; a thread using multiple models or modes appears in each, so conversation counts are not additive. If several models generate output within one turn, its timing is listed under “Mixed models (timing)”. If one model uses several modes within a turn, its timing is listed under “Mixed modes (timing)” because separate durations cannot be recovered. If one model in one mode crosses context thresholds within a turn, its timing is listed under “Mixed contexts (timing)”. Tool calls follow the context class of matching model and mode usage in their turn; ambiguous calls are listed under “Mixed contexts (tools)”. Mode and tier totals retain this activity once.</p>
<p class="definition"><b>Model tiers.</b> Budget: Luna, Terra, GPT-5.4-mini, Spark, codex-auto-review, and Claude Haiku. Medium: Sol, GPT-5.4, GPT-5.5, and Claude Sonnet. High: Astra, Claude Opus, Fable, and Mythos. Models outside these groups are Unclassified. Tier metrics are calculated from underlying activity, with each conversation counted once per tier. Turns using several models in the same tier retain their timing in that tier; turns spanning tiers have timing under “Mixed tiers (timing)”. Per-model pricing and the Fast premium still apply. The model selector and comparison table show entries with recorded tokens in the selected window, harness, tier, and mode. Zero-token entries, including shared timing and tool-call buckets, remain included in aggregate totals and coverage.</p>
<p class="definition"><b>Mode attribution.</b> Logged service tier “default” is Normal; “priority” or “fast” is Fast. Settings persist until changed. Per the selected assumption, unknown mode—including missing evidence, explicit null, and “auto”—is counted as Normal in all metrics and costs. Other explicit tiers have their own bucket. Tokens and calls follow their recorded tier or the latest logged settings. This combines logged mode with the Normal assumption; a backend fallback cannot be detected without a response tier. A turn with usage in several modes has its timing under “Mixed modes (timing)” because separate durations are unavailable. Conversation durations and calls include only activity attributed to the selected mode.</p>
<p class="definition"><b>Harness coverage.</b> Claude Code reads project JSONL files, Copilot reads CLI session events, and OpenCode reads its message and tool tables. T3 Code links saved native sessions to the Codex, Claude Code, and OpenCode readers; linked sessions count once under T3 when selected. Missing native logs and other T3 providers are outside coverage. Copilot shutdown-only totals are assigned to shutdown, which does not establish when individual requests occurred. Timing samples require logged timing evidence.</p>
<p class="definition"><b>Coverage.</b> Logs in the listed input directories include archived and active sessions. Copies sharing a conversation ID are merged; repeated usage responses, tool calls, and turn completions are counted once. Active logs are read while they may still be growing; the report cutoff limits included activity. Unfinished turns contribute recorded tokens and calls, with completion timings excluded. An older-window label does not imply a complete year of available history. Missing durations are excluded, so conversation duration can be partial.</p>
<p class="definition"><b>Cost estimate.</b> Current standard API rates are applied to every historical window, with an assumed 50% premium on OpenAI token categories recorded in Fast mode. Normal, including assumed Normal activity, uses base rates. Other explicit tiers also use base rates; their actual premiums are unknown. Codex 5.3 Spark uses GPT-5.4-mini rates and codex-auto-review uses GPT-5.6-luna rates as user-selected proxies, not published prices for those models. These are API-equivalent estimates, not subscription bills. Claude cache writes include separate 5-minute and 1-hour rates when logged; Claude Fast uses its published model-specific premium. Unpublished Fast rates remain unpriced. OpenRouter catalog rates price matched models lacking an embedded rate table. OpenCode input/cache and output/reasoning counters are normalized to avoid overlap. Recorded harness costs and billing units are shown separately below. Subscription charges, tool fees, and regional uplifts are excluded. Reasoning is split out of output; cache reads/writes are split out of input. OpenAI long-context rates apply above 272,000 input tokens where published. Older Claude Sonnet rates change above 200,000; Claude 4.6+ uses standard rates throughout its context window. OpenRouter context thresholds come from the catalog. Aggregate counters without per-request sizes assume normal-context rates, including the base OpenRouter rates without context overrides. Recorded speed-mode premiums still apply where known; actual long-context costs may be higher.</p>
</div><div><h3>Selected-window coverage</h3><div class="table-wrap"><table id="coverage"></table></div><h3 style="margin-top:18px">Unpriced usage</h3><div id="unpriced"></div></div></div>
<details><summary>Token usage by model</summary><div class="table-wrap"><table id="models"></table></div></details>
<details><summary>Parser diagnostics</summary><div class="table-wrap"><table id="diagnostics"></table></div><ul class="warnings" id="warnings"></ul></details>
<details><summary>Exact metrics for every window</summary><div class="table-wrap"><table id="all-metrics"></table></div></details>
<h3 style="margin-top:18px">Recorded costs &amp; billing units</h3><p>Harness totals for the selected window; tier, model, and mode filters do not apply. Recorded USD, credits, and request counters are separate from the API estimate above.</p><div class="table-wrap"><table id="recorded-billing"></table></div><h3 style="margin-top:18px">Matched OpenRouter prices</h3><p>Base catalog rates in USD per million tokens; context overrides are applied per request in estimated costs.</p><div class="table-wrap"><table id="router-prices"></table></div><p class="sources" id="sources"></p>
</section>
<footer class="footer"><span id="footer"></span><span>Generated locally · No conversation content embedded</span></footer>
</main><div class="tooltip" id="tooltip" role="tooltip" hidden></div>
<script type="application/json" id="report-data">__REPORT_DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('report-data').textContent);
const colors=['#3975e7','#159e99','#8c67db','#e99b36','#df7896'];
const metricDefs=[['ttft','Time to first token','seconds','Explicit first-token timing per completed turn'],['throughput','Effective throughput','tokens / second','Output tokens over full turn duration'],['length','Conversation length','minutes','Active duration per conversation'],['tools','Tool calls','calls / conversation','Model-issued calls per conversation']];
const chartStats=[['avg','Average',colors[0]],['median','Median',colors[3]],['p75','P75',colors[4]],['p95','P95',colors[1]],['p99','P99',colors[2]]];
const selectedChartStats=new Set(['p95']);
const visibleChartStats=()=>chartStats.filter(([stat])=>selectedChartStats.has(stat));
const statDefs=[['avg','Average'],['p75','P75'],['p95','P95'],['p99','P99'],['min','Minimum'],['median','Median'],['max','Maximum']];
let selected=0,selectedModel='',selectedMode='',selectedTier='',selectedHarness='',selectedStatistic='avg',selectedGranularity='daily';
const firstDate=data.trend_periods.daily[0].start.slice(0,10),cutoffDate=data.generated.slice(0,10);
let trendStart=firstDate,trendEnd=cutoffDate;
const harnessGroup=()=>selectedHarness?data.by_harness[selectedHarness]:data;
const activeGroup=()=>selectedTier?harnessGroup().by_tier[selectedTier]:harnessGroup();
const activeModels=()=>selectedMode?activeGroup().by_mode[selectedMode].by_model:activeGroup().by_model;
const visibleModels=()=>Object.entries(activeModels()).filter(([,windows])=>windows[selected].total_tokens>0);
const activeTrends=()=>{const group=selectedMode?activeGroup().by_mode[selectedMode]:activeGroup();return (selectedModel?group.by_model_trends[selectedModel]:group.trends)[selectedGranularity]};
const activeWindows=()=>selectedModel?activeModels()[selectedModel]:selectedMode?activeGroup().by_mode[selectedMode].windows:activeGroup().windows;
const $=id=>document.getElementById(id), number=n=>Number(n).toLocaleString('en-US',{maximumFractionDigits:2}), money=n=>Number(n).toLocaleString('en-US',{style:'currency',currency:'USD',minimumFractionDigits:2,maximumFractionDigits:2});
const compact=n=>Number(n).toLocaleString('en-US',{notation:'compact',maximumFractionDigits:1});
const text=(tag,value,cls)=>{const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e};
const fmtDate=value=>new Date(value).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:data.timezone});
const fmtTime=value=>new Date(value).toLocaleString('en-US',{dateStyle:'medium',timeStyle:'short',timeZone:data.timezone});
const displayMetric=(key,value)=>value===null?'N/A':number(key==='length'?value/60:value);
const comparisonMetric=(key,m)=>displayMetric(key,m[selectedStatistic]);
const comparisonLabel=()=>statDefs.find(([stat])=>stat===selectedStatistic)[1];
const svgNode=(tag,attrs={},value)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,v] of Object.entries(attrs))e.setAttribute(key,String(v));if(value!==undefined)e.textContent=value;return e};
function svgBase(label,w,h){const svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':label});svg.append(svgNode('title',{},label));return svg}
function tip(node,label){node.setAttribute('tabindex','0');node.setAttribute('aria-label',label);node.append(svgNode('title',{},label));const show=(event)=>{const rect=node.getBoundingClientRect();const x=event.clientX||rect.x+rect.width/2,y=event.clientY||rect.y;$('tooltip').textContent=label;$('tooltip').hidden=false;$('tooltip').style.left=`${Math.max(8,Math.min(x+12,window.innerWidth-285))}px`;$('tooltip').style.top=`${Math.max(8,Math.min(y+14,window.innerHeight-80))}px`};node.addEventListener('pointermove',show);node.addEventListener('focus',show);for(const kind of ['pointerleave','blur'])node.addEventListener(kind,()=>$('tooltip').hidden=true)}
function table(target,headers,rows){target.replaceChildren();const head=document.createElement('thead'),tr=document.createElement('tr');headers.forEach((h,i)=>tr.append(text('th',h,i?'num':'')));head.append(tr);target.append(head);const body=document.createElement('tbody');rows.forEach(row=>{const r=document.createElement('tr');row.forEach((v,i)=>r.append(text('td',v,i?'num':'')));body.append(r)});target.append(body)}
function metricChart(key,title,unit){
    const statistics=visibleChartStats();
    if(!statistics.length)return text('div','Select a statistic above to display the chart.','empty');
    const active=activeTrends();
    const firstDay=data.trend_periods.daily.find(p=>p.start.slice(0,10)===trendStart),lastDay=data.trend_periods.daily.find(p=>p.start.slice(0,10)===trendEnd),rangeStart=Date.parse(firstDay.start);
    const entries=data.trend_periods[selectedGranularity].map((period,index)=>({period,metrics:active[index]})).filter(({period})=>period.start.slice(0,10)<=trendEnd&&(Date.parse(period.end)>rangeStart||Date.parse(period.end)===rangeStart&&!period.end_exclusive));
    const periods=entries.map(({period})=>period),values=entries.map(({metrics})=>metrics);
    const maxima=values.flatMap(v=>v?statistics.map(([stat])=>v[key][stat]):[]).filter(v=>v!==null).map(v=>key==='length'?v/60:v);
    if(!maxima.length)return text('div','No valid samples in these periods.','empty');
    const svg=svgBase(`${title}: ${selectedGranularity} ${statistics.map(([,label])=>label).join(', ')} from ${trendStart} to ${trendEnd}`,540,230);
    const max=maxima.reduce((largest,value)=>Math.max(largest,value),0.001)*1.12,top=18,bottom=190,left=48,right=530;
    const hourly=selectedGranularity==='hourly',dateNumber=date=>Date.parse(date+'T00:00:00Z');
    const start=hourly?rangeStart:dateNumber(trendStart);
    const end=hourly?Date.parse(lastDay.end):dateNumber(trendEnd);
    const x=date=>end===start?(left+right)/2:left+(right-left)*(date-start)/(end-start);
    for(let i=0;i<=4;i++){
        const y=bottom-(bottom-top)*i/4;
        svg.append(svgNode('line',{x1:left,y1:y,x2:right,y2:y,stroke:'#e9eef4','stroke-dasharray':i?'3 4':'0'}));
        svg.append(svgNode('text',{x:left-7,y:y+3,'text-anchor':'end',fill:'#8290a1','font-size':10},compact(max*i/4)));
    }
    for(const [stat,label,color] of statistics){
        let path='',connected=false;
        const points=[];
        values.forEach((metrics,index)=>{
            const m=metrics?.[key];
            if(!m||m[stat]===null){connected=false;return}
            const value=key==='length'?m[stat]/60:m[stat],at=hourly?Date.parse(periods[index].start):dateNumber(periods[index].start.slice(0,10)),cx=x(Math.max(start,at)),cy=bottom-value/max*(bottom-top);
            path+=`${connected?'L':'M'}${cx},${cy} `;connected=true;
            const point=svgNode('circle',{cx,cy,r:hourly||selectedGranularity==='daily'?2.5:3.5,fill:color,stroke:'#fff','stroke-width':1});
            const period=periods[index];
            tip(point,`${period.label} · ${fmtTime(period.start)} – ${fmtTime(period.end)}${period.end_exclusive?' (end excluded)':''} · ${label}: ${number(value)} ${unit} · ${number(m.count)} samples`);
            points.push(point);
        });
        svg.append(svgNode('path',{d:path,fill:'none',stroke:color,'stroke-width':2,'stroke-linejoin':'round'}),...points);
    }
    const days=Math.round((dateNumber(trendEnd)-dateNumber(trendStart))/86400000),ticks=hourly?(end>start?Math.max(1,Math.min(5,Math.floor((end-start)/3600000))):0):Math.min(5,days);
    for(let i=0;i<=ticks;i++){
        const at=hourly?(i===ticks?end:start+Math.round((end-start)*i/ticks/3600000)*3600000):start+(ticks?Math.round(days*i/ticks):0)*86400000;
        const label=hourly?new Date(at).toLocaleString('en-US',{hour:'numeric',minute:'2-digit',...(days?{month:'short',day:'numeric'}:{}),timeZone:data.timezone}):new Date(at).toLocaleDateString('en-US',{month:'short',day:'numeric',...(days>365?{year:'2-digit'}:{}),timeZone:'UTC'});
        svg.append(svgNode('text',{x:x(at),y:213,'text-anchor':ticks===0?'middle':i===0?'start':i===ticks?'end':'middle',fill:'#8290a1','font-size':10},label));
    }
    return svg;
}
function legend(){const el=text('div','','legend');visibleChartStats().forEach(([,name,color])=>{const s=text('span',name),dot=text('i','','swatch');dot.style.background=color;s.prepend(dot);el.append(s)});return el}
function performance(window){$('trend-caption').textContent=`${trendStart} – ${trendEnd} · calendar period statistics · ${visibleChartStats().map(([,label])=>label).join(', ')||'no statistics selected'}`;$('performance').replaceChildren();for(const [key,title,unit,desc] of metricDefs){const panel=text('article','','panel'),top=text('div','','panel-top'),heading=text('div','');heading.append(text('h3',title),text('p',desc));top.append(heading,text('span',unit,'unit'));panel.append(top,legend());const chart=text('div','','chart');chart.append(metricChart(key,title,unit));panel.append(chart);const stats=text('div','','stats');for(const [stat,label] of statDefs){const e=text('div','','stat');e.append(text('span',label.toUpperCase()),text('strong',displayMetric(key,window.metrics[key][stat])));stats.append(e)}panel.append(stats,text('div',`${number(window.metrics[key].count)} valid ${key==='length'||key==='tools'?'conversation':'turn'} samples · ${window.label}`,'samples'));$('performance').append(panel)}}
function donut(categories,field,total,title){const svg=svgBase(title,190,190),cx=95,cy=95,r=68,length=2*Math.PI*r;svg.append(svgNode('circle',{cx,cy,r,fill:'none',stroke:'#edf1f6','stroke-width':20}));let offset=0;categories.forEach((cat,i)=>{const value=Number(cat[field]);if(value<=0||total<=0)return;const segment=value/total*length,circle=svgNode('circle',{cx,cy,r,fill:'none',stroke:colors[i],'stroke-width':20,'stroke-dasharray':`${segment} ${length-segment}`,'stroke-dashoffset':-offset,transform:'rotate(-90 95 95)'});tip(circle,`${cat.name}: ${field==='cost'?money(value):number(value)} (${number(value/total*100)}%)`);svg.append(circle);offset+=segment});svg.append(svgNode('text',{x:95,y:94,'text-anchor':'middle',fill:'#172a3c','font-size':22,'font-weight':650},field==='cost'?money(total):compact(total)),svgNode('text',{x:95,y:116,'text-anchor':'middle',fill:'#8290a1','font-size':10},field==='cost'?'ESTIMATED USD':'TOTAL TOKENS'));return svg}
function breakdowns(window){$('breakdowns').replaceChildren();for(const field of ['tokens','cost']){const panel=text('article','','panel'),title=field==='tokens'?'Token composition':'Cost composition';panel.append(text('h3',title),text('p',field==='tokens'?'Separate categories · each token counted once':`Estimated API-equivalent token cost${window.partial_cost?' · partial':''}`));const body=text('div','','breakdown-body');body.append(donut(window.categories,field,field==='tokens'?window.total_tokens:Number(window.cost),title));const wrap=text('div','','table-wrap'),t=document.createElement('table');table(t,['Category',field==='tokens'?'Tokens':'USD'],[]);const tbody=t.querySelector('tbody');window.categories.forEach((cat,i)=>{const tr=document.createElement('tr'),label=text('td',cat.name,'label'),dot=text('i','','swatch');dot.style.background=colors[i];dot.style.marginRight='7px';label.prepend(dot);const value=field==='tokens'?number(cat.tokens):money(cat.cost);tr.append(label,text('td',value+(field==='cost'&&cat.unpriced_tokens?' *':''),'num'));tbody.append(tr)});const foot=document.createElement('tfoot'),row=document.createElement('tr');row.append(text('td','Total'),text('td',field==='tokens'?number(window.total_tokens):money(window.cost),'num'));foot.append(row);t.append(foot);wrap.append(t);body.append(wrap);panel.append(body);if(field==='cost'&&window.partial_cost)panel.append(text('p',`* ${number(window.unpriced_tokens)} tokens excluded from the API estimate; see unpriced usage and recorded billing below.`));$('breakdowns').append(panel)}}
function selectWindow(index){selected=index;modelOptions();recordedBilling();$('tooltip').hidden=true;const w=activeWindows()[index];$('model-scope').textContent=`${selectedHarness||'All harnesses'} · ${selectedTier||'All tiers'} · ${selectedModel||'All models'} · ${selectedMode||'All modes'}`;[...$('tabs').children].forEach((b,i)=>b.setAttribute('aria-pressed',i===index));$('range').textContent=w.end_exclusive?`${fmtDate(w.start)} · full calendar day`:`${fmtDate(w.start)} – ${fmtTime(w.end)}`;$('notice').hidden=!w.partial_cost&&!Object.keys(data.quality).some(k=>k.startsWith('Malformed')||k==='Unreadable files'||k==='Invalid usage records');$('notice').textContent=w.partial_cost?`Partial cost estimate: ${number(w.unpriced_tokens)} tokens lack a verified rate or the category detail needed to calculate cost. Their usage is included in token totals.`:'Some records could not be read. Review parser diagnostics below.';$('cards').replaceChildren();const cards=[['Conversations',number(w.conversations),'Active threads, including subagents',colors[0]],['Total tokens',compact(w.total_tokens),`${number(w.total_tokens)} recorded tokens`,colors[1]],['Active duration',number(w.active_seconds/3600)+' h','Completed turn durations, summed',colors[2]],['Estimated cost',money(w.cost),w.partial_cost?'Partial estimate · USD':'USD · API-equivalent estimate',colors[3]]];for(const [label,value,note,color] of cards){const c=text('article','','card');c.style.setProperty('--accent',color);c.append(text('div',label,'card-label'),text('div',value,'card-value'),text('div',note,'card-note'));if(label==='Total tokens'){const totals=document.createElement('dl');totals.className='token-totals';for(const [name,count] of [['Input (includes cached)',w.input_tokens],['Output (includes reasoning)',w.output_tokens],['Cached input',w.cached_input_tokens]]){const row=document.createElement('div');row.append(text('dt',name),text('dd',number(count)));totals.append(row)}c.append(totals)}$('cards').append(c)}performance(w);breakdowns(w);tierComparison();modelComparison();exactMetrics();const coverage=[['Completed turns',w.coverage['Completed turns']||0],['First-token timing samples',w.metrics.ttft.count],['Missing first-token timing',w.coverage['Missing first-token timing']||0],['Missing turn duration',w.coverage['Missing turn duration']||0],['Throughput samples',w.metrics.throughput.count],['Missing throughput samples',w.coverage['Missing throughput samples']||0],['Conversation duration samples',w.metrics.length.count],['Aborted turns',w.coverage['Aborted turns']||0],['Unfinished turns started in window',w.coverage['Unfinished turns']||0],['Usage responses',w.coverage['Usage responses']||0],['Tool calls',w.tool_calls]];table($('coverage'),['Measurement','Count'],coverage.map(([k,v])=>[k,number(v)]));$('unpriced').replaceChildren();if(w.partial_cost){const t=document.createElement('table');table(t,['Model','Unpriced tokens'],Object.entries(w.unpriced).map(([k,v])=>[k,number(v)]));$('unpriced').append(t)}else $('unpriced').append(text('p','All recorded usage in this window has a published or assumed rate.'));table($('models'),['Model','Total tokens'],Object.entries(w.models).filter(([,tokens])=>tokens>0).map(([k,v])=>[k,number(v)]))}
for(const [stat,label] of statDefs){const option=text('option',label);option.value=stat;$('comparison-stat-select').append(option);}
for(const [stat,label,color] of chartStats){
    const option=text('label',''),input=document.createElement('input'),swatch=text('i','','swatch');
    input.type='checkbox';input.value=stat;input.checked=selectedChartStats.has(stat);swatch.style.background=color;
    option.append(input,swatch,document.createTextNode(label));$('trend-stat-options').append(option);
    input.addEventListener('change',()=>{
        if(input.checked)selectedChartStats.add(stat);else selectedChartStats.delete(stat);
        $('tooltip').hidden=true;performance(activeWindows()[selected]);
    });
}
for(const id of ['trend-start','trend-end']){
    const input=$(id);input.min=firstDate;input.max=cutoffDate;input.value=id==='trend-start'?trendStart:trendEnd;
    input.addEventListener('change',()=>{
        const start=$('trend-start'),end=$('trend-end'),error=$('trend-range-error');
        error.hidden=start.validity.valid&&end.validity.valid&&start.value<=end.value;
        if(!error.hidden){error.textContent=`Choose dates from ${firstDate} to ${cutoffDate}, with Start on or before End.`;return}
        trendStart=start.value;trendEnd=end.value;$('tooltip').hidden=true;performance(activeWindows()[selected]);
    });
}
$('granularity-select').addEventListener('change',()=>{selectedGranularity=$('granularity-select').value;$('tooltip').hidden=true;performance(activeWindows()[selected]);});
$('comparison-stat-select').value=selectedStatistic;
$('comparison-stat-select').addEventListener('change',()=>{selectedStatistic=$('comparison-stat-select').value;tierComparison();modelComparison();});
$('subtitle').textContent=`${number(data.files)} log files · ${number(data.threads)} threads · ${data.timezone}`;
data.windows.forEach((w,i)=>{const b=text('button',w.label);b.type='button';b.setAttribute('aria-pressed',i===0);b.addEventListener('click',()=>selectWindow(i));$('tabs').append(b)});
$('pricing-date').textContent=`Pricing: ${data.pricing_date}`;
table($('diagnostics'),['Diagnostic','Count'],Object.entries(data.quality).map(([k,v])=>[k,number(v)]));data.warnings.forEach(w=>$('warnings').append(text('li',w)));
function exactMetrics(){const rows=[];for(const w of activeWindows())for(const [key,title,unit] of metricDefs){const m=w.metrics[key];rows.push([`${w.label} · ${title} (${unit})`,displayMetric(key,m.avg),displayMetric(key,m.min),displayMetric(key,m.median),displayMetric(key,m.max),displayMetric(key,m.p75),displayMetric(key,m.p95),displayMetric(key,m.p99),number(m.count)])}table($('all-metrics'),['Window / metric','Average','Minimum','Median','Maximum','P75','P95','P99','Samples'],rows);}
function modelComparison(){const entries=visibleModels();const rows=entries.map(([model,windows])=>{const w=windows[selected];return [model,number(w.conversations),comparisonMetric('ttft',w.metrics.ttft),comparisonMetric('throughput',w.metrics.throughput),comparisonMetric('length',w.metrics.length),comparisonMetric('tools',w.metrics.tools),number(w.total_tokens),money(w.cost)+(w.partial_cost?' (partial)':'')]});table($('model-comparison'),['Model','Conversations','First token (s)','Throughput (tokens/s)','Length (min)','Calls / conversation','Total tokens','Cost (USD)'],rows);[...$('model-comparison').querySelectorAll('tbody tr')].forEach((row,i)=>row.dataset.selected=entries[i][0]===selectedModel);$('comparison-caption').textContent=`${data.windows[selected].label} · ${comparisonLabel()} per metric · ${selectedTier||'All tiers'} · ${selectedMode||'All modes'} · models with recorded tokens in this window`;}
function modelOptions(){const select=$('model-select');select.replaceChildren(text('option','All models'));select.firstChild.value='';const models=visibleModels().map(([model])=>model);for(const model of models){const option=text('option',model);option.value=model;select.append(option);}if(!models.includes(selectedModel))selectedModel='';select.value=selectedModel;}
$('model-select').addEventListener('change',()=>{selectedModel=$('model-select').value;selectWindow(selected);});
function modeOptions(){const select=$('mode-select');select.replaceChildren(text('option','All modes'));select.firstChild.value='';for(const mode of Object.keys(harnessGroup().by_mode)){const option=text('option',mode);option.value=mode;select.append(option);}if(!(selectedMode in harnessGroup().by_mode))selectedMode='';select.value=selectedMode;}
modeOptions();
$('mode-select').addEventListener('change',()=>{selectedMode=$('mode-select').value;selectWindow(selected);});
function tierComparison(){const entries=Object.entries(harnessGroup().by_tier).map(([tier,group])=>{const scope=selectedMode?group.by_mode[selectedMode]:group;return [tier,(selectedModel?scope.by_model[selectedModel]:scope.windows)?.[selected]];}).filter(([,w])=>w);const rows=entries.map(([tier,w])=>[tier,number(w.conversations),comparisonMetric('ttft',w.metrics.ttft),comparisonMetric('throughput',w.metrics.throughput),comparisonMetric('length',w.metrics.length),comparisonMetric('tools',w.metrics.tools),number(w.total_tokens),money(w.cost)+(w.partial_cost?' (partial)':'')]);table($('tier-comparison'),['Tier','Conversations','First token (s)','Throughput (tokens/s)','Length (min)','Calls / conversation','Total tokens','Cost (USD)'],rows);[...$('tier-comparison').querySelectorAll('tbody tr')].forEach((row,i)=>row.dataset.selected=entries[i][0]===selectedTier);$('tier-caption').textContent=`${data.windows[selected].label} · ${comparisonLabel()} per metric · ${selectedMode||'All modes'} · ${selectedModel||'All models'} · all tiers`;}
function tierOptions(){const select=$('tier-select');select.replaceChildren(text('option','All tiers'));select.firstChild.value='';for(const tier of Object.keys(harnessGroup().by_tier)){const option=text('option',tier);option.value=tier;select.append(option);}if(!(selectedTier in harnessGroup().by_tier))selectedTier='';select.value=selectedTier;}
tierOptions();
$('tier-select').addEventListener('change',()=>{selectedTier=$('tier-select').value;selectWindow(selected);});
$('sources').append(document.createTextNode(`Rates verified ${data.pricing_date}: `));const link=text('a','OpenAI API pricing');link.href=data.pricing_source;link.rel='noreferrer';$('sources').append(link,document.createTextNode('. Rates are embedded in the script and are not updated automatically. Input directories: '+data.sources.join(', ')));
const anthropicLink=text('a','Anthropic API pricing');anthropicLink.href=data.anthropic_pricing_source;anthropicLink.rel='noreferrer';$('sources').append(document.createTextNode(` · Anthropic verified ${data.anthropic_pricing_date}: `),anthropicLink);if(data.openrouter){const routerLink=text('a','OpenRouter model catalog');routerLink.href=data.openrouter.source;routerLink.rel='noreferrer';const origin=data.openrouter.bundled?' · bundled snapshot':data.openrouter.snapshot_file?' · supplied snapshot':' · live catalog';const date=data.openrouter.retrieved?` retrieved ${fmtTime(data.openrouter.retrieved)}`:' (retrieval date unknown)';const error=data.openrouter.error?` · live fetch unavailable: ${data.openrouter.error}; using bundled prices`:'';$('sources').append(document.createTextNode(' · '),routerLink,document.createTextNode(origin+date+error));}
$('sources').append(document.createTextNode(' · '));const modeLink=text('a','Fast mode documentation');modeLink.href='https://developers.openai.com/api/docs/guides/fast-mode';modeLink.rel='noreferrer';$('sources').append(modeLink);
$('footer').textContent=`Report cutoff: ${fmtTime(data.generated)} (${data.timezone})`;
for(const harness of Object.keys(data.by_harness)){const option=text('option',harness);option.value=harness;$('harness-select').append(option);}
$('harness-select').addEventListener('change',()=>{selectedHarness=$('harness-select').value;modeOptions();tierOptions();selectWindow(selected);});
function recordedBilling(){const units=harnessGroup().windows[selected].recorded_billing;table($('recorded-billing'),['Recorded measurement','Amount'],Object.entries(units).map(([unit,amount])=>[unit,unit.endsWith('USD')?money(amount):number(amount)]));table($('router-prices'),['Model','OpenRouter ID','Input / MTok','Cache read / MTok','Output / MTok'],Object.entries(data.openrouter_rates).filter(([model])=>visibleModels().some(([name])=>name===model)).map(([model,r])=>[model,r.id,money(r.input),r.cached===null?'N/A':money(r.cached),money(r.output)]));}
selectWindow(0);
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
