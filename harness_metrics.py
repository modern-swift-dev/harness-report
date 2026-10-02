#!/usr/bin/env python3
"""Build an offline coding-agent usage report. Python 3.10+; no dependencies.

    python3 harness_metrics.py [directory] --output report.html

Supports Codex, Claude Code, Copilot CLI, and OpenCode local storage.
With no directory, discovers installed harnesses; use --harness to select them.
Codex discovery includes ~/.codex/sessions and ~/.codex/archived_sessions.
An explicit directory reads only that Codex archive plus supplied source paths.
Calendar windows use --timezone (Toronto by default, or UTC without timezone data).
OpenRouter prices are fetched once when generating a report unless --offline
or --openrouter-prices is supplied. Reports embed a snapshot and work offline.
Local storage is read only; costs are API estimates, not subscription bills.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
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
from typing import Any, Iterable
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


class Harness(str, Enum):
    CODEX = "codex"
    CLAUDE = "claude"
    COPILOT = "copilot"
    OPENCODE = "opencode"


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
    # A session aggregate cannot establish per-request long-context pricing.
    if aggregate and price and price.is_long_context(usage.input):
        price = None
    multiplier = (price.fast_multiplier if price else None) if mode == SpeedMode.FAST else Decimal(1)
    selected = (price.long if price.long and usage.input > price.threshold else price.short) if price else None
    if price:
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


@dataclass
class Window:
    label: str
    start: datetime
    end: datetime
    end_exclusive: bool = False
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


@dataclass
class ModeBreakdown:
    windows: list[Window]
    by_model: dict[str, list[Window]] = field(default_factory=dict)


@dataclass
class TierBreakdown:
    windows: list[Window]
    by_model: dict[str, list[Window]] = field(default_factory=dict)
    by_mode: dict[SpeedMode, ModeBreakdown] = field(default_factory=dict)


def empty_windows(windows: list[Window]) -> list[Window]:
    return [Window(w.label, w.start, w.end, end_exclusive=w.end_exclusive) for w in windows]


def add_thread(windows: list[Window], thread: Thread,
               by_model: dict[str, list[Window]] | None = None,
               by_mode: dict[SpeedMode, ModeBreakdown] | None = None,
               by_tier: dict[ModelTier, TierBreakdown] | None = None,
               catalog: dict[str, Price] | None = None) -> None:
    def targets(model: str | None, mode: SpeedMode, tier: ModelTier,
                long_context: bool = False) -> list[Window]:
        result = list(windows)
        name = report_model(model, mode, long_context)
        if by_model is not None:
            if name not in by_model:
                by_model[name] = empty_windows(windows)
            result += by_model[name]
        if by_mode is not None:
            if mode not in by_mode:
                by_mode[mode] = ModeBreakdown(empty_windows(windows))
            group = by_mode[mode]
            if name not in group.by_model:
                group.by_model[name] = empty_windows(windows)
            result += group.windows + group.by_model[name]
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
            result += (tier_group.windows + tier_group.by_model[name] +
                       mode_group.windows + mode_group.by_model[name])
        return result

    for turn in thread.turns.values():
        usage_events = turn.usage
        for record in usage_events:
            model = record.model or turn.model
            long_context = is_long_context(model, record.usage, catalog, record.aggregate)
            name = report_model(model, record.mode, long_context)
            costs, unpriced = price_usage(model, record.usage, record.mode, catalog,
                                          aggregate=record.aggregate)
            for window in targets(model, record.mode, model_tier(model), long_context):
                if not window.contains(record.at):
                    continue
                if record.usage.total:
                    window.conversations.add(thread.id)
                window.tokens.update(record.usage.categories())
                window.models[name] += record.usage.total
                for category, cost in costs.items():
                    window.costs[category] += cost
                window.unpriced_categories.update(unpriced)
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
        usage_contexts = {is_long_context(r.model or turn.model, r.usage, catalog, r.aggregate)
                          for r in usage_events}
        if len(usage_contexts) > 1 and len(usage_models) == 1 and len(usage_modes) == 1:
            timing_model = "Mixed contexts (timing)"
        usage_tiers = {model_tier(r.model or turn.model) for r in usage_events}
        timing_tier = (ModelTier.MIXED if len(usage_tiers) > 1 else
                       next(iter(usage_tiers)) if usage_tiers else model_tier(turn.model))
        turn_windows = targets(timing_model, timing_mode, timing_tier, usage_contexts == {True})
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
        call_contexts = {is_long_context(r.model or turn.model, r.usage, catalog, r.aggregate)
                         for r in turn.usage
                         if (r.model or turn.model) == model and r.mode == call.mode} if turn else set()
        call_model = "Mixed contexts (tools)" if len(call_contexts) > 1 else model
        for window in targets(call_model, call.mode, model_tier(model), call_contexts == {True}):
            if window.contains(call.at):
                window.conversations.add(thread.id)
                window.call_counts[thread.id] += 1
    # Native billing counters are retained separately from token-rate estimates.
    # Per-model attribution is unavailable for some harness summaries.
    for event in thread.billing:
        for window in windows:
            if window.contains(event.at):
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


def summarize(window: Window, report_zone: tzinfo = TIMEZONE) -> dict[str, Any]:
    categories = [c for c in Category if c != Category.CACHE_WRITE or window.tokens[c]]
    return {
        "label": window.label, "start": window.start.astimezone(report_zone).isoformat(),
        "end": window.end.astimezone(report_zone).isoformat(),
        "end_exclusive": window.end_exclusive,
        "conversations": len(window.conversations), "total_tokens": sum(window.tokens.values()),
        "active_seconds": sum(window.durations.values()), "tool_calls": sum(window.call_counts.values()),
        "cost": str(sum(window.costs.values(), Decimal(0))),
        "partial_cost": bool(window.unpriced), "unpriced_tokens": sum(window.unpriced.values()),
        "metrics": {"ttft": distribution(window.ttft), "throughput": distribution(window.throughput),
                    "length": distribution(window.durations.values()),
                    "tools": distribution(window.call_counts[c] for c in window.conversations)},
        "categories": [{"name": c.value, "tokens": window.tokens[c],
                        "cost": str(window.costs[c]), "unpriced_tokens": window.unpriced_categories[c]}
                       for c in categories],
        "unpriced": dict(sorted(window.unpriced.items())),
        "models": dict(sorted(window.models.items())), "coverage": dict(window.coverage),
        "recorded_billing": {unit: str(amount) for unit, amount in sorted(window.billing.items())},
    }


def build_breakdown(threads: Iterable[Thread], now: datetime,
                    catalog: dict[str, Price] | None = None,
                    report_zone: tzinfo = TIMEZONE) -> dict[str, Any]:
    windows = make_windows(now, report_zone)
    by_model: dict[str, list[Window]] = {}
    by_mode = {mode: ModeBreakdown(empty_windows(windows))
               for mode in (SpeedMode.NORMAL, SpeedMode.FAST)}
    by_tier = {tier: TierBreakdown(empty_windows(windows))
               for tier in (ModelTier.BUDGET, ModelTier.MEDIUM, ModelTier.HIGH, ModelTier.UNCLASSIFIED)}
    for thread in threads:
        add_thread(windows, thread, by_model, by_mode, by_tier, catalog)
    active_models = {name: ws for name, ws in sorted(by_model.items()) if any(w.conversations for w in ws)}
    tier_summaries: dict[str, Any] = {}
    for tier, group in by_tier.items():
        members = {name: ws for name, ws in sorted(group.by_model.items()) if any(w.conversations for w in ws)}
        modes: dict[str, Any] = {}
        for mode in by_mode:
            mode_group = group.by_mode.get(mode, ModeBreakdown(empty_windows(windows)))
            modes[mode.value] = {"windows": [summarize(w, report_zone) for w in mode_group.windows],
                                 "by_model": {name: [summarize(w, report_zone) for w in mode_group.by_model.get(name, empty_windows(windows))]
                                              for name in members}}
        tier_summaries[tier.value] = {"windows": [summarize(w, report_zone) for w in group.windows],
                                     "by_model": {name: [summarize(w, report_zone) for w in ws] for name, ws in members.items()},
                                     "by_mode": modes}
    return {"by_tier": tier_summaries,
            "windows": [summarize(w, report_zone) for w in windows],
            "by_model": {model: [summarize(w, report_zone) for w in model_windows]
                         for model, model_windows in active_models.items()},
            "by_mode": {mode.value: {"windows": [summarize(w, report_zone) for w in group.windows],
                        "by_model": {model: [summarize(w, report_zone) for w in group.by_model.get(model, empty_windows(windows))]
                                     for model in active_models}}
                        for mode, group in by_mode.items()}}


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


def collect_report(root: Path, now: datetime | None = None, progress: bool = False,
                   additional_roots: Iterable[Path] = (),
                   harness_roots: dict[Harness, list[Path]] | None = None,
                   include_codex: bool = True, catalog: dict[str, Price] | None = None,
                   catalog_metadata: dict[str, Any] | None = None,
                   report_zone: tzinfo = TIMEZONE) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("report cutoff must include a timezone")
    now = now.astimezone(timezone.utc)
    quality = Quality()
    sources = {h: list(dict.fromkeys(p.resolve() for p in paths)) for h, paths in (harness_roots or {}).items()}
    if include_codex:
        sources[Harness.CODEX] = list(dict.fromkeys(p.resolve() for p in (root, *additional_roots)))
    roots = list(dict.fromkeys(p for paths in sources.values() for p in paths))
    threads: list[Thread] = []
    all_paths: set[Path] = set()
    readers = {Harness.CODEX: read_thread, Harness.CLAUDE: read_claude, Harness.COPILOT: read_copilot}
    for harness, directories in sources.items():
        if harness in readers:
            paths = sorted({p.resolve() for directory in directories for p in directory.rglob("*.jsonl")})
            groups: dict[str, list[Path]] = defaultdict(list)
            for path in paths:
                groups[session_identity(path, harness)].append(path)
            for index, (thread_id, files) in enumerate(groups.items(), 1):
                try:
                    threads.append(readers[harness](thread_id, files, quality))
                except (OSError, UnicodeError) as error:
                    quality.warn("Unreadable files", files[0], 0, str(error))
                if progress and (index % 250 == 0 or index == len(groups)):
                    print(f"{harness.value}: read {index:,}/{len(groups):,} conversations", file=sys.stderr)
        else:
            filename = "opencode.db"
            paths = sorted({(directory / filename if directory.is_dir() else directory).resolve() for directory in directories})
            threads.extend(read_opencode(paths, quality))
        all_paths.update(paths)
    for thread in threads:
        thread.id = f"{thread.harness.value}:{thread.id}"
    report = build_breakdown(threads, now, catalog, report_zone)
    report.update({"generated": now.astimezone(report_zone).isoformat(), "timezone": str(report_zone),
                   "source": str(root.resolve()), "sources": [str(p) for p in roots],
                   "files": len(all_paths), "threads": len(threads),
                   "pricing_date": PRICING_DATE, "pricing_source": PRICING_SOURCE,
                   "anthropic_pricing_date": ANTHROPIC_PRICING_DATE,
                   "anthropic_pricing_source": ANTHROPIC_PRICING_SOURCE,
                   "openrouter": catalog_metadata,
                   "price_proxies": PRICE_PROXIES, "fast_cost_multiplier": str(FAST_COST_MULTIPLIER),
                   "unknown_mode_assumption": SpeedMode.NORMAL.value,
                   "by_harness": {h.value: build_breakdown([t for t in threads if t.harness == h], now, catalog, report_zone) for h in sources},
                   "quality": dict(quality.counts), "warnings": quality.warnings})
    # Include only relevant matched catalog rows in the offline artifact.
    models = {report_model(record.model or turn.model, record.mode,
                           is_long_context(record.model or turn.model, record.usage, catalog, record.aggregate)):
              record.model or turn.model
              for thread in threads for turn in thread.turns.values() for record in turn.usage}
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", type=Path,
                        help="Codex archive directory; supplying one disables automatic local source discovery")
    parser.add_argument("--output", type=Path, default=Path("harness_metrics.html"), help="HTML output (default: harness_metrics.html)")
    parser.add_argument("--timezone", type=report_timezone, default=TIMEZONE,
                        help=f"calendar window timezone (default: {TIMEZONE}; UTC works without timezone data)")
    parser.add_argument("--harness", action="append", choices=[h.value for h in Harness],
                        help="include this harness (repeatable; default: all installed harnesses)")
    parser.add_argument("--claude-dir", type=Path, help="Claude projects directory (default: ~/.claude/projects)")
    parser.add_argument("--copilot-dir", type=Path, help="Copilot session-state directory (default: ~/.copilot/session-state)")
    parser.add_argument("--opencode-dir", type=Path, help="OpenCode data directory or opencode.db path")
    parser.add_argument("--openrouter-prices", type=Path, help="saved /api/v1/models JSON catalog; avoids a network request")
    parser.add_argument("--offline", action="store_true", help="skip live OpenRouter pricing; embedded/supplied prices still work")
    args = parser.parse_args()
    root = args.directory if args.directory is not None else Path.cwd()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    if args.output.suffix.lower() != ".html":
        parser.error("output must have an .html extension")
    selected = {Harness(name) for name in args.harness} if args.harness else set(Harness)
    home = Path.home()
    defaults = {Harness.CLAUDE: Path(os.environ.get("CLAUDE_CONFIG_DIR", str(home / ".claude"))) / "projects",
                Harness.COPILOT: home / ".copilot" / "session-state",
                Harness.OPENCODE: Path(os.environ.get("XDG_DATA_HOME", str(home / ".local" / "share"))) / "opencode"}
    supplied = {Harness.CLAUDE: args.claude_dir, Harness.COPILOT: args.copilot_dir,
                Harness.OPENCODE: args.opencode_dir}
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
        if existing:
            harness_roots[harness] = existing
    catalog: dict[str, Price] = {}
    catalog_metadata: dict[str, Any] | None = None
    try:
        if args.openrouter_prices:
            catalog = openrouter_prices(json.loads(args.openrouter_prices.read_text(encoding="utf-8")))
            catalog_metadata = {"source": OPENROUTER_SOURCE, "snapshot_file": str(args.openrouter_prices.resolve()),
                                "retrieved": None}
        elif harness_roots and not args.offline:
            print("Fetching OpenRouter model prices…", file=sys.stderr)
            request = Request(OPENROUTER_SOURCE, headers={"User-Agent": "coding-agent-metrics/1.0"})
            with urlopen(request, timeout=10) as response:
                catalog = openrouter_prices(json.load(response))
            catalog_metadata = {"source": OPENROUTER_SOURCE, "retrieved": datetime.now(timezone.utc).isoformat()}
    except (OSError, ValueError) as error:
        if args.openrouter_prices:
            parser.error(f"unable to load OpenRouter catalog: {error}")
        print(f"OpenRouter prices unavailable: {error}; unmatched usage remains unpriced", file=sys.stderr)
        catalog_metadata = {"source": OPENROUTER_SOURCE, "error": str(error), "retrieved": None}
    try:
        codex_roots = [home / ".codex" / name for name in ("sessions", "archived_sessions")]
        report = collect_report(root, progress=True,
                                additional_roots=[path for path in codex_roots if path.is_dir()] if args.directory is None else [],
                                harness_roots=harness_roots, include_codex=Harness.CODEX in selected,
                                catalog=catalog, catalog_metadata=catalog_metadata, report_zone=args.timezone)
        args.output.write_text(render_report(report), encoding="utf-8")
    except OSError as error:
        print(f"Unable to write report: {error}", file=sys.stderr)
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
.filters{display:flex;gap:16px;flex-wrap:wrap}.model-filter{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:12px}.model-filter select{font:inherit;font-weight:600;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 34px 10px 12px;max-width:100%}.model-filter select:focus-visible{outline:3px solid #3975e780;outline-offset:3px}.model-comparison{margin-bottom:28px}.comparison-controls{align-items:center;flex-wrap:wrap}.model-comparison table{min-width:800px;margin:16px 0 10px}.model-comparison th,.model-comparison td{white-space:nowrap;padding:13px 10px}.model-comparison td:first-child{font-weight:600}.model-comparison th:first-child,.model-comparison td:first-child{position:sticky;left:0;background:#fff}.model-comparison tr[data-selected=true],.model-comparison tr[data-selected=true] td:first-child{background:#f1f5fd}.filter-scope{color:var(--muted);font-size:12px;margin:-6px 0 18px}
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
<div class="section-head"><h2>Performance across windows</h2><p>Average, median &amp; tail percentiles · overlapping periods are compared independently</p></div>
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
<p class="definition"><b>Distribution statistics.</b> Average is the arithmetic mean. Median is the middle sample, or the average of the two middle samples for an even count. Minimum and maximum are observed extremes. P75, P95, and P99 use nearest rank. All statistics use the same valid samples; each completed turn receives equal weight for timing and throughput.</p>
<p class="definition"><b>Window boundaries.</b> Tokens and calls use record time; turn metrics use completion time. Today starts at midnight in the report timezone and ends at the report cutoff. Yesterday is the preceding calendar day in that timezone, excluding today's midnight. Rolling windows are exact 24-hour days. The full duration of a turn finishing in the window is assigned to that window.</p>
<p class="definition"><b>Model attribution.</b> Fast usage has a separate model entry with a “-fast” suffix. Requests crossing a published context-pricing threshold have a “-long” suffix, including cached input when selecting the threshold; combined usage has “-fast-long”. Usage below the threshold keeps the model name unless Fast. Models without context pricing and aggregate records without per-request sizes do not receive “-long”. Tokens, calls, timing, and costs are separated by recorded mode, with the Fast premium applied to the underlying model's rates. Tokens and calls use their recorded model, falling back to the turn model. Timing uses the model generating that turn. Conversation duration and tool counts include only that model's activity; a thread using multiple models or modes appears in each, so conversation counts are not additive. If several models generate output within one turn, its timing is listed under “Mixed models (timing)”. If one model uses several modes within a turn, its timing is listed under “Mixed modes (timing)” because separate durations cannot be recovered. If one model in one mode crosses context thresholds within a turn, its timing is listed under “Mixed contexts (timing)”. Tool calls follow the context class of matching model and mode usage in their turn; ambiguous calls are listed under “Mixed contexts (tools)”. Mode and tier totals retain this activity once.</p>
<p class="definition"><b>Model tiers.</b> Budget: Luna, Terra, GPT-5.4-mini, Spark, codex-auto-review, and Claude Haiku. Medium: Sol, GPT-5.4, GPT-5.5, and Claude Sonnet. High: Astra, Claude Opus, Fable, and Mythos. Models outside these groups are Unclassified. Tier metrics are calculated from underlying activity, with each conversation counted once per tier. Turns using several models in the same tier retain their timing in that tier; turns spanning tiers have timing under “Mixed tiers (timing)”. Per-model pricing and the Fast premium still apply. The model selector and comparison table show entries with recorded tokens in the selected window, harness, tier, and mode. Zero-token entries, including shared timing and tool-call buckets, remain included in aggregate totals and coverage.</p>
<p class="definition"><b>Mode attribution.</b> Logged service tier “default” is Normal; “priority” or “fast” is Fast. Settings persist until changed. Per the selected assumption, unknown mode—including missing evidence, explicit null, and “auto”—is counted as Normal in all metrics and costs. Other explicit tiers have their own bucket. Tokens and calls follow their recorded tier or the latest logged settings. This combines logged mode with the Normal assumption; a backend fallback cannot be detected without a response tier. A turn with usage in several modes has its timing under “Mixed modes (timing)” because separate durations are unavailable. Conversation durations and calls include only activity attributed to the selected mode.</p>
<p class="definition"><b>Harness coverage.</b> Claude Code reads project JSONL files, Copilot reads CLI session events, and OpenCode reads its message and tool tables. Copilot shutdown-only totals are assigned to shutdown, which does not establish when individual requests occurred. Timing samples require logged timing evidence.</p>
<p class="definition"><b>Coverage.</b> Logs in the listed input directories include archived and active sessions. Copies sharing a conversation ID are merged; repeated usage responses, tool calls, and turn completions are counted once. Active logs are read while they may still be growing; the report cutoff limits included activity. Unfinished turns contribute recorded tokens and calls, with completion timings excluded. An older-window label does not imply a complete year of available history. Missing durations are excluded, so conversation duration can be partial.</p>
<p class="definition"><b>Cost estimate.</b> Current standard API rates are applied to every historical window, with an assumed 50% premium on OpenAI token categories recorded in Fast mode. Normal, including assumed Normal activity, uses base rates. Other explicit tiers also use base rates; their actual premiums are unknown. Codex 5.3 Spark uses GPT-5.4-mini rates and codex-auto-review uses GPT-5.6-luna rates as user-selected proxies, not published prices for those models. These are API-equivalent estimates, not subscription bills. Claude cache writes include separate 5-minute and 1-hour rates when logged; Claude Fast uses its published model-specific premium. Unpublished Fast rates remain unpriced. OpenRouter catalog rates price matched models lacking an embedded rate table. OpenCode input/cache and output/reasoning counters are normalized to avoid overlap. Recorded harness costs and billing units are shown separately below. Subscription charges, tool fees, and regional uplifts are excluded. Reasoning is split out of output; cache reads/writes are split out of input. OpenAI long-context rates apply above 272,000 input tokens where published. Older Claude Sonnet rates change above 200,000; Claude 4.6+ uses standard rates throughout its context window. OpenRouter context thresholds come from the catalog. Aggregate counters remain unpriced when missing per-request sizes prevent selecting the correct context rate.</p>
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
const statDefs=[['avg','Average'],['p75','P75'],['p95','P95'],['p99','P99'],['min','Minimum'],['median','Median'],['max','Maximum']];
let selected=0,selectedModel='',selectedMode='',selectedTier='',selectedHarness='',selectedStatistic='avg';
const harnessGroup=()=>selectedHarness?data.by_harness[selectedHarness]:data;
const activeGroup=()=>selectedTier?harnessGroup().by_tier[selectedTier]:harnessGroup();
const activeModels=()=>selectedMode?activeGroup().by_mode[selectedMode].by_model:activeGroup().by_model;
const visibleModels=()=>Object.entries(activeModels()).filter(([,windows])=>windows[selected].total_tokens>0);
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
function metricChart(key,title,unit){const windows=activeWindows();const values=windows.map(w=>w.metrics[key]);const maxima=values.flatMap(v=>chartStats.map(([stat])=>v[stat])).filter(v=>v!==null).map(v=>key==='length'?v/60:v);if(!maxima.length)return text('div','No valid samples in these windows.','empty');const svg=svgBase(`${title}: average, median, P75, P95 and P99 for ${windows.length} windows`,540,230),max=Math.max(...maxima,0.001)*1.12,top=18,bottom=190,left=48,right=530;for(let i=0;i<=4;i++){const y=bottom-(bottom-top)*i/4;svg.append(svgNode('line',{x1:left,y1:y,x2:right,y2:y,stroke:'#e9eef4','stroke-dasharray':i?'3 4':'0'}));svg.append(svgNode('text',{x:left-7,y:y+3,'text-anchor':'end',fill:'#8290a1','font-size':10},compact(max*i/4)))}const group=(right-left)/windows.length,barWidth=Math.min(10,group*.14),stride=barWidth+Math.min(3,group*.025);values.forEach((v,index)=>{const center=left+group*(index+.5);if(index===selected)svg.append(svgNode('rect',{x:center-group*.45,y:top-5,width:group*.9,height:bottom-top+10,rx:6,fill:'#f1f5fd'}));chartStats.forEach(([stat,label,color],s)=>{if(v[stat]===null)return;const value=key==='length'?v[stat]/60:v[stat],height=Math.max(1,value/max*(bottom-top)),bar=svgNode('rect',{x:center-stride*chartStats.length/2+s*stride,y:bottom-height,width:barWidth,height,rx:3,fill:color,opacity:index===selected?1:.65});tip(bar,`${windows[index].label} · ${label}: ${number(value)} ${unit} · ${number(v.count)} samples`);svg.append(bar)});svg.append(svgNode('text',{x:center,y:213,'text-anchor':'middle',fill:index===selected?'#172a3c':'#8290a1','font-size':10,'font-weight':index===selected?700:400},windows[index].label==='Yesterday'?'Yest.':windows[index].label.replace(/^Last (\d+) days$/,'$1d')))});return svg}
function legend(){const el=text('div','','legend');chartStats.forEach(([,name,color])=>{const s=text('span',name),dot=text('i','','swatch');dot.style.background=color;s.prepend(dot);el.append(s)});return el}
function performance(window){$('performance').replaceChildren();for(const [key,title,unit,desc] of metricDefs){const panel=text('article','','panel'),top=text('div','','panel-top'),heading=text('div','');heading.append(text('h3',title),text('p',desc));top.append(heading,text('span',unit,'unit'));panel.append(top,legend());const chart=text('div','','chart');chart.append(metricChart(key,title,unit));panel.append(chart);const stats=text('div','','stats');for(const [stat,label] of statDefs){const e=text('div','','stat');e.append(text('span',label.toUpperCase()),text('strong',displayMetric(key,window.metrics[key][stat])));stats.append(e)}panel.append(stats,text('div',`${number(window.metrics[key].count)} valid ${key==='length'||key==='tools'?'conversation':'turn'} samples · ${window.label}`,'samples'));$('performance').append(panel)}}
function donut(categories,field,total,title){const svg=svgBase(title,190,190),cx=95,cy=95,r=68,length=2*Math.PI*r;svg.append(svgNode('circle',{cx,cy,r,fill:'none',stroke:'#edf1f6','stroke-width':20}));let offset=0;categories.forEach((cat,i)=>{const value=Number(cat[field]);if(value<=0||total<=0)return;const segment=value/total*length,circle=svgNode('circle',{cx,cy,r,fill:'none',stroke:colors[i],'stroke-width':20,'stroke-dasharray':`${segment} ${length-segment}`,'stroke-dashoffset':-offset,transform:'rotate(-90 95 95)'});tip(circle,`${cat.name}: ${field==='cost'?money(value):number(value)} (${number(value/total*100)}%)`);svg.append(circle);offset+=segment});svg.append(svgNode('text',{x:95,y:94,'text-anchor':'middle',fill:'#172a3c','font-size':22,'font-weight':650},field==='cost'?money(total):compact(total)),svgNode('text',{x:95,y:116,'text-anchor':'middle',fill:'#8290a1','font-size':10},field==='cost'?'ESTIMATED USD':'TOTAL TOKENS'));return svg}
function breakdowns(window){$('breakdowns').replaceChildren();for(const field of ['tokens','cost']){const panel=text('article','','panel'),title=field==='tokens'?'Token composition':'Cost composition';panel.append(text('h3',title),text('p',field==='tokens'?'Separate categories · each token counted once':`Estimated API-equivalent token cost${window.partial_cost?' · partial':''}`));const body=text('div','','breakdown-body');body.append(donut(window.categories,field,field==='tokens'?window.total_tokens:Number(window.cost),title));const wrap=text('div','','table-wrap'),t=document.createElement('table');table(t,['Category',field==='tokens'?'Tokens':'USD'],[]);const tbody=t.querySelector('tbody');window.categories.forEach((cat,i)=>{const tr=document.createElement('tr'),label=text('td',cat.name,'label'),dot=text('i','','swatch');dot.style.background=colors[i];dot.style.marginRight='7px';label.prepend(dot);const value=field==='tokens'?number(cat.tokens):money(cat.cost);tr.append(label,text('td',value+(field==='cost'&&cat.unpriced_tokens?' *':''),'num'));tbody.append(tr)});const foot=document.createElement('tfoot'),row=document.createElement('tr');row.append(text('td','Total'),text('td',field==='tokens'?number(window.total_tokens):money(window.cost),'num'));foot.append(row);t.append(foot);wrap.append(t);body.append(wrap);panel.append(body);if(field==='cost'&&window.partial_cost)panel.append(text('p',`* ${number(window.unpriced_tokens)} tokens excluded from the API estimate; see unpriced usage and recorded billing below.`));$('breakdowns').append(panel)}}
function selectWindow(index){selected=index;modelOptions();recordedBilling();$('tooltip').hidden=true;const w=activeWindows()[index];$('model-scope').textContent=`${selectedHarness||'All harnesses'} · ${selectedTier||'All tiers'} · ${selectedModel||'All models'} · ${selectedMode||'All modes'}`;[...$('tabs').children].forEach((b,i)=>b.setAttribute('aria-pressed',i===index));$('range').textContent=w.end_exclusive?`${fmtDate(w.start)} · full calendar day`:`${fmtDate(w.start)} – ${fmtTime(w.end)}`;$('notice').hidden=!w.partial_cost&&!Object.keys(data.quality).some(k=>k.startsWith('Malformed')||k==='Unreadable files'||k==='Invalid usage records');$('notice').textContent=w.partial_cost?`Partial cost estimate: ${number(w.unpriced_tokens)} tokens lack a verified rate or the category detail needed to calculate cost. Their usage is included in token totals.`:'Some records could not be read. Review parser diagnostics below.';$('cards').replaceChildren();const cards=[['Conversations',number(w.conversations),'Active threads, including subagents',colors[0]],['Total tokens',compact(w.total_tokens),`${number(w.total_tokens)} recorded tokens`,colors[1]],['Active duration',number(w.active_seconds/3600)+' h','Completed turn durations, summed',colors[2]],['Estimated cost',money(w.cost),w.partial_cost?'Partial estimate · USD':'USD · API-equivalent estimate',colors[3]]];for(const [label,value,note,color] of cards){const c=text('article','','card');c.style.setProperty('--accent',color);c.append(text('div',label,'card-label'),text('div',value,'card-value'),text('div',note,'card-note'));$('cards').append(c)}performance(w);breakdowns(w);tierComparison();modelComparison();exactMetrics();const coverage=[['Completed turns',w.coverage['Completed turns']||0],['First-token timing samples',w.metrics.ttft.count],['Missing first-token timing',w.coverage['Missing first-token timing']||0],['Missing turn duration',w.coverage['Missing turn duration']||0],['Throughput samples',w.metrics.throughput.count],['Missing throughput samples',w.coverage['Missing throughput samples']||0],['Conversation duration samples',w.metrics.length.count],['Aborted turns',w.coverage['Aborted turns']||0],['Unfinished turns started in window',w.coverage['Unfinished turns']||0],['Usage responses',w.coverage['Usage responses']||0],['Tool calls',w.tool_calls]];table($('coverage'),['Measurement','Count'],coverage.map(([k,v])=>[k,number(v)]));$('unpriced').replaceChildren();if(w.partial_cost){const t=document.createElement('table');table(t,['Model','Unpriced tokens'],Object.entries(w.unpriced).map(([k,v])=>[k,number(v)]));$('unpriced').append(t)}else $('unpriced').append(text('p','All recorded usage in this window has a published or assumed rate.'));table($('models'),['Model','Total tokens'],Object.entries(w.models).filter(([,tokens])=>tokens>0).map(([k,v])=>[k,number(v)]))}
for(const [stat,label] of statDefs){const option=text('option',label);option.value=stat;$('comparison-stat-select').append(option);}
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
const anthropicLink=text('a','Anthropic API pricing');anthropicLink.href=data.anthropic_pricing_source;anthropicLink.rel='noreferrer';$('sources').append(document.createTextNode(` · Anthropic verified ${data.anthropic_pricing_date}: `),anthropicLink);if(data.openrouter){const routerLink=text('a','OpenRouter model catalog');routerLink.href=data.openrouter.source;routerLink.rel='noreferrer';$('sources').append(document.createTextNode(' · '),routerLink,document.createTextNode(data.openrouter.error?` unavailable: ${data.openrouter.error}`:data.openrouter.retrieved?` retrieved ${fmtTime(data.openrouter.retrieved)}`:' · supplied snapshot (retrieval date unknown)'));}
$('sources').append(document.createTextNode(' · '));const modeLink=text('a','Fast mode documentation');modeLink.href='https://developers.openai.com/api/docs/guides/fast-mode';modeLink.rel='noreferrer';$('sources').append(modeLink);
$('footer').textContent=`Report cutoff: ${fmtTime(data.generated)} (${data.timezone})`;
for(const harness of Object.keys(data.by_harness)){const option=text('option',harness);option.value=harness;$('harness-select').append(option);}
$('harness-select').addEventListener('change',()=>{selectedHarness=$('harness-select').value;modeOptions();tierOptions();selectWindow(selected);});
function recordedBilling(){const units=harnessGroup().windows[selected].recorded_billing;table($('recorded-billing'),['Recorded measurement','Amount'],Object.entries(units).map(([unit,amount])=>[unit,unit.endsWith('USD')?money(amount):number(amount)]));table($('router-prices'),['Model','OpenRouter ID','Input / MTok','Cache read / MTok','Output / MTok'],Object.entries(data.openrouter_rates).filter(([model])=>visibleModels().some(([name])=>name===model)).map(([model,r])=>[model,r.id,money(r.input),r.cached===null?'N/A':money(r.cached),money(r.output)]));}
selectWindow(0);
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
