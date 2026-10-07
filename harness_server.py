#!/usr/bin/env python3
"""Serve the local usage dashboard at http://localhost:3050.

Install requirements-server.txt first. Source, pricing, cache, and timezone
options match the static generator; the dashboard shows all retained cache data.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from enum import Enum
import json
from pathlib import Path
import socket
import sqlite3
import sys
from threading import RLock
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import uuid4

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import FileResponse, Response
    from pydantic import BaseModel
    import uvicorn
except ImportError as error:
    raise ImportError("Install server dependencies: python3 -m pip install -r requirements-server.txt") from error

import harness_metrics as metrics


class ReportWindow(str, Enum):
    TODAY = "today"
    YESTERDAY = "yesterday"
    LAST_7 = "last_7_days"
    LAST_14 = "last_14_days"
    LAST_30 = "last_30_days"
    LAST_60 = "last_60_days"
    LAST_90 = "last_90_days"
    LAST_180 = "last_180_days"
    LAST_365 = "last_365_days"


class Distribution(BaseModel):
    count: int
    avg: float | None
    min: float | None
    median: float | None
    max: float | None
    p75: float | None
    p95: float | None
    p99: float | None


class MetricSummary(BaseModel):
    ttft: Distribution
    throughput: Distribution
    length: Distribution
    tools: Distribution


class TrendPoint(MetricSummary):
    total_tokens: int
    cost: str
    partial_cost: bool


class Period(BaseModel):
    label: str
    start: str
    end: str
    end_exclusive: bool


class CategorySummary(BaseModel):
    name: metrics.Category
    tokens: int
    cost: str
    unpriced_tokens: int


class WindowSummary(Period):
    conversations: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    active_seconds: float
    tool_calls: int
    cost: str
    partial_cost: bool
    unpriced_tokens: int
    metrics: MetricSummary
    categories: list[CategorySummary]
    unpriced: dict[str, int]
    models: dict[str, int]
    coverage: dict[str, int]
    recorded_billing: dict[str, str]


class CatalogMetadata(BaseModel):
    source: str
    retrieved: str | None = None
    snapshot_file: str | None = None
    bundled: bool = False
    error: str | None = None


class RouterRate(BaseModel):
    id: str
    input: str
    cached: str | None
    output: str


class Metadata(BaseModel):
    snapshot: str
    generated: str
    timezone: str
    first_date: date
    cutoff_date: date
    files: int
    threads: int
    sources: list[str]
    harnesses: list[metrics.Harness]
    windows: list[Period]
    window_ids: list[ReportWindow]
    pricing_date: str
    pricing_source: str
    anthropic_pricing_date: str
    anthropic_pricing_source: str
    openrouter: CatalogMetadata
    quality: dict[str, int]
    warnings: list[str]


class Scope(BaseModel):
    harness: metrics.Harness | None = None
    tier: metrics.ModelTier | None = None
    model: str | None = None
    mode: metrics.SpeedMode | None = None


class DashboardResponse(BaseModel):
    snapshot: str
    scope: Scope
    windows: list[WindowSummary]
    models: dict[str, WindowSummary]
    tiers: dict[str, WindowSummary]
    modes: list[metrics.SpeedMode]
    tier_choices: list[metrics.ModelTier]
    recorded_billing: dict[str, str]
    openrouter_rates: dict[str, RouterRate]


class TrendResponse(BaseModel):
    snapshot: str
    scope: Scope
    granularity: metrics.Granularity
    periods: list[Period]
    points: list[TrendPoint | None]
    range_start: str
    range_end: str


@dataclass(frozen=True)
class ServerConfig:
    sources: dict[metrics.Harness, list[Path]]
    cache_path: Path
    report_zone: tzinfo
    catalog: dict[str, metrics.Price]
    catalog_metadata: dict[str, Any]


@dataclass
class Snapshot:
    metadata: Metadata
    report: dict[str, Any]
    rates: dict[str, RouterRate]
    database_version: int = 0
    file_identity: tuple[int, int] = (0, 0)


class DashboardService:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.lock = RLock()
        self.snapshot: Snapshot | None = None
        self.startup_error: str | None = None
        self.monitor: sqlite3.Connection | None = None
        self.trend_cache: OrderedDict[tuple[date, date, metrics.Granularity, str], TrendResponse] = OrderedDict()
        self.trend_cache_bytes = 0

    def close(self) -> None:
        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None

    def database_identity(self) -> tuple[int, int]:
        stat = self.config.cache_path.stat()
        return stat.st_dev, stat.st_ino

    def read_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.config.cache_path.resolve().as_uri() + "?mode=ro", uri=True)

    def current(self, identifier: str | None = None) -> Snapshot:
        snapshot = self.snapshot
        if snapshot is None:
            detail = f" Startup refresh failed: {self.startup_error}" if self.startup_error else ""
            raise HTTPException(503, f"No dashboard snapshot is available. Refresh the sources.{detail}")
        if identifier is not None and identifier != snapshot.metadata.snapshot:
            raise HTTPException(409, "Snapshot changed. Reload the dashboard.")
        try:
            unchanged = (self.monitor is not None and self.database_identity() == snapshot.file_identity
                         and self.monitor.execute("PRAGMA data_version").fetchone()[0] == snapshot.database_version)
        except (OSError, sqlite3.Error):
            unchanged = False
        if not unchanged:
            raise HTTPException(409, "The database changed outside this server. Use Refresh to load it.")
        return snapshot

    def refresh(self) -> Metadata:
        with self.lock:
            now = datetime.now(timezone.utc)
            pinned: list[tuple[int, tuple[int, int]]] = []

            def pin_version(writer: sqlite3.Connection) -> None:
                # The writer's data_version changes only for other connections' commits, so an
                # unchanged value around the monitor read proves it observed exactly this import.
                before = writer.execute("PRAGMA data_version").fetchone()[0]
                if self.monitor is None or (self.snapshot is not None
                                            and self.database_identity() != self.snapshot.file_identity):
                    self.close()
                    self.monitor = sqlite3.connect(self.config.cache_path.resolve().as_uri() + "?mode=ro",
                                                   uri=True, check_same_thread=False)
                pinned.append((self.monitor.execute("PRAGMA data_version").fetchone()[0], self.database_identity()))
                if writer.execute("PRAGMA data_version").fetchone()[0] != before:
                    raise ValueError("The database changed during refresh. Retry Refresh.")

            # Compute before the importer commits, so failed recalculation rolls back its updates.
            with metrics.prepare_sources(self.config.sources, metrics.Quality(), True,
                                         self.config.cache_path, now, after_commit=pin_version) as prepared:
                if prepared.uncached_sources:
                    shown = ", ".join(prepared.uncached_sources[:3])
                    more = len(prepared.uncached_sources) - 3
                    raise ValueError("Some sources changed or could not be read and were not cached: "
                                     f"{shown}{f' and {more:,} more' if more > 0 else ''}. "
                                     "Check source access and retry Refresh; the previous snapshot is retained.")
                print("Calculating summaries for all cached conversations…", file=sys.stderr)
                candidate = self.calculate_snapshot(prepared.cache, now)
            candidate.database_version, candidate.file_identity = pinned[0]
            self.snapshot = candidate
            self.startup_error = None
            self.trend_cache.clear()
            self.trend_cache_bytes = 0
            return candidate.metadata

    def calculate_snapshot(self, reader: metrics.MetricsReader, now: datetime) -> Snapshot:
        windows = metrics.make_windows(now, self.config.report_zone)
        lookup = metrics.WindowLookup(windows)
        overall = metrics.empty_breakdown(windows)
        harnesses: dict[metrics.Harness, metrics.Breakdown] = {}
        quality = metrics.Quality()
        first: datetime | None = None
        count = 0
        models: dict[str, str | None] = {}
        for thread in reader.threads(quality):
            count += 1
            at = metrics.first_datapoint(thread, now)
            if at is not None and (first is None or at < first):
                first = at
            thread.id = f"{thread.harness.value}:{thread.id}"
            if thread.harness not in harnesses:
                harnesses[thread.harness] = metrics.empty_breakdown(windows)
            group = harnesses[thread.harness]
            metrics.add_thread(group.windows, thread, group.by_model, group.by_mode, group.by_tier,
                               self.config.catalog, lookup)
            for turn in thread.turns.values():
                for record in turn.usage:
                    model = record.model or turn.model
                    name = metrics.report_model(model, record.mode,
                                                metrics.is_long_context(model, record.usage,
                                                                        self.config.catalog, record.aggregate))
                    models[name] = model
        by_harness = {}
        for harness, group in harnesses.items():
            metrics.merge_breakdown(overall, group)
            by_harness[harness.value] = metrics.summarize_breakdown(group, self.config.report_zone)
        report = metrics.summarize_breakdown(overall, self.config.report_zone)
        report['by_harness'] = by_harness
        paths: set[str] = set()
        for (manifest,) in reader.connection.execute("SELECT manifest FROM cache_groups"):
            paths.update(path for path, _ in json.loads(manifest) if not path.endswith('-wal'))
        rates = {}
        for name, model in models.items():
            identifier = metrics.router_model(model, self.config.catalog)
            if identifier and name in report['by_model']:
                price = self.config.catalog[identifier].short
                rates[name] = RouterRate(id=identifier, input=str(price.input),
                                         cached=str(price.cached) if price.cached is not None else None,
                                         output=str(price.output))
        local = now.astimezone(self.config.report_zone)
        metadata = Metadata(
            snapshot=uuid4().hex, generated=local.isoformat(), timezone=str(self.config.report_zone),
            first_date=(first or now).astimezone(self.config.report_zone).date(), cutoff_date=local.date(),
            files=len(paths), threads=count, sources=sorted({str(Path(path).parent) for path in paths}),
            harnesses=sorted(harnesses, key=lambda harness: harness.value),
            windows=[Period(**{key: summary[key] for key in Period.model_fields}) for summary in report['windows']],
            window_ids=list(ReportWindow), pricing_date=metrics.PRICING_DATE, pricing_source=metrics.PRICING_SOURCE,
            anthropic_pricing_date=metrics.ANTHROPIC_PRICING_DATE,
            anthropic_pricing_source=metrics.ANTHROPIC_PRICING_SOURCE,
            openrouter=CatalogMetadata(**self.config.catalog_metadata),
            quality=dict(quality.counts), warnings=quality.warnings)
        return Snapshot(metadata, report, rates)

    def harness_group(self, snapshot: Snapshot, scope: Scope) -> dict[str, Any]:
        if scope.harness is None:
            return snapshot.report
        group = snapshot.report['by_harness'].get(scope.harness.value)
        if group is None:
            raise HTTPException(422, "Harness has no cached conversations.")
        return group

    def selected_group(self, snapshot: Snapshot, scope: Scope) -> dict[str, Any]:
        group = self.harness_group(snapshot, scope)
        if scope.tier is not None:
            group = group['by_tier'].get(scope.tier.value)
            if group is None:
                raise HTTPException(422, "Tier is unavailable in this scope.")
        if scope.mode is not None:
            group = group['by_mode'].get(scope.mode.value)
            if group is None:
                raise HTTPException(422, "Mode is unavailable in this scope.")
        return group

    def dashboard(self, identifier: str, window: ReportWindow, scope: Scope) -> DashboardResponse:
        with self.lock:
            snapshot = self.current(identifier)
            index = list(ReportWindow).index(window)
            harness = self.harness_group(snapshot, scope)
            scope = scope.model_copy(update={
                'tier': scope.tier if scope.tier is None or scope.tier.value in harness['by_tier'] else None,
                'mode': scope.mode if scope.mode is None or scope.mode.value in harness['by_mode'] else None,
            })
            group = self.selected_group(snapshot, scope)
            models = {name: WindowSummary(**windows[index]) for name, windows in group['by_model'].items()
                      if windows[index]['total_tokens'] > 0}
            if scope.model is not None and scope.model not in snapshot.report['by_model']:
                raise HTTPException(422, "Unknown model.")
            effective = scope.model_copy(update={'model': scope.model if scope.model in models else None})
            summaries = group['by_model'][effective.model] if effective.model else group['windows']
            tiers = {}
            for name, tier in harness['by_tier'].items():
                comparison = tier['by_mode'].get(scope.mode.value) if scope.mode else tier
                if comparison is not None:
                    values = comparison['by_model'].get(effective.model) if effective.model else comparison['windows']
                    if values is not None:
                        tiers[name] = WindowSummary(**values[index])
            return DashboardResponse(
                snapshot=identifier, scope=effective, windows=[WindowSummary(**summary) for summary in summaries],
                models=models, tiers=tiers, modes=list(harness['by_mode']), tier_choices=list(harness['by_tier']),
                recorded_billing=harness['windows'][index]['recorded_billing'],
                openrouter_rates={name: rate for name, rate in snapshot.rates.items() if name in models})

    def trends(self, identifier: str, start: date, end: date, granularity: metrics.Granularity,
               scope: Scope) -> TrendResponse:
        with self.lock:
            snapshot = self.current(identifier)
            metadata = snapshot.metadata
            self.selected_group(snapshot, scope)
            if scope.model is not None and scope.model not in snapshot.report['by_model']:
                raise HTTPException(422, "Unknown model.")
            if not metadata.first_date <= start <= end <= metadata.cutoff_date:
                raise HTTPException(422, "Choose dates within available history, with Start on or before End.")
            key = start, end, granularity, scope.model_dump_json()
            if key in self.trend_cache:
                self.trend_cache.move_to_end(key)
                return self.trend_cache[key]
            now = datetime.fromisoformat(metadata.generated).astimezone(timezone.utc)
            zone = self.config.report_zone
            start_local = datetime.combine(start, datetime.min.time(), zone)
            end_local = datetime.combine(end + timedelta(days=1), datetime.min.time(), zone)
            first_local = datetime.combine(metadata.first_date, datetime.min.time(), zone)
            period_start, period_end = start_local, end_local
            if granularity == metrics.Granularity.WEEKLY:
                period_start -= timedelta(days=period_start.weekday())
                period_end += timedelta(days=(-period_end.weekday()) % 7)
            elif granularity == metrics.Granularity.MONTHLY:
                period_start = period_start.replace(day=1)
                if period_end.day != 1:
                    period_end = (period_end.replace(year=period_end.year + 1, month=1, day=1)
                                  if period_end.month == 12 else period_end.replace(month=period_end.month + 1, day=1))
            windows = metrics.make_trend_windows(min(period_end.astimezone(timezone.utc), now), zone,
                                                max(period_start, first_local), (granularity,))
            range_start = start_local.astimezone(timezone.utc)
            windows = [window for window in windows
                       if window.start.astimezone(zone).date() <= end
                       and (window.end > range_start or window.end == range_start and not window.end_exclusive)]
            lookup = metrics.WindowLookup(windows)
            span = (windows[0].start, windows[-1].end) if windows else (range_start, range_start)
            metric_scope = metrics.MetricScope(scope.model, scope.mode, scope.tier)
            with closing(self.read_connection()) as connection:
                connection.execute('BEGIN')
                reader = metrics.MetricsReader(connection)
                for thread in reader.threads(metrics.Quality(), scope.harness, span):
                    thread.id = f"{thread.harness.value}:{thread.id}"
                    metrics.add_thread(windows, thread, catalog=self.config.catalog, lookup=lookup, scope=metric_scope)
            self.current(identifier)
            response = TrendResponse(
                snapshot=identifier, scope=scope, granularity=granularity,
                periods=[Period(label=window.label, start=window.start.astimezone(zone).isoformat(),
                                end=window.end.astimezone(zone).isoformat(), end_exclusive=window.end_exclusive)
                         for window in windows],
                points=[TrendPoint(**point) if (point := metrics.trend_summary(window)) is not None else None
                        for window in windows], range_start=range_start.isoformat(),
                range_end=min(end_local.astimezone(timezone.utc), now).isoformat())
            size = len(response.model_dump_json().encode('utf-8'))
            # Bound both entry count and serialized size; a large series can be returned uncached.
            if size <= 8 * 1024 * 1024:
                self.trend_cache[key] = response
                self.trend_cache_bytes += size
                while len(self.trend_cache) > 32 or self.trend_cache_bytes > 8 * 1024 * 1024:
                    _, discarded = self.trend_cache.popitem(last=False)
                    self.trend_cache_bytes -= len(discarded.model_dump_json().encode('utf-8'))
            return response


def create_app(config: ServerConfig) -> FastAPI:
    service = DashboardService(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            try:
                await run_in_threadpool(service.refresh)
            except (OSError, sqlite3.Error, ValueError) as error:
                # Serve the dashboard so the failure is visible and Refresh can retry it.
                service.startup_error = str(error)
                print(f"Initial refresh failed: {error}", file=sys.stderr)
            yield
        finally:
            service.close()

    # No CDN-backed documentation pages are needed for the offline dashboard.
    app = FastAPI(title='Harness dashboard', lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.service = service

    @app.middleware('http')
    async def no_browser_cache(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/', response_class=FileResponse)
    def dashboard_page() -> FileResponse:
        return FileResponse(Path(__file__).resolve().with_name('harness_dashboard.html'))

    @app.get('/api/metadata', response_model=Metadata)
    def metadata() -> Metadata:
        with service.lock:
            return service.current().metadata

    @app.get('/api/dashboard', response_model=DashboardResponse)
    def dashboard(snapshot: str = Query(...), window: ReportWindow = ReportWindow.TODAY,
                  harness: metrics.Harness | None = None, tier: metrics.ModelTier | None = None,
                  model: str | None = None, mode: metrics.SpeedMode | None = None) -> DashboardResponse:
        return service.dashboard(snapshot, window, Scope(harness=harness, tier=tier, model=model, mode=mode))

    @app.get('/api/trends', response_model=TrendResponse)
    def trends(snapshot: str = Query(...), start: date = Query(...), end: date = Query(...),
               granularity: metrics.Granularity = metrics.Granularity.DAILY,
               harness: metrics.Harness | None = None, tier: metrics.ModelTier | None = None,
               model: str | None = None, mode: metrics.SpeedMode | None = None) -> TrendResponse:
        return service.trends(snapshot, start, end, granularity, Scope(harness=harness, tier=tier, model=model, mode=mode))

    @app.post('/api/refresh', response_model=Metadata)
    def refresh() -> Metadata:
        try:
            return service.refresh()
        except (OSError, sqlite3.Error, ValueError) as error:
            raise HTTPException(503, f"Refresh failed: {error}") from error

    return app


def main() -> int:
    parser = metrics.argument_parser(static=False, description=__doc__)
    args = parser.parse_args()
    sources = metrics.source_paths(args, parser)
    catalog, catalog_metadata = metrics.load_catalog(args, parser)
    config = ServerConfig(sources, (args.cache or metrics.default_cache_path()).resolve(), args.timezone,
                          catalog, catalog_metadata)
    # Bind before importing sources so an occupied port causes no cache changes.
    with closing(socket.socket()) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('127.0.0.1', 3050))
            listener.listen(128)
        except OSError as error:
            print(f"Unable to start dashboard on http://localhost:3050: {error}", file=sys.stderr)
            return 1
        print('Dashboard: http://localhost:3050 (Ctrl+C to stop)', flush=True)
        server = uvicorn.Server(uvicorn.Config(create_app(config), log_level='info'))
        try:
            server.run(sockets=[listener])
        except KeyboardInterrupt:
            pass
        return 0 if server.started else 1


if __name__ == '__main__':
    raise SystemExit(main())
