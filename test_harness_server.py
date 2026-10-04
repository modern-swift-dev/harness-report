"""Optional server tests: install requirements-server.txt, then run unittest discovery."""
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import signal
import sqlite3
import subprocess
import sys
import tempfile
from threading import Thread
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import harness_metrics as metrics
import test_harness_metrics as fixtures
from test_harness_metrics import NOW, START, complete, event, modern, prefix, record, usage

SERVER_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('fastapi', 'uvicorn'))
if SERVER_AVAILABLE:
    import harness_server as server


@unittest.skipUnless(SERVER_AVAILABLE, 'Optional FastAPI/Uvicorn dependencies are not installed')
class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root / 'codex'
        self.logs.mkdir()
        self.cache = self.root / 'metrics.sqlite3'
        helper = fixtures.HarnessTests()
        helper.root = self.root
        self.claude = self.root / 'claude'
        self.claude.mkdir()
        self.write(self.claude / 'session.jsonl', [helper.claude()])
        self.copilot = self.root / 'copilot'
        self.copilot.mkdir()
        self.write(self.copilot / 'session.jsonl', [helper.copilot('session.start', 'start', {'sessionId': 'root'}),
                                                  helper.copilot('session.shutdown', 'end', {'modelMetrics': {
                                                      'gpt-6-astra': {'usage': {'inputTokens': 300, 'outputTokens': 30}}}})])
        self.opencode = helper.opencode_db()
        self.write(self.logs / 'one.jsonl', prefix() + [modern(),
                   record('response_item', {'type': 'function_call', 'call_id': 'tool'}), complete()])
        old = START - timedelta(days=5)
        self.write(self.logs / 'old.jsonl', prefix(thread='old', at=old) + [
            modern(at=old + timedelta(seconds=1)), complete(at=old + timedelta(seconds=10))])
        rows = prefix(thread='mixed')
        rows[0]['payload']['service_tier'] = 'priority'
        self.write(self.logs / 'mixed.jsonl', rows + [
            modern(counts=usage(input=272000)), modern(response='long', counts=usage(input=272001)),
            record('response_item', {'type': 'function_call', 'call_id': 'mixed'}), complete()])
        self.sources = {metrics.Harness.CODEX: [self.logs], metrics.Harness.CLAUDE: [self.claude],
                        metrics.Harness.COPILOT: [self.copilot], metrics.Harness.OPENCODE: [self.opencode]}
        self.catalog = metrics.openrouter_prices(json.loads(metrics.BUNDLED_OPENROUTER_PRICES.read_text()))
        self.config = server.ServerConfig(self.sources, self.cache, timezone.utc, self.catalog,
                                          {'source': metrics.OPENROUTER_SOURCE, 'retrieved': NOW.isoformat()})
        self.service = server.DashboardService(self.config)
        self.addCleanup(self.service.close)
        self.clock = patch('harness_server.datetime', wraps=datetime)
        self.clock.start().now.return_value = NOW
        self.addCleanup(self.clock.stop)
        self.metadata = self.service.refresh()

    def write(self, path, rows):
        path.write_text('\n'.join(json.dumps(row) if isinstance(row, dict) else row for row in rows) + '\n')

    def static_report(self, now=NOW, zone=timezone.utc):
        return metrics.collect_report(self.logs, now, harness_roots={h: paths for h, paths in self.sources.items()
                                      if h != metrics.Harness.CODEX}, catalog=self.catalog, report_zone=zone)

    def static_group(self, report, scope):
        group = report['by_harness'][scope.harness.value] if scope.harness else report
        if scope.tier:
            group = group['by_tier'][scope.tier.value]
        if scope.mode:
            group = group['by_mode'][scope.mode.value]
        return group

    def assert_trend_parity(self, report, scope, interval, start=None, end=None):
        start = start or self.metadata.first_date
        end = end or self.metadata.cutoff_date
        actual = self.service.trends(self.metadata.snapshot, start, end, interval, scope)
        group = self.static_group(report, scope)
        points = group['by_model_trends'][scope.model][interval.value] if scope.model else group['trends'][interval.value]
        first = datetime.combine(start, datetime.min.time(), self.config.report_zone).astimezone(timezone.utc)
        expected = [(period, point) for period, point in zip(report['trend_periods'][interval.value], points)
                    if period['start'][:10] <= end.isoformat()
                    and (datetime.fromisoformat(period['end']) > first
                         or datetime.fromisoformat(period['end']) == first and not period['end_exclusive'])]
        self.assertEqual([period.model_dump() for period in actual.periods], [period for period, _ in expected])
        self.assertEqual([point.model_dump() if point else None for point in actual.points], [point for _, point in expected])
        return actual

    def test_t3_refresh_reclassifies_cached_usage_without_double_counting(self):
        helper = fixtures.HarnessTests()
        helper.root = self.root
        t3 = helper.t3_db((("codex", "root"), ("claude", "same-id")))
        before = self.service.dashboard(self.metadata.snapshot, server.ReportWindow.TODAY, server.Scope())
        self.sources[metrics.Harness.T3] = [t3]
        metadata = self.service.refresh()
        self.assertIn(metrics.Harness.T3, metadata.harnesses)
        combined = self.service.dashboard(metadata.snapshot, server.ReportWindow.TODAY, server.Scope())
        self.assertEqual(combined.windows, before.windows)
        scope = server.Scope(harness=metrics.Harness.T3)
        actual = self.service.dashboard(metadata.snapshot, server.ReportWindow.TODAY, scope)
        expected = self.static_report()['by_harness']['t3']['windows'][0]
        self.assertEqual(actual.windows[0].model_dump(), expected)
        self.assertEqual(actual.windows[0].total_tokens, 1900)
        with closing(sqlite3.connect(t3)) as db, db:
            db.execute("INSERT INTO provider_session_runtime VALUES (?, ?)", ('codex', '{'))
        metadata = self.service.refresh()
        self.assertEqual(metadata.quality['Malformed records'], 1)

    def test_t3_unreadable_database_retains_previous_snapshot(self):
        path = self.root / 'invalid.sqlite'
        path.write_text('not a SQLite database')
        self.sources[metrics.Harness.T3] = [path]
        with self.assertRaisesRegex(ValueError, 'could not be read'):
            self.service.refresh()
        self.assertEqual(self.service.current().metadata.snapshot, self.metadata.snapshot)

    def test_summary_and_combined_filters_match_static_report(self):
        report = self.static_report()
        for harness in (None, *self.sources):
            base = report['by_harness'][harness.value] if harness else report
            for tier in (None, *[metrics.ModelTier(name) for name in base['by_tier']]):
                tier_group = base['by_tier'][tier.value] if tier else base
                for mode in (None, *[metrics.SpeedMode(name) for name in base['by_mode']]):
                    group = tier_group['by_mode'][mode.value] if mode else tier_group
                    scope = server.Scope(harness=harness, tier=tier, mode=mode)
                    with self.subTest(scope=scope):
                        actual = self.service.dashboard(self.metadata.snapshot, server.ReportWindow.TODAY, scope)
                        self.assertEqual([window.model_dump() for window in actual.windows], group['windows'])
                        expected = {name: windows[0] for name, windows in group['by_model'].items()
                                    if windows[0]['total_tokens'] > 0}
                        self.assertEqual({name: window.model_dump() for name, window in actual.models.items()}, expected)
                        expected_tiers = {name: (tier_scope['by_mode'][mode.value] if mode else tier_scope)['windows'][0]
                                          for name, tier_scope in base['by_tier'].items()}
                        self.assertEqual({name: summary.model_dump() for name, summary in actual.tiers.items()}, expected_tiers)
                        self.assertEqual({name: rate.model_dump() for name, rate in actual.openrouter_rates.items()},
                                         {name: rate for name, rate in report['openrouter_rates'].items() if name in expected})
                        for model in expected:
                            model_scope = scope.model_copy(update={'model': model})
                            actual = self.service.dashboard(self.metadata.snapshot, server.ReportWindow.TODAY, model_scope)
                            self.assertEqual([window.model_dump() for window in actual.windows], group['by_model'][model])
                            self.assertEqual(actual.recorded_billing, base['windows'][0]['recorded_billing'])

    def test_trends_preserve_all_intervals_and_mixed_context_attribution(self):
        report = self.static_report()
        scopes = [server.Scope(), server.Scope(harness=metrics.Harness.CLAUDE),
                  server.Scope(harness=metrics.Harness.CODEX, tier=metrics.ModelTier.MEDIUM, mode=metrics.SpeedMode.FAST),
                  server.Scope(model='gpt-6.1-sol-fast-long'), server.Scope(model='Mixed contexts (timing)'),
                  server.Scope(model='Mixed contexts (tools)')]
        for scope in scopes:
            for interval in metrics.Granularity:
                with self.subTest(scope=scope, interval=interval):
                    self.assert_trend_parity(report, scope, interval)
        timing = self.service.trends(self.metadata.snapshot, self.metadata.first_date, self.metadata.cutoff_date,
                                     metrics.Granularity.DAILY, scopes[-2])
        self.assertEqual(sum(point.ttft.count for point in timing.points if point), 1)

    def test_mixed_models_modes_and_tiers_keep_complete_turn_attribution(self):
        self.write(self.logs / 'mixed-models.jsonl', prefix(thread='mixed-models') + [modern(),
                   record('turn_context', {'turn_id': 't1', 'model': 'gpt-6-astra'}, START + timedelta(seconds=2)),
                   modern(response='other-model', at=START + timedelta(seconds=3)), complete()])
        self.write(self.logs / 'mixed-modes.jsonl', prefix(thread='mixed-modes') + [modern(),
                   event('thread_settings_applied', START + timedelta(seconds=2),
                         thread_settings={'service_tier': 'priority'}),
                   modern(response='other-mode', at=START + timedelta(seconds=3)), complete()])
        self.metadata = self.service.refresh()
        report = self.static_report()
        scopes = [server.Scope(model='Mixed models (timing)'), server.Scope(model='Mixed modes (timing)'),
                  server.Scope(tier=metrics.ModelTier.MIXED), server.Scope(mode=metrics.SpeedMode.MIXED),
                  server.Scope(tier=metrics.ModelTier.MEDIUM, mode=metrics.SpeedMode.MIXED),
                  server.Scope(model='gpt-6.1-sol', tier=metrics.ModelTier.MEDIUM),
                  server.Scope(model='gpt-6-astra', tier=metrics.ModelTier.HIGH)]
        for scope in scopes:
            for interval in metrics.Granularity:
                with self.subTest(scope=scope, interval=interval):
                    self.assert_trend_parity(report, scope, interval)

    def test_date_range_keeps_whole_week_and_month_and_only_requested_interval(self):
        report = self.static_report()
        start = self.metadata.cutoff_date - timedelta(days=1)
        for interval in metrics.Granularity:
            actual = self.assert_trend_parity(report, server.Scope(), interval, start, start)
            self.assertNotIn('by_model', actual.model_dump())
            self.assertNotIn('by_harness', actual.model_dump())
            if interval == metrics.Granularity.HOURLY:
                self.assertEqual(len(actual.points), 24)
            if interval == metrics.Granularity.MONTHLY:
                self.assertEqual(actual.points[0].ttft.count, report['trends']['monthly'][0]['ttft']['count'])

    def test_selected_model_resets_if_new_scope_has_no_tokens(self):
        result = self.service.dashboard(self.metadata.snapshot, server.ReportWindow.YESTERDAY,
                                         server.Scope(model='gpt-6.1-sol'))
        self.assertIsNone(result.scope.model)
        self.assertEqual(result.models, {})
        result = self.service.dashboard(self.metadata.snapshot, server.ReportWindow.TODAY,
                                         server.Scope(harness=metrics.Harness.CLAUDE, tier=metrics.ModelTier.MIXED,
                                                      mode=metrics.SpeedMode.MIXED))
        self.assertIsNone(result.scope.tier)
        self.assertIsNone(result.scope.mode)

    def test_refresh_adds_activity_without_duplicates_and_invalidates_old_snapshot(self):
        self.service.trends(self.metadata.snapshot, self.metadata.first_date, self.metadata.cutoff_date,
                            metrics.Granularity.HOURLY, server.Scope())
        self.assertTrue(self.service.trend_cache)
        self.write(self.logs / 'new.jsonl', prefix(thread='new') + [modern(), complete()])
        refreshed = self.service.refresh()
        self.assertEqual(refreshed.threads, self.metadata.threads + 1)
        self.assertFalse(self.service.trend_cache)
        self.assertEqual(self.service.refresh().threads, refreshed.threads)
        with self.assertRaises(server.HTTPException) as raised:
            self.service.dashboard(self.metadata.snapshot, server.ReportWindow.TODAY, server.Scope())
        self.assertEqual(raised.exception.status_code, 409)

    def test_all_database_groups_survive_removed_sources_and_narrow_refresh(self):
        (self.logs / 'old.jsonl').unlink()
        self.sources.clear()
        self.sources[metrics.Harness.CLAUDE] = [self.claude]
        self.metadata = self.service.refresh()
        self.assertEqual(self.metadata.threads, 6)
        result = self.service.dashboard(self.metadata.snapshot, server.ReportWindow.LAST_7, server.Scope())
        self.assertEqual(result.windows[2].conversations, 6)
        self.assertIn(metrics.Harness.CODEX, self.metadata.harnesses)

    def test_failed_recalculation_rolls_back_import_and_keeps_previous_view(self):
        old = self.service.dashboard(self.metadata.snapshot, server.ReportWindow.TODAY, server.Scope())
        self.write(self.logs / 'new.jsonl', prefix(thread='new') + [modern(), complete()])
        with patch.object(self.service, 'calculate_snapshot', side_effect=ValueError('recalculation failed')):
            with self.assertRaisesRegex(ValueError, 'recalculation failed'):
                self.service.refresh()
        self.assertEqual(self.service.current().metadata.snapshot, self.metadata.snapshot)
        self.assertEqual(self.service.dashboard(self.metadata.snapshot, server.ReportWindow.TODAY, server.Scope()), old)
        with closing(sqlite3.connect(self.cache)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM cache_threads').fetchone()[0], self.metadata.threads)

    def test_unstable_import_keeps_previous_snapshot(self):
        with patch('harness_metrics.source_manifest', side_effect=lambda *_: str(time.monotonic_ns())):
            with self.assertRaisesRegex(ValueError, 'were not cached'):
                self.service.refresh()
        self.assertEqual(self.service.current().metadata.snapshot, self.metadata.snapshot)

    def test_malformed_records_retain_usage_and_diagnostics(self):
        path = self.logs / 'one.jsonl'
        with path.open('a') as stream:
            stream.write('{bad json\n')
        self.metadata = self.service.refresh()
        self.assertEqual(self.metadata.quality['Malformed lines'], 1)
        self.assertEqual(self.service.dashboard(self.metadata.snapshot, server.ReportWindow.TODAY,
                                               server.Scope(model='gpt-6.1-sol')).windows[0].total_tokens, 1100)

    def test_api_reads_do_not_modify_database_and_cached_series_are_reused(self):
        before = hashlib.sha256(self.cache.read_bytes()).digest()
        metadata = self.service.current().metadata
        args = metadata.snapshot, metadata.first_date, metadata.cutoff_date, metrics.Granularity.DAILY, server.Scope()
        first = self.service.trends(*args)
        with patch.object(metrics.MetricsReader, 'threads', side_effect=AssertionError('unexpected database scan')):
            self.assertIs(self.service.trends(*args), first)
            self.service.dashboard(metadata.snapshot, server.ReportWindow.TODAY, server.Scope())
        self.assertEqual(hashlib.sha256(self.cache.read_bytes()).digest(), before)

    def test_external_database_commits_are_detected(self):
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute('UPDATE cache_turns SET duration=duration+1 WHERE completed=1')
        with self.assertRaises(server.HTTPException) as raised:
            self.service.current(self.metadata.snapshot)
        self.assertEqual(raised.exception.status_code, 409)
        self.metadata = self.service.refresh()
        self.assertEqual(self.service.current().metadata.snapshot, self.metadata.snapshot)

    def test_incompatible_database_is_rejected_without_changes(self):
        cache = self.root / 'unrelated.sqlite3'
        with closing(sqlite3.connect(cache)) as connection, connection:
            connection.execute('CREATE TABLE unrelated(value TEXT)')
        before = cache.read_bytes()
        config = server.ServerConfig(self.sources, cache, timezone.utc, self.catalog, {'source': metrics.OPENROUTER_SOURCE})
        service = server.DashboardService(config)
        with self.assertRaisesRegex(ValueError, 'different database'):
            service.refresh()
        self.assertEqual(cache.read_bytes(), before)
        self.assertEqual(self.metadata.snapshot, self.service.current().metadata.snapshot)

    def test_empty_database_has_valid_windows_and_chart_gaps(self):
        config = server.ServerConfig({}, self.root / 'empty.sqlite3', timezone.utc, self.catalog,
                                     {'source': metrics.OPENROUTER_SOURCE})
        service = server.DashboardService(config)
        self.addCleanup(service.close)
        metadata = service.refresh()
        result = service.dashboard(metadata.snapshot, server.ReportWindow.TODAY, server.Scope())
        self.assertEqual(result.windows[0].total_tokens, 0)
        trends = service.trends(metadata.snapshot, metadata.first_date, metadata.cutoff_date,
                                metrics.Granularity.HOURLY, server.Scope())
        self.assertTrue(trends.periods)
        self.assertTrue(all(point is None for point in trends.points))

    def test_trend_cache_is_bounded(self):
        for index in range(33):
            start = self.metadata.first_date
            interval = list(metrics.Granularity)[index % 4]
            end = start + timedelta(days=index // 4 % 6)
            scope = server.Scope(harness=metrics.Harness.CLAUDE) if index >= 24 else server.Scope()
            self.service.trends(self.metadata.snapshot, start, end, interval, scope)
        self.assertLessEqual(len(self.service.trend_cache), 32)
        self.assertLessEqual(self.service.trend_cache_bytes, 8 * 1024 * 1024)

    def test_midnight_cutoff_and_dst_series_match_static(self):
        try:
            zone = metrics.report_timezone('America/Toronto')
        except Exception:
            self.skipTest('Toronto timezone data unavailable')
        for cutoff in (datetime(2026, 3, 9, 4, tzinfo=timezone.utc),
                       datetime(2026, 11, 2, 5, tzinfo=timezone.utc)):
            begin = cutoff - timedelta(days=2)
            path = self.logs / 'dst.jsonl'
            self.write(path, prefix(thread='dst', at=begin) + [modern(at=begin), complete(at=begin + timedelta(seconds=10))])
            config = server.ServerConfig({metrics.Harness.CODEX: [self.logs]}, self.root / f'dst-{cutoff.month}.sqlite3',
                                          zone, self.catalog, {'source': metrics.OPENROUTER_SOURCE})
            service = server.DashboardService(config)
            try:
                with patch('harness_server.datetime', wraps=datetime) as clock:
                    clock.now.return_value = cutoff
                    metadata = service.refresh()
                report = metrics.collect_report(self.logs, cutoff, catalog=self.catalog, report_zone=zone)
                saved_service, saved_metadata, saved_config = self.service, self.metadata, self.config
                try:
                    self.service, self.metadata, self.config = service, metadata, config
                    self.assert_trend_parity(report, server.Scope(), metrics.Granularity.HOURLY,
                                              begin.astimezone(zone).date(), metadata.cutoff_date)
                    selected = metadata.cutoff_date - timedelta(days=1)
                    series = self.assert_trend_parity(report, server.Scope(), metrics.Granularity.HOURLY, selected, selected)
                    self.assertEqual(len(series.points), 23 if cutoff.month == 3 else 25)
                    for window in server.ReportWindow:
                        actual = service.dashboard(metadata.snapshot, window, server.Scope())
                        self.assertEqual([summary.model_dump() for summary in actual.windows], report['windows'])
                finally:
                    self.service, self.metadata, self.config = saved_service, saved_metadata, saved_config
            finally:
                service.close()

    def test_http_endpoints_validation_lean_html_and_refresh_failure(self):
        app = server.create_app(self.config)
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        http_server = server.uvicorn.Server(server.uvicorn.Config(app, log_level='error'))
        thread = Thread(target=http_server.run, kwargs={'sockets': [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not http_server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(http_server.started)
            url = f'http://127.0.0.1:{port}'
            def request(path, method='GET'):
                try:
                    with urlopen(Request(url + path, method=method), timeout=10) as response:
                        return response.status, response.headers, response.read()
                except HTTPError as error:
                    with error:
                        return error.code, error.headers, error.read()
            status, headers, body = request('/')
            self.assertEqual(status, 200)
            self.assertLess(len(body), 60000)
            self.assertNotIn(b'__REPORT_DATA__', body)
            self.assertNotIn(b'id="report-data"', body)
            self.assertEqual(headers['Cache-Control'], 'no-store')
            status, _, body = request('/api/metadata')
            self.assertEqual(status, 200)
            identifier = json.loads(body)['snapshot']
            status, _, _ = request('/api/dashboard?' + urlencode({'snapshot': identifier, 'harness': 'invalid'}))
            self.assertEqual(status, 422)
            status, _, _ = request('/api/dashboard?' + urlencode({'snapshot': identifier, 'window': 'invalid'}))
            self.assertEqual(status, 422)
            for query in ({'start': '2026-09-30', 'end': '2026-09-25'}, {'start': 'invalid', 'end': '2026-09-30'},
                          {'start': '2026-09-24', 'end': '2026-09-30'}):
                status, _, _ = request('/api/trends?' + urlencode({'snapshot': identifier, **query}))
                self.assertEqual(status, 422)
            status, _, _ = request('/api/dashboard?snapshot=obsolete')
            self.assertEqual(status, 409)
            status, _, body = request('/api/dashboard?' + urlencode({'snapshot': identifier}))
            self.assertEqual(status, 200)
            self.assertNotIn('trends', json.loads(body))
            query = {'snapshot': identifier, 'start': '2026-09-30', 'end': '2026-09-30', 'granularity': 'daily'}
            status, _, body = request('/api/trends?' + urlencode(query))
            self.assertEqual(status, 200)
            self.assertEqual(len(json.loads(body)['points']), 1)
            with patch.object(app.state.service, 'calculate_snapshot', side_effect=ValueError('test failure')):
                status, _, body = request('/api/refresh', 'POST')
                self.assertEqual(status, 503)
                self.assertIn(b'test failure', body)
            status, _, body = request('/api/metadata')
            self.assertEqual(json.loads(body)['snapshot'], identifier)
            status, _, body = request('/api/refresh', 'POST')
            self.assertEqual(status, 200)
            self.assertNotEqual(json.loads(body)['snapshot'], identifier)
            status, _, _ = request('/api/dashboard?snapshot=' + identifier)
            self.assertEqual(status, 409)
        finally:
            http_server.should_exit = True
            thread.join(timeout=10)
            listener.close()
            self.assertFalse(thread.is_alive())

    def test_cli_rejects_static_only_flags_and_occupied_port_before_import(self):
        for flag in ('--no-cache', '--output'):
            process = subprocess.run([sys.executable, 'harness_server.py', flag], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('unrecognized arguments', process.stderr)
        with closing(socket.socket()) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(('127.0.0.1', 3050))
                listener.listen(1)
            except OSError:
                self.skipTest('Port 3050 is already occupied')
            before = self.cache.read_bytes()
            process = subprocess.run([sys.executable, 'harness_server.py', str(self.logs), '--cache', str(self.cache),
                                      '--timezone', 'UTC'], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('Unable to start dashboard', process.stderr)
            self.assertEqual(self.cache.read_bytes(), before)

    def test_cli_starts_on_port_3050_and_serves_the_retained_database(self):
        with closing(socket.socket()) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', 3050))
            except OSError:
                self.skipTest('Port 3050 is already occupied')
        logfile = self.root / 'server.log'
        with logfile.open('w') as output:
            process = subprocess.Popen([sys.executable, 'harness_server.py', str(self.logs), '--cache', str(self.cache),
                                        '--offline', '--timezone', 'UTC'], stdout=output, stderr=subprocess.STDOUT)
            try:
                metadata = None
                deadline = time.monotonic() + 10
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        with urlopen('http://127.0.0.1:3050/api/metadata', timeout=1) as response:
                            metadata = json.load(response)
                            break
                    except OSError:
                        time.sleep(.01)
                self.assertIsNotNone(metadata, logfile.read_text())
                self.assertEqual(metadata['threads'], self.metadata.threads)
                self.assertEqual(metadata['timezone'], 'UTC')
                self.assertEqual(set(metadata['harnesses']), {harness.value for harness in self.sources})
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    self.fail('Server did not stop after Ctrl+C')
            self.assertEqual(process.returncode, 0, logfile.read_text())


if __name__ == '__main__':
    unittest.main()
