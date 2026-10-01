"""Run with: python3 -m unittest discover -v"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from contextlib import closing, redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import harness_metrics as metrics


NOW = datetime(2026, 9, 30, 16, tzinfo=timezone.utc)
START = NOW - timedelta(minutes=10)


def usage(input=1000, output=100, cached=400, reasoning=20, write=0):
    return {"input_tokens": input, "output_tokens": output,
            "cached_input_tokens": cached, "reasoning_output_tokens": reasoning,
            "cache_write_input_tokens": write, "total_tokens": input + output}


def record(kind, payload, at=START):
    return {"timestamp": at.isoformat(), "type": kind, "payload": payload}


def event(kind, at=START, **fields):
    return record("event_msg", {"type": kind, **fields}, at)


def prefix(thread="root", turn="t1", model="gpt-6.1-sol", at=START):
    return [record("session_meta", {"id": thread}),
            event("task_started", at, turn_id=turn),
            record("turn_context", {"turn_id": turn, "model": model}, at)]


def modern(turn="t1", response="r1", counts=None, at=START + timedelta(seconds=1)):
    return record("token_usage_record", {"turn_id": turn, "response_id": response,
                  "usage": counts or usage()}, at)


def legacy(counts, at=START + timedelta(seconds=1), last=None):
    return event("token_count", at, info={"total_token_usage": counts,
                                         "last_token_usage": last or counts})


def complete(turn="t1", at=START + timedelta(seconds=10), **fields):
    return event("task_complete", at, turn_id=turn, duration_ms=10_000,
                 time_to_first_token_ms=500, **fields)


class ReportTests(unittest.TestCase):
    def setUp(self):
        clock = patch("harness_metrics.datetime", wraps=datetime)
        clock.start().now.return_value = NOW
        self.addCleanup(clock.stop)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows, name="logs/session.jsonl"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows) + "\n")
        return path

    def report(self):
        return metrics.collect_report(self.root, NOW)

    def test_explicit_timing_and_cost_categories(self):
        self.write(prefix() + [modern(), complete()])
        w = self.report()["windows"][0]
        self.assertEqual(w["total_tokens"], 1100)
        self.assertEqual(w["metrics"]["ttft"]["avg"], .5)
        self.assertEqual(w["metrics"]["throughput"]["avg"], 10)
        self.assertEqual(w["metrics"]["length"]["avg"], 10)
        self.assertEqual(w["metrics"]["tools"]["avg"], 0)
        self.assertEqual(sum(c["tokens"] for c in w["categories"]), 1100)
        # 600 * $2 + 400 * $.1 + (80+20) * $10, per million.
        self.assertEqual(Decimal(w["cost"]), Decimal(".00224"))
        self.assertEqual(sum(Decimal(c["cost"]) for c in w["categories"]), Decimal(w["cost"]))

    def test_nearest_rank_and_empty_distributions(self):
        d = metrics.distribution(range(1, 101))
        self.assertEqual(d, {"count": 100, "avg": 50.5, "min": 1, "median": 50.5, "max": 100, "p75": 75, "p95": 95, "p99": 99})
        self.assertEqual(metrics.distribution([8])["p99"], 8)
        self.assertIsNone(metrics.distribution([])["avg"])

    def test_min_max_and_median_with_even_odd_and_repeated_samples(self):
        for values, expected in [([9, 1, 5], (1, 5, 9)), ([9, 1, 5, 3], (1, 4, 9)),
                                 ([0, 0, 8], (0, 0, 8)), ([8], (8, 8, 8))]:
            with self.subTest(values=values):
                stats = metrics.distribution(iter(values))
                self.assertEqual((stats["min"], stats["median"], stats["max"]), expected)
                self.assertEqual(stats["count"], len(values))
        empty = metrics.distribution([])
        self.assertEqual(empty["count"], 0)
        self.assertTrue(all(value is None for key, value in empty.items() if key != "count"))

    def test_p75_uses_nearest_rank_including_small_and_empty_samples(self):
        for values, expected in [([1, 2, 3, 4], 3), ([1, 2, 3, 4, 5], 4),
                                 ([9, 1, 1, 4], 4), ([8], 8), ([], None)]:
            with self.subTest(values=values):
                self.assertEqual(metrics.distribution(values)["p75"], expected)

    def test_counters_repeats_resets_and_modern_precedence(self):
        first = usage()
        second = usage(2200, 220, 1000, 40)
        reset = usage(500, 50, 100, 10)
        self.write(prefix() + [legacy(first), legacy(first),
                   legacy(second, START + timedelta(seconds=2), usage(1200, 120, 600, 20)),
                   legacy(reset, START + timedelta(seconds=3)), complete()])
        w = self.report()["windows"][0]
        self.assertEqual(w["total_tokens"], 2970)
        self.assertAlmostEqual(w["metrics"]["throughput"]["avg"], 27)
        self.assertEqual(self.report()["quality"]["Token counter resets"], 1)
        self.write(prefix() + [modern(), modern(), legacy(first), complete()])
        self.assertEqual(self.report()["windows"][0]["total_tokens"], 1100)

    def test_mixed_legacy_and_modern_turns(self):
        self.write(prefix() + [legacy(usage()), complete(),
                   event("task_started", START + timedelta(seconds=20), turn_id="t2"),
                   record("turn_context", {"turn_id": "t2", "model": "gpt-5.5"}),
                   modern("t2", "r2", usage(2000, 200, 400, 0)),
                   legacy(usage(3000, 300, 800, 20), START + timedelta(seconds=30)),
                   complete("t2", START + timedelta(seconds=40))])
        w = self.report()["windows"][0]
        self.assertEqual(w["total_tokens"], 3300)
        self.assertEqual(w["metrics"]["length"]["avg"], 20)
        self.assertEqual(w["models"], {"gpt-5.5": 2200, "gpt-6.1-sol": 1100})

    def test_inherited_history_excluded_and_subagent_separate(self):
        parent = prefix() + [modern(), complete()]
        self.write(parent)
        child = [record("session_meta", {"id": "child", "session_id": "root",
                  "subagent_history_start_ordinal": len(parent) + 1})] + parent
        child += [event("task_started", turn_id="child-turn"),
                  record("turn_context", {"turn_id": "child-turn", "model": "gpt-6.1-sol"}),
                  modern("child-turn", "r2"), complete("child-turn")]
        self.write(child, "child.jsonl")
        r = self.report()
        self.assertEqual(r["windows"][0]["conversations"], 2)
        self.assertEqual(r["windows"][0]["total_tokens"], 2200)
        self.assertGreater(r["quality"]["Inherited records excluded"], 0)

    def test_copied_parent_metadata_without_boundary(self):
        self.write([record("session_meta", {"id": "child"})] + prefix() + [modern(), complete(),
                   event("thread_settings_applied", thread_id="child", thread_settings={"model": "gpt-6.1-sol"}),
                   event("task_started", turn_id="child-turn"),
                   modern("child-turn", "child-response"), complete("child-turn")])
        self.assertEqual(self.report()["windows"][0]["total_tokens"], 1100)

    def test_old_subagent_history_ordinal_does_not_hide_own_events(self):
        rows = prefix(thread="child", model="codex-auto-review") + [modern(), complete()]
        rows[0]["payload"].update({"subagent_history_start_ordinal": len(rows),
                                   "forked_from_id": "parent"})
        self.write(rows)
        w = self.report()["windows"][0]
        self.assertEqual(w["conversations"], 1)
        self.assertEqual(w["total_tokens"], 1100)
        self.assertFalse(w["partial_cost"])
        self.assertEqual(w["models"], {"codex-auto-review": 1100})
        self.assertEqual(Decimal(w["cost"]), Decimal(".000248"))

    def test_rethreaded_parent_prefix_is_a_baseline_not_child_usage(self):
        copied = usage(10000, 1000, 8000, 200)
        child_total = usage(11000, 1100, 8400, 220)
        rows = [record("session_meta", {"id": "child", "forked_from_id": "parent",
                                        "subagent_history_start_ordinal": 999}),
                legacy(copied), event("task_started", turn_id="rollout-2"),
                record("response_item", {"type": "function_call", "call_id": "parent-call"}),
                event("task_started", turn_id="child-turn"),
                record("turn_context", {"turn_id": "child-turn", "model": "gpt-6.1-sol"}),
                legacy(child_total, START + timedelta(seconds=2), usage()), complete("child-turn")]
        self.write(rows)
        w = self.report()["windows"][0]
        self.assertEqual(w["total_tokens"], 1100)
        self.assertEqual(w["tool_calls"], 0)
        self.assertEqual(w["models"], {"gpt-6.1-sol": 1100})

    def test_tool_calls_and_duplicate_files(self):
        calls = [record("response_item", {"type": kind, "call_id": f"c{i}"})
                 for i, kind in enumerate(sorted(metrics.CALL_TYPES))]
        rows = prefix() + calls + calls + [
            record("response_item", {"type": "function_call_output", "call_id": "c0"}),
            event("item_completed", item={"type": "McpToolCall", "id": "c0"}), modern(), complete()]
        self.write(rows)
        self.write(rows, "duplicate.jsonl")
        w = self.report()["windows"][0]
        self.assertEqual(w["tool_calls"], 4)
        self.assertEqual(w["metrics"]["tools"]["avg"], 4)
        self.assertEqual(w["total_tokens"], 1100)
        self.assertEqual(w["metrics"]["ttft"]["count"], 1)

    def test_archived_and_live_copies_merge_new_turns_without_double_counting(self):
        original = prefix() + [modern(),
            record("response_item", {"type": "function_call", "call_id": "first"}), complete()]
        self.write(original, "archive/nested/original.jsonl")
        self.write(original + [event("task_started", turn_id="t2"), modern("t2", "r2"),
            record("response_item", {"type": "function_call", "call_id": "second"}), complete("t2")],
            "sessions/year/month/day/copy.jsonl")
        r = metrics.collect_report(self.root / "archive", NOW, additional_roots=[self.root / "sessions"])
        w = r["windows"][0]
        self.assertEqual(r["files"], 2)
        self.assertEqual(r["threads"], 1)
        self.assertEqual(w["conversations"], 1)
        self.assertEqual(w["total_tokens"], 2200)
        self.assertEqual(w["tool_calls"], 2)
        self.assertEqual(w["active_seconds"], 20)
        self.assertEqual(w["metrics"]["ttft"]["count"], 2)
        self.assertEqual(Decimal(w["cost"]), Decimal(".00448"))
        self.assertEqual(r["by_tier"]["Medium"]["windows"][0]["total_tokens"], 2200)
        self.assertEqual(r["sources"], [str((self.root / "archive").resolve()), str((self.root / "sessions").resolve())])

    def test_archived_and_live_legacy_copies_share_cumulative_baselines(self):
        original = prefix() + [legacy(usage()), complete()]
        self.write(original, "archive/original.jsonl")
        self.write(original + [event("task_started", START + timedelta(seconds=20), turn_id="t2"),
            legacy(usage(2000, 200, 800, 40), START + timedelta(seconds=21), usage()),
            complete("t2", START + timedelta(seconds=30))], "sessions/copy.jsonl")
        r = metrics.collect_report(self.root / "archive", NOW, additional_roots=[self.root / "sessions"])
        self.assertEqual(r["windows"][0]["total_tokens"], 2200)
        self.assertEqual(r["windows"][0]["coverage"]["Usage responses"], 2)
        self.assertEqual(r["windows"][0]["metrics"]["ttft"]["count"], 2)
        self.assertEqual(r["threads"], 1)

    def test_live_unfinished_turn_contributes_usage_and_calls(self):
        self.write(prefix() + [modern(), complete()], "archive/done.jsonl")
        rows = prefix(thread="live", model="codex-auto-review")
        rows[0]["payload"]["service_tier"] = "fast"
        self.write(rows + [modern(response="live-response"),
            record("response_item", {"type": "function_call", "call_id": "live-call"})], "sessions/pending.jsonl")
        r = metrics.collect_report(self.root / "archive", NOW, additional_roots=[self.root / "sessions"])
        w = r["windows"][0]
        self.assertEqual(w["conversations"], 2)
        self.assertEqual(w["total_tokens"], 2200)
        self.assertEqual(w["tool_calls"], 1)
        self.assertEqual(w["metrics"]["ttft"]["count"], 1)
        self.assertEqual(w["active_seconds"], 10)
        self.assertEqual(w["coverage"]["Unfinished turns"], 1)
        self.assertEqual(Decimal(r["by_tier"]["Budget"]["by_mode"]["Fast"]["windows"][0]["cost"]), Decimal(".000372"))

    def test_overlapping_source_directories_read_each_file_once(self):
        self.write(prefix() + [modern(), complete()], "nested/session.jsonl")
        r = metrics.collect_report(self.root, NOW,
                                   additional_roots=[self.root / "nested", self.root])
        self.assertEqual(r["files"], 1)
        self.assertEqual(r["threads"], 1)
        self.assertEqual(r["windows"][0]["total_tokens"], 1100)
        self.assertNotIn("Duplicate usage records excluded", r["quality"])
        self.assertEqual(r["sources"], [str(self.root.resolve()), str((self.root / "nested").resolve())])

    def test_cli_explicit_archive_excludes_home_sessions_when_present(self):
        self.write(prefix() + [modern(), complete()], "archive/session.jsonl")
        self.write(prefix(thread="live") + [modern(response="live"), complete()],
                   "home/.codex/sessions/year/month/day/live.jsonl")
        self.write(prefix(thread="archived") + [modern(response="archived"), complete()],
                   "home/.codex/archived_sessions/archived.jsonl")
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "home"), \
             patch("sys.argv", ["harness_metrics.py", str(self.root / "archive"), "--output", str(output)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        html = output.read_text()
        r = json.loads(html.split('<script type="application/json" id="report-data">')[1].split('</script>')[0])
        self.assertEqual(r["files"], 1)
        self.assertEqual(r["threads"], 1)
        self.assertEqual(r["sources"], [str((self.root / "archive").resolve())])
        self.assertIn('data.sources.join', html)

    def test_cli_explicit_archive_excludes_other_installed_harnesses(self):
        self.write(prefix() + [modern(), complete()], "archive/session.jsonl")
        self.write([{"type": "assistant", "sessionId": "claude", "timestamp": START.isoformat(),
                     "message": {"model": "claude-sonnet-4-6", "usage": {"input_tokens": 100, "output_tokens": 10}}}],
                   "home/.claude/projects/session.jsonl")
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "home"), \
             patch("sys.argv", ["harness_metrics.py", str(self.root / "archive"), "--offline", "--output", str(output)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual(set(report["by_harness"]), {"codex"})
        self.assertEqual(report["windows"][0]["total_tokens"], 1100)

    def test_cli_without_directory_discovers_installed_harnesses(self):
        self.write(prefix() + [modern(), complete()], "archive/session.jsonl")
        self.write(prefix(thread="live") + [modern(response="live"), complete()],
                   "home/.codex/sessions/live.jsonl")
        self.write([{"type": "assistant", "sessionId": "claude", "timestamp": START.isoformat(),
                     "message": {"model": "claude-sonnet-4-6", "usage": {"input_tokens": 100, "output_tokens": 10}}}],
                   "home/.claude/projects/session.jsonl")
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "home"), \
             patch("harness_metrics.Path.cwd", return_value=self.root / "archive"), \
             patch("sys.argv", ["harness_metrics.py", "--offline", "--output", str(output)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual(set(report["by_harness"]), {"codex", "claude"})
        self.assertEqual(report["files"], 3)
        self.assertEqual(report["windows"][0]["total_tokens"], 2310)

    def test_cli_default_codex_includes_archived_sessions_and_deduplicates_copies(self):
        working = self.root / "working"
        working.mkdir()
        live = prefix(thread="live") + [modern(response="live"), complete()]
        self.write(live, "home/.codex/sessions/live.jsonl")
        self.write(live, "home/.codex/archived_sessions/copy.jsonl")
        self.write(prefix(thread="archived") + [modern(response="archived"), complete()],
                   "home/.codex/archived_sessions/nested/archived.jsonl")
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "home"), \
             patch("harness_metrics.Path.cwd", return_value=working), \
             patch("sys.argv", ["harness_metrics.py", "--harness", "codex", "--offline", "--output", str(output)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual(report["sources"], [str(working.resolve()),
                         str((self.root / "home/.codex/sessions").resolve()),
                         str((self.root / "home/.codex/archived_sessions").resolve())])
        self.assertEqual(report["files"], 3)
        self.assertEqual(report["threads"], 2)
        self.assertEqual(report["windows"][0]["total_tokens"], 2200)
        self.assertEqual(report["windows"][0]["active_seconds"], 20)

    def test_cli_default_codex_reads_archived_sessions_without_active_sessions(self):
        working = self.root / "working"
        working.mkdir()
        self.write(prefix(thread="archived") + [modern(), complete()],
                   "home/.codex/archived_sessions/archived.jsonl")
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "home"), \
             patch("harness_metrics.Path.cwd", return_value=working), \
             patch("sys.argv", ["harness_metrics.py", "--offline", "--output", str(output)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual(report["sources"], [str(working.resolve()),
                         str((self.root / "home/.codex/archived_sessions").resolve())])
        self.assertEqual(report["files"], 1)
        self.assertEqual(report["windows"][0]["total_tokens"], 1100)

    def test_malformed_nested_codex_fields_are_skipped_without_losing_usage(self):
        bad_records = [
            event("thread_settings_applied", thread_settings=None),
            event("thread_settings_applied", thread_settings={"model": 123}),
            record("turn_context", {"turn_id": "t1", "model": 123}),
            record("response_item", {"type": "function_call", "internal_chat_message_metadata_passthrough": [1]}),
            event("token_count", info=[1]),
            record("token_usage_record", {"turn_id": "t1", "response_id": [1], "usage": usage()}),
            record("event_msg", {"type": ["task_complete"]}),
        ]
        self.write(prefix() + bad_records + [
            record("response_item", {"type": "function_call", "call_id": "valid-call",
                                     "internal_chat_message_metadata_passthrough": None}),
            modern(), complete()])
        with redirect_stderr(io.StringIO()):
            report = self.report()
        self.assertEqual(report["quality"]["Malformed records"], len(bad_records))
        self.assertEqual(report["windows"][0]["total_tokens"], 1100)
        self.assertEqual(report["windows"][0]["active_seconds"], 10)
        self.assertEqual(report["windows"][0]["tool_calls"], 1)

    def test_cli_utc_timezone_controls_all_window_boundaries(self):
        self.write(prefix() + [modern(), complete()])
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "home"), \
             patch("sys.argv", ["harness_metrics.py", str(self.root), "--timezone", "UTC", "--offline", "--output", str(output)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        html = output.read_text()
        report = json.loads(html.split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual(report["timezone"], "UTC")
        midnight = "2026-09-30T00:00:00+00:00"
        for windows in [report["windows"], report["by_model"]["gpt-6.1-sol"],
                        report["by_mode"]["Normal"]["windows"], report["by_tier"]["Medium"]["windows"],
                        report["by_harness"]["codex"]["windows"]]:
            self.assertEqual(windows[0]["start"], midnight)
        self.assertNotIn("Toronto midnight", html)

    def test_cli_without_timezone_database_can_show_help_and_generate_report(self):
        script = str(Path(metrics.__file__).resolve())
        output = self.root / "portable.html"
        environment = {**os.environ, "PYTHONTZPATH": ""}
        for args in [["--help"], [str(self.root), "--harness", "codex", "--offline", "--output", str(output)]]:
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-S", "-B", script, *args], env=environment,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual(report["timezone"], "UTC")

    def test_cli_invalid_timezone_reports_argument_error(self):
        with patch("sys.argv", ["harness_metrics.py", str(self.root), "--timezone", "Invalid/Timezone"]), \
             redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as result:
                metrics.main()
        self.assertEqual(result.exception.code, 2)
        self.assertIn("unknown or unavailable timezone", errors.getvalue())

    def test_cli_missing_home_sessions_keeps_primary_directory(self):
        self.write(prefix() + [modern(), complete()], "archive/session.jsonl")
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "missing-home"), \
             patch("sys.argv", ["harness_metrics.py", str(self.root / "archive"), "--output", str(output)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        html = output.read_text()
        r = json.loads(html.split('<script type="application/json" id="report-data">')[1].split('</script>')[0])
        self.assertEqual(r["files"], 1)
        self.assertEqual(r["sources"], [str((self.root / "archive").resolve())])

    def test_cross_window_activity_and_full_duration_assignment(self):
        midnight = metrics.make_windows(NOW)[0].start
        self.write(prefix(at=midnight - timedelta(seconds=15)) + [
            modern(at=midnight - timedelta(seconds=1)),
            modern(response="r2", at=midnight),
            complete(at=midnight + timedelta(seconds=1))])
        r = self.report()
        self.assertEqual(r["windows"][0]["total_tokens"], 1100)
        self.assertEqual(r["windows"][0]["active_seconds"], 10)
        last_30 = next(w for w in r["windows"] if w["label"] == "Last 30 days")
        self.assertEqual(last_30["total_tokens"], 2200)

    @unittest.skipUnless(str(metrics.TIMEZONE) == "America/Toronto", "Toronto timezone data is unavailable")
    def test_toronto_midnight_and_exact_rolling_days(self):
        now = datetime(2026, 9, 1, 3, 59, tzinfo=timezone.utc)  # August in Toronto.
        windows = metrics.make_windows(now)
        self.assertEqual(windows[0].start, datetime(2026, 8, 31, 4, tzinfo=timezone.utc))
        self.assertEqual(windows[-1].start, now - timedelta(days=365))
        self.assertTrue(windows[-1].contains(windows[-1].start))
        self.assertFalse(windows[-1].contains(windows[-1].start - timedelta(microseconds=1)))

    def test_missing_timing_aborted_unfinished_and_fallback_duration(self):
        self.write(prefix() + [modern(), event("task_complete", START + timedelta(seconds=10), turn_id="t1"),
            event("task_started", START + timedelta(seconds=20), turn_id="t2"),
            modern("t2", "r2"), event("turn_aborted", START + timedelta(seconds=25), turn_id="t2"),
            event("task_started", START + timedelta(seconds=30), turn_id="t3"), modern("t3", "r3")])
        w = self.report()["windows"][0]
        self.assertEqual(w["total_tokens"], 3300)
        self.assertIsNone(w["metrics"]["ttft"]["avg"])
        self.assertEqual(w["metrics"]["length"]["avg"], 10)
        self.assertEqual(w["metrics"]["throughput"]["count"], 1)
        self.assertEqual(w["coverage"]["Aborted turns"], 1)
        self.assertEqual(w["coverage"]["Unfinished turns"], 1)

    def test_unknown_pricing_and_long_context_boundary(self):
        self.write(prefix(model="unpublished-model") + [modern(), complete()])
        w = self.report()["windows"][0]
        self.assertTrue(w["partial_cost"])
        self.assertEqual(w["unpriced_tokens"], 1100)
        self.assertEqual(Decimal(w["cost"]), 0)
        for input, expected in [(272000, ".544"), (272001, "1.088004")]:
            costs, missing = metrics.price_usage("gpt-6.1-sol", metrics.Usage(input, 0))
            self.assertEqual(costs[metrics.Category.INPUT], Decimal(expected))
            self.assertFalse(missing)

    def test_cache_writes_are_disjoint_and_unknown_rate_is_partial(self):
        u = metrics.Usage(1000, 100, 400, 20, 100)
        costs, missing = metrics.price_usage("gpt-6.1-sol", u)
        self.assertEqual(sum(u.categories().values()), 1100)
        self.assertEqual(costs[metrics.Category.CACHE_WRITE], Decimal(".00025"))
        self.assertFalse(missing)
        costs, missing = metrics.price_usage("gpt-5.5", u)
        self.assertEqual(missing, {metrics.Category.CACHE_WRITE: 100})

    def test_malformed_and_invalid_usage(self):
        self.write(prefix() + ['{bad-json', '[]', modern(counts=usage(cached=2000)),
                              modern(response="valid"), complete()])
        r = self.report()
        self.assertEqual(r["windows"][0]["total_tokens"], 1100)
        self.assertEqual(r["quality"]["Malformed lines"], 1)
        self.assertEqual(r["quality"]["Malformed records"], 1)
        self.assertEqual(r["quality"]["Invalid usage records"], 1)

    def test_empty_report_and_safe_offline_html(self):
        r = self.report()
        self.assertEqual(r["files"], 0)
        self.assertEqual(r["windows"][0]["total_tokens"], 0)
        self.assertIsNone(r["windows"][0]["metrics"]["tools"]["avg"])
        r["source"] = '</script><script>alert("bad")</script>'
        html = metrics.render_report(r)
        self.assertNotIn(r["source"], html)
        self.assertNotIn('__REPORT_DATA__', html)
        self.assertNotIn('<script src=', html)
        self.assertNotIn('fetch(', html)
        self.assertIn('\\u003c/script\\u003e', html)

    def test_malformed_header_does_not_abort_report(self):
        self.write([record("session_meta", [])] + prefix()[1:] + [modern(), complete()])
        r = self.report()
        self.assertEqual(r["files"], 1)
        self.assertEqual(r["windows"][0]["total_tokens"], 1100)
        self.assertEqual(r["quality"]["Malformed records"], 1)

    def test_by_model_splits_multi_model_conversations_and_reconciles_totals(self):
        t2_start = START + timedelta(seconds=20)
        self.write(prefix() + [modern(), modern(response="r2"),
            record("response_item", {"type": "function_call", "call_id": "a1"}), complete(),
            event("task_started", t2_start, turn_id="t2"),
            record("turn_context", {"turn_id": "t2", "model": "gpt-5.5"}, t2_start),
            modern("t2", "r3", usage(6000, 600, 2400, 120), t2_start + timedelta(seconds=1)),
            record("response_item", {"type": "function_call", "call_id": "b1"}, t2_start),
            record("response_item", {"type": "custom_tool_call", "call_id": "b2"}, t2_start),
            event("task_complete", t2_start + timedelta(seconds=30), turn_id="t2",
                  duration_ms=30000, time_to_first_token_ms=2000)])
        self.write(prefix(thread="second", turn="t3") + [
            modern("t3", "r4", usage(3000, 300, 1200, 60)),
            event("task_complete", START + timedelta(seconds=30), turn_id="t3",
                  duration_ms=30000, time_to_first_token_ms=1000)], "second.jsonl")
        r = self.report()
        a = r["by_model"]["gpt-6.1-sol"][0]
        b = r["by_model"]["gpt-5.5"][0]
        total = r["windows"][0]
        self.assertEqual(a["conversations"], 2)
        self.assertEqual(b["conversations"], 1)
        self.assertEqual(total["conversations"], 2)
        self.assertEqual(a["metrics"]["ttft"], {"count": 2, "avg": .75, "min": .5, "median": .75, "max": 1, "p75": 1, "p95": 1, "p99": 1})
        self.assertEqual(b["metrics"]["ttft"]["avg"], 2)
        self.assertEqual(a["metrics"]["throughput"]["avg"], 15)
        self.assertEqual(b["metrics"]["throughput"]["avg"], 20)
        self.assertEqual(a["metrics"]["length"]["avg"], 20)
        self.assertEqual(b["metrics"]["length"]["avg"], 30)
        self.assertEqual(a["metrics"]["tools"]["avg"], .5)
        self.assertEqual(b["metrics"]["tools"]["avg"], 2)
        self.assertEqual(a["total_tokens"], 5500)
        self.assertEqual(b["total_tokens"], 6600)
        for index, total in enumerate(r["windows"]):
            models = [w[index] for w in r["by_model"].values()]
            for key in ["total_tokens", "tool_calls", "active_seconds"]:
                self.assertEqual(sum(w[key] for w in models), total[key])
            self.assertEqual(sum(Decimal(w["cost"]) for w in models), Decimal(total["cost"]))
            for metric in ["ttft", "throughput"]:
                self.assertEqual(sum(w["metrics"][metric]["count"] for w in models), total["metrics"][metric]["count"])

    def test_by_model_handles_legacy_unknown_missing_and_empty_windows(self):
        old = NOW - timedelta(days=60)
        self.write(prefix(model=None, at=old) + [legacy(usage(), old + timedelta(seconds=1)),
                   complete(at=old + timedelta(seconds=10))])
        self.write(prefix(thread="no-usage", model="gpt-5.5") + [complete()], "no-usage.jsonl")
        r = self.report()
        unknown = r["by_model"]["Unknown model"]
        self.assertEqual(unknown[0]["conversations"], 0)
        self.assertIsNone(unknown[0]["metrics"]["ttft"]["avg"])
        last_90 = next(w for w in unknown if w["label"] == "Last 90 days")
        self.assertEqual(last_90["total_tokens"], 1100)
        self.assertTrue(last_90["partial_cost"])
        self.assertEqual(last_90["metrics"]["throughput"]["avg"], 10)
        no_usage = r["by_model"]["gpt-5.5"][0]
        self.assertEqual(no_usage["total_tokens"], 0)
        self.assertEqual(no_usage["metrics"]["ttft"]["avg"], .5)
        self.assertIsNone(no_usage["metrics"]["throughput"]["avg"])

    def test_multiple_models_inside_one_turn_do_not_duplicate_timing(self):
        self.write(prefix() + [modern(),
            event("thread_settings_applied", thread_id="root", thread_settings={"model": "gpt-5.5"}),
            modern(response="r2"), complete()])
        r = self.report()
        self.assertEqual(r["windows"][0]["active_seconds"], 10)
        for model in ["gpt-6.1-sol", "gpt-5.5"]:
            w = r["by_model"][model][0]
            self.assertEqual(w["total_tokens"], 1100)
            self.assertEqual(w["active_seconds"], 0)
            self.assertIsNone(w["metrics"]["throughput"]["avg"])
        mixed = r["by_model"]["Mixed models (timing)"][0]
        self.assertEqual(mixed["total_tokens"], 0)
        self.assertEqual(mixed["active_seconds"], 10)
        self.assertEqual(mixed["metrics"]["throughput"]["avg"], 20)

    def test_today_yesterday_and_short_rolling_boundaries(self):
        windows = {w.label: w for w in metrics.make_windows(NOW)}
        self.assertEqual(next(iter(windows)), "Today")
        self.assertEqual(set(windows), {"Today", "Yesterday",
                         "Last 7 days", "Last 14 days", "Last 30 days",
                         "Last 60 days", "Last 90 days", "Last 180 days", "Last 365 days"})
        today = windows["Today"]
        yesterday = windows["Yesterday"]
        midnight_hour = 4 if str(metrics.TIMEZONE) == "America/Toronto" else 0
        self.assertEqual(today.start, datetime(2026, 9, 30, midnight_hour, tzinfo=timezone.utc))
        self.assertEqual(today.end, NOW)
        self.assertEqual(yesterday.start, datetime(2026, 9, 29, midnight_hour, tzinfo=timezone.utc))
        self.assertEqual(yesterday.end, today.start)
        self.assertTrue(yesterday.contains(yesterday.start))
        self.assertTrue(yesterday.contains(today.start - timedelta(microseconds=1)))
        self.assertFalse(yesterday.contains(today.start))
        self.assertTrue(today.contains(today.start))
        self.assertFalse(today.contains(NOW + timedelta(microseconds=1)))
        for days in [7, 14, 60]:
            w = windows[f"Last {days} days"]
            self.assertEqual(w.start, NOW - timedelta(days=days))
            self.assertTrue(w.contains(w.start))
            self.assertFalse(w.contains(w.start - timedelta(microseconds=1)))

    @unittest.skipUnless(str(metrics.TIMEZONE) == "America/Toronto", "Toronto timezone data is unavailable")
    def test_yesterday_uses_calendar_day_across_dst_and_month_boundary(self):
        for cutoff, hours in [(datetime(2026, 3, 9, 16, tzinfo=timezone.utc), 23),
                              (datetime(2026, 11, 2, 17, tzinfo=timezone.utc), 25)]:
            with self.subTest(cutoff=cutoff):
                windows = {w.label: w for w in metrics.make_windows(cutoff)}
                yesterday = windows["Yesterday"]
                self.assertEqual((yesterday.end - yesterday.start).total_seconds(), hours * 3600)
                self.assertEqual(yesterday.start.astimezone(metrics.TIMEZONE).hour, 0)
                self.assertEqual(yesterday.end.astimezone(metrics.TIMEZONE).hour, 0)
                self.assertEqual((windows["Last 7 days"].end - windows["Last 7 days"].start).total_seconds(), 7 * 86400)
        windows = {w.label: w for w in metrics.make_windows(datetime(2026, 10, 1, 16, tzinfo=timezone.utc))}
        self.assertEqual(windows["Yesterday"].start, datetime(2026, 9, 30, 4, tzinfo=timezone.utc))

    def test_midnight_activity_is_counted_once_in_overall_and_model_windows(self):
        windows = {w.label: w for w in metrics.make_windows(NOW)}
        yesterday_start = windows["Yesterday"].start
        midnight = windows["Today"].start
        self.write(prefix(at=yesterday_start - timedelta(seconds=2)) + [
            modern(response="before", at=yesterday_start - timedelta(seconds=1)),
            modern(response="yesterday", at=yesterday_start),
            record("response_item", {"type": "function_call", "call_id": "yesterday-call"}, yesterday_start),
            modern(response="today", at=midnight),
            record("response_item", {"type": "function_call", "call_id": "today-call"}, midnight),
            complete(at=midnight + timedelta(seconds=10))])
        r = self.report()
        for summaries in [r["windows"], r["by_model"]["gpt-6.1-sol"]]:
            by_label = {w["label"]: w for w in summaries}
            today, yesterday = by_label["Today"], by_label["Yesterday"]
            self.assertEqual(today["total_tokens"], 1100)
            self.assertEqual(yesterday["total_tokens"], 1100)
            self.assertEqual(today["tool_calls"], 1)
            self.assertEqual(yesterday["tool_calls"], 1)
            self.assertEqual(today["active_seconds"], 10)
            self.assertEqual(yesterday["active_seconds"], 0)
            self.assertEqual(today["metrics"]["ttft"]["count"], 1)
            self.assertEqual(yesterday["metrics"]["ttft"]["count"], 0)
            self.assertTrue(yesterday["end_exclusive"])

    def test_recorded_mode_mapping_and_partial_settings(self):
        for tier, mode in [("default", metrics.SpeedMode.NORMAL), ("priority", metrics.SpeedMode.FAST),
                           ("fast", metrics.SpeedMode.FAST), (None, metrics.SpeedMode.NORMAL),
                           ("auto", metrics.SpeedMode.NORMAL), ("flex", metrics.SpeedMode.OTHER)]:
            with self.subTest(tier=tier):
                self.assertEqual(metrics.recorded_mode({"service_tier": tier}, metrics.SpeedMode.FAST), mode)
        self.assertEqual(metrics.recorded_mode({}, metrics.SpeedMode.FAST), metrics.SpeedMode.FAST)
        self.assertEqual(metrics.recorded_mode({}), metrics.SpeedMode.NORMAL)
        rows = prefix()
        rows[0]["payload"]["service_tier"] = "priority"
        self.write(rows + [event("thread_settings_applied", thread_settings={"model": "gpt-5.5"}),
                          modern(), complete()])
        r = self.report()
        self.assertEqual(r["by_mode"]["Fast"]["windows"][0]["total_tokens"], 1100)
        self.assertEqual(r["by_mode"]["Fast"]["by_model"]["gpt-5.5"][0]["total_tokens"], 1100)

    def test_mode_switch_between_turns_and_model_intersection(self):
        rows = prefix()
        rows[0]["payload"]["service_tier"] = "default"
        self.write(rows + [modern(), complete(),
            record("response_item", {"type": "function_call", "call_id": "normal"}),
            event("thread_settings_applied", thread_settings={"service_tier": "priority", "model": "gpt-5.5"}),
            event("task_started", turn_id="t2"), modern("t2", "r2"), complete("t2"),
            record("response_item", {"type": "function_call", "call_id": "fast"})])
        r = self.report()
        normal = r["by_mode"]["Normal"]["windows"][0]
        fast = r["by_mode"]["Fast"]["windows"][0]
        for w in [normal, fast]:
            self.assertEqual(w["conversations"], 1)
            self.assertEqual(w["total_tokens"], 1100)
            self.assertEqual(w["tool_calls"], 1)
            self.assertEqual(w["metrics"]["ttft"], {"count": 1, "avg": .5, "min": .5, "median": .5, "max": .5, "p75": .5, "p95": .5, "p99": .5})
            self.assertEqual(w["metrics"]["length"]["avg"], 10)
            self.assertEqual(w["metrics"]["throughput"]["avg"], 10)
        self.assertEqual(Decimal(normal["cost"]), Decimal(".00224"))
        self.assertEqual(Decimal(fast["cost"]), Decimal(".0093"))
        self.assertEqual(r["by_mode"]["Normal"]["by_model"]["gpt-5.5"][0]["total_tokens"], 0)
        self.assertEqual(r["windows"][0]["conversations"], 1)
        for i, total in enumerate(r["windows"]):
            modes = [g["windows"][i] for g in r["by_mode"].values()]
            for key in ["total_tokens", "tool_calls", "active_seconds"]:
                self.assertEqual(sum(w[key] for w in modes), total[key])
            self.assertEqual(sum(Decimal(w["cost"]) for w in modes), Decimal(total["cost"]))
            for metric in ["ttft", "throughput"]:
                self.assertEqual(sum(w["metrics"][metric]["count"] for w in modes), total["metrics"][metric]["count"])
            for group in r["by_mode"].values():
                self.assertEqual(sum(ws[i]["total_tokens"] for ws in group["by_model"].values()), group["windows"][i]["total_tokens"])

    def test_mode_switch_within_turn_does_not_duplicate_timing(self):
        rows = prefix()
        rows[0]["payload"]["service_tier"] = "default"
        self.write(rows + [modern(), event("thread_settings_applied", thread_settings={"service_tier": "fast"}),
                           modern(response="r2"), complete()])
        r = self.report()
        for mode in ["Normal", "Fast"]:
            w = r["by_mode"][mode]["windows"][0]
            self.assertEqual(w["total_tokens"], 1100)
            self.assertEqual(w["active_seconds"], 0)
            self.assertIsNone(w["metrics"]["ttft"]["avg"])
        mixed = r["by_mode"]["Mixed modes (timing)"]["windows"][0]
        self.assertEqual(mixed["total_tokens"], 0)
        self.assertEqual(mixed["active_seconds"], 10)
        self.assertEqual(mixed["metrics"]["throughput"]["avg"], 20)

    def test_unknown_and_explicit_normal_merge_before_timing_and_percentiles(self):
        self.write(prefix() + [modern(),
            event("thread_settings_applied", thread_settings={"service_tier": "default"}),
            modern(response="r2"), complete(),
            event("task_started", turn_id="t2"),
            modern("t2", "r3"), complete("t2")])
        r = self.report()
        normal = r["by_mode"]["Normal"]["windows"][0]
        self.assertEqual(normal["total_tokens"], 3300)
        self.assertEqual(normal["conversations"], 1)
        self.assertEqual(normal["metrics"]["length"], {"count": 1, "avg": 20, "min": 20, "median": 20, "max": 20, "p75": 20, "p95": 20, "p99": 20})
        self.assertEqual(normal["metrics"]["ttft"]["count"], 2)
        self.assertEqual(normal["metrics"]["throughput"]["avg"], 15)
        self.assertEqual(normal["cost"], r["windows"][0]["cost"])
        self.assertNotIn("Unknown", r["by_mode"])
        self.assertNotIn("Mixed modes (timing)", r["by_mode"])
        self.assertEqual(r["by_mode"]["Normal"]["by_model"]["gpt-6.1-sol"][0], normal)

    def test_response_tier_override_and_explicit_unknown(self):
        rows = prefix()
        rows[0]["payload"]["service_tier"] = "priority"
        response = modern()
        response["payload"]["service_tier"] = "default"
        self.write(rows + [response, modern(response="r2"),
            event("thread_settings_applied", thread_settings={"service_tier": None}),
            modern(response="r3"), complete()])
        r = self.report()
        self.assertEqual(r["by_mode"]["Normal"]["windows"][0]["total_tokens"], 2200)
        self.assertEqual(r["by_mode"]["Fast"]["windows"][0]["total_tokens"], 1100)
        self.assertNotIn("Unknown", r["by_mode"])
        self.assertEqual(Decimal(r["windows"][0]["cost"]), Decimal(".00784"))

    def test_spark_proxy_and_fast_costs_for_every_category(self):
        u = metrics.Usage(1000, 100, 400, 20)
        base, missing = metrics.price_usage("gpt-5.4-mini", u)
        proxy, proxy_missing = metrics.price_usage("gpt-5.3-codex-spark", u)
        fast, _ = metrics.price_usage("gpt-5.3-codex-spark", u, metrics.SpeedMode.FAST)
        self.assertEqual(proxy, base)
        self.assertFalse(missing or proxy_missing)
        for category, cost in base.items():
            self.assertEqual(fast[category], cost * Decimal("1.5"))
        base_write, _ = metrics.price_usage("gpt-6.1-sol", metrics.Usage(1000, 100, 400, 20, 100))
        fast_write, _ = metrics.price_usage("gpt-6.1-sol", metrics.Usage(1000, 100, 400, 20, 100), metrics.SpeedMode.FAST)
        self.assertEqual(fast_write[metrics.Category.CACHE_WRITE], base_write[metrics.Category.CACHE_WRITE] * Decimal("1.5"))
        self.write(prefix(model="gpt-5.3-codex-spark") + [modern(), complete()])
        w = self.report()["windows"][0]
        self.assertFalse(w["partial_cost"])
        self.assertEqual(w["models"], {"gpt-5.3-codex-spark": 1100})
        self.assertEqual(Decimal(w["cost"]), Decimal(".00093"))

    def test_auto_review_luna_proxy_long_context_and_fast_premium(self):
        for input in [1000, 272000, 272001]:
            for mode in [metrics.SpeedMode.NORMAL, metrics.SpeedMode.FAST]:
                with self.subTest(input=input, mode=mode):
                    u = metrics.Usage(input, 100, 400, 20, 100)
                    base, missing = metrics.price_usage("gpt-5.6-luna", u)
                    proxy, proxy_missing = metrics.price_usage("codex-auto-review", u, mode)
                    multiplier = Decimal("1.5") if mode == metrics.SpeedMode.FAST else Decimal(1)
                    self.assertEqual(proxy, {category: cost * multiplier for category, cost in base.items()})
                    self.assertFalse(missing or proxy_missing)
        rows = prefix(model="codex-auto-review")
        rows[0]["payload"]["service_tier"] = "priority"
        self.write(rows + [modern(), complete()])
        report = self.report()
        self.assertEqual(report["price_proxies"]["codex-auto-review"], "gpt-5.6-luna")
        w = report["by_mode"]["Fast"]["by_model"]["codex-auto-review"][0]
        self.assertFalse(w["partial_cost"])
        self.assertEqual(Decimal(w["cost"]), Decimal(".000372"))
        self.assertIn('codex-auto-review uses GPT-5.6-luna rates', metrics.render_report(report))

    def test_unknown_mode_is_normal_and_preserves_midnight_boundaries(self):
        midnight = metrics.make_windows(NOW)[0].start
        self.write(prefix() + [modern(at=midnight - timedelta(microseconds=1)),
                              modern(response="r2", at=midnight), complete()])
        r = self.report()
        group = r["by_mode"]["Normal"]
        for windows in [group["windows"], group["by_model"]["gpt-6.1-sol"]]:
            by_label = {w["label"]: w for w in windows}
            self.assertEqual(by_label["Yesterday"]["total_tokens"], 1100)
            self.assertEqual(by_label["Today"]["total_tokens"], 1100)
            self.assertTrue(by_label["Yesterday"]["end_exclusive"])
        self.assertNotIn("Unknown", r["by_mode"])
        self.assertEqual(r["unknown_mode_assumption"], "Normal")
        self.assertEqual(r["by_mode"]["Fast"]["windows"][0]["total_tokens"], 0)

    def test_inherited_fast_settings_do_not_classify_child_usage(self):
        rows = [record("session_meta", {"id": "child"}), record("session_meta", {"id": "parent"}),
                event("thread_settings_applied", thread_settings={"service_tier": "priority"}),
                modern(), event("task_started", thread_id="child", turn_id="child-turn"),
                record("turn_context", {"turn_id": "child-turn", "model": "gpt-6.1-sol"}),
                modern("child-turn", "child-response"), complete("child-turn")]
        self.write(rows)
        r = self.report()
        self.assertEqual(r["by_mode"]["Normal"]["windows"][0]["total_tokens"], 1100)
        self.assertEqual(r["by_mode"]["Fast"]["windows"][0]["total_tokens"], 0)

    def test_short_rolling_windows_select_activity_by_record_time(self):
        self.write(prefix(at=NOW - timedelta(days=15)) + [
            modern(response="old", at=NOW - timedelta(days=14, microseconds=1)),
            modern(response="fourteen", at=NOW - timedelta(days=14)),
            modern(response="seven", at=NOW - timedelta(days=7)),
            complete()])
        r = self.report()
        for summaries in [r["windows"], r["by_model"]["gpt-6.1-sol"]]:
            by_label = {w["label"]: w for w in summaries}
            self.assertEqual(by_label["Last 7 days"]["total_tokens"], 1100)
            self.assertEqual(by_label["Last 14 days"]["total_tokens"], 2200)
            self.assertEqual(by_label["Last 30 days"]["total_tokens"], 3300)

    def test_model_tier_membership_including_pricing_proxies(self):
        expected = {"gpt-5.6-luna": "Budget", "gpt-6-luna": "Budget", "gpt-5.6-terra": "Budget",
                    "gpt-5.4-mini": "Budget", "gpt-5.3-codex-spark": "Budget", "codex-auto-review": "Budget",
                    "gpt-5.6-sol": "Medium", "gpt-6-sol": "Medium", "gpt-6.1-sol": "Medium",
                    "gpt-5.4": "Medium", "gpt-5.5": "Medium", "gpt-6-astra": "High",
                    None: "Unclassified", "unpublished-model": "Unclassified", "gpt-5.4-pro": "Unclassified"}
        for model, tier in expected.items():
            with self.subTest(model=model):
                self.assertEqual(metrics.model_tier(model).value, tier)

    def test_tier_metrics_combine_samples_and_deduplicate_conversations(self):
        second_start = START + timedelta(seconds=20)
        self.write(prefix(model="gpt-6-luna") + [modern(),
            record("response_item", {"type": "function_call", "call_id": "a"}), complete(),
            event("thread_settings_applied", thread_settings={"model": "gpt-5.6-terra", "service_tier": "fast"}),
            event("task_started", second_start, turn_id="t2"),
            modern("t2", "r2", usage(2000, 200, 800, 40), second_start + timedelta(seconds=1)),
            record("response_item", {"type": "function_call", "call_id": "b"}, second_start),
            record("response_item", {"type": "custom_tool_call", "call_id": "c"}, second_start),
            event("task_complete", second_start + timedelta(seconds=30), turn_id="t2",
                  duration_ms=30000, time_to_first_token_ms=2000)])
        self.write(prefix(thread="second", turn="t3", model="gpt-6-luna") + [
            modern("t3", "r3", usage(3000, 300, 1200, 60)),
            event("task_complete", START + timedelta(seconds=20), turn_id="t3",
                  duration_ms=20000, time_to_first_token_ms=1000)], "second.jsonl")
        r = self.report()
        tier = r["by_tier"]["Budget"]
        w = tier["windows"][0]
        self.assertEqual(w["conversations"], 2)
        self.assertEqual(w["total_tokens"], 6600)
        self.assertEqual(w["metrics"]["length"], {"count": 2, "avg": 30, "min": 20, "median": 30, "max": 40, "p75": 40, "p95": 40, "p99": 40})
        self.assertEqual(w["metrics"]["tools"], {"count": 2, "avg": 1.5, "min": 0, "median": 1.5, "max": 3, "p75": 3, "p95": 3, "p99": 3})
        self.assertAlmostEqual(w["metrics"]["ttft"]["avg"], 7 / 6)
        self.assertEqual(w["metrics"]["ttft"]["p95"], 2)
        self.assertEqual(w["metrics"]["throughput"]["p99"], 15)
        self.assertEqual(tier["by_mode"]["Fast"]["windows"][0]["active_seconds"], 30)
        self.assertEqual(tier["by_mode"]["Fast"]["by_model"]["gpt-5.6-terra"][0]["total_tokens"], 2200)
        self.assertEqual(tier["by_mode"]["Normal"]["by_model"]["gpt-5.6-terra"][0]["total_tokens"], 0)
        self.assertEqual(sum(ws[0]["conversations"] for ws in tier["by_model"].values()), 3)
        for i, total in enumerate(r["windows"]):
            tiers = [g["windows"][i] for g in r["by_tier"].values()]
            for key in ["total_tokens", "tool_calls", "active_seconds"]:
                self.assertEqual(sum(w[key] for w in tiers), total[key])
            self.assertEqual(sum(Decimal(w["cost"]) for w in tiers), Decimal(total["cost"]))
            for metric in ["ttft", "throughput"]:
                self.assertEqual(sum(w["metrics"][metric]["count"] for w in tiers), total["metrics"][metric]["count"])
            for group in r["by_tier"].values():
                self.assertEqual(sum(ws[i]["total_tokens"] for ws in group["by_model"].values()), group["windows"][i]["total_tokens"])
                self.assertEqual(sum(Decimal(m["windows"][i]["cost"]) for m in group["by_mode"].values()), Decimal(group["windows"][i]["cost"]))

    def test_same_tier_model_switch_preserves_whole_turn_timing(self):
        self.write(prefix(model="gpt-6-luna") + [modern(),
            event("thread_settings_applied", thread_settings={"model": "gpt-5.6-terra"}),
            modern(response="r2"), complete()])
        r = self.report()
        w = r["by_tier"]["Budget"]["windows"][0]
        self.assertEqual(w["metrics"]["ttft"]["count"], 1)
        self.assertEqual(w["active_seconds"], 10)
        self.assertEqual(w["metrics"]["throughput"]["avg"], 20)
        self.assertEqual(r["by_tier"]["Budget"]["by_model"]["Mixed models (timing)"][0]["active_seconds"], 10)
        self.assertNotIn("Mixed tiers (timing)", r["by_tier"])

    def test_cross_tier_turn_separates_usage_and_timing(self):
        self.write(prefix(model="gpt-6-luna") + [modern(),
            record("response_item", {"type": "function_call", "call_id": "a"}),
            event("thread_settings_applied", thread_settings={"model": "gpt-6-astra", "service_tier": "fast"}),
            modern(response="r2"),
            record("response_item", {"type": "function_call", "call_id": "b"}), complete()])
        r = self.report()
        for tier in ["Budget", "High"]:
            w = r["by_tier"][tier]["windows"][0]
            self.assertEqual(w["total_tokens"], 1100)
            self.assertEqual(w["tool_calls"], 1)
            self.assertEqual(w["active_seconds"], 0)
            self.assertIsNone(w["metrics"]["ttft"]["avg"])
        mixed = r["by_tier"]["Mixed tiers (timing)"]
        self.assertEqual(mixed["windows"][0]["active_seconds"], 10)
        self.assertEqual(mixed["windows"][0]["total_tokens"], 0)
        self.assertEqual(mixed["by_mode"]["Mixed modes (timing)"]["windows"][0]["metrics"]["throughput"]["avg"], 20)
        self.assertEqual(r["by_tier"]["High"]["by_mode"]["Fast"]["windows"][0]["cost"],
                         r["by_model"]["gpt-6-astra"][0]["cost"])

    def test_tier_unclassified_empty_and_calendar_boundary_windows(self):
        midnight = metrics.make_windows(NOW)[0].start
        self.write(prefix(model=None) + [modern(at=midnight - timedelta(microseconds=1)),
                                        modern(response="r2", at=midnight), complete()])
        r = self.report()
        group = r["by_tier"]["Unclassified"]
        for windows in [group["windows"], group["by_model"]["Unknown model"],
                        group["by_mode"]["Normal"]["windows"],
                        group["by_mode"]["Normal"]["by_model"]["Unknown model"]]:
            by_label = {w["label"]: w for w in windows}
            self.assertEqual(by_label["Today"]["total_tokens"], 1100)
            self.assertEqual(by_label["Yesterday"]["total_tokens"], 1100)
            self.assertTrue(by_label["Yesterday"]["end_exclusive"])
            self.assertTrue(by_label["Today"]["partial_cost"])
        for tier in ["Budget", "Medium", "High"]:
            self.assertFalse(r["by_tier"][tier]["by_model"])
            self.assertIsNone(r["by_tier"][tier]["windows"][0]["metrics"]["length"]["avg"])

    def test_last_60_days_boundary_in_overall_model_and_mode_windows(self):
        boundary = NOW - timedelta(days=60)
        self.write(prefix(at=boundary - timedelta(days=1)) + [
            modern(response="outside", at=boundary - timedelta(microseconds=1)),
            modern(response="boundary", at=boundary),
            modern(response="within", at=NOW - timedelta(days=31)),
            modern(response="recent"), complete()])
        r = self.report()
        for windows in [r["windows"], r["by_model"]["gpt-6.1-sol"],
                        r["by_mode"]["Normal"]["windows"],
                        r["by_mode"]["Normal"]["by_model"]["gpt-6.1-sol"]]:
            by_label = {w["label"]: w for w in windows}
            self.assertEqual(by_label["Last 30 days"]["total_tokens"], 1100)
            self.assertEqual(by_label["Last 60 days"]["total_tokens"], 3300)
            self.assertEqual(by_label["Last 90 days"]["total_tokens"], 4400)
            self.assertEqual(by_label["Last 60 days"]["metrics"]["ttft"]["count"], 1)
            self.assertEqual(by_label["Last 60 days"]["active_seconds"], 10)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        clock = patch("harness_metrics.datetime", wraps=datetime)
        clock.start().now.return_value = NOW
        self.addCleanup(clock.stop)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return path

    def report(self, sources, catalog=None):
        return metrics.collect_report(self.root / "codex", NOW, harness_roots=sources, catalog=catalog)

    def claude(self, output=100, request="r1", message="m1", at=START):
        return {"type": "assistant", "sessionId": "same-id", "timestamp": at.isoformat(), "requestId": request,
                "message": {"id": message, "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
                            "usage": {"input_tokens": 100, "output_tokens": output,
                                      "cache_read_input_tokens": 400, "cache_creation_input_tokens": 200,
                                      "cache_creation": {"ephemeral_1h_input_tokens": 50}},
                            "content": [{"type": "tool_use", "id": "tool1", "input": {"secret": "private text"}}]}}

    def copilot(self, kind, identifier, data, at=START):
        return {"type": kind, "id": identifier, "timestamp": at.isoformat(), "data": data}

    def opencode_db(self, name="opencode/opencode.db", model="claude-sonnet-4-6", provider="anthropic"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE message (id TEXT, session_id TEXT, data TEXT)")
            db.execute("CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, data TEXT)")
            data = {"role": "assistant", "modelID": model, "providerID": provider, "cost": .012,
                    "time": {"created": int(START.timestamp() * 1000), "completed": int((START + timedelta(seconds=10)).timestamp() * 1000)},
                    "tokens": {"input": 100, "output": 80, "reasoning": 20, "cache": {"read": 400, "write": 200}}}
            db.execute("INSERT INTO message VALUES (?, ?, ?)", ("m1", "same-id", json.dumps(data)))
            db.execute("INSERT INTO part VALUES (?, ?, ?, ?)", ("p1", "m1", "same-id", json.dumps({"type": "tool", "callID": "c1", "state": {"status": "completed"}})))
        return path

    def catalog(self):
        return metrics.openrouter_prices({"data": [
            {"id": "anthropic/claude-haiku-4.5", "pricing": {"prompt": ".000001", "completion": ".000005", "input_cache_read": ".0000001"}},
            {"id": "vendor/new-model", "pricing": {"prompt": ".000002", "completion": ".000010", "input_cache_read": ".0000002", "input_cache_write": ".0000025", "input_cache_write_1h": ".000004",
                    "overrides": [{"min_prompt_tokens": 1000, "prompt": ".000004"}, {"min_prompt_tokens": 2000, "prompt": ".000008"}]}}]})

    def test_claude_cache_normalization_streaming_and_copy_deduplication(self):
        rows = [{"type": "user", "sessionId": "same-id", "uuid": "user1", "timestamp": START.isoformat(), "message": {"content": "private text"}},
                self.claude(output=1), self.claude(message="m2"),
                {"type": "system", "subtype": "turn_duration", "durationMs": 10000, "timestamp": (START + timedelta(seconds=10)).isoformat()}]
        self.write(rows, "claude/project/session.jsonl")
        self.write(rows, "claude/project/copy.jsonl")
        r = self.report({metrics.Harness.CLAUDE: [self.root / "claude"]})
        w = r["windows"][0]
        self.assertEqual(w["total_tokens"], 800)
        self.assertEqual(w["tool_calls"], 1)
        self.assertEqual(w["active_seconds"], 10)
        self.assertEqual(Decimal(w["cost"]), Decimal(".0027825"))
        self.assertFalse(w["partial_cost"])
        self.assertNotIn("private text", metrics.render_report(r))
        self.assertEqual(r["by_harness"]["claude"]["windows"][0], w)

    def test_claude_subagents_with_shared_session_id_are_separate(self):
        self.write([self.claude()], "claude/project/main.jsonl")
        self.write([self.claude(request="child", message="child")], "claude/project/main/subagents/agent-child.jsonl")
        r = self.report({metrics.Harness.CLAUDE: [self.root / "claude"]})
        self.assertEqual(r["threads"], 2)
        self.assertEqual(r["windows"][0]["total_tokens"], 1600)

    def test_claude_invalid_usage_is_diagnosed(self):
        row = self.claude()
        row["message"]["usage"]["cache_creation_input_tokens"] = -1
        self.write([row, {**row, "message": {"usage": None}}], "claude/session.jsonl")
        with redirect_stderr(io.StringIO()):
            r = self.report({metrics.Harness.CLAUDE: [self.root / "claude"]})
        self.assertEqual(r["windows"][0]["total_tokens"], 0)
        self.assertEqual(r["quality"]["Invalid usage records"], 2)

    def test_copilot_per_response_usage_ignores_shutdown_double_count(self):
        usage_data = {"model": "claude-sonnet-4.6", "inputTokens": 700, "outputTokens": 100,
                      "cacheReadTokens": 400, "cacheWriteTokens": 200, "reasoningTokens": 20,
                      "apiCallId": "response", "timeToFirstTokenMs": 500, "cost": 3,
                      "copilotUsage": {"totalNanoAiu": 1000000}}
        rows = [self.copilot("session.start", "start", {"sessionId": "same-id"}),
                self.copilot("assistant.turn_start", "turn", {"turnId": "t1"}),
                self.copilot("assistant.usage", "usage", usage_data),
                self.copilot("assistant.usage", "copy", usage_data),
                self.copilot("tool.execution_start", "tool", {"toolCallId": "c1"}),
                self.copilot("assistant.turn_end", "end", {"turnId": "t1"}, START + timedelta(seconds=10)),
                self.copilot("session.shutdown", "shutdown", {"modelMetrics": {"claude-sonnet-4.6": {"usage": usage_data}}})]
        self.write(rows, "copilot/session/events.jsonl")
        r = self.report({metrics.Harness.COPILOT: [self.root / "copilot"]})
        w = r["windows"][0]
        self.assertEqual(w["total_tokens"], 800)
        self.assertEqual(w["tool_calls"], 1)
        self.assertEqual(w["metrics"]["ttft"]["avg"], .5)
        self.assertEqual(Decimal(w["cost"]), Decimal(".00267"))
        self.assertEqual(w["recorded_billing"], {"Copilot nano-AIU": "1000000", "Copilot premium requests": "3"})

    def test_copilot_invalid_models_are_skipped_without_losing_valid_usage(self):
        rows = [self.copilot("session.start", "start", {"sessionId": "same-id"}),
                self.copilot("session.model_change", "bad-model", {"newModel": 123}),
                self.copilot("assistant.usage", "bad-usage", {"model": [1], "inputTokens": 100, "outputTokens": 10}),
                self.copilot("assistant.usage", "good", {"model": "gpt-6.1-sol", "inputTokens": 100, "outputTokens": 10})]
        self.write(rows, "copilot/events.jsonl")
        with redirect_stderr(io.StringIO()):
            report = self.report({metrics.Harness.COPILOT: [self.root / "copilot"]})
        self.assertEqual(report["quality"]["Malformed records"], 2)
        self.assertEqual(report["windows"][0]["total_tokens"], 110)

    def test_copilot_shutdown_only_uses_latest_cumulative_summary(self):
        def summary(count):
            return {"modelMetrics": {"gpt-6.1-sol": {"usage": {"inputTokens": count, "outputTokens": 100,
                    "cacheReadTokens": 400, "cacheWriteTokens": 0}, "totalNanoAiu": count,
                    "requests": {"cost": 2}}}}
        self.write([self.copilot("session.start", "s", {"sessionId": "same-id"}),
                    self.copilot("session.shutdown", "a", summary(1000)),
                    self.copilot("session.shutdown", "b", summary(2000))], "copilot/events.jsonl")
        r = self.report({metrics.Harness.COPILOT: [self.root / "copilot"]})
        self.assertEqual(r["windows"][0]["total_tokens"], 2100)
        self.assertEqual(r["windows"][0]["coverage"]["Aggregate snapshots"], 1)
        self.assertEqual(r["windows"][0]["recorded_billing"]["Copilot nano-AIU"], "2000")

    def test_copilot_summary_reconciles_usage_missing_from_detailed_events(self):
        counts = {"model": "gpt-6.1-sol", "inputTokens": 1000, "outputTokens": 100,
                  "cacheReadTokens": 400, "cacheWriteTokens": 0}
        summary = {"modelMetrics": {"gpt-6.1-sol": {"usage": {**counts, "inputTokens": 2000,
                   "outputTokens": 200, "cacheReadTokens": 800}}}}
        self.write([self.copilot("session.start", "s", {"sessionId": "same-id"}),
                    self.copilot("assistant.usage", "u", counts),
                    self.copilot("session.shutdown", "end", summary)], "copilot/events.jsonl")
        r = self.report({metrics.Harness.COPILOT: [self.root / "copilot"]})
        self.assertEqual(r["windows"][0]["total_tokens"], 2200)
        self.assertEqual(Decimal(r["windows"][0]["cost"]), Decimal(".00448"))
        self.assertEqual(r["windows"][0]["coverage"]["Aggregate snapshots"], 1)

    def test_opencode_normalizes_disjoint_categories_and_preserves_recorded_cost(self):
        db = self.opencode_db()
        before = db.read_bytes()
        r = self.report({metrics.Harness.OPENCODE: [db, db.parent]})
        self.assertEqual(db.read_bytes(), before)
        w = r["windows"][0]
        self.assertEqual(w["total_tokens"], 800)
        self.assertEqual(w["tool_calls"], 1)
        self.assertEqual(w["metrics"]["throughput"]["avg"], 10)
        self.assertEqual(w["recorded_billing"], {"OpenCode recorded USD": "0.012"})
        self.assertEqual(Decimal(w["cost"]), Decimal(".00267"))

    def test_opencode_unknown_model_uses_openrouter_catalog(self):
        db = self.opencode_db(model="vendor/new-model", provider="openrouter")
        r = self.report({metrics.Harness.OPENCODE: [db]}, self.catalog())
        self.assertFalse(r["windows"][0]["partial_cost"])
        self.assertEqual(Decimal(r["windows"][0]["cost"]), Decimal(".00178"))

    def test_harness_totals_reconcile_and_session_ids_do_not_collide(self):
        self.write(prefix(thread="same-id") + [modern(), complete()], "codex/session.jsonl")
        self.write([self.claude()], "claude/session.jsonl")
        r = self.report({metrics.Harness.CLAUDE: [self.root / "claude"],
                         metrics.Harness.OPENCODE: [self.opencode_db()]}, self.catalog())
        for i, window in enumerate(r["windows"]):
            members = [h["windows"][i] for h in r["by_harness"].values()]
            for key in ["total_tokens", "tool_calls", "active_seconds", "conversations", "unpriced_tokens"]:
                self.assertEqual(sum(w[key] for w in members), window[key])
            self.assertEqual(sum(Decimal(w["cost"]) for w in members), Decimal(window["cost"]))
        self.assertEqual(r["windows"][0]["conversations"], 3)

    def test_openrouter_per_token_rates_cache_hour_and_context_overrides(self):
        prices = self.catalog()
        costs, missing = metrics.price_usage("vendor/new-model", metrics.Usage(700, 100, 400, 20, 200, 50), catalog=prices)
        self.assertFalse(missing)
        self.assertEqual(sum(costs.values()), Decimal(".001855"))
        for count, rate in [(999, 2), (1000, 4), (2000, 8)]:
            costs, _ = metrics.price_usage("vendor/new-model", metrics.Usage(count, 0), catalog=prices)
            self.assertEqual(costs[metrics.Category.INPUT], Decimal(count * rate) / metrics.MILLION)
        self.assertIsNone(metrics.router_model("Claude Future 99", prices))
        self.assertEqual(metrics.router_model("claude-haiku-4-5-20251001", prices), "anthropic/claude-haiku-4.5")
        self.assertEqual(metrics.model_tier("Claude Haiku 4.5"), metrics.ModelTier.BUDGET)
        self.assertEqual(metrics.model_tier("anthropic/claude-sonnet-4.6"), metrics.ModelTier.MEDIUM)

    def test_claude_fast_is_not_charged_codex_multiplier(self):
        normal, _ = metrics.price_usage("claude-opus-5-5", metrics.Usage(100, 10))
        fast, _ = metrics.price_usage("claude-opus-5-5", metrics.Usage(100, 10), metrics.SpeedMode.FAST)
        self.assertEqual(sum(fast.values()), sum(normal.values()) * 2)
        _, missing = metrics.price_usage("claude-sonnet-4-6", metrics.Usage(100, 10), metrics.SpeedMode.FAST)
        self.assertEqual(sum(missing.values()), 110)

    def test_cli_explicit_harness_offline_and_source_override(self):
        self.write([self.claude()], "claude/session.jsonl")
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "home"), \
             patch("sys.argv", ["harness_metrics.py", str(self.root), "--harness", "claude", "--claude-dir", str(self.root / "claude"), "--offline", "--output", str(output)]), \
             patch("harness_metrics.urlopen") as request, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
            request.assert_not_called()
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual(set(report["by_harness"]), {"claude"})
        self.assertEqual(next(w for w in report["windows"] if w["label"] == "Last 7 days")["total_tokens"], 800)

    def test_cli_openrouter_failure_retains_report_and_error(self):
        db = self.opencode_db(model="unknown-model", provider="test")
        output = self.root / "report.html"
        with patch("sys.argv", ["harness_metrics.py", str(self.root), "--harness", "opencode", "--opencode-dir", str(db), "--output", str(output)]), \
             patch("harness_metrics.urlopen", side_effect=OSError("offline")), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual(report["openrouter"]["error"], "offline")
        self.assertTrue(next(w for w in report["windows"] if w["label"] == "Last 7 days")["partial_cost"])

    def test_cli_missing_default_copilot_directory_is_not_an_error(self):
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "missing-home"), \
             patch("sys.argv", ["harness_metrics.py", str(self.root), "--harness", "copilot", "--output", str(output)]), \
             patch("harness_metrics.urlopen") as request, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(metrics.main(), 0)
            request.assert_not_called()
        self.assertEqual(errors.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
