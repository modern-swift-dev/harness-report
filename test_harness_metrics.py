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
        home = patch("harness_metrics.Path.home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)

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

    def test_inherited_cumulative_counters_set_the_fork_baseline(self):
        parent = prefix() + [legacy(usage(10_000, 1_000)), complete()]
        child_turn = [event("task_started", turn_id="child-turn"),
                      record("turn_context", {"turn_id": "child-turn", "model": "gpt-6.1-sol"}),
                      legacy(usage(11_000, 1_100), last=usage(1_000, 100)), complete("child-turn")]
        variants = {
            "ordinal": [record("session_meta", {"id": "child", "forked_from_id": "root",
                                                "subagent_history_start_ordinal": len(parent) + 1})] + parent + child_turn,
            "copied metadata": [record("session_meta", {"id": "child"})] + parent + [
                event("thread_settings_applied", thread_id="child", thread_settings={"model": "gpt-6.1-sol"})] + child_turn,
        }
        for name, rows in variants.items():
            with self.subTest(name):
                self.write(rows, "logs/child.jsonl")
                w = self.report()["windows"][0]
                self.assertEqual(w["total_tokens"], 1_100)

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
        # The working directory holds a decoy log that is not an implicit archive.
        self.assertEqual(report["files"], 2)
        self.assertEqual(report["windows"][0]["total_tokens"], 1210)

    def test_cli_without_directory_or_installed_codex_ignores_working_directory_logs(self):
        working = self.root / "working"
        working.mkdir()
        self.write(prefix(thread="decoy") + [modern(), complete()], "working/decoy.jsonl")
        output = self.root / "report.html"
        with patch("harness_metrics.Path.home", return_value=self.root / "home"), \
             patch("harness_metrics.Path.cwd", return_value=working), \
             patch("sys.argv", ["harness_metrics.py", "--offline", "--no-cache", "--output", str(output)]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual((report["files"], report["threads"], report["sources"]), (0, 0, []))

    def test_cli_default_codex_includes_archived_sessions_and_deduplicates_copies(self):
        working = self.root / "working"
        working.mkdir()
        self.write(prefix(thread="decoy") + [modern(response="decoy"), complete()], "working/decoy.jsonl")
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
        self.assertEqual(report["sources"], [str((self.root / "home/.codex/sessions").resolve()),
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
        self.assertEqual(report["sources"], [str((self.root / "home/.codex/archived_sessions").resolve())])
        self.assertEqual(report["files"], 1)
        self.assertEqual(report["windows"][0]["total_tokens"], 1100)

    def test_codex_invalid_utf8_line_is_skipped_without_losing_usage(self):
        path = self.root / "logs/session.jsonl"
        path.parent.mkdir(parents=True)
        rows = [json.dumps(r).encode() for r in prefix() + [modern(), complete()]]
        path.write_bytes(b"\n".join(rows[:2] + [b"\xff\xfe broken"] + rows[2:]) + b"\n")
        with redirect_stderr(io.StringIO()):
            r = self.report()
        self.assertEqual(r["windows"][0]["total_tokens"], 1100)
        self.assertEqual(r["quality"]["Malformed lines"], 1)
        self.assertNotIn("Unreadable files", r["quality"])

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

    def test_codex_token_count_with_null_info_is_not_malformed(self):
        self.write(prefix() + [event("token_count", info=None, rate_limits={}), modern(), complete()])
        with redirect_stderr(io.StringIO()) as stderr:
            report = self.report()
        self.assertEqual(stderr.getvalue(), "")
        self.assertNotIn("Malformed records", report["quality"])
        self.assertEqual(report["windows"][0]["total_tokens"], 1100)

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

    def test_token_summary_includes_cache_writes_and_reasoning_without_double_counting(self):
        self.write(prefix() + [modern(counts=usage(input=1000, output=100, cached=400, reasoning=20, write=100)), complete()])
        report = self.report()
        groups = [report, report["by_harness"]["codex"],
                  {"windows": report["by_model"]["gpt-6.1-sol"]},
                  report["by_mode"]["Normal"], report["by_tier"]["Medium"]]
        for group in groups:
            for window in group["windows"]:
                with self.subTest(label=window["label"], tokens=window["total_tokens"]):
                    expected = (1000, 100, 400) if window["total_tokens"] else (0, 0, 0)
                    self.assertEqual((window["input_tokens"], window["output_tokens"], window["cached_input_tokens"]), expected)
                    self.assertEqual(window["total_tokens"], window["input_tokens"] + window["output_tokens"])

    def test_long_model_entries_separate_context_and_fast_activity(self):
        for legacy_usage in (False, True):
            with self.subTest(legacy_usage=legacy_usage):
                rows = prefix()
                expected = {}
                cumulative = usage(input=0, output=0, cached=0, reasoning=0)
                for index, (mode, input_tokens) in enumerate([
                    ("default", 272000), ("default", 272001),
                    ("priority", 272000), ("priority", 272001),
                ]):
                    turn = f"t{index + 1}"
                    at = START + timedelta(seconds=index * 20)
                    counts = usage(input=input_tokens, cached=272000)
                    cumulative = {key: cumulative[key] + count for key, count in counts.items()}
                    rows += [event("thread_settings_applied", at,
                                   thread_settings={"service_tier": mode}),
                             event("task_started", at, turn_id=turn)]
                    rows.append(legacy(cumulative, at + timedelta(seconds=1), last=counts) if legacy_usage else
                                modern(turn, f"r{index}", counts, at + timedelta(seconds=1)))
                    rows += [record("response_item", {"type": "function_call", "call_id": f"c{index}"},
                                    at + timedelta(seconds=2)), complete(turn, at + timedelta(seconds=10))]
                    speed = metrics.SpeedMode.FAST if mode == "priority" else metrics.SpeedMode.NORMAL
                    name = "gpt-6.1-sol" + ("-fast" if mode == "priority" else "")
                    name += "-long" if input_tokens > 272000 else ""
                    expected[name] = (input_tokens + 100, speed, metrics.Usage(input_tokens, 100, 272000, 20))
                self.write(rows)
                report = self.report()
                for group in [report, report["by_harness"]["codex"], report["by_tier"]["Medium"]]:
                    self.assertEqual(set(group["by_model"]), set(expected))
                    for name, (tokens, speed, counts) in expected.items():
                        window = group["by_model"][name][0]
                        costs, missing = metrics.price_usage("gpt-6.1-sol", counts, speed)
                        self.assertFalse(missing)
                        self.assertEqual(window["total_tokens"], tokens)
                        self.assertEqual(Decimal(window["cost"]), sum(costs.values()))
                        self.assertEqual(window["tool_calls"], 1)
                        self.assertEqual(window["active_seconds"], 10)
                        self.assertEqual(window["metrics"]["ttft"]["count"], 1)
                        self.assertEqual(window["models"], {name: tokens})
                        self.assertEqual(group["by_mode"][speed.value]["by_model"][name][0], window)
                    for i, total in enumerate(group["windows"]):
                        members = [windows[i] for windows in group["by_model"].values()]
                        for key in ["total_tokens", "tool_calls", "active_seconds", "unpriced_tokens"]:
                            self.assertEqual(sum(w[key] for w in members), total[key])
                        self.assertEqual(sum(Decimal(w["cost"]) for w in members), Decimal(total["cost"]))
                self.assertEqual(report["windows"][0]["conversations"], 1)
                self.assertEqual(report["by_tier"]["Medium"]["windows"][0]["conversations"], 1)
                self.assertIn("gpt-6.1-sol-fast-long", metrics.render_report(report))

    def test_mixed_context_turn_keeps_timing_and_ambiguous_tools_separate(self):
        for speed in [metrics.SpeedMode.NORMAL, metrics.SpeedMode.FAST]:
            with self.subTest(speed=speed):
                rows = prefix()
                rows[0]["payload"]["service_tier"] = "priority" if speed == metrics.SpeedMode.FAST else "default"
                self.write(rows + [modern(counts=usage(input=272000)),
                    modern(response="r2", counts=usage(input=272001), at=START + timedelta(seconds=3)),
                    record("response_item", {"type": "function_call", "call_id": "mixed"}), complete()])
                report = self.report()
                name = "gpt-6.1-sol-fast" if speed == metrics.SpeedMode.FAST else "gpt-6.1-sol"
                group = report["by_mode"][speed.value]
                for model in [name, name + "-long"]:
                    window = group["by_model"][model][0]
                    self.assertEqual(window["active_seconds"], 0)
                    self.assertEqual(window["tool_calls"], 0)
                    self.assertEqual(window["metrics"]["ttft"]["count"], 0)
                timing = group["by_model"]["Mixed contexts (timing)"][0]
                self.assertEqual(timing["active_seconds"], 10)
                self.assertEqual(timing["metrics"]["ttft"]["count"], 1)
                self.assertEqual(timing["metrics"]["throughput"]["avg"], 20)
                self.assertEqual(timing["total_tokens"], 0)
                self.assertEqual(group["by_model"]["Mixed contexts (tools)"][0]["tool_calls"], 1)
                self.assertEqual(report["by_tier"]["Medium"]["windows"][0]["active_seconds"], 10)
                members = [windows[0] for windows in report["by_model"].values()]
                total = report["windows"][0]
                for key in ["total_tokens", "tool_calls", "active_seconds"]:
                    self.assertEqual(sum(w[key] for w in members), total[key])
                self.assertEqual(sum(Decimal(w["cost"]) for w in members), Decimal(total["cost"]))

    def test_tool_context_classification_scales_with_usage_not_tool_calls(self):
        usage_count = 32
        call_count = 64
        rows = prefix() + [modern(response=f"r{i}", counts=usage(input=272000 + i % 2))
                           for i in range(usage_count)]
        rows += [record("response_item", {"type": "function_call", "call_id": f"c{i}"})
                 for i in range(call_count)]
        self.write(rows + [complete()])
        with patch("harness_metrics.is_long_context", wraps=metrics.is_long_context) as classify:
            report = self.report()
        # Allow repeated summary passes, but never a usage scan for every call.
        self.assertLessEqual(classify.call_count, usage_count * 6)
        for group in [report, report["by_harness"]["codex"], report["by_mode"]["Normal"],
                      report["by_tier"]["Medium"]]:
            self.assertEqual(group["windows"][0]["total_tokens"], usage_count * 272100 + usage_count // 2)
            self.assertEqual(group["windows"][0]["tool_calls"], call_count)
            self.assertEqual(group["by_model"]["Mixed contexts (tools)"][0]["tool_calls"], call_count)
            self.assertEqual(group["by_model"]["gpt-6.1-sol"][0]["tool_calls"], 0)
            self.assertEqual(group["by_model"]["gpt-6.1-sol-long"][0]["tool_calls"], 0)

    def test_usage_categories_are_not_rebuilt_for_each_window(self):
        self.write(prefix() + [modern(counts=usage(write=100)), complete()])
        with patch("harness_metrics.Usage.categories", autospec=True,
                   side_effect=metrics.Usage.categories) as categorize:
            report = self.report()
        self.assertLessEqual(categorize.call_count, 3)
        self.assertEqual({entry["name"]: entry["tokens"] for entry in report["windows"][0]["categories"]},
                         {metrics.Category.INPUT.value: 500, metrics.Category.CACHE_READ.value: 400,
                          metrics.Category.OUTPUT.value: 80, metrics.Category.REASONING.value: 20,
                          metrics.Category.CACHE_WRITE.value: 100})

    def test_long_context_labels_follow_model_specific_prices_and_proxies(self):
        for model, input_tokens, long_context in [
            ("claude-sonnet-4-5", 200000, False), ("claude-sonnet-4-5", 200001, True),
            ("claude-sonnet-4-6", 300000, False), ("gpt-5.4-mini", 300000, False),
            ("unpublished-model", 300000, False), ("codex-auto-review", 272001, True),
        ]:
            with self.subTest(model=model, input_tokens=input_tokens):
                self.write(prefix(model=model) + [modern(counts=usage(input=input_tokens)), complete()])
                report = self.report()
                name = model + ("-long" if long_context else "")
                self.assertEqual(set(report["by_model"]), {name})
                self.assertEqual(report["windows"][0]["models"], {name: input_tokens + 100})

    def test_aggregate_usage_does_not_get_long_suffix(self):
        turn = metrics.Turn("t1", start=START, end=START + timedelta(seconds=10), duration=10,
                            model="gpt-6.1-sol", completed=True,
                            modern=[metrics.UsageEvent(START, metrics.Usage(300000, 100),
                                                       "gpt-6.1-sol", "aggregate", aggregate=True)])
        report = metrics.build_breakdown([metrics.Thread("aggregate", turns={turn.id: turn})], NOW)
        self.assertEqual(set(report["by_model"]), {"gpt-6.1-sol"})
        window = report["by_model"]["gpt-6.1-sol"][0]
        self.assertFalse(window["partial_cost"])
        self.assertEqual(window["unpriced"], {})
        self.assertEqual(Decimal(window["cost"]), Decimal(".601"))
        self.assertEqual(window["coverage"]["Aggregate snapshots"], 1)

    def test_aggregate_base_context_rates_preserve_recorded_speed_premium(self):
        for mode, expected in [(metrics.SpeedMode.NORMAL, ".601"), (metrics.SpeedMode.FAST, ".9015")]:
            with self.subTest(mode=mode):
                costs, missing = metrics.price_usage("gpt-6.1-sol", metrics.Usage(300000, 100), mode, aggregate=True)
                self.assertEqual(sum(costs.values()), Decimal(expected))
                self.assertFalse(missing)

    def test_malformed_and_invalid_usage(self):
        self.write(prefix() + ['{bad-json', '[]', modern(counts=usage(cached=2000)),
                              modern(response="valid"), complete()])
        r = self.report()
        self.assertEqual(r["windows"][0]["total_tokens"], 1100)
        self.assertEqual(r["quality"]["Malformed lines"], 1)
        self.assertEqual(r["quality"]["Malformed records"], 1)
        self.assertEqual(r["quality"]["Invalid usage records"], 1)

    def test_trend_periods_partition_history_and_include_cutoff_once(self):
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        windows = metrics.make_trend_windows(cutoff, timezone.utc, start=cutoff - timedelta(days=365))
        lookup = metrics.WindowLookup(windows)
        for granularity in metrics.Granularity:
            periods = [w for w in windows if w.granularity == granularity]
            self.assertEqual(periods[0].start, cutoff - timedelta(days=365))
            self.assertEqual(periods[-1].end, cutoff)
            self.assertFalse(periods[-1].end_exclusive)
            for previous, following in zip(periods, periods[1:]):
                self.assertEqual(previous.end, following.start)
                self.assertTrue(previous.end_exclusive)
                self.assertEqual(sum(windows[i].granularity == granularity
                                     for i in lookup.matching(following.start)), 1)
            self.assertEqual(sum(w.contains(cutoff) for w in periods), 1)
            self.assertEqual(sum(w.contains(cutoff - timedelta(days=365)) for w in periods), 1)
        monthly = [w for w in windows if w.granularity == metrics.Granularity.MONTHLY]
        self.assertEqual(monthly[-2].label, "2025-12-01")
        self.assertEqual(monthly[-1].label, "2026-01-01")

    def test_trend_calendar_days_handle_dst_and_leap_month(self):
        leap = metrics.make_trend_windows(datetime(2024, 3, 2, tzinfo=timezone.utc), timezone.utc,
                                          start=datetime(2024, 2, 1, tzinfo=timezone.utc))
        february = next(w for w in leap if w.granularity == metrics.Granularity.MONTHLY
                        and w.label == "2024-02-01")
        self.assertEqual(february.end - february.start, timedelta(days=29))
        try:
            toronto = metrics.ZoneInfo("America/Toronto")
        except metrics.ZoneInfoNotFoundError:
            self.skipTest("Toronto timezone data unavailable")
        periods = metrics.make_trend_windows(datetime(2026, 11, 3, tzinfo=timezone.utc), toronto,
                                            start=datetime(2026, 3, 1, tzinfo=timezone.utc))
        daily = {w.label: w for w in periods if w.granularity == metrics.Granularity.DAILY}
        self.assertEqual(daily["2026-03-08"].end - daily["2026-03-08"].start, timedelta(hours=23))
        self.assertEqual(daily["2026-11-01"].end - daily["2026-11-01"].start, timedelta(hours=25))
        for w in periods:
            if w.granularity == metrics.Granularity.WEEKLY:
                self.assertEqual(datetime.fromisoformat(w.label).weekday(), 0)

    def test_trends_use_period_samples_and_conversations_with_filters(self):
        cutoff = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
        starts = [datetime(2026, 10, day, 10, tzinfo=timezone.utc) for day in (5, 6)]
        turns = {}
        calls = {}
        for index, (start, duration, ttft) in enumerate(zip(starts, (10, 30), (1, 3))):
            turn = metrics.Turn(str(index), start=start, end=start + timedelta(seconds=duration),
                                duration=duration, ttft=ttft, model="gpt-6.1-sol", completed=True,
                                modern=[metrics.UsageEvent(start, metrics.Usage(100, 100),
                                                           "gpt-6.1-sol", str(index))])
            turns[turn.id] = turn
            calls[str(index)] = metrics.ToolCall(start, turn.id, turn.model)
        other = metrics.Turn("other", start=starts[1], end=starts[1] + timedelta(seconds=20),
                             duration=20, ttft=5, model="gpt-6-astra", completed=True,
                             mode=metrics.SpeedMode.FAST,
                             modern=[metrics.UsageEvent(starts[1], metrics.Usage(100, 100),
                                                        "gpt-6-astra", "other", mode=metrics.SpeedMode.FAST)])
        report = metrics.build_breakdown([metrics.Thread("one", turns=turns, calls=calls),
                                          metrics.Thread("two", turns={other.id: other})],
                                         cutoff, report_zone=timezone.utc)
        def point(scope, granularity, label):
            index = next(i for i, p in enumerate(report["trend_periods"][granularity])
                         if p["label"] == label)
            return scope["trends"][granularity][index]

        weekly = point(report, "weekly", "2026-10-05")
        self.assertEqual(weekly["ttft"]["count"], 3)
        self.assertEqual(weekly["ttft"]["avg"], 3)
        self.assertEqual(weekly["length"]["avg"], 30)
        self.assertEqual(weekly["tools"]["avg"], 1)
        self.assertAlmostEqual(weekly["throughput"]["avg"], (10 + 100 / 30 + 5) / 3)
        self.assertEqual(point(report, "daily", "2026-10-06")["ttft"]["avg"], 4)
        self.assertEqual(point(report, "hourly", "2026-10-06T10:00+00:00")["ttft"]["avg"], 4)
        self.assertEqual(point(report, "monthly", "2026-10-01")["length"]["avg"], 30)
        self.assertIsNone(point(report, "daily", "2026-10-07"))
        normal = report["by_mode"]["Normal"]
        self.assertEqual(point(normal, "weekly", "2026-10-05")["length"]["avg"], 40)
        high_fast = report["by_tier"]["High"]["by_mode"]["Fast"]
        self.assertEqual(point(high_fast, "weekly", "2026-10-05")["ttft"]["avg"], 5)
        self.assertEqual(point(high_fast, "hourly", "2026-10-06T10:00+00:00")["ttft"]["avg"], 5)
        index = next(i for i, p in enumerate(report["trend_periods"]["weekly"])
                     if p["label"] == "2026-10-05")
        self.assertEqual(normal["by_model_trends"]["gpt-6.1-sol"]["weekly"][index]["tools"]["avg"], 2)

    def test_hourly_trends_assign_crossing_turn_and_calls_to_exact_hour(self):
        boundary = datetime(2026, 10, 3, 10, tzinfo=timezone.utc)
        start = boundary - timedelta(minutes=2)
        turn = metrics.Turn("crossing", start=start, end=boundary, completed=True,
                            duration=120, ttft=2, model="gpt-6.1-sol",
                            modern=[metrics.UsageEvent(start, metrics.Usage(100, 240),
                                                       "gpt-6.1-sol", "response")])
        thread = metrics.Thread("one", turns={turn.id: turn},
                                calls={"call": metrics.ToolCall(boundary, turn.id, turn.model)})
        report = metrics.build_breakdown([thread], boundary + timedelta(hours=2), report_zone=timezone.utc)
        periods = report["trend_periods"]["hourly"]
        points = dict(zip((p["label"] for p in periods), report["trends"]["hourly"]))
        self.assertEqual(points["2026-10-03T09:00+00:00"]["ttft"]["count"], 0)
        self.assertEqual(points["2026-10-03T09:00+00:00"]["total_tokens"], 340)
        completed = points["2026-10-03T10:00+00:00"]
        self.assertEqual(completed["total_tokens"], 0)
        self.assertEqual(completed["ttft"]["avg"], 2)
        self.assertEqual(completed["throughput"]["avg"], 2)
        self.assertEqual(completed["length"]["avg"], 120)
        self.assertEqual(completed["tools"]["avg"], 1)
        self.assertIsNone(points["2026-10-03T11:00+00:00"])
        self.assertEqual(periods[-1]["start"], periods[-1]["end"])
        self.assertFalse(periods[-1]["end_exclusive"])
        self.assertIn('<option value="hourly">Hourly</option>', metrics.render_report(
            {**report, "generated": boundary.isoformat()}))

    def test_token_trends_total_usage_once_per_period_and_respect_filters(self):
        cutoff = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
        events = [metrics.UsageEvent(at, metrics.Usage(100, 40, cached=20, reasoning=10, write=15),
                                    model, str(index), mode=mode)
                  for index, (at, model, mode) in enumerate([
                      (cutoff - timedelta(days=3), "gpt-6.1-sol", metrics.SpeedMode.NORMAL),
                      (cutoff - timedelta(days=1), "gpt-6-astra", metrics.SpeedMode.FAST),
                      (cutoff, "gpt-6.1-sol", metrics.SpeedMode.NORMAL),
                      (cutoff + timedelta(seconds=1), "gpt-6.1-sol", metrics.SpeedMode.NORMAL)])]
        turn = metrics.Turn("unfinished", start=events[0].at, model="gpt-6.1-sol", modern=events)
        report = metrics.build_breakdown([metrics.Thread("one", turns={turn.id: turn})],
                                         cutoff, report_zone=timezone.utc)
        for interval in metrics.Granularity:
            key = interval.value
            with self.subTest(interval=key):
                self.assertEqual(sum(p["total_tokens"] for p in report["trends"][key] if p), 420)
                self.assertEqual(sum(p["total_tokens"] for p in report["by_mode"]["Normal"]["trends"][key] if p), 280)
                self.assertEqual(sum(p["total_tokens"] for p in report["by_tier"]["High"]["by_mode"]["Fast"]["trends"][key] if p), 140)
                self.assertEqual(sum(p["total_tokens"] for p in report["by_model_trends"]["gpt-6.1-sol"][key] if p), 280)
        daily = dict(zip((p["label"] for p in report["trend_periods"]["daily"]), report["trends"]["daily"]))
        self.assertIsNone(daily["2026-10-01"])
        self.assertEqual(daily["2026-10-03"]["total_tokens"], 140)
        self.assertEqual(daily["2026-10-03"]["ttft"]["count"], 0)

    def test_trend_costs_match_window_costs_and_flag_unpriced_periods(self):
        cutoff = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
        events = [metrics.UsageEvent(at, metrics.Usage(100, 40, cached=20, reasoning=10, write=15),
                                    model, str(index), mode=mode)
                  for index, (at, model, mode) in enumerate([
                      (cutoff - timedelta(days=3), "gpt-6.1-sol", metrics.SpeedMode.NORMAL),
                      (cutoff - timedelta(days=1), "gpt-6-astra", metrics.SpeedMode.FAST),
                      (cutoff - timedelta(hours=2), "unknown-model", metrics.SpeedMode.NORMAL),
                      (cutoff, "gpt-6.1-sol", metrics.SpeedMode.NORMAL)])]
        turn = metrics.Turn("unfinished", start=events[0].at, model="gpt-6.1-sol", modern=events)
        report = metrics.build_breakdown([metrics.Thread("one", turns={turn.id: turn})],
                                         cutoff, report_zone=timezone.utc)
        scopes = {"all": (report["windows"], report["trends"]),
                  "fast": (report["by_mode"]["Fast"]["windows"], report["by_mode"]["Fast"]["trends"]),
                  "model": (report["by_model"]["gpt-6.1-sol"], report["by_model_trends"]["gpt-6.1-sol"])}
        for name, (windows, trends) in scopes.items():
            for interval in metrics.Granularity:
                with self.subTest(scope=name, interval=interval.value):
                    points = [p for p in trends[interval.value] if p]
                    self.assertEqual(sum((Decimal(p["cost"]) for p in points), Decimal(0)),
                                     Decimal(windows[-1]["cost"]))
        self.assertGreater(Decimal(report["by_mode"]["Fast"]["windows"][-1]["cost"]), 0)
        daily = dict(zip((p["label"] for p in report["trend_periods"]["daily"]), report["trends"]["daily"]))
        self.assertTrue(daily["2026-10-03"]["partial_cost"])
        self.assertFalse(daily["2026-09-30"]["partial_cost"])
        self.assertGreater(Decimal(daily["2026-09-30"]["cost"]), 0)

    def test_hourly_trends_preserve_dst_skipped_and_repeated_hours(self):
        try:
            toronto = metrics.ZoneInfo("America/Toronto")
        except metrics.ZoneInfoNotFoundError:
            self.skipTest("Toronto timezone data unavailable")
        for month, day, expected in ((3, 8, 23), (11, 1, 25)):
            start = datetime(2026, month, day, tzinfo=toronto)
            end = start + timedelta(days=1)
            periods = [w for w in metrics.make_trend_windows(end.astimezone(timezone.utc), toronto, start)
                       if w.granularity == metrics.Granularity.HOURLY and w.start < w.end]
            self.assertEqual(len(periods), expected)
            self.assertTrue(all(w.end - w.start == timedelta(hours=1) for w in periods))
            self.assertEqual(len({w.label for w in periods}), expected)
            if month == 3:
                self.assertFalse(any("T02:00" in w.label for w in periods))
            else:
                self.assertEqual([w.label for w in periods if "T01:00" in w.label],
                                 ["2026-11-01T01:00-04:00", "2026-11-01T01:00-05:00"])
        turns = {}
        for hour, ttft in ((5, 1), (6, 3)):
            at = datetime(2026, 11, 1, hour, 10, tzinfo=timezone.utc)
            turns[str(hour)] = metrics.Turn(str(hour), start=at, end=at + timedelta(seconds=10),
                                            completed=True, duration=10, ttft=ttft,
                                            modern=[metrics.UsageEvent(at, metrics.Usage(hour * 100, 40),
                                                                       "gpt-6.1-sol", str(hour))])
        report = metrics.build_breakdown([metrics.Thread("one", turns=turns)],
                                         datetime(2026, 11, 1, 8, tzinfo=timezone.utc), report_zone=toronto)
        repeated = [(p["label"], point["ttft"]["avg"]) for p, point in
                    zip(report["trend_periods"]["hourly"], report["trends"]["hourly"])
                    if "T01:00" in p["label"]]
        self.assertEqual(repeated, [("2026-11-01T01:00-04:00", 1), ("2026-11-01T01:00-05:00", 3)])
        token_totals = [point["total_tokens"] for period, point in
                        zip(report["trend_periods"]["hourly"], report["trends"]["hourly"])
                        if "T01:00" in period["label"]]
        self.assertEqual(token_totals, [540, 640])

    def test_hourly_empty_buckets_share_storage_without_leaking_between_filters(self):
        cutoff = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
        periods = metrics.make_windows(cutoff, timezone.utc) + metrics.make_trend_windows(cutoff, timezone.utc)
        first, second, overall = (metrics.empty_breakdown(periods) for _ in range(3))
        index = next(i for i, w in enumerate(periods) if w.granularity == metrics.Granularity.HOURLY)
        self.assertIs(first.windows[index], second.windows[index])
        self.assertIs(first.windows[index], first.by_mode[metrics.SpeedMode.FAST].windows[index])
        at = periods[index].start + timedelta(minutes=10)
        thread = metrics.Thread("one", turns={"t": metrics.Turn(
            "t", start=at, end=at + timedelta(seconds=10), completed=True,
            duration=10, ttft=1, model="gpt-6.1-sol")},
            billing=[metrics.BillingEvent(at, "USD", Decimal(1))])
        metrics.add_thread(first.windows, thread, first.by_model, first.by_mode, first.by_tier)
        self.assertIsNot(first.windows[index], second.windows[index])
        self.assertEqual(first.windows[index].ttft, [1])
        self.assertEqual(second.windows[index].ttft, [])
        self.assertEqual(second.windows[index].billing, {})
        self.assertEqual(first.by_mode[metrics.SpeedMode.FAST].windows[index].ttft, [])
        metrics.merge_breakdown(overall, first)
        metrics.merge_breakdown(overall, second)
        self.assertEqual(overall.windows[index].ttft, [1])
        self.assertEqual(overall.by_model["gpt-6.1-sol"][index].ttft, [1])
        self.assertEqual(overall.windows[index].billing["USD"], Decimal(1))
        self.assertEqual(second.windows[index].ttft, [])

    def test_trend_history_starts_at_first_datapoint_beyond_one_year(self):
        oldest = datetime(2023, 4, 12, 14, tzinfo=timezone.utc)
        self.write(prefix(thread="old", at=oldest) + [modern(at=oldest + timedelta(seconds=1)),
                                                     complete(at=oldest + timedelta(seconds=10))], "old.jsonl")
        self.write(prefix(thread="new") + [modern(), complete()], "new.jsonl")
        report = self.report()
        for granularity, periods in report["trend_periods"].items():
            self.assertEqual(periods[0]["start"][:10], "2023-04-12")
            self.assertEqual(periods[-1]["end"], report["generated"])
            first_point = next(point for point in report["trends"][granularity]
                               if point and point["ttft"]["count"])
            self.assertEqual(first_point["ttft"]["count"], 1)
        self.assertEqual(report["windows"][0]["total_tokens"], 1100)
        self.assertEqual(report["windows"][-1]["total_tokens"], 1100)
        html = metrics.render_report(report)
        self.assertIn('id="trend-start"', html)
        self.assertIn('id="trend-end"', html)
        self.assertNotIn("Past 365 days", html)

    def test_trend_start_uses_local_date_and_ignores_future_and_billing_only_activity(self):
        at = datetime(2026, 9, 20, 2, tzinfo=timezone.utc)
        turn = metrics.Turn("early", start=at, end=at + timedelta(seconds=10), completed=True, duration=10)
        future = metrics.Turn("future", start=NOW + timedelta(days=1))
        thread = metrics.Thread("one", turns={turn.id: turn, future.id: future},
                                billing=[metrics.BillingEvent(at - timedelta(days=1000), "USD", Decimal(1))])
        report = metrics.build_breakdown([thread], NOW)
        expected_date = at.astimezone(metrics.TIMEZONE).date().isoformat()
        self.assertEqual(report["trend_periods"]["daily"][0]["start"][:10], expected_date)
        empty = metrics.build_breakdown([], NOW)
        self.assertEqual(len(empty["trend_periods"]["daily"]), 1)
        self.assertIsNone(empty["trends"]["daily"][0])

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

    def test_by_model_separates_normal_and_fast_usage_with_original_rates(self):
        second_start = START + timedelta(seconds=20)
        for legacy_usage in (False, True):
            with self.subTest(legacy_usage=legacy_usage):
                first = legacy(usage()) if legacy_usage else modern()
                second = (legacy(usage(2000, 200, 800, 40), second_start + timedelta(seconds=1), last=usage())
                          if legacy_usage else modern("t2", "r2", at=second_start + timedelta(seconds=1)))
                self.write(prefix() + [first,
                    record("response_item", {"type": "function_call", "call_id": "normal"}), complete(),
                    event("thread_settings_applied", second_start,
                          thread_settings={"service_tier": "priority"}),
                    event("task_started", second_start, turn_id="t2"), second,
                    record("response_item", {"type": "function_call", "call_id": "fast"}, second_start),
                    complete("t2", second_start + timedelta(seconds=10))])
                r = self.report()
                self.assertEqual(set(r["by_model"]), {"gpt-6.1-sol", "gpt-6.1-sol-fast"})
                for group in (r, r["by_harness"]["codex"], r["by_tier"]["Medium"]):
                    normal = group["by_model"]["gpt-6.1-sol"][0]
                    fast = group["by_model"]["gpt-6.1-sol-fast"][0]
                    for window in (normal, fast):
                        self.assertEqual(window["conversations"], 1)
                        self.assertEqual(window["total_tokens"], 1100)
                        self.assertEqual(window["tool_calls"], 1)
                        self.assertEqual(window["active_seconds"], 10)
                        self.assertEqual(window["metrics"]["ttft"]["count"], 1)
                        self.assertEqual(window["metrics"]["throughput"]["avg"], 10)
                        self.assertFalse(window["partial_cost"])
                    self.assertEqual(Decimal(fast["cost"]), Decimal(normal["cost"]) * Decimal("1.5"))
                    self.assertEqual(group["by_mode"]["Normal"]["by_model"]["gpt-6.1-sol-fast"][0]["total_tokens"], 0)
                    self.assertEqual(group["by_mode"]["Fast"]["by_model"]["gpt-6.1-sol"][0]["total_tokens"], 0)
                    for i, total in enumerate(group["windows"]):
                        members = [ws[i] for ws in group["by_model"].values()]
                        for key in ("total_tokens", "tool_calls", "active_seconds"):
                            self.assertEqual(sum(w[key] for w in members), total[key])
                        self.assertEqual(sum(Decimal(w["cost"]) for w in members), Decimal(total["cost"]))
                self.assertEqual(r["windows"][0]["models"], {"gpt-6.1-sol": 1100, "gpt-6.1-sol-fast": 1100})

    def test_fast_model_entry_preserves_catalog_identity_and_unpriced_usage(self):
        rows = prefix(model="vendor/new-model")
        rows[0]["payload"]["service_tier"] = "priority"
        self.write(rows + [modern(), complete()])
        catalog = metrics.openrouter_prices({"data": [{"id": "vendor/new-model",
            "pricing": {"prompt": ".000002", "completion": ".00001"}}]})
        report = metrics.collect_report(self.root, NOW, catalog=catalog)
        self.assertEqual(set(report["by_model"]), {"vendor/new-model-fast"})
        self.assertEqual(report["openrouter_rates"]["vendor/new-model-fast"]["id"], "vendor/new-model")
        window = report["by_model"]["vendor/new-model-fast"][0]
        self.assertEqual(window["total_tokens"], 1100)
        self.assertEqual(window["unpriced"], {"vendor/new-model-fast": 1100})
        self.assertTrue(window["partial_cost"])

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
        self.assertEqual(r["by_mode"]["Fast"]["by_model"]["gpt-5.5-fast"][0]["total_tokens"], 1100)

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
        self.assertEqual(r["by_mode"]["Normal"]["by_model"]["gpt-5.5-fast"][0]["total_tokens"], 0)
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
        for model in ("gpt-6.1-sol", "gpt-6.1-sol-fast"):
            self.assertEqual(r["by_model"][model][0]["total_tokens"], 1100)
            self.assertEqual(r["by_model"][model][0]["active_seconds"], 0)
            self.assertIsNone(r["by_model"][model][0]["metrics"]["ttft"]["avg"])
        self.assertEqual(r["by_model"]["Mixed modes (timing)"][0]["active_seconds"], 10)
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
        w = report["by_mode"]["Fast"]["by_model"]["codex-auto-review-fast"][0]
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
                    "gpt-5.4-nano": "Budget", "gpt-5-mini": "Budget", "gpt-5-nano": "Budget",
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
        self.assertEqual(tier["by_mode"]["Fast"]["by_model"]["gpt-5.6-terra-fast"][0]["total_tokens"], 2200)
        self.assertEqual(tier["by_mode"]["Normal"]["by_model"]["gpt-5.6-terra-fast"][0]["total_tokens"], 0)
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
                         r["by_model"]["gpt-6-astra-fast"][0]["cost"])

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


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "logs"
        self.root.mkdir()
        self.cache = Path(self.temp.name) / "cache" / "metrics.sqlite3"

    def write(self, name, rows):
        path = self.root / name
        path.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows) + "\n")
        return path

    def report(self, **kwargs):
        return metrics.collect_report(self.root, NOW, cache_path=self.cache, **kwargs)

    def test_failed_report_keeps_checkpointed_groups_but_server_style_failure_rolls_back(self):
        for name in ("one", "two", "three"):
            self.write(f"{name}.jsonl", prefix(thread=name) + [modern(), complete()])
        def cached_groups():
            with closing(sqlite3.connect(self.cache)) as connection:
                return connection.execute("SELECT count(*) FROM cache_groups").fetchone()[0]
        with patch("harness_metrics.CHECKPOINT_GROUPS", 2), \
             patch("harness_metrics.add_thread", side_effect=ValueError("report failed")):
            with self.assertRaises(ValueError):
                self.report()
        self.assertEqual(cached_groups(), 2)
        self.cache.unlink()
        with self.assertRaises(ValueError):
            with metrics.prepare_sources({metrics.Harness.CODEX: [self.root]}, metrics.Quality(), False,
                                         self.cache, NOW):
                raise ValueError("caller failed")
        self.assertEqual(cached_groups(), 0)

    def test_cache_skips_parsing_and_identity_reads_and_appends_new_conversations(self):
        self.write("one.jsonl", prefix(thread="one") + [modern(), complete()])
        expected = metrics.collect_report(self.root, NOW)
        self.assertEqual(self.report(), expected)
        with patch("harness_metrics.read_thread", wraps=metrics.read_thread) as read, \
             patch("harness_metrics.session_identity", wraps=metrics.session_identity) as identity:
            self.assertEqual(self.report(), expected)
            self.assertEqual(read.call_count, 0)
            self.assertEqual(identity.call_count, 0)
            self.write("two.jsonl", prefix(thread="two") + [modern(), complete()])
            updated = self.report()
            self.assertEqual(read.call_count, 1)
            self.assertEqual(identity.call_count, 1)
            self.assertEqual(updated["threads"], 2)
            self.assertEqual(updated["windows"][0]["total_tokens"], 2200)
        self.assertEqual(updated, metrics.collect_report(self.root, NOW))

    def test_cache_refreshes_appended_turns_and_rewritten_files_without_duplicates(self):
        path = self.write("one.jsonl", prefix() + [modern(), complete()])
        self.report()
        more = prefix(turn="t2")[1:] + [modern("t2", "r2"), complete("t2")]
        with path.open("a") as stream:
            stream.write("\n".join(json.dumps(r) for r in more) + "\n")
        self.assertEqual(self.report()["windows"][0]["total_tokens"], 2200)
        self.assertEqual(self.report(), metrics.collect_report(self.root, NOW))
        self.write("one.jsonl", prefix() + [modern(counts=usage(input=2000)), complete()])
        self.assertEqual(self.report()["windows"][0]["total_tokens"], 2100)
        self.assertEqual(self.report(), metrics.collect_report(self.root, NOW))

    def test_cache_merges_copies_and_handles_removed_files_and_source_scope(self):
        rows = prefix() + [modern(), complete()]
        original = self.write("one.jsonl", rows)
        self.report()
        self.write("copy.jsonl", rows)
        self.assertEqual(self.report()["windows"][0]["total_tokens"], 1100)
        original.unlink()
        self.assertEqual(self.report(), metrics.collect_report(self.root, NOW))
        other = self.root / "other"
        other.mkdir()
        self.assertEqual(metrics.collect_report(other, NOW, cache_path=self.cache)["threads"], 0)
        (self.root / "copy.jsonl").unlink()
        self.assertEqual(self.report()["threads"], 0)

    def test_cache_preserves_diagnostics_and_recomputes_cutoff_timezone_and_prices(self):
        self.write("one.jsonl", prefix() + ['{bad', modern(), complete()])
        self.report()
        self.assertEqual(self.report(), metrics.collect_report(self.root, NOW))
        future = NOW + timedelta(days=7)
        self.assertEqual(metrics.collect_report(self.root, future, cache_path=self.cache, report_zone=timezone.utc),
                         metrics.collect_report(self.root, future, report_zone=timezone.utc))
        rates = metrics.Rates(Decimal("20"), Decimal("10"), Decimal("100"))
        with patch.dict(metrics.PRICES, {"gpt-6.1-sol": metrics.Price(rates)}):
            self.assertEqual(self.report(), metrics.collect_report(self.root, NOW))

    def test_cached_trend_start_includes_old_selected_sources_only(self):
        old = datetime(2022, 5, 8, 12, tzinfo=timezone.utc)
        self.write("old.jsonl", prefix(thread="old", at=old) + [modern(at=old), complete(at=old)])
        self.write("new.jsonl", prefix(thread="new") + [modern(), complete()])
        expected = metrics.collect_report(self.root, NOW)
        self.assertEqual(self.report(), expected)
        with patch("harness_metrics.read_thread", side_effect=AssertionError("old logs parsed")):
            self.assertEqual(self.report(), expected)
        (self.root / "old.jsonl").unlink()
        self.assertEqual(self.report()["trend_periods"]["daily"][0]["start"][:10],
                         START.astimezone(metrics.TIMEZONE).date().isoformat())

    def test_cache_does_not_store_files_that_change_while_being_parsed(self):
        path = self.write("one.jsonl", prefix() + [modern(), complete()])
        original = metrics.read_thread
        def growing(thread_id, paths, quality):
            thread = original(thread_id, paths, quality)
            with path.open("a") as stream:
                stream.write(json.dumps(modern(response="r2")) + "\n")
            return thread
        with patch("harness_metrics.read_thread", growing):
            self.report()
        with patch("harness_metrics.read_thread", wraps=original) as read:
            report = self.report()
            self.assertEqual(read.call_count, 1)
            self.assertEqual(report["windows"][0]["total_tokens"], 2200)
        self.assertEqual(self.report(), metrics.collect_report(self.root, NOW))

    def test_cli_cache_options_enable_reuse_and_allow_bypass(self):
        self.write("one.jsonl", prefix() + [modern(), complete()])
        output = Path(self.temp.name) / "report.html"
        base = ["harness_metrics.py", str(self.root), "--offline", "--output", str(output)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with patch("sys.argv", base + ["--cache", str(self.cache)]):
                self.assertEqual(metrics.main(), 0)
            with patch("sys.argv", base + ["--cache", str(self.cache)]), \
                 patch("harness_metrics.read_thread", wraps=metrics.read_thread) as read:
                self.assertEqual(metrics.main(), 0)
                self.assertEqual(read.call_count, 0)
            with patch("sys.argv", base + ["--no-cache"]), \
                 patch("harness_metrics.sqlite3.connect", wraps=sqlite3.connect) as connections:
                self.assertEqual(metrics.main(), 0)
                self.assertTrue(all(Path(call.args[0]) != self.cache for call in connections.call_args_list))

    def test_cache_rejects_unrelated_existing_database_without_changes(self):
        self.cache.parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.execute("INSERT INTO unrelated VALUES ('preserve')")
            connection.execute("PRAGMA user_version=1")
        original = self.cache.read_bytes()
        with self.assertRaisesRegex(ValueError, "different database"):
            self.report()
        self.assertEqual(self.cache.read_bytes(), original)

    def test_outdated_cache_version_is_rebuilt_from_sources(self):
        self.write("one.jsonl", prefix(thread="one") + [modern(), complete()])
        expected = self.report()
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute("UPDATE cache_usage SET input='999999'")
            connection.execute(f"PRAGMA user_version={metrics.CACHE_VERSION - 1}")
        with closing(sqlite3.connect(self.cache)) as connection:
            with self.assertRaisesRegex(ValueError, "older parser"):
                metrics.MetricsReader(connection)
        with patch("harness_metrics.read_thread", wraps=metrics.read_thread) as read:
            self.assertEqual(self.report(), expected)
            self.assertEqual(read.call_count, 1)
        with closing(sqlite3.connect(self.cache)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], metrics.CACHE_VERSION)

    def test_outdated_version_with_foreign_tables_is_not_dropped(self):
        self.write("one.jsonl", prefix(thread="one") + [modern(), complete()])
        self.report()
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.execute(f"PRAGMA user_version={metrics.CACHE_VERSION - 1}")
        original = self.cache.read_bytes()
        with self.assertRaisesRegex(ValueError, "older parser"):
            self.report()
        self.assertEqual(self.cache.read_bytes(), original)

    def test_bucket_lookup_matches_exact_boundaries_and_crossing_turns(self):
        periods = metrics.make_windows(NOW, timezone.utc) + metrics.make_trend_windows(
            NOW, timezone.utc, start=NOW - timedelta(days=1000))
        lookup = metrics.WindowLookup(periods)
        timestamps = {NOW, NOW + timedelta(seconds=1), NOW - timedelta(days=366)}
        sampled = periods[::max(1, len(periods) // 40)] + periods[:12] + periods[-12:]
        for period in sampled:
            timestamps.update((period.start, period.end, period.start - timedelta(microseconds=1)))
        for at in timestamps:
            self.assertEqual(sorted(lookup.matching(at)),
                             [i for i, window in enumerate(periods) if window.contains(at)])

    def test_aggregation_boundary_checks_do_not_scale_with_history_bucket_count(self):
        at = NOW - timedelta(seconds=20)
        turn = metrics.Turn("t", start=at, end=NOW, model="gpt-6.1-sol", completed=True,
                            duration=20, ttft=.5,
                            modern=[metrics.UsageEvent(at, metrics.Usage(1000, 100), "gpt-6.1-sol", "r")])
        thread = metrics.Thread("one", turns={turn.id: turn},
                                calls={"call": metrics.ToolCall(at, turn.id, turn.model)})
        old_at = NOW - timedelta(days=1000)
        old = metrics.Thread("old", turns={"old": metrics.Turn(
            "old", start=old_at, end=old_at + timedelta(seconds=20), model=turn.model,
            completed=True, duration=20, ttft=.5)})
        contains = metrics.Window.contains
        checks = []
        def counted(window, timestamp):
            checks.append(timestamp)
            return contains(window, timestamp)
        with patch.object(metrics.Window, "contains", counted):
            result = metrics.build_breakdown([old, thread], NOW)
        self.assertLess(len(checks), 400)
        self.assertGreater(len(result["trend_periods"]["daily"]), 1000)
        self.assertEqual(result["windows"][0]["total_tokens"], 1100)


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
        home = patch("harness_metrics.Path.home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)

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

    def t3_db(self, sessions=(("codex", "same-id"),), name="t3/state.sqlite", v2=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            if v2:
                db.execute("CREATE TABLE orchestration_v2_projection_provider_threads (provider TEXT, payload_json TEXT)")
                for provider, identifier in sessions:
                    db.execute("INSERT INTO orchestration_v2_projection_provider_threads VALUES (?, ?)",
                               (provider, json.dumps({"nativeThreadRef": {"nativeId": identifier}})))
            else:
                db.execute("CREATE TABLE provider_session_runtime (provider_name TEXT, resume_cursor_json TEXT)")
                for provider, identifier in sessions:
                    field = "threadId" if provider == "codex" else "sessionId"
                    db.execute("INSERT INTO provider_session_runtime VALUES (?, ?)",
                               (provider, json.dumps({field: identifier})))
        return path

    def test_t3_attributes_native_sessions_once_and_preserves_metrics(self):
        self.write(prefix(thread="same-id") + [modern(), complete()], "codex/t3.jsonl")
        self.write(prefix(thread="standalone") + [modern(), complete()], "codex/standalone.jsonl")
        self.write([self.claude()], "claude/session.jsonl")
        sources = {metrics.Harness.CLAUDE: [self.root / "claude"],
                   metrics.Harness.OPENCODE: [self.opencode_db()]}
        expected = self.report(sources)
        db = self.t3_db((("codex", "same-id"), ("claude", "same-id"), ("opencode", "same-id")))
        before = db.read_bytes()
        sources[metrics.Harness.T3] = [db, db.parent]
        report = self.report(sources)
        self.assertEqual(db.read_bytes(), before)
        self.assertEqual(report["windows"], expected["windows"])
        self.assertEqual(report["by_model"], expected["by_model"])
        self.assertEqual(report["by_harness"]["t3"]["windows"][0]["total_tokens"], 2700)
        self.assertEqual(report["by_harness"]["codex"]["windows"][0]["total_tokens"], 1100)
        self.assertEqual(report["by_harness"]["claude"]["windows"][0]["total_tokens"], 0)
        self.assertEqual(report["by_harness"]["opencode"]["windows"][0]["total_tokens"], 0)

    def test_t3_only_discovers_linked_native_logs_and_ignores_other_sessions(self):
        self.write(prefix(thread="same-id") + [modern(), complete()], "home/.codex/sessions/t3.jsonl")
        self.write(prefix(thread="other") + [modern(), complete()], "home/.codex/sessions/other.jsonl")
        db = self.t3_db()
        report = metrics.collect_report(self.root, NOW, include_codex=False,
                                        harness_roots={metrics.Harness.T3: [db]})
        self.assertEqual(set(report["by_harness"]), {"t3"})
        window = report["windows"][0]
        self.assertEqual(window["total_tokens"], 1100)
        self.assertEqual(window["conversations"], 1)
        self.assertEqual(window["metrics"]["ttft"]["avg"], .5)
        self.assertEqual(Decimal(window["cost"]), Decimal(".00224"))

    def test_t3_v2_native_references_do_not_use_internal_thread_ids(self):
        self.write(prefix(thread="native") + [modern(), complete()], "codex/native.jsonl")
        db = self.t3_db((("codex", "native"),), name="t3/statev2.sqlite", v2=True)
        with closing(sqlite3.connect(db)) as connection, connection:
            connection.execute("INSERT INTO orchestration_v2_projection_provider_threads VALUES (?, ?)",
                               ("codex", json.dumps({"nativeThreadRef": {"nativeId": None}})))
        report = self.report({metrics.Harness.T3: [db.parent]})
        self.assertNotIn("Malformed records", report["quality"])
        self.assertEqual(report["by_harness"]["t3"]["windows"][0]["total_tokens"], 1100)
        self.assertEqual(report["by_harness"]["codex"]["windows"][0]["total_tokens"], 0)

    def test_t3_claude_subagents_follow_parent_session(self):
        self.write([self.claude()], "claude/session/subagents/agent-one.jsonl")
        db = self.t3_db((("claude", "same-id"),))
        report = self.report({metrics.Harness.T3: [db], metrics.Harness.CLAUDE: [self.root / "claude"]})
        self.assertEqual(report["by_harness"]["t3"]["windows"][0]["total_tokens"], 800)
        self.assertEqual(report["by_harness"]["claude"]["windows"][0]["total_tokens"], 0)

    def test_t3_malformed_bindings_retain_valid_native_usage(self):
        self.write(prefix(thread="same-id") + [modern(), complete()], "codex/session.jsonl")
        db = self.t3_db()
        with closing(sqlite3.connect(db)) as connection, connection:
            for raw in ("{", "[]", '{"threadId": 123}'):
                connection.execute("INSERT INTO provider_session_runtime VALUES (?, ?)", ("codex", raw))
        with redirect_stderr(io.StringIO()):
            report = self.report({metrics.Harness.T3: [db]})
        self.assertEqual(report["quality"]["Malformed records"], 3)
        self.assertEqual(report["by_harness"]["t3"]["windows"][0]["total_tokens"], 1100)

    def test_t3_cache_refreshes_bindings_and_native_usage_without_duplicates(self):
        self.write(prefix(thread="same-id") + [modern(), complete()], "codex/session.jsonl")
        db = self.t3_db()
        cache = self.root / "metrics.sqlite3"
        sources = {metrics.Harness.T3: [db]}
        def report():
            return metrics.collect_report(self.root / "codex", NOW, harness_roots=sources, cache_path=cache)
        expected = report()
        with patch("harness_metrics.read_thread", side_effect=AssertionError("unchanged log parsed")):
            self.assertEqual(report(), expected)
        with closing(sqlite3.connect(db)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA wal_autocheckpoint=0")
            connection.execute("DELETE FROM provider_session_runtime")
            connection.commit()
            changed = report()
        self.assertEqual(changed["by_harness"]["t3"]["windows"][0]["total_tokens"], 0)
        self.assertEqual(changed["by_harness"]["codex"]["windows"][0]["total_tokens"], 1100)
        with closing(sqlite3.connect(cache)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM cache_threads").fetchone()[0], 1)
        # Changing source selection must restore native attribution in a warm cache.
        restored = metrics.collect_report(self.root / "codex", NOW, cache_path=cache)
        self.assertEqual(set(restored["by_harness"]), {"codex"})
        self.assertEqual(restored["windows"][0]["total_tokens"], 1100)

    def test_t3_cache_cannot_overwrite_source_database(self):
        db = self.t3_db()
        before = db.read_bytes()
        with self.assertRaisesRegex(ValueError, "separate from harness storage"):
            metrics.collect_report(self.root, NOW, include_codex=False,
                                   harness_roots={metrics.Harness.T3: [db.parent]}, cache_path=db)
        self.assertEqual(db.read_bytes(), before)

    def test_cli_t3_default_discovery_and_explicit_source(self):
        db = self.t3_db(name="home/.t3/userdata/state.sqlite")
        self.write(prefix(thread="same-id") + [modern(), complete()], "home/.codex/sessions/t3.jsonl")
        for options in ([], [str(self.root), "--t3-dir", str(db)]):
            with self.subTest(options=options):
                output = self.root / "report.html"
                with patch("sys.argv", ["harness_metrics.py", *options, "--harness", "t3", "--no-cache",
                                        "--timezone", "UTC", "--output", str(output)]), \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(metrics.main(), 0)
                report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
                self.assertEqual(set(report["by_harness"]), {"t3"})
                self.assertEqual(report["windows"][0]["total_tokens"], 1100)
        parser = metrics.argument_parser()
        paths = metrics.source_paths(parser.parse_args([str(self.root)]), parser)
        self.assertNotIn(metrics.Harness.T3, paths)

    def catalog(self):
        return metrics.openrouter_prices({"data": [
            {"id": "anthropic/claude-haiku-4.5", "pricing": {"prompt": ".000001", "completion": ".000005", "input_cache_read": ".0000001"}},
            {"id": "vendor/new-model", "pricing": {"prompt": ".000002", "completion": ".000010", "input_cache_read": ".0000002", "input_cache_write": ".0000025", "input_cache_write_1h": ".000004",
                    "overrides": [{"min_prompt_tokens": 1000, "prompt": ".000004"}, {"min_prompt_tokens": 2000, "prompt": ".000008"}]}}]})

    def test_sqlite_cache_reuses_all_harness_readers_and_preserves_metrics(self):
        self.write(prefix() + [modern(), complete()], "codex/session.jsonl")
        self.write([self.claude()], "claude/session.jsonl")
        self.write([self.copilot("session.start", "start", {"sessionId": "copilot"}),
                    self.copilot("assistant.usage", "usage", {"model": "gpt-6.1-sol", "inputTokens": 100,
                                                              "outputTokens": 10})], "copilot/events.jsonl")
        sources = {metrics.Harness.CLAUDE: [self.root / "claude"],
                   metrics.Harness.COPILOT: [self.root / "copilot"],
                   metrics.Harness.OPENCODE: [self.opencode_db()]}
        cache = self.root / "cache.sqlite3"
        expected = self.report(sources)
        cached = metrics.collect_report(self.root / "codex", NOW, harness_roots=sources, cache_path=cache)
        self.assertEqual(cached, expected)
        with patch("harness_metrics.read_thread", side_effect=AssertionError("Codex parsed")), \
             patch("harness_metrics.read_claude", side_effect=AssertionError("Claude parsed")), \
             patch("harness_metrics.read_copilot", side_effect=AssertionError("Copilot parsed")), \
             patch("harness_metrics.read_opencode", side_effect=AssertionError("OpenCode parsed")):
            warm = metrics.collect_report(self.root / "codex", NOW, harness_roots=sources, cache_path=cache)
        self.assertEqual(warm, expected)
        self.assertNotIn(b"private text", cache.read_bytes())

    def test_sqlite_cache_invalidates_on_opencode_wal_changes(self):
        path = self.opencode_db()
        cache = self.root / "metrics.sqlite3"
        sources = {metrics.Harness.OPENCODE: [path]}
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA wal_autocheckpoint=0")
            connection.commit()
            report = metrics.collect_report(self.root, NOW, harness_roots=sources, include_codex=False, cache_path=cache)
            row = connection.execute("SELECT data FROM message WHERE id='m1'").fetchone()
            data = json.loads(row[0])
            data["tokens"]["input"] = 300
            connection.execute("UPDATE message SET data=? WHERE id='m1'", (json.dumps(data),))
            connection.commit()
            with patch("harness_metrics.read_opencode", wraps=metrics.read_opencode) as read:
                updated = metrics.collect_report(self.root, NOW, harness_roots=sources, include_codex=False, cache_path=cache)
                self.assertEqual(read.call_count, 1)
            self.assertEqual(updated["windows"][0]["total_tokens"], report["windows"][0]["total_tokens"] + 200)
            self.assertEqual(updated, metrics.collect_report(self.root, NOW, harness_roots=sources, include_codex=False))

    def test_sqlite_cache_cannot_write_to_harness_database(self):
        path = self.opencode_db()
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "separate from harness storage"):
            metrics.collect_report(self.root, NOW, harness_roots={metrics.Harness.OPENCODE: [path]}, cache_path=path)
        self.assertEqual(path.read_bytes(), original)

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

    def test_claude_turn_without_turn_duration_is_timed_from_prompt(self):
        user = {"type": "user", "uuid": "u1", "sessionId": "same-id", "timestamp": START.isoformat(),
                "message": {"role": "user", "content": "hello"}}
        self.write([user, self.claude(output=100, at=START + timedelta(seconds=4))], "claude/session.jsonl")
        w = self.report({metrics.Harness.CLAUDE: [self.root / "claude"]})["windows"][0]
        self.assertEqual(w["active_seconds"], 4)
        self.assertEqual(w["metrics"]["throughput"]["count"], 1)
        self.assertEqual(w["metrics"]["throughput"]["avg"], 25)
        self.assertNotIn("Missing turn duration", w["coverage"])

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

    def test_claude_malformed_type_and_stop_reason_are_skipped(self):
        bad_stop = self.claude(request="r2", message="m2")
        bad_stop["message"]["stop_reason"] = ["end_turn"]
        rows = [{"type": ["assistant"], "timestamp": START.isoformat()}, bad_stop, self.claude()]
        self.write(rows, "claude/session.jsonl")
        r = self.report({metrics.Harness.CLAUDE: [self.root / "claude"]})
        self.assertEqual(r["windows"][0]["total_tokens"], 1_600)

    def test_claude_invalid_usage_still_completes_the_turn(self):
        user = {"type": "user", "uuid": "u1", "sessionId": "same-id", "timestamp": START.isoformat(),
                "message": {"role": "user", "content": "hello"}}
        final = self.claude(request="r2", message="m2", at=START + timedelta(seconds=5))
        final["message"]["usage"] = {"input_tokens": -1}
        self.write([user, final], "claude/session.jsonl")
        with redirect_stderr(io.StringIO()):
            r = self.report({metrics.Harness.CLAUDE: [self.root / "claude"]})
        self.assertEqual(r["quality"]["Invalid usage records"], 1)
        self.assertEqual(r["windows"][0]["coverage"]["Completed turns"], 1)
        self.assertNotIn("Unfinished turns", r["windows"][0]["coverage"])

    def test_session_identity_tolerates_invalid_utf8_after_first_record(self):
        path = self.root / "codex/invalid.jsonl"
        path.parent.mkdir(parents=True)
        path.write_bytes(json.dumps(record("session_meta", {"id": "real-id"})).encode() + b"\n\xff\xfe bad\n")
        self.assertEqual(metrics.session_identity(path, metrics.Harness.CODEX), "real-id")

    def test_harness_filter_excludes_t3_threads_from_shared_opencode_group(self):
        db = self.opencode_db()
        with closing(sqlite3.connect(db)) as connection, connection:
            row = connection.execute("SELECT data FROM message WHERE id='m1'").fetchone()
            connection.execute("INSERT INTO message VALUES (?, ?, ?)", ("m2", "native-only", row[0]))
        sources = {metrics.Harness.OPENCODE: [db], metrics.Harness.T3: [self.t3_db((("opencode", "same-id"),))]}
        cache = self.root / "cache.sqlite3"
        metrics.collect_report(self.root / "codex", NOW, harness_roots=sources, cache_path=cache)
        with closing(sqlite3.connect(cache)) as connection:
            reader = metrics.MetricsReader(connection)
            for harness in (metrics.Harness.OPENCODE, metrics.Harness.T3):
                with self.subTest(harness):
                    threads = list(reader.threads(metrics.Quality(), harness))
                    self.assertEqual({thread.harness for thread in threads}, {harness})
                    self.assertEqual(len(threads), 1)

    def test_claude_invalid_utf8_line_is_skipped_without_losing_usage(self):
        path = self.root / "claude/session.jsonl"
        path.parent.mkdir(parents=True)
        first, last = self.claude(request="r1", message="m1"), self.claude(request="r2", message="m2")
        path.write_bytes(json.dumps(first).encode() + b"\n\xff\xfe broken\n" + json.dumps(last).encode() + b"\n")
        with redirect_stderr(io.StringIO()):
            r = self.report({metrics.Harness.CLAUDE: [self.root / "claude"]})
        self.assertEqual(r["windows"][0]["total_tokens"], 1600)
        self.assertEqual(r["quality"]["Malformed lines"], 1)
        self.assertNotIn("Unreadable files", r["quality"])

    def test_claude_synthetic_messages_do_not_affect_models_or_usage_counts(self):
        synthetic = self.claude(request="synthetic", message="synthetic")
        synthetic["message"].update(model="<synthetic>", usage={"input_tokens": 0, "output_tokens": 0},
                                    content=[{"type": "text", "text": "API Error"}])
        self.write([self.claude(), synthetic], "claude/session.jsonl")
        r = self.report({metrics.Harness.CLAUDE: [self.root / "claude"]})
        w = r["windows"][0]
        self.assertEqual(w["total_tokens"], 800)
        self.assertEqual(w["coverage"]["Usage responses"], 1)
        self.assertEqual(list(w["models"]), ["claude-sonnet-4-6"])
        self.assertNotIn("<synthetic>", metrics.render_report(r))

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

    def test_copilot_throughput_uses_persisted_message_output_without_usage_events(self):
        def turn(n, start, seconds, calls):
            at = START + timedelta(seconds=start)
            messages = [self.copilot("assistant.message", f"m{n}-{call}-{chunk}",
                                     {"turnId": "0", "messageId": f"m{n}-{call}-{chunk}", "apiCallId": f"c{n}-{call}",
                                      "model": "claude-sonnet-4.6", "outputTokens": output, "chunkIndex": chunk}, at)
                        for call, output in enumerate(calls) for chunk in range(2)]
            return [self.copilot("assistant.turn_start", f"start{n}", {"turnId": "0", "interactionId": f"i{n}"}, at),
                    *messages,
                    self.copilot("assistant.turn_end", f"end{n}", {"turnId": "0"}, at + timedelta(seconds=seconds))]
        summary = {"modelMetrics": {"claude-sonnet-4.6": {"usage": {"inputTokens": 600, "outputTokens": 400}}}}
        self.write([self.copilot("session.start", "s", {"sessionId": "same-id"}), *turn(1, 0, 4, [100]),
                    *turn(2, 10, 6, [100, 200]), self.copilot("session.shutdown", "x", summary, START + timedelta(seconds=20))],
                   "copilot/events.jsonl")
        w = self.report({metrics.Harness.COPILOT: [self.root / "copilot"]})["windows"][0]
        self.assertEqual(w["total_tokens"], 1000)
        self.assertEqual(w["active_seconds"], 10)
        self.assertEqual(w["metrics"]["throughput"]["count"], 2)
        self.assertEqual(w["metrics"]["throughput"]["avg"], 37.5)
        self.assertNotIn("Missing throughput samples", w["coverage"])

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

    def test_copilot_shutdown_billing_counts_models_without_detailed_usage(self):
        detailed = {"model": "gpt-6.1-sol", "inputTokens": 1000, "outputTokens": 100, "cost": 1,
                    "copilotUsage": {"totalNanoAiu": 500}}
        summary = {"modelMetrics": {
            "gpt-6.1-sol": {"usage": {"inputTokens": 1000, "outputTokens": 100}, "totalNanoAiu": 500,
                            "requests": {"cost": 1}},
            "claude-sonnet-4.6": {"usage": {"inputTokens": 300, "outputTokens": 30}, "totalNanoAiu": 700,
                                  "requests": {"cost": 2}}}}
        self.write([self.copilot("session.start", "s", {"sessionId": "same-id"}),
                    self.copilot("assistant.usage", "u", detailed),
                    self.copilot("session.shutdown", "end", summary)], "copilot/events.jsonl")
        r = self.report({metrics.Harness.COPILOT: [self.root / "copilot"]})
        self.assertEqual(r["windows"][0]["total_tokens"], 1430)
        self.assertEqual(r["windows"][0]["recorded_billing"],
                         {"Copilot nano-AIU": "1200", "Copilot premium requests": "3"})

    def test_copilot_large_shutdown_totals_use_normal_context_rates(self):
        for has_detailed_usage, expected in [(False, ".7506"), (True, "1.0012")]:
            with self.subTest(has_detailed_usage=has_detailed_usage):
                rows = []
                if has_detailed_usage:
                    rows.append(self.copilot("assistant.usage", "usage", {
                        "model": "gpt-5.4", "inputTokens": 100000, "outputTokens": 100,
                        "cacheReadTokens": 400, "reasoningTokens": 20}))
                rows.append(self.copilot("session.shutdown", "shutdown", {"modelMetrics": {
                    "gpt-5.4": {"usage": {"inputTokens": 400000 if has_detailed_usage else 300000,
                        "outputTokens": 200 if has_detailed_usage else 100,
                        "cacheReadTokens": 800 if has_detailed_usage else 400,
                        "reasoningTokens": 40 if has_detailed_usage else 20}}}}))
                self.write(rows, "copilot/events.jsonl")
                report = self.report({metrics.Harness.COPILOT: [self.root / "copilot"]})
                window = report["windows"][0]
                self.assertFalse(window["partial_cost"])
                self.assertEqual(window["unpriced"], {})
                self.assertEqual(Decimal(window["cost"]), Decimal(expected))
                self.assertEqual(window["total_tokens"], 400200 if has_detailed_usage else 300100)
                self.assertEqual(window["coverage"]["Aggregate snapshots"], 1)
                self.assertEqual(set(report["by_model"]), {"gpt-5.4"})

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

    def test_report_aggregates_each_thread_once_and_preserves_combined_statistics(self):
        rows = prefix(thread="same-id") + [modern(), complete()]
        rows += [event("task_started", turn_id="t2"),
                 event("thread_settings_applied", thread_settings={"service_tier": "priority"}),
                 modern("t2", "r2", usage(input=300000, output=200))]
        ending = complete("t2")
        ending["payload"]["duration_ms"] = 20000
        self.write(rows + [ending], "codex/session.jsonl")
        claude = self.claude(output=700)
        claude["message"]["model"] = "unpriced-model"
        self.write([{"type": "user", "sessionId": "same-id", "uuid": "user1",
                     "timestamp": START.isoformat()}, claude,
                    {"type": "system", "subtype": "turn_duration", "durationMs": 5000,
                     "timestamp": (START + timedelta(seconds=5)).isoformat()}], "claude/session.jsonl")
        sources = {metrics.Harness.CLAUDE: [self.root / "claude"],
                   metrics.Harness.OPENCODE: [self.opencode_db()],
                   metrics.Harness.COPILOT: [self.root / "empty-copilot"]}
        with patch("harness_metrics.add_thread", wraps=metrics.add_thread) as aggregate:
            report = self.report(sources, self.catalog())
        threads = {call.args[1].id: call.args[1] for call in aggregate.call_args_list}
        self.assertEqual(len(threads), 3)
        self.assertEqual(aggregate.call_count, len(threads))
        expected = metrics.build_breakdown(threads.values(), NOW, self.catalog())
        for key in expected:
            self.assertEqual(report[key], expected[key], key)
        for harness in [*sources, metrics.Harness.CODEX]:
            expected_harness = metrics.build_breakdown(
                [thread for thread in threads.values() if thread.harness == harness], NOW, self.catalog())
            self.assertEqual(report["by_harness"][harness.value], expected_harness, harness)
        self.assertEqual(report["windows"][0]["metrics"]["length"]["median"], 10)
        self.assertEqual(report["windows"][0]["metrics"]["length"]["p95"], 30)
        self.assertTrue(report["windows"][0]["partial_cost"])

    def test_single_harness_report_aggregates_each_thread_once(self):
        self.write(prefix() + [modern(), complete()], "codex/session.jsonl")
        with patch("harness_metrics.add_thread", wraps=metrics.add_thread) as aggregate:
            report = self.report({})
        self.assertEqual(aggregate.call_count, 1)
        expected = metrics.build_breakdown([aggregate.call_args.args[1]], NOW)
        self.assertEqual(report["by_harness"]["codex"], expected)
        for key in expected:
            self.assertEqual(report[key], expected[key], key)

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

    def test_long_catalog_entries_keep_original_identity_and_unpriced_fast_usage(self):
        prices = self.catalog()
        for count, long_context in [(999, False), (1000, True), (2000, True)]:
            for speed in [metrics.SpeedMode.NORMAL, metrics.SpeedMode.FAST]:
                with self.subTest(count=count, speed=speed):
                    rows = prefix(model="vendor/new-model")
                    rows[0]["payload"]["service_tier"] = "priority" if speed == metrics.SpeedMode.FAST else "default"
                    self.write(rows + [modern(counts=usage(input=count)), complete()], "codex/session.jsonl")
                    report = metrics.collect_report(self.root, NOW, catalog=prices)
                    name = "vendor/new-model" + ("-fast" if speed == metrics.SpeedMode.FAST else "")
                    name += "-long" if long_context else ""
                    self.assertEqual(set(report["by_model"]), {name})
                    self.assertEqual(report["openrouter_rates"][name]["id"], "vendor/new-model")
                    window = report["by_model"][name][0]
                    self.assertEqual(window["total_tokens"], count + 100)
                    costs, missing = metrics.price_usage("vendor/new-model", metrics.Usage(count, 100, 400, 20),
                                                        speed, prices)
                    self.assertEqual(Decimal(window["cost"]), sum(costs.values(), Decimal(0)))
                    self.assertEqual(window["unpriced_tokens"], sum(missing.values()))
                    self.assertEqual(window["unpriced"], {name: count + 100} if missing else {})
        self.assertFalse(metrics.is_long_context("vendor/new-model", metrics.Usage(2000, 0), prices, aggregate=True))
        costs, missing = metrics.price_usage("vendor/new-model", metrics.Usage(2000, 0), catalog=prices, aggregate=True)
        self.assertEqual(sum(costs.values(), Decimal(0)), Decimal(".004"))
        self.assertFalse(missing)

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
        db = self.opencode_db(model="gemini-2.5-pro", provider="google")
        output = self.root / "report.html"
        with patch("sys.argv", ["harness_metrics.py", str(self.root), "--harness", "opencode", "--opencode-dir", str(db), "--live-prices", "--output", str(output)]), \
             patch("harness_metrics.urlopen", side_effect=OSError("offline")), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertEqual(report["openrouter"]["error"], "offline")
        self.assertTrue(report["openrouter"]["bundled"])
        window = next(w for w in report["windows"] if w["label"] == "Last 7 days")
        self.assertFalse(window["partial_cost"])
        self.assertEqual(Decimal(window["cost"]), Decimal(".00125"))

    def test_bundled_catalog_preserves_cache_and_context_rates(self):
        snapshot = json.loads(metrics.BUNDLED_OPENROUTER_PRICES.read_text(encoding="utf-8"))
        catalog = metrics.openrouter_prices(snapshot)
        self.assertEqual(snapshot["source"], metrics.OPENROUTER_SOURCE)
        self.assertIsNotNone(datetime.fromisoformat(snapshot["retrieved"]).tzinfo)
        self.assertEqual(len(catalog), len(snapshot["data"]))
        price = catalog["google/gemini-2.5-pro"]
        self.assertEqual(price.short.cached, Decimal(".125"))
        self.assertEqual(price.short.write, Decimal(".375"))
        for count, input_rate in [(199999, "1.25"), (200000, "2.5")]:
            with self.subTest(input_tokens=count):
                costs, missing = metrics.price_usage("gemini-2.5-pro", metrics.Usage(count, 0), catalog=catalog)
                self.assertEqual(costs[metrics.Category.INPUT], Decimal(count) * Decimal(input_rate) / metrics.MILLION)
                self.assertFalse(missing)
        embedded = metrics.model_price("gpt-6.1-sol", catalog)
        self.assertEqual(embedded, metrics.PRICES["gpt-6.1-sol"])

    def test_cli_defaults_to_bundled_prices_from_another_working_directory(self):
        db = self.opencode_db(model="gemini-2.5-pro", provider="google")
        output = self.root / "report.html"
        original = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original)
        for options in [[], ["--offline"]]:
            with self.subTest(options=options), \
                 patch("sys.argv", ["harness_metrics.py", str(self.root), "--harness", "opencode", "--opencode-dir", str(db), "--output", str(output), *options]), \
                 patch("harness_metrics.urlopen") as request, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(metrics.main(), 0)
                request.assert_not_called()
            report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
            self.assertTrue(report["openrouter"]["bundled"])
            self.assertEqual(report["openrouter"]["snapshot_file"], str(metrics.BUNDLED_OPENROUTER_PRICES))
            self.assertIsNotNone(report["openrouter"]["retrieved"])
            window = next(w for w in report["windows"] if w["label"] == "Last 7 days")
            self.assertEqual(window["total_tokens"], 800)
            self.assertEqual(Decimal(window["cost"]), Decimal(".00125"))
            self.assertFalse(window["partial_cost"])

    def test_cli_supplied_and_live_catalogs_replace_bundled_prices(self):
        snapshot = {"data": [{"id": "vendor/new-model", "pricing": {
            "prompt": ".000002", "completion": ".000010", "input_cache_read": ".0000002",
            "input_cache_write": ".0000025"}}]}
        path = self.root / "prices.json"
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        db = self.opencode_db(model="vendor/new-model", provider="openrouter")
        output = self.root / "report.html"
        for options in [["--openrouter-prices", str(path), "--offline"], ["--live-prices"]]:
            with self.subTest(options=options), \
                 patch("sys.argv", ["harness_metrics.py", str(self.root), "--harness", "opencode", "--opencode-dir", str(db), "--output", str(output), *options]), \
                 patch("harness_metrics.urlopen", return_value=io.StringIO(json.dumps(snapshot))) as request, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(metrics.main(), 0)
                if "--live-prices" in options:
                    request.assert_called_once()
                else:
                    request.assert_not_called()
            report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
            self.assertFalse(report["openrouter"].get("bundled", False))
            window = next(w for w in report["windows"] if w["label"] == "Last 7 days")
            self.assertFalse(window["partial_cost"])
            self.assertEqual(Decimal(window["cost"]), Decimal(".00178"))

    def test_cli_invalid_live_catalog_falls_back_to_bundle(self):
        output = self.root / "report.html"
        with patch("sys.argv", ["harness_metrics.py", str(self.root), "--live-prices", "--output", str(output)]), \
             patch("harness_metrics.urlopen", return_value=io.StringIO('{"data": []}')), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        report = json.loads(output.read_text().split('id="report-data">')[1].split('</script>')[0])
        self.assertTrue(report["openrouter"]["bundled"])
        self.assertIn("no valid model prices", report["openrouter"]["error"])

    def test_cli_missing_or_invalid_local_catalog_stops_without_network(self):
        path = self.root / "prices.json"
        output = self.root / "report.html"
        for contents in [None, "{invalid", '{"data": []}', '{"data": {}}']:
            if contents is not None:
                path.write_text(contents, encoding="utf-8")
            for supplied in [False, True]:
                options = ["--openrouter-prices", str(path)] if supplied else []
                with self.subTest(contents=contents, supplied=supplied), \
                     patch("harness_metrics.BUNDLED_OPENROUTER_PRICES", path), \
                     patch("sys.argv", ["harness_metrics.py", str(self.root), "--output", str(output), *options]), \
                     patch("harness_metrics.urlopen") as request, redirect_stderr(io.StringIO()) as errors, \
                     self.assertRaises(SystemExit) as stopped:
                    metrics.main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn(str(path), errors.getvalue())
                request.assert_not_called()
                self.assertFalse(output.exists())

    def test_cli_live_pricing_rejects_conflicting_options(self):
        for options in [["--offline"], ["--openrouter-prices", "prices.json"]]:
            with self.subTest(options=options), \
                 patch("sys.argv", ["harness_metrics.py", "--live-prices", *options]), \
                 patch("harness_metrics.urlopen") as request, redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as stopped:
                metrics.main()
            self.assertEqual(stopped.exception.code, 2)
            request.assert_not_called()

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
