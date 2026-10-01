# Repository Guidelines

## Project Structure & Module Organization

- `harness_metrics.py` contains the CLI, harness-specific readers, typed metric models, pricing calculations, and report rendering. HTML, CSS, and JavaScript assets are embedded in its `HTML` constant.
- `test_harness_metrics.py` contains unit and integration tests with synthetic JSONL records and SQLite fixtures.
- The project uses Python 3.10+ and the standard library. There are no separate asset directories, build scripts, or dependency manifests.

## Build, Test, and Development Commands

Run commands from the repository root; no build step is required.

- `python3 harness_metrics.py --help` — inspect source, pricing, and timezone options.
- `python3 harness_metrics.py /path/to/archive --offline --timezone UTC --output report.html` — generate a standalone report from a Codex archive.
- `python3 -m unittest discover -v` — run the complete test suite.
- `python3 -m unittest -v test_harness_metrics.ReportTests` — run report-focused tests.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_CASE` constants. Preserve type annotations, enums, and dataclasses for domain values; use `Decimal` for costs. Validate external record fields before accessing nested values. Keep changes scoped and prefer the simplest implementation. No formatter, linter, or static type checker is configured.

## Testing Guidelines

Tests use `unittest` and `unittest.mock`; name modules `test_*.py` and methods `test_*`. Use temporary directories, fixed timestamps, synthetic records, and mocked network calls. Add regression tests for parser, pricing, window-boundary, or CLI behavior changes. Check totals and diagnostics, including retained valid usage after malformed records. No numeric coverage threshold is configured. Toronto-specific tests skip when timezone data is unavailable.

## Commit & Pull Request Guidelines

Git history contains no commits, so commit conventions are not established. Use concise imperative subjects, such as `Fix malformed usage record handling`. Keep each change focused. PR descriptions should explain the problem, resulting behavior, and validation commands. Include screenshots for report presentation changes and link an issue when applicable.

## Data Handling & Agent Instructions

Keep harness storage read-only and report assets self-contained. Reports expose source paths; use synthetic data in shared artifacts. Explicit archive directories disable automatic local source discovery. Use UTC when timezone data is unavailable.

Flag uncertainty and clarify unresolved requirements before implementation. Proceed with authorized reversible work; assessment-only requests should produce findings without code edits.
