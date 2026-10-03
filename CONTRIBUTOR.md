# Contributing to Harness Report

Contributions can improve harness compatibility, usage accounting, pricing, report presentation, tests, or documentation. Read [AGENTS.md](AGENTS.md) for repository guidelines and [README.md](README.md) for user-facing behavior.

## Set up your workspace

Fork the repository on GitHub, clone your fork, and create a branch for your change. Python 3.10 or later is required. The static generator uses the standard library; no dependency installation or build step is needed. The optional dashboard server uses the dependencies in `requirements-server.txt`.

Run commands from the repository root:

```sh
python3 --version
python3 harness_metrics.py --help
python3 -m unittest discover -v
```

Keep one fix or feature per branch. For larger changes, describe the proposed behavior in an issue before implementation so scope and requirements can be resolved.

## Find the relevant code

| File | Responsibility |
| --- | --- |
| `harness_metrics.py` | CLI, storage readers, typed domain models, pricing, aggregation, and rendering. |
| `test_harness_metrics.py` | Synthetic JSONL and SQLite fixtures, unit tests, and CLI integration tests. |
| `harness_server.py` | FastAPI server, refresh snapshots, typed responses, and lazy chart queries. |
| `harness_dashboard.html` | Lean dashboard with local styles and JavaScript that requests the REST API. |
| `test_harness_server.py` | Optional API, cache consistency, and static-calculation parity tests. |
| `README.md` | Installation, usage, report interpretation, and troubleshooting. |
| `AGENTS.md` | Repository coding and workflow guidelines. |

Within the main module, `read_thread`, `read_claude`, `read_copilot`, and `read_opencode` parse their respective formats. `collect_report` discovers and groups records; `build_breakdown` calculates summaries; `render_report` embeds them in the `HTML` template. HTML, CSS, and JavaScript live in that template rather than separate asset files.

## Make a focused change

Use four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_CASE` constants. Preserve type annotations, enums, and dataclasses for domain values. Use `Decimal` for money and timezone-aware timestamps for activity.

Validate external fields before using them. Keep source storage read-only, retain valid usage around malformed records, and preserve deduplication and consistent totals across breakdowns. Report rendering must remain self-contained and safely embed strings from external data.

For harness changes, include a minimal synthetic fixture representing the storage format. For pricing changes, record the authoritative source and verification date alongside the rates and test context thresholds, cache categories, and mode premiums where applicable. Update CLI help and the README when options or observable behavior change.

`openrouter_prices.json` is the default supplemental pricing catalog and must be distributed alongside `harness_metrics.py`. To refresh it from a network with OpenRouter access, fetch `https://openrouter.ai/api/v1/models`, retain each valid model's `id` and complete `pricing` object (including cache rates and `overrides`), and record the URL as `source` and the UTC fetch time as `retrieved` beside the `data` array. Validate the replacement with `openrouter_prices()` and run the test suite. Keep embedded OpenAI and Anthropic rates authoritative for models they cover.

No formatter, linter, or static type checker is configured. Follow nearby code and avoid unrelated formatting or refactoring.

## Test your change

Run a focused test while developing, then the full suite:

```sh
python3 -m unittest -v test_harness_metrics.ReportTests.test_explicit_timing_and_cost_categories
python3 -m unittest discover -v
```

Test modules use `test_*.py`; test methods use `test_*`. Use `unittest`, temporary directories, fixed timestamps, synthetic records, and mocked network requests. Add regression checks that assert the corrected behavior, including totals, diagnostics, source isolation, or boundary handling as relevant. No numeric coverage threshold is configured.

Install `requirements-server.txt` in a virtual environment to include server tests, then run `python3 -m unittest discover -v`. Server tests use real HTTP requests against an ephemeral loopback port and synthetic databases, with no additional test dependencies. They skip when FastAPI/Uvicorn are unavailable. Use `python3 harness_server.py` to inspect the dashboard at `http://localhost:3050`; use synthetic sources and a separate cache for screenshots.

For timezone or startup changes, also test without timezone data:

```sh
PYTHONTZPATH='' python3 -S -B -m unittest discover -v
```

This command disables site packages and system timezone lookup. Toronto-specific tests skip when that data is unavailable; UTC behavior should pass.

For report presentation changes, generate a report from synthetic fixtures and open it in a browser. Check empty and populated windows, filter combinations, and narrow layouts. Use synthetic reports for screenshots.

## Report a problem or submit a pull request

Bug reports should include the command, Python version, operating system, affected harness, expected behavior, and actual behavior. Attach the smallest synthetic or redacted record that reproduces the issue. Exclude private conversations, credentials, and personal paths.

Use a concise imperative commit subject, such as `Fix malformed usage record handling`. Describe the concrete problem and resulting behavior in the pull request, list validation commands and results, and link a related issue when one exists. Include screenshots for visible report changes and mention any changed CLI defaults or compatibility limits.

Review the diff before submitting. Keep real session logs, generated personal reports, and credentials out of commits, and address review feedback within the original scope.
