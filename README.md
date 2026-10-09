# Harness Report

Generate an interactive HTML usage report from local **Codex, Claude Code, Copilot CLI, OpenCode, and T3 Code** session data. Explore token counts, estimated API costs, tool calls, and timing statistics by harness, model, model tier, and speed mode.

The script reads session storage without modifying it. Each static report includes its data and visual assets and can be opened offline. An optional local dashboard loads summaries and chart series on demand from the SQLite metrics cache.

## Requirements

- Python 3.10 or later; static reports require no third-party Python packages. The optional server dependencies are listed in `requirements-server.txt`.
- Local session logs or an OpenCode/T3 Code database in a supported format.
- A browser with JavaScript enabled to view the report.

Named timezones require timezone data available to Python. `UTC` always works; the default is `America/Toronto` when available, otherwise UTC.

## Quick start

Download this repository, or clone it:

```sh
git clone https://github.com/modern-swift-dev/harness-report.git
cd harness-report
python3 harness_metrics.py --offline --output report.html
```

Open `report.html` in your browser. The report is a snapshot: rerun the command to include new activity. An existing output file is overwritten.

The CLI maintains a SQLite cache at `~/.cache/harness-report/metrics.sqlite3` (or under `XDG_CACHE_HOME` when set). The first run parses the selected sources. Later runs add new conversations and skip parsing unchanged files, including their session headers. Files are checked using their identity, size, and modification/change timestamps. Changed conversations are refreshed atomically, so appended turns, rewrites, and duplicate archive copies do not double-count activity. Active conversations are reparsed when they change; the cache does not assume they are finished.

The cache contains normalized usage, timing, calls, billing, and parser diagnostics, without message text or tool arguments. Prices, timezones, and the report cutoff are applied when generating each report, so changing them does not require reimporting logs. Reports include only the currently selected source files; unrelated or removed sources retained in the cache are excluded. OpenCode cache invalidation includes its SQLite WAL file; a changed OpenCode database is refreshed as a whole.

Use `--cache /path/to/metrics.sqlite3` to choose a separate cache database, or `--no-cache` to reparse sources without persistent caching. A temporary SQLite working database keeps memory bounded when persistent caching is disabled and is removed after generation. Keep the cache separate from harness databases. The cache is disposable: a cache written by an older parser version is cleared and rebuilt from the sources on the next import. A cache from a newer version, or a database containing other tables, is rejected unchanged; use a new cache path or bypass it. Source logs remain read-only. Progress output shows how many conversations were cached or parsed.

Without a directory argument, the script discovers these installed sources when present. The current directory is not scanned:

| Harness | Default source |
| --- | --- |
| Codex | `~/.codex/sessions` and `~/.codex/archived_sessions` |
| Claude Code | `~/.claude/projects` |
| Copilot CLI | `~/.copilot/session-state` |
| OpenCode | `~/.local/share/opencode/opencode.db` |
| T3 Code | `~/.t3/userdata/state.sqlite` and `statev2.sqlite`, linked to native provider logs |

`CLAUDE_CONFIG_DIR` changes the Claude configuration root; its `projects` subdirectory is scanned. `XDG_DATA_HOME` changes OpenCode's data root; the script looks under `opencode`.

## Local dashboard

With Homebrew installed, set up Python and all server dependencies, then start the dashboard:

```sh
make setup
make serve
```

`Brewfile` declares Python 3.14, including SQLite support. `make setup` installs missing Homebrew dependencies without upgrading existing formulae, creates `.venv`, and installs the Python libraries from `requirements-server.txt`. `make serve` starts the server using that environment.

Without Homebrew, use any installed Python 3.10 or later to create the environment and install the server dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-server.txt
make serve
```

Open [http://localhost:3050](http://localhost:3050). The server binds to `127.0.0.1:3050` and needs no internet connection with the default pricing. Press Ctrl+C to stop it. An occupied port stops startup before any source import.

The dashboard uses the same interface as the static report, served without embedded metrics (approximately 90 KB, 27 KB compressed). The browser loads all reporting summaries once per refresh, then requests chart series for the chosen dates, interval, and filters; series needed by one view are sent together and calculated in a single pass over the cache. The server calculates reporting summaries once per refresh and keeps a bounded cache of requested chart series. Startup and refresh can take tens of seconds for large databases; later view requests reuse those summaries.

Startup imports configured sources into the persistent cache without generating a static report. **Refresh sources** repeats the import and recalculates summaries; controls pause until it finishes, then the page reloads with the same window, filters, and dates. Dates left at the first or latest day follow new history. There is no automatic polling. Failed refreshes retain the previous snapshot. Sources that change while being read or cannot be imported cause a refresh error that names the affected files; check access and retry. If the startup import fails, the server still starts and the dashboard shows the error until a refresh succeeds. Malformed individual records are skipped with diagnostics while valid usage is retained.

The dashboard includes **every conversation retained in the chosen metrics database**, including cached sources that no longer exist. Source arguments and `--harness` select what to import during refresh; use the dashboard filters to control what is displayed. Changing source options does not remove older cached conversations. Choose a separate `--cache` path to isolate an archive.

Source discovery, timezone, cache, and pricing options match the static generator:

```sh
python3 harness_server.py /path/to/codex-archive --cache /path/to/archive-metrics.sqlite3 --timezone UTC
python3 harness_server.py --harness claude --claude-dir /path/to/claude-projects
python3 harness_server.py --help
```

The server requires persistent caching and does not accept `--no-cache` or `--output`. Rates are selected at startup and stay fixed until restart. Changes made by another importer are detected, including commits that land while a refresh completes; use Refresh to load a consistent snapshot. API reads access SQLite read-only; imports update only the metrics cache, never harness storage. The cache must remain separate from harness databases.

The local REST API provides:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/metadata` | Current snapshot ID, cutoff, history, choices, pricing, and diagnostics. |
| `GET /api/report` | Metadata plus every reporting summary and catalog rate, as used by the dashboard. Trend series are omitted. |
| `GET /api/dashboard` | Reporting summaries, model/tier comparisons, billing, and catalog rates for selected filters. |
| `GET /api/trends` | One filtered series for the requested dates and interval, retaining gaps. |
| `POST /api/series` | Up to 64 filtered series for one date range, calculated together. |
| `POST /api/refresh` | Import configured sources and publish a new snapshot. |

Dashboard queries require `snapshot` and accept `window` (`today`, `yesterday`, or `last_N_days` for the supported windows), `harness`, `tier`, `model`, and `mode`. Trend queries require `snapshot`, `start`, and `end` (ISO dates), and also accept the scope filters and `granularity` (`hourly`, `daily`, `weekly`, or `monthly`). Series requests take a JSON body with `snapshot`, `start`, `end`, and `items`, each with a `granularity` and a `scope` object of the same filters; a filter combination with no summaries returns empty points rather than 422. Enum values for tier and mode match their dashboard labels. `/openapi.json` documents the typed request and response schemas. Invalid queries return 422; stale snapshots return 409. Both HTML and API responses disable browser caching.

Static HTML generation is unchanged and does not require these dependencies.

## Choose your sources

Read only a Codex archive, including JSONL files in nested directories:

```sh
python3 harness_metrics.py /path/to/codex-archive --offline --timezone UTC --output codex.html
```

An explicit directory disables automatic local source discovery. Other harnesses are included only through explicit source options and must be selected if you supply `--harness`.

Read a Claude archive without including Codex:

```sh
python3 harness_metrics.py --harness claude --claude-dir /path/to/claude-projects --offline --output claude.html
```

Read Copilot CLI events or an OpenCode database:

```sh
python3 harness_metrics.py --harness copilot --copilot-dir /path/to/session-state --offline --output copilot.html
python3 harness_metrics.py --harness opencode --opencode-dir /path/to/opencode.db --offline --output opencode.html
```

Read only sessions linked to T3 Code:

```sh
python3 harness_metrics.py --harness t3 --offline --output t3.html
python3 harness_metrics.py --harness t3 --t3-dir /path/to/state.sqlite --offline --output t3.html
```

[T3 Code](https://github.com/pingdotgg/t3code) drives native providers. The importer reads saved native session references from its legacy `state.sqlite` or current `statev2.sqlite` database, then uses the existing Codex, Claude Code, and OpenCode readers for usage and timing. Both T3 state and provider storage stay read-only. Linked sessions appear under `t3` instead of their provider when T3 is selected, so combined totals count them once. Claude subagents follow their parent's session reference. Other providers, sessions without saved native references, and missing native logs are outside T3 coverage; conversation text and context-window counters are not used to estimate missing usage.

T3-only imports look for linked logs in the default native provider locations listed above. A copied T3 database alone is insufficient. To use a different provider archive, select that provider too and supply its source option; for example, `--harness t3 --harness claude --t3-dir /path/to/state.sqlite --claude-dir /path/to/projects`. The provider filter retains sessions unrelated to T3. A Codex directory argument supplies the Codex logs when Codex is selected. Both the static report and local dashboard support these options.

Select multiple installed harnesses by repeating `--harness`:

```sh
python3 harness_metrics.py --harness codex --harness claude --offline --output combined.html
```

## Options

```sh
python3 harness_metrics.py --help
```

| Option | Purpose |
| --- | --- |
| `directory` | Optional Codex archive directory; disables automatic discovery. |
| `--harness NAME` | Select `codex`, `claude`, `copilot`, `opencode`, or `t3`; repeat to combine. |
| `--claude-dir PATH` | Claude projects directory containing JSONL logs. |
| `--copilot-dir PATH` | Copilot session directory containing JSONL events. |
| `--opencode-dir PATH` | OpenCode data directory or database file. |
| `--t3-dir PATH` | T3 Code base/userdata directory or state database file. |
| `--output PATH` | HTML output; defaults to `harness_metrics.html`. The parent directory must exist. |
| `--timezone NAME` | Timezone for calendar windows, such as `UTC` or `Europe/Paris`. |
| `--cache PATH` | Persistent SQLite metrics cache; defaults to the user cache directory. |
| `--no-cache` | Reparse sources without reading or writing the persistent metrics cache. |
| `--offline` | Use local pricing only (the default); cannot be combined with `--live-prices`. |
| `--openrouter-prices PATH` | Use a saved OpenRouter `/api/v1/models` JSON response instead of the bundled catalog. |
| `--live-prices` | Fetch OpenRouter prices; use the bundled catalog if the request fails. |

Quote paths containing spaces. For home-directory examples, use shell-expanded paths such as `"$HOME/.claude/projects"`.

## Pricing and network access

Report generation works without network access by default. OpenAI and Anthropic rates are embedded in the script, and [openrouter_prices.json](openrouter_prices.json) supplies additional model prices from the [public OpenRouter catalog](https://openrouter.ai/api/v1/models). The bundled file records its source and retrieval timestamp and preserves cache rates and context overrides. Keep it alongside `harness_metrics.py` when copying or packaging the tool; it is located relative to the script, regardless of the working directory.

To explicitly fetch live prices:

```sh
python3 harness_metrics.py --live-prices --output report.html
```

This request retrieves prices without uploading session logs. A failed request uses the bundled prices and records the error in the report. Live pricing cannot be combined with `--offline` or `--openrouter-prices`.

To use a saved catalog:

```sh
python3 harness_metrics.py --openrouter-prices /path/to/models.json --output report.html
```

The JSON must contain the catalog's `data` array with valid model prices. A supplied catalog can also be used with `--offline`. A missing or invalid local catalog stops generation with its path in the error message. Prices are snapshots and are not updated automatically; the report shows the catalog's retrieval timestamp when available.

Costs are **API-equivalent estimates, not subscription bills**. Embedded rates are snapshots applied to historical usage, and some models use documented proxies. OpenRouter prices supplement models without embedded rates. Unpriced tokens remain in usage totals and make the cost estimate partial. Recorded billing measurements appear separately.

## Read the report

Choose a reporting window and filter by harness, tier, model, or mode. Comparison tables offer average, median, minimum, maximum, P75, P95, and P99 statistics.

**Total tokens over time** shows input plus output tokens per calendar period, including cached input, cache writes, and reasoning, with zero for periods without recorded usage. It shares the Start, End, Hourly/Daily/Weekly/Monthly controls and harness, tier, model, and mode filters with the performance charts in both static reports and the local dashboard. Weekly and monthly totals cover the entire overlapping calendar period, with partial first and current periods. Performance statistic selections do not affect this token line.

The four performance charts are stacked at full width. **Displayed performance statistics** lets you choose any combination of Average, Median, P75, P95, and P99 lines, with only **P95** selected by default. Selections apply to all four charts, persist when changing dates, intervals, or filters, and set the vertical scale using only the visible lines. Their **Start** and **End** date controls default to the first recorded activity date and today (the report cutoff date). All available history is included, including activity older than a year. Choose Hourly, Daily, Weekly, or Monthly points; date choices persist when changing intervals or filters. Hourly points use the report timezone and show times within the selected dates; repeated daylight saving hours remain separate, with their UTC offset in the tooltip. Weekly and monthly points show statistics for entire calendar periods overlapping the selected range, with partial first and current periods. Missing samples appear as gaps. Statistics beneath each chart still describe the selected reporting window.

The top token summary shows total input, output, and cached input for the selected filters. Input includes cached input and cache writes; output includes reasoning. Total tokens equals input plus output. Cached input counts cache reads and is already included in input.

The model selector and comparison table show only entries with recorded tokens in the selected window, harness, tier, and mode. Zero-token entries, including shared timing and tool-call buckets, still contribute to aggregate totals and coverage. Changing filters resets the model selection to All models if that model has no tokens in the new scope.

Fast usage has its own model entry with a `-fast` suffix, such as `gpt-6.1-sol-fast`; Normal usage keeps the original name. Each entry has separate tokens, costs, tool calls, and timing. OpenAI Fast costs include the 50% premium. A turn that uses both modes keeps its timing under **Mixed modes (timing)** because separate durations are unavailable.

Requests crossing a model's context-pricing threshold have a `-long` suffix, such as `gpt-6.1-sol-long`; Fast requests above the threshold use `gpt-6.1-sol-fast-long`. Thresholds use total input, including cached input, and follow embedded model prices or matched OpenRouter context overrides. Models without context pricing and aggregate records lacking per-request sizes keep their original names. Aggregate records use normal-context rates without long-context surcharges or OpenRouter context overrides; recorded speed-mode premiums still apply where known. This assumption can underestimate actual long-context costs. Tokens and costs follow each request. A turn mixing context classes for one model and mode keeps shared timing under **Mixed contexts (timing)**. Tool calls follow matching turn usage when it has one context class; ambiguous calls appear under **Mixed contexts (tools)**. Mode and tier totals retain this activity once.

- **Today and Yesterday** follow the report timezone. Rolling windows cover exact 24-hour days.
- **Effective throughput** divides output tokens, including reasoning, by full turn duration, including tool execution and waiting.
- **Active duration** sums completed turn durations; concurrent activity can overlap.
- **Conversation counts** include subagents and can overlap across models, tiers, and modes.

Inspect coverage, unpriced usage, and parser diagnostics before interpreting comparisons. Available log history determines coverage; a yearly window does not guarantee a full year of data.

## Troubleshooting and privacy

- **Empty report:** verify source paths and selected harnesses, try a longer window, and inspect parser diagnostics. Missing default sources are skipped.
- **Unavailable timezone:** use `--timezone UTC` or provide timezone data to Python.
- **Output error:** use an `.html` filename in an existing, writable directory.
- **Partial costs or missing timing:** check the report's unpriced usage and sample counts; logs may lack the required evidence.

Reports contain source paths, model identifiers, usage, and diagnostic messages. Review them before sharing. The readers discard conversation text and tool arguments from metrics; use synthetic data for public examples and bug reports.

## Contributing

See [CONTRIBUTOR.md](CONTRIBUTOR.md) for development and contribution instructions.
