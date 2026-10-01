# Harness Report

Generate an interactive HTML usage report from local **Codex, Claude Code, Copilot CLI, and OpenCode** session data. Explore token counts, estimated API costs, tool calls, and timing statistics by harness, model, model tier, and speed mode.

The script reads session storage without modifying it. Each report includes its data and visual assets and can be opened offline.

## Requirements

- Python 3.10 or later; no third-party Python packages are required.
- Local session logs or an OpenCode database in a supported format.
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

Without a directory argument, the script scans the current directory for Codex JSONL logs and discovers these installed sources when present:

| Harness | Default source |
| --- | --- |
| Codex | `~/.codex/sessions` and `~/.codex/archived_sessions`, plus the current directory |
| Claude Code | `~/.claude/projects` |
| Copilot CLI | `~/.copilot/session-state` |
| OpenCode | `~/.local/share/opencode/opencode.db` |

`CLAUDE_CONFIG_DIR` changes the Claude configuration root; its `projects` subdirectory is scanned. `XDG_DATA_HOME` changes OpenCode's data root; the script looks under `opencode`.

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
| `--harness NAME` | Select `codex`, `claude`, `copilot`, or `opencode`; repeat to combine. |
| `--claude-dir PATH` | Claude projects directory containing JSONL logs. |
| `--copilot-dir PATH` | Copilot session directory containing JSONL events. |
| `--opencode-dir PATH` | OpenCode data directory or database file. |
| `--output PATH` | HTML output; defaults to `harness_metrics.html`. The parent directory must exist. |
| `--timezone NAME` | Timezone for calendar windows, such as `UTC` or `Europe/Paris`. |
| `--offline` | Skip fetching OpenRouter prices. |
| `--openrouter-prices PATH` | Load a saved OpenRouter `/api/v1/models` JSON response. |

Quote paths containing spaces. For home-directory examples, use shell-expanded paths such as `"$HOME/.claude/projects"`.

## Pricing and network access

Embedded OpenAI and Anthropic rates work offline. When non-Codex sources are included and neither `--offline` nor `--openrouter-prices` is supplied, the script fetches the public OpenRouter model catalog to price additional matched models. That request retrieves prices; it does not upload session logs.

To use a saved catalog:

```sh
python3 harness_metrics.py --openrouter-prices /path/to/models.json --output report.html
```

The JSON must contain the catalog's `data` array. A supplied catalog can also be used with `--offline`. A failed live fetch still produces a report; an invalid supplied catalog stops generation.

Costs are **API-equivalent estimates, not subscription bills**. Embedded rates are snapshots applied to historical usage, and some models use documented proxies. OpenRouter prices supplement models without embedded rates. Unpriced tokens remain in usage totals and make the cost estimate partial. Recorded billing measurements appear separately.

## Read the report

Choose a reporting window and filter by harness, tier, model, or mode. Comparison tables offer average, median, minimum, maximum, P75, P95, and P99 statistics.

- **Today and Yesterday** follow the report timezone. Rolling windows cover exact 24-hour days.
- **First-token time** requires explicit logged timing; missing samples are excluded.
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
