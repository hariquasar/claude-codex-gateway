# claude-codex-gateway

You have an LLM endpoint — a **base URL** and maybe an **API key** — that speaks
the OpenAI chat-completions API. Claude Code and Codex don't work against it
directly: Claude Code expects the Anthropic Messages API (`/v1/messages`) and
Codex expects the OpenAI Responses API (`/v1/responses`).

This script runs a local [LiteLLM](https://github.com/BerriAI/litellm) gateway
in front of your endpoint and translates both into plain chat completions:

```
Claude Code ──/v1/messages──┐
                            ├──► gateway (127.0.0.1:4009) ──/chat/completions──► your base URL
Codex ───────/v1/responses──┘
```

## Quick start

```bash
python3 gateway.py            # Windows: py gateway.py
```

On first run it asks for:

1. **Base URL** — e.g. `https://api.example.com/v1` (a trailing `/chat/completions` is stripped)
2. **API key** — hidden prompt; press Enter if your endpoint needs none
3. **Model** — fetched from `<base URL>/models` and shown as a numbered list;
   pick a number or type any custom model name. If the endpoint has no
   `/models`, you just type the name.

All three are saved under `~/.local/share/claude-codex-gateway/` (private,
0600), so later runs start straight away. When it is ready it prints the
settings to paste into Claude Code and Codex.

Requirements: Python 3.9+ and internet access. `uv`, a suitable Python and
LiteLLM are installed automatically in private environments. No Docker.

## Options

| Flag | Purpose |
| --- | --- |
| `--base-url URL` | Upstream base URL (saved) |
| `--model NAME` | Use this model name, skip the picker (saved) |
| `--choose-model` | Show the model picker again |
| `--list-models` | Print the upstream model list and exit |
| `--api-key-file PATH` | Read the key from a plain-text, `.env` (`UPSTREAM_API_KEY=...`) or macOS `.rtf` file |
| `--port N` | Local port (default 4009; the next free port is used if taken) |
| `--require-local-key` | Require a generated local key instead of the placeholder `local` |
| `--no-dashboard` | Lightweight mode: no PostgreSQL, no admin UI |
| `--no-cache` / `--cache-ttl S` | Response cache (on by default, in memory, 300 s) |
| `--no-auto-prompt-cache` | Disable automatic prompt-cache markers |
| `--open-ui` | Open the dashboard in a browser |

The key can also come from the `UPSTREAM_API_KEY` environment variable.

### Model type

The model name decides how requests are shaped. Names containing `claude` are
treated as Claude models and get automatic Anthropic prompt-cache markers
(cheaper repeated context); every other model is sent as a plain OpenAI-style
request with no Anthropic-only fields.

## Client setup

The gateway prints these with your real port, model and key filled in.

**Claude Code** — `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:4009",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_MODEL": "your-model",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "your-model",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "your-model",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "your-model",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1"
  }
}
```

**Codex** — `~/.codex/config.toml`:

```toml
model = "your-model"
model_provider = "local-gateway"

[model_providers.local-gateway]
name = "Local gateway"
base_url = "http://127.0.0.1:4009/v1"
wire_api = "responses"
```

With `--require-local-key`, Claude Code uses the printed key as
`ANTHROPIC_AUTH_TOKEN`, and Codex reads it from `GATEWAY_API_KEY`
(`env_key = "GATEWAY_API_KEY"`).

## Dashboard

By default the full LiteLLM admin UI runs at `http://127.0.0.1:4009/ui/` with
spend logs in a private embedded PostgreSQL. The login is printed at start-up
and saved to `dashboard-login.txt` in the state directory. Use
`--no-dashboard` if you don't need it.

## Security notes

- The gateway listens on `127.0.0.1` only.
- Without `--require-local-key`, any local process can use your upstream key
  through the gateway. Use `--require-local-key` on shared machines.
- Startup error output redacts keys; the full log is `gateway.log` in the
  state directory.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```
