#!/usr/bin/env python3
"""Make any OpenAI-compatible endpoint usable from Claude Code and Codex.

Run: python3 gateway.py
Windows: py gateway.py

You supply an upstream base URL (anything that speaks OpenAI chat completions)
and, optionally, an API key. A local LiteLLM gateway then translates:
  Claude Code  ->  /v1/messages      (Anthropic Messages API)
  Codex        ->  /v1/responses     (OpenAI Responses API)
into plain chat-completion calls the upstream understands.

The model is picked from the upstream's /models list, or typed by hand
(--model NAME). Claude-family models get automatic prompt-cache markers.
The base URL, model and key are saved locally after the first run.

Automatically installs uv (if missing), a supported Python, and LiteLLM in
private environments. Requires an existing Python 3.9+ and internet access.
Docker is not required. Response caching is on by default (in memory, 300 s).
The full LiteLLM dashboard with a private PostgreSQL database is set up by
default; --no-dashboard runs the lightweight gateway without a database.
Keep the terminal open; Ctrl+C stops the gateway.
"""
import argparse
from contextlib import contextmanager, nullcontext
import errno
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
import webbrowser

VERSION = "1.106.0"
KEY_NAME = "UPSTREAM_API_KEY"
NO_KEY = "no-key"  # sent upstream when the endpoint needs no API key
CACHE_TTL = 300
POSTGRES_VERSION = "18.6.3"
PRISMA_VERSION = "0.15.0"


def open_private_text(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        # Windows has no fchmod; access there follows the user's folder ACL.
        if os.name != "nt":
            os.fchmod(fd, 0o600)
        return os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise


def write_private(path, text):
    with open_private_text(path) as handle:
        handle.write(text)


def read_env_key(path):
    """Return the saved key ("" means "saved as no key"), or None if absent."""
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == KEY_NAME:
                return value.strip().strip("\"'")
    return None


def load_api_key(args, state):
    key = os.environ.get(KEY_NAME)
    if args.api_key_file:
        path = args.api_key_file.expanduser()
        if path.suffix.lower() == ".rtf":
            if sys.platform != "darwin":
                raise RuntimeError("RTF key files are supported only on macOS. "
                                   "Use a plain-text key file or the hidden prompt.")
            key = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)],
                                 capture_output=True, text=True, check=True).stdout.strip().lstrip("﻿")
        else:
            key = read_env_key(path)
            if key is None:
                key = path.read_text(encoding="utf-8-sig").strip()
    if key is None:
        key = read_env_key(state / "upstream.env")
    if key is None:
        key = getpass.getpass("Upstream API key (hidden, press Enter if none): ")
    key = key.strip()
    if any(c.isspace() for c in key):
        raise RuntimeError("The API key must not contain spaces or line breaks.")
    write_private(state / "upstream.env", KEY_NAME + "=" + key + "\n")
    return key


def normalize_base_url(url):
    url = url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/models"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise RuntimeError("The base URL must start with http:// or https://, e.g. https://example.com/v1")
    return url


def fetch_models(base_url, api_key):
    """Return the model ids the upstream lists at GET <base>/models."""
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    request = urllib.request.Request(base_url + "/models", headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        body = json.loads(response.read().decode("utf-8"))
    return parse_models(body)


def parse_models(body):
    entries = body.get("data", []) if isinstance(body, dict) else []
    return sorted({m["id"] for m in entries if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]})


def model_type(model):
    """Claude-family models accept Anthropic prompt-cache markers; others do not."""
    return "claude" if "claude" in model.lower() else "generic"


def choose_model(base_url, api_key):
    try:
        models = fetch_models(base_url, api_key)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"Could not fetch the model list from {base_url}/models ({exc}).", flush=True)
        models = []
    if models:
        print("Models available upstream:", flush=True)
        for number, name in enumerate(models, 1):
            print(f"  {number:>3}. {name}", flush=True)
        answer = input("Pick a number, or type a custom model name: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1]
    else:
        answer = input("Model name: ").strip()
    if not answer:
        raise RuntimeError("A model name is required.")
    return answer


def load_settings(state):
    path = state / "settings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def ensure_uv(state):
    installed = shutil.which("uv")
    if installed:
        return installed
    bootstrap = state / "bootstrap"
    bindir = bootstrap / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    uv = bindir / ("uv.exe" if os.name == "nt" else "uv")
    if uv.exists():
        return str(uv)
    print("Installing uv in a private environment...", file=sys.stderr, flush=True)
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(bootstrap)],
                       check=True, stdout=sys.stderr, stderr=sys.stderr)
    subprocess.run([str(python), "-m", "pip", "install", "uv"],
                   check=True, stdout=sys.stderr, stderr=sys.stderr)
    return str(uv)


def setup_runtime(state, dashboard=False):
    runtime = state / ("dashboard-venv" if dashboard else "venv")
    bindir = runtime / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    gateway = bindir / ("litellm.exe" if os.name == "nt" else "litellm")
    check = ("from importlib.metadata import version; import importlib.util; "
             f"assert version('litellm') == '{VERSION}'; "
             "assert importlib.util.find_spec('fastapi') is not None")
    if dashboard:
        check += (f"; assert version('embedded-postgres') == '{POSTGRES_VERSION}'"
                  f"; assert version('prisma') == '{PRISMA_VERSION}'"
                  "; import sys; assert sys.version_info[:2] == (3,12); import psycopg")
    if python.exists() and gateway.exists():
        result = subprocess.run([str(python), "-c", check], capture_output=True)
        if result.returncode == 0:
            return gateway
    print("Preparing the private LiteLLM environment...", file=sys.stderr, flush=True)
    uv = ensure_uv(state)
    runtime_ok = False
    if python.exists():
        predicate = "sys.version_info[:2] == (3,12)" if dashboard else "(3,10) <= sys.version_info[:2] < (3,15)"
        result = subprocess.run([str(python), "-c", "import sys; assert " + predicate], capture_output=True)
        runtime_ok = result.returncode == 0
    if not runtime_ok:
        command = [uv, "venv", "--clear", "--python", "3.12" if dashboard else ">=3.10,<3.15", str(runtime)]
        subprocess.run(command, check=True, stdout=sys.stderr, stderr=sys.stderr)
    command = [uv, "pip", "install", "--python", str(python)]
    requirements = [f"litellm[proxy]=={VERSION}"]
    if dashboard:
        requirements += [f"embedded-postgres=={POSTGRES_VERSION}", f"prisma=={PRISMA_VERSION}", "psycopg[binary]>=3.2,<4"]
    subprocess.run(command + requirements, check=True, stdout=sys.stderr, stderr=sys.stderr)
    return gateway


def gateway_config(base_url, model, require_local_key=True, cache_enabled=True, cache_ttl=CACHE_TTL,
                   dashboard=False, auto_prompt_cache=True):
    # JSON strings are valid YAML scalars, so model names like "llama3:8b" stay intact.
    q = json.dumps
    auth_line = ("  master_key: os.environ/GATEWAY_LOCAL_PROXY_KEY\n" if require_local_key
                 else "  dangerously_permit_weak_or_unset_master_key: true\n")
    cache_lines = (f"  cache: true\n  cache_params:\n    type: local\n    ttl: {cache_ttl}\n"
                   if cache_enabled else "  cache: false\n")
    # A Claude model behind an OpenAI-compatible URL is not detected as Anthropic
    # by LiteLLM, so the prompt-cache breakpoints are declared explicitly.
    prompt_lines = ("      cache_control_injection_points:\n"
                    "        - location: message\n          role: system\n"
                    "          control: {type: ephemeral}\n") if auto_prompt_cache else ""
    if dashboard:
        auth_line = ("  master_key: os.environ/LITELLM_MASTER_KEY\n"
                     "  database_url: os.environ/DATABASE_URL\n"
                     "  store_prompts_in_spend_logs: false\n")
    # use_chat_completions_api: Codex calls /v1/responses; send it upstream as a
    # chat completion instead of assuming the upstream has a Responses API.
    return f"""model_list:
  - model_name: {q(model)}
    litellm_params:
      model: {q("openai/" + model)}
      api_base: {q(base_url)}
      api_key: os.environ/{KEY_NAME}
      use_chat_completions_api: true
{prompt_lines}      drop_params: true
      additional_drop_params:
        - prompt_cache_key
litellm_settings:
{cache_lines}  enable_anthropic_prompt_caching: {str(auto_prompt_cache).lower()}
  anthropic_prompt_caching_ttl: 5m
  use_chat_completions_url_for_anthropic_messages: true
  drop_params: true
  num_retries: 0
  request_timeout: 120
  set_verbose: false
general_settings:
{auth_line}  disable_spend_logs: {str(not dashboard).lower()}
"""


def saved_dashboard_credentials(state):
    path = state / "dashboard-secrets.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if any(not isinstance(value.get(k), str) or len(value[k]) < 24
               for k in ("master_key", "ui_password", "salt_key", "database_password", "client_key")):
            raise RuntimeError("Saved dashboard credentials are incomplete. Restore dashboard-secrets.json from backup.")
        return value
    value = {"master_key": "sk-admin-" + secrets.token_hex(32),
             "ui_password": secrets.token_urlsafe(24), "salt_key": secrets.token_hex(32),
             "database_password": secrets.token_hex(32), "client_key": "sk-local-" + secrets.token_hex(24)}
    write_private(path, json.dumps(value, indent=2) + "\n")
    return value


@contextmanager
def dashboard_database(state, credentials):
    import embedded_postgres
    import litellm
    import psycopg

    data = state / "postgres-data"
    binaries = Path(embedded_postgres.__file__).parent / "pginstall/bin"
    suffix = ".exe" if os.name == "nt" else ""
    log_path = state / "database-setup.log"
    secret_values = tuple(credentials.values())

    def run(command, env=None):
        with open_private_text(log_path) as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=log)
        if result.returncode:
            raise startup_error("Database setup failed.", log_path, secret_values)

    if not (data / "PG_VERSION").exists():
        # A partial init is kept for inspection rather than deleting existing data.
        if data.exists() and any(data.iterdir()):
            raise RuntimeError(f"Incomplete PostgreSQL initialization in {data}. Preserve it before retrying with a fresh --state-dir.")
        password_file = state / "database-init-password"
        write_private(password_file, credentials["database_password"] + "\n")
        print("Initializing the private dashboard database...", flush=True)
        try:
            run([str(binaries / ("initdb" + suffix)), "-D", str(data), "-U", "postgres",
                 "--encoding=UTF8", "--auth-host=scram-sha-256", "--auth-local=scram-sha-256",
                 "--pwfile=" + str(password_file)])
        finally:
            password_file.unlink(missing_ok=True)

    with embedded_postgres.get_server(data, cleanup_mode="stop") as server:
        info = server.get_postmaster_info()
        query = {"host": str(info.socket_dir)} if info.socket_dir else {}
        if info.port:
            query["port"] = str(info.port)
        host = info.hostname or "localhost"
        netloc = "postgres:" + credentials["database_password"] + "@" + host
        if info.port:
            netloc += ":" + str(info.port)
        def uri(database):
            return urllib.parse.urlunsplit(("postgresql", netloc, "/" + database,
                                             urllib.parse.urlencode(query), ""))
        with psycopg.connect(uri("postgres"), autocommit=True) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("litellm",)).fetchone()
            if not exists:
                conn.execute("CREATE DATABASE litellm")
        database_url = uri("litellm")
        schema = state / "dashboard-schema.prisma"
        source = (Path(litellm.__file__).parent / "proxy/schema.prisma").read_text(encoding="utf-8")
        source = re.sub(r"binaryTargets\s*=\s*\[[^\]]*\]", 'binaryTargets = ["native"]', source)
        fingerprint = hashlib.sha256(source.encode()).hexdigest()
        marker = state / "dashboard-schema-version"
        write_private(schema, source)
        env = dict(os.environ, DATABASE_URL=database_url, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        check = subprocess.run([sys.executable, "-c", "from prisma import Prisma"], capture_output=True)
        changed = not marker.exists() or marker.read_text().strip() != fingerprint
        if changed or check.returncode:
            print("Preparing the dashboard database client (first run may take a few minutes)...", flush=True)
            run([sys.executable, "-m", "prisma", "generate", "--schema", str(schema)], env)
        with psycopg.connect(database_url) as conn:
            has_tables = conn.execute("SELECT to_regclass('public.\"LiteLLM_UserTable\"')").fetchone()[0] is not None
        if changed or not has_tables:
            print("Preparing dashboard tables...", flush=True)
            # No --accept-data-loss: upgrades that would remove data must fail safely.
            run([sys.executable, "-m", "prisma", "db", "push", "--skip-generate", "--schema", str(schema)], env)
            write_private(marker, fingerprint + "\n")
        yield database_url


def dashboard_runner(state):
    path = state / "dashboard-server.py"
    # Keep LiteLLM's own authentication on every admin endpoint. Only the local
    # generation routes translate the requested placeholder into a scoped key.
    write_private(path, '''import os
import uvicorn
from litellm.proxy.proxy_server import app

class LocalClientToken:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        routes = {"/v1/messages", "/messages", "/v1/messages/count_tokens", "/messages/count_tokens",
                  "/v1/chat/completions", "/chat/completions", "/v1/responses", "/responses",
                  "/v1/models", "/models"}
        if scope["type"] == "http" and scope.get("path") in routes and os.environ.get("GATEWAY_ALLOW_LOCAL_TOKEN") == "1":
            headers = list(scope.get("headers", []))
            auth = {k.lower(): v for k, v in headers}
            token = auth.get(b"authorization", auth.get(b"x-api-key", b""))
            if token in (b"", b"local", b"Bearer local", b"bearer local"):
                headers = [(k, v) for k, v in headers if k.lower() not in (b"authorization", b"x-api-key", b"x-litellm-api-key")]
                headers.append((b"authorization", ("Bearer " + os.environ["GATEWAY_CLIENT_KEY"]).encode()))
                scope = dict(scope, headers=headers)
        await self.app(scope, receive, send)

app.add_middleware(LocalClientToken)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["GATEWAY_PORT"]), workers=1, loop="asyncio")
''')
    return path


def provision_client_key(base, credentials):
    if is_ready(base, credentials["client_key"]):
        return
    # No "models" restriction: the key keeps working after the model is changed.
    payload = json.dumps({"key": credentials["client_key"], "key_alias": "local-coding-client",
                          "user_id": "local-coding-user"}).encode()
    request = urllib.request.Request(base + "/key/generate", data=payload,
              headers={"Authorization": "Bearer " + credentials["master_key"], "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Could not provision the local coding token (HTTP {exc.code}). See gateway.log.") from None
    if not is_ready(base, credentials["client_key"]):
        raise RuntimeError("The saved local coding token could not be verified.")


def is_ready(base, token):
    headers = {"Authorization": "Bearer " + token} if token else {}
    request = urllib.request.Request(base + "/v1/models", headers=headers)
    try:
        # A company HTTP proxy must not intercept the local readiness check.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def choose_port(preferred):
    for port in range(preferred, 65536):
        with socket.socket() as sock:
            if os.name == "nt":
                # Windows SO_REUSEADDR can accept a port another process owns.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE or (
                    os.name == "nt" and getattr(exc, "winerror", None) in (10013, 10048)
                ):
                    continue
                raise RuntimeError(f"Cannot use port {port}: {exc.strerror}") from exc
            return port
    raise RuntimeError(f"No free port is available from {preferred} through 65535.")


def startup_error(message, log_file, secret_values):
    try:
        with log_file.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 12000))
            detail = handle.read().decode("utf-8", errors="replace")
        detail = "\n".join(detail.splitlines()[-30:])
        for value in secret_values:
            if value:
                detail = detail.replace(value, "[REDACTED]")
        detail = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", detail)
        detail = re.sub(r"(?i)(bearer\s+)[^\s\"'<>]+", r"\1[REDACTED]", detail)
        detail = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", detail)
    except OSError:
        detail = "The startup log could not be read."
    return RuntimeError(f"{message}\nPrivate log: {log_file}\n"
                        f"Last startup messages (API keys redacted):\n{detail}")


def claude_code_settings(base, token, model):
    model_names = ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                   "ANTHROPIC_DEFAULT_HAIKU_MODEL")
    # Pin the main model and every tier alias to the one model the gateway serves.
    env = dict.fromkeys(model_names, model)
    env.update({"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": token,
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1"})
    return json.dumps({"env": env}, indent=2)


def codex_config(base, model, require_token):
    lines = [f"model = {json.dumps(model)}",
             'model_provider = "local-gateway"',
             "",
             "[model_providers.local-gateway]",
             'name = "Local gateway"',
             f"base_url = {json.dumps(base + '/v1')}",
             'wire_api = "responses"']
    if require_token:
        lines.append('env_key = "GATEWAY_API_KEY"  # export GATEWAY_API_KEY=<API key above>')
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", help="Upstream OpenAI-compatible base URL, e.g. https://example.com/v1 (saved)")
    parser.add_argument("--model", help="Model name to serve; skips the picker (saved)")
    parser.add_argument("--choose-model", action="store_true", help="Show the model picker again instead of the saved model")
    parser.add_argument("--list-models", action="store_true", help="Print the upstream's model list and exit")
    parser.add_argument("--api-key-file", type=Path, help="Read the upstream key from a plain-text, .env, or macOS RTF file")
    parser.add_argument("--port", type=int, default=4009)
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/share/claude-codex-gateway")
    parser.add_argument("--require-local-key", action="store_true", help="Require a generated local API key instead of accepting the fixed placeholder \"local\"")
    parser.add_argument("--no-cache", action="store_true", help="Disable LiteLLM response caching")
    parser.add_argument("--no-auto-prompt-cache", action="store_true", help="Disable automatic prompt-cache markers for Claude models (client-supplied markers are still forwarded)")
    parser.add_argument("--cache-ttl", type=int, default=CACHE_TTL, metavar="SECONDS", help="Response cache lifetime in seconds (default: 300)")
    parser.add_argument("--open-ui", action="store_true", help="Open the dashboard (or API testing UI with --no-dashboard)")
    parser.add_argument("--no-dashboard", action="store_true", help="Use the lightweight gateway without the database or admin dashboard")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.cache_ttl <= 0:
        parser.error("--cache-ttl must be greater than zero")
    state = args.state_dir.expanduser().resolve()
    if os.name == "nt":
        # Inherit the user's profile ACL. A 0700 directory created by an
        # elevated Windows Python can hide PostgreSQL from its restricted token.
        state.mkdir(parents=True, exist_ok=True)
    else:
        state.mkdir(parents=True, mode=0o700, exist_ok=True)
        state.chmod(0o700)

    settings = load_settings(state)
    base_url = normalize_base_url(args.base_url or settings.get("base_url") or input("Upstream base URL (e.g. https://example.com/v1): "))
    if args.list_models:
        for name in fetch_models(base_url, load_api_key(args, state)):
            print(name)
        return 0

    dashboard = not args.no_dashboard
    gateway = setup_runtime(state, dashboard)
    runtime_python = gateway.parent / ("python.exe" if os.name == "nt" else "python")
    if dashboard and Path(sys.prefix).resolve() != gateway.parent.parent.resolve():
        # Re-enter in the managed interpreter so database dependencies need not
        # be installed in the user's original Python. Pass the resolved URL on
        # so it is not asked for twice.
        argv = [str(runtime_python), str(Path(__file__).resolve()), *sys.argv[1:]]
        if not args.base_url:
            argv += ["--base-url", base_url]
        os.execv(str(runtime_python), argv)
    port = choose_port(args.port)
    if port != args.port:
        print(f"Warning: port {args.port} is unavailable. Using port {port}. "
              f"New base URL: http://127.0.0.1:{port}", file=sys.stderr, flush=True)
    api_key = load_api_key(args, state)
    saved_model = None if args.choose_model or settings.get("base_url") != base_url else settings.get("model")
    model = (args.model or saved_model or choose_model(base_url, api_key)).strip()
    write_private(state / "settings.json", json.dumps({"base_url": base_url, "model": model}, indent=2) + "\n")
    credentials = saved_dashboard_credentials(state) if dashboard else None
    local_key = ""
    if dashboard:
        local_key = credentials["client_key"] if args.require_local_key else "local"
    elif args.require_local_key:
        local_key_file = state / "local-api-key"
        local_key = local_key_file.read_text().strip() if local_key_file.exists() else "sk-local-" + secrets.token_hex(24)
        if not re.fullmatch(r"sk-local-[0-9a-f]{48}", local_key):
            raise RuntimeError("The saved local API key has an unexpected format.")
        write_private(local_key_file, local_key + "\n")
    context = dashboard_database(state, credentials) if dashboard else nullcontext(None)
    with context as database_url:
        return serve(args, state, gateway, runtime_python, port, base_url, model, api_key, local_key, credentials, database_url)


def serve(args, state, gateway, runtime_python, port, base_url, model, api_key, local_key, credentials, database_url):
    dashboard = credentials is not None
    auto_prompt_cache = model_type(model) == "claude" and not args.no_auto_prompt_cache
    config = state / "config.yaml"
    write_private(config, gateway_config(base_url, model, args.require_local_key, not args.no_cache, args.cache_ttl,
                                         dashboard, auto_prompt_cache))
    env = dict(os.environ)
    for name in ("LITELLM_MASTER_KEY", "GATEWAY_LOCAL_PROXY_KEY", "DATABASE_URL", "DIRECT_URL", "DATABASE_URL_READ_REPLICA", "WORKER_CONFIG"):
        env.pop(name, None)
    env.update({KEY_NAME: api_key or NO_KEY,
                "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False", "DO_NOT_TRACK": "1"})
    if local_key and not dashboard:
        env["GATEWAY_LOCAL_PROXY_KEY"] = local_key
    base = f"http://127.0.0.1:{port}"
    if dashboard:
        env.update({"LITELLM_MASTER_KEY": credentials["master_key"],
                    "LITELLM_SALT_KEY": credentials["salt_key"],
                    "UI_USERNAME": "admin", "UI_PASSWORD": credentials["ui_password"],
                    "DATABASE_URL": database_url, "CONFIG_FILE_PATH": str(config),
                    "GATEWAY_CLIENT_KEY": credentials["client_key"], "GATEWAY_PORT": str(port),
                    "GATEWAY_ALLOW_LOCAL_TOKEN": "0" if args.require_local_key else "1"})
        command = [str(runtime_python), str(dashboard_runner(state))]
        readiness_token = credentials["master_key"]
    else:
        command = [str(gateway), "--config", str(config), "--host", "127.0.0.1", "--port", str(port), "--num_workers", "1"]
        readiness_token = local_key
    redactions = (api_key, local_key) + (tuple(credentials.values()) if credentials else ())
    log_file = state / "gateway.log"
    print("Starting LiteLLM directly in Python (no Docker required)...", flush=True)
    with open_private_text(log_file) as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 60
            while not is_ready(base, readiness_token):
                if server.poll() is not None:
                    raise startup_error(f"LiteLLM could not start (exit code {server.returncode}).",
                                        log_file, redactions)
                if time.monotonic() >= deadline:
                    raise startup_error("LiteLLM startup timed out after 60 seconds.",
                                        log_file, redactions)
                time.sleep(0.3)
            if dashboard:
                provision_client_key(base, credentials)
            token = local_key or "local"
            print("\nGateway is ready.\n", flush=True)
            print(f"Upstream:              {base_url}", flush=True)
            print(f"Model:                 {model} ({model_type(model)})", flush=True)
            print(f"Base URL (Claude Code): {base}", flush=True)
            print(f"Base URL (OpenAI/Codex): {base}/v1", flush=True)
            print(f"API key:               {token}\n", flush=True)
            print(f"Response cache:        {'OFF' if args.no_cache else f'ON (in memory, {args.cache_ttl} seconds; clears on restart)'}", flush=True)
            print(f"Auto prompt caching:   {'ON (5-minute markers)' if auto_prompt_cache else 'OFF'}\n", flush=True)
            if dashboard:
                login = f"Dashboard: {base}/ui/\nUsername: admin\nPassword: {credentials['ui_password']}\n"
                login_file = state / "dashboard-login.txt"
                write_private(login_file, login)
                print(login, flush=True)
                print(f"Dashboard login saved: {login_file}\n", flush=True)
            print("Claude Code: add to ~/.claude/settings.json", flush=True)
            print(claude_code_settings(base, token, model), flush=True)
            print("\nCodex: add to ~/.codex/config.toml", flush=True)
            print(codex_config(base, model, args.require_local_key), flush=True)
            print("\nKeep this terminal open. Press Ctrl+C to stop the gateway.", flush=True)
            if args.open_ui:
                try:
                    opened = webbrowser.open(base + ("/ui/" if dashboard else "/"), new=2)
                except (OSError, webbrowser.Error):
                    opened = False
                if not opened:
                    print(f"Open this address in your browser: {base}{'/ui/' if dashboard else '/'}", flush=True)
            return server.wait()
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nGateway stopped.")
        raise SystemExit(0)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print("Setup error: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
