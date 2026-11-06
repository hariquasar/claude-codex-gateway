import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gateway

try:
    import yaml
except ImportError:  # PyYAML is optional; structure checks run when it is present.
    yaml = None


class NormalizeBaseUrl(unittest.TestCase):
    def test_strips_trailing_slash_and_endpoint_paths(self):
        for raw in ("https://api.example.com/v1/", "https://api.example.com/v1/chat/completions",
                    "https://api.example.com/v1/responses", "https://api.example.com/v1/models"):
            self.assertEqual(gateway.normalize_base_url(raw), "https://api.example.com/v1")

    def test_rejects_non_http(self):
        for raw in ("api.example.com/v1", "ftp://x/v1", ""):
            with self.assertRaises(RuntimeError):
                gateway.normalize_base_url(raw)


class Models(unittest.TestCase):
    def test_parse_models_openai_shape(self):
        body = {"data": [{"id": "b"}, {"id": "a"}, {"id": "a"}, {"id": ""}, {"name": "x"}, "junk"]}
        self.assertEqual(gateway.parse_models(body), ["a", "b"])

    def test_parse_models_unexpected_shape(self):
        self.assertEqual(gateway.parse_models([1, 2]), [])
        self.assertEqual(gateway.parse_models({"models": ["a"]}), [])

    def test_model_type(self):
        self.assertEqual(gateway.model_type("Claude-Sonnet-4"), "claude")
        self.assertEqual(gateway.model_type("anthropic/claude-opus"), "claude")
        self.assertEqual(gateway.model_type("gpt-4o"), "generic")
        self.assertEqual(gateway.model_type("llama3:8b"), "generic")

    def test_fetch_models_sends_key_only_when_given(self):
        seen = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({"data": [{"id": "m1"}]}).encode()

        def fake_urlopen(request, timeout):
            seen.append(request)
            return Response()

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            self.assertEqual(gateway.fetch_models("https://h/v1", "k"), ["m1"])
            gateway.fetch_models("https://h/v1", "")
        self.assertEqual(seen[0].full_url, "https://h/v1/models")
        self.assertEqual(seen[0].get_header("Authorization"), "Bearer k")
        self.assertIsNone(seen[1].get_header("Authorization"))

    def test_choose_model_by_number_or_custom_name(self):
        with mock.patch.object(gateway, "fetch_models", return_value=["a", "b"]), \
                mock.patch("builtins.print"):
            with mock.patch("builtins.input", return_value="2"):
                self.assertEqual(gateway.choose_model("u", ""), "b")
            with mock.patch("builtins.input", return_value="my-custom"):
                self.assertEqual(gateway.choose_model("u", ""), "my-custom")
            with mock.patch("builtins.input", return_value="9"):
                self.assertEqual(gateway.choose_model("u", ""), "9")

    def test_choose_model_falls_back_to_manual_entry(self):
        with mock.patch.object(gateway, "fetch_models", side_effect=OSError("down")), \
                mock.patch("builtins.print"), mock.patch("builtins.input", return_value="typed"):
            self.assertEqual(gateway.choose_model("u", ""), "typed")
        with mock.patch.object(gateway, "fetch_models", return_value=[]), \
                mock.patch("builtins.print"), mock.patch("builtins.input", return_value=" "):
            with self.assertRaises(RuntimeError):
                gateway.choose_model("u", "")


class ApiKey(unittest.TestCase):
    def args(self, key_file=None):
        return SimpleNamespace(api_key_file=key_file)

    def test_blank_key_is_saved_and_not_asked_again(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {}, clear=True):
            state = Path(tmp)
            with mock.patch("getpass.getpass", return_value="  ") as prompt:
                self.assertEqual(gateway.load_api_key(self.args(), state), "")
                self.assertEqual(gateway.load_api_key(self.args(), state), "")
            self.assertEqual(prompt.call_count, 1)
            self.assertEqual(gateway.read_env_key(state / "upstream.env"), "")

    def test_env_and_file_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            with mock.patch.dict("os.environ", {gateway.KEY_NAME: "from-env"}):
                self.assertEqual(gateway.load_api_key(self.args(), state), "from-env")
            plain = state / "key.txt"
            plain.write_text("﻿plain-key\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                self.assertEqual(gateway.load_api_key(self.args(plain), state), "plain-key")
            dotenv = state / "key.env"
            dotenv.write_text(f"OTHER=1\n{gateway.KEY_NAME}='quoted'\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                self.assertEqual(gateway.load_api_key(self.args(dotenv), state), "quoted")

    def test_rejects_whitespace_inside_key(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {gateway.KEY_NAME: "a b"}):
            with self.assertRaises(RuntimeError):
                gateway.load_api_key(self.args(), Path(tmp))


class Config(unittest.TestCase):
    def test_generic_model_has_bridge_and_no_cache_markers(self):
        text = gateway.gateway_config("https://h/v1", "llama3:8b", auto_prompt_cache=False)
        self.assertIn('model_name: "llama3:8b"', text)
        self.assertIn('model: "openai/llama3:8b"', text)
        self.assertIn('api_base: "https://h/v1"', text)
        self.assertIn("use_chat_completions_api: true", text)
        self.assertIn("use_chat_completions_url_for_anthropic_messages: true", text)
        self.assertNotIn("cache_control_injection_points", text)
        self.assertIn("enable_anthropic_prompt_caching: false", text)

    def test_claude_model_gets_cache_markers(self):
        text = gateway.gateway_config("https://h/v1", "claude-x", auto_prompt_cache=True)
        self.assertIn("cache_control_injection_points", text)
        self.assertIn("enable_anthropic_prompt_caching: true", text)

    def test_auth_modes(self):
        self.assertIn("GATEWAY_LOCAL_PROXY_KEY", gateway.gateway_config("https://h/v1", "m", require_local_key=True))
        self.assertIn("dangerously_permit_weak_or_unset_master_key",
                      gateway.gateway_config("https://h/v1", "m", require_local_key=False))
        self.assertIn("LITELLM_MASTER_KEY", gateway.gateway_config("https://h/v1", "m", dashboard=True))

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_yaml_structure(self):
        cfg = yaml.safe_load(gateway.gateway_config("https://h/v1", 'odd: "name"', auto_prompt_cache=True))
        entry = cfg["model_list"][0]
        self.assertEqual(entry["model_name"], 'odd: "name"')
        self.assertEqual(entry["litellm_params"]["model"], 'openai/odd: "name"')
        self.assertTrue(entry["litellm_params"]["use_chat_completions_api"])


class ClientSnippets(unittest.TestCase):
    def test_claude_code_pins_every_tier(self):
        env = json.loads(gateway.claude_code_settings("http://127.0.0.1:4009", "local", "m"))["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:4009")
        for name in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                     "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            self.assertEqual(env[name], "m")

    def test_codex_config(self):
        text = gateway.codex_config("http://127.0.0.1:4009", "m", require_token=False)
        self.assertIn('base_url = "http://127.0.0.1:4009/v1"', text)
        self.assertIn('wire_api = "responses"', text)
        self.assertNotIn("env_key", text)
        self.assertIn("env_key", gateway.codex_config("http://127.0.0.1:4009", "m", require_token=True))


if __name__ == "__main__":
    unittest.main()
