"""Offline checks for the actual DeepSeek transport and agent tool dispatch."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from philoagent.web_tools import DeepSeekWebError, REQUEST_TIMEOUT, deepseek_fetch, deepseek_search


UPSTREAM = Path(__file__).resolve().parents[1] / "vendor" / "MemEvolve" / "Flash-Searcher-main"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data, ensure_ascii=False).encode("utf-8")
    result.headers["Content-Type"] = "application/json"
    return result


SEARCH_RESPONSE = {"results": [{
    "title": "Primary source", "url": "https://example.org/paper?token=private#section",
    "snippet": "A source-backed claim.", "summary": "A summary", "date": "2026-10-29",
    "site_name": "Example", "content": None,
}]}


class DeepSeekWebTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-secret"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        transport = patch("philoagent.web_tools.requests.post")
        self.post = transport.start()
        self.addCleanup(transport.stop)
        self.post.return_value = response(SEARCH_RESPONSE)

    def test_search_request_and_source_shape(self):
        results, error = deepseek_search("知识来源", count=7)
        self.assertEqual(error, "")
        self.assertEqual(results[0]["link"], SEARCH_RESPONSE["results"][0]["url"])
        self.assertEqual(results[0]["title"], "Primary source")
        self.assertEqual(results[0]["snippet"], "A source-backed claim.")
        args, kwargs = self.post.call_args
        self.assertEqual(args, ("https://api.deepseek.com/v1/tools/search",))
        self.assertEqual(kwargs["json"], {"query": "知识来源", "count": 7, "need_content": False})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer offline-test-secret")
        self.assertEqual(kwargs["timeout"], REQUEST_TIMEOUT)
        self.assertFalse(kwargs["allow_redirects"])

    def test_fetch_request_keeps_title_and_markdown(self):
        self.post.return_value = response({"title": "Fetched page", "content": "## Evidence\n\nA claim."})
        content = deepseek_fetch("https://example.org/paper")
        self.assertIn("# Fetched page", content)
        self.assertIn("Source: https://example.org/paper", content)
        self.assertIn("## Evidence\n\nA claim.", content)
        self.assertEqual(self.post.call_args.args, ("https://api.deepseek.com/v1/tools/fetch",))
        self.assertEqual(self.post.call_args.kwargs["json"], {"url": "https://example.org/paper"})

    def test_explicit_credential_override_and_versioned_base(self):
        os.environ.update(DEEPSEEK_API_KEY="dedicated-secret", DEEPSEEK_BASE_URL="https://api.deepseek.com/v1/")
        deepseek_search("source")
        self.assertEqual(self.post.call_args.args[0], "https://api.deepseek.com/v1/tools/search")
        self.assertEqual(self.post.call_args.kwargs["headers"]["Authorization"], "Bearer dedicated-secret")

    def test_non_deepseek_model_base_never_receives_web_key(self):
        os.environ["OPENAI_BASE_URL"] = "https://untrusted.invalid/v1"
        deepseek_search("source")
        self.assertEqual(self.post.call_args.args[0], "https://api.deepseek.com/v1/tools/search")

    def test_unsafe_endpoints_are_rejected_without_request(self):
        for base in ("https://untrusted.invalid", "http://api.deepseek.com", "https://api.deepseek.com.evil.invalid",
                     "https://user@api.deepseek.com", "https://@api.deepseek.com", "https://api.deepseek.com:444",
                     "https://api.deepseek.com/v1?token=secret", "https://api.deepseek.com/unrelated"):
            with self.subTest(base=base):
                os.environ["DEEPSEEK_BASE_URL"] = base
                with self.assertRaisesRegex(DeepSeekWebError, "DEEPSEEK_BASE_URL"):
                    deepseek_search("source")
        self.post.assert_not_called()

    def test_missing_credentials_fail_before_network(self):
        os.environ.pop("OPENAI_API_KEY")
        with self.assertRaisesRegex(DeepSeekWebError, "DEEPSEEK_API_KEY or OPENAI_API_KEY"):
            deepseek_search("source")
        self.post.assert_not_called()

    def test_query_and_count_boundaries(self):
        for query in ("", "  ", "中" * 129, None):
            with self.subTest(query_length=len(query) if isinstance(query, str) else None):
                self.assertTrue(deepseek_search(query)[1])
        for count in (0, 51, True, "5", 1.5):
            with self.subTest(count=count):
                self.assertTrue(deepseek_search("source", count=count)[1])
        self.post.assert_not_called()
        deepseek_search("中" * 128, count=50)
        self.assertEqual(self.post.call_args.kwargs["json"]["count"], 50)

    def test_fetch_rejects_non_http_credentials_and_oversize_urls(self):
        for url in ("file:///etc/passwd", "https://user:secret@example.org", "https://@example.org",
                    "https://example.org/\nprivate", "https://example.org:bad", "https://", "https://x/" + "a" * 8192):
            with self.subTest(url=url[:55]):
                self.assertTrue(deepseek_fetch(url).startswith("Error reading page:"))
        self.post.assert_not_called()

    def test_http_failures_are_classified_without_body_or_key(self):
        for operation in ("search", "fetch"):
            for status, message in ((400, "invalid request"), (401, "authentication"), (402, "insufficient balance"),
                                    (403, "forbidden"), (404, "page unavailable" if operation == "fetch" else "request failed"),
                                    (429, "rate limit"), (500, "service unavailable"), (302, "request failed")):
                with self.subTest(operation=operation, status=status):
                    self.post.return_value = response({"error": "sensitive-provider-body"}, status)
                    with self.assertRaises(DeepSeekWebError) as raised:
                        deepseek_search("source") if operation == "search" else deepseek_fetch("https://example.org")
                    text = str(raised.exception)
                    self.assertIn(message, text)
                    self.assertEqual("TERMINATE_QUOTA" in text, status in (402, 429))
                    self.assertNotIn("sensitive-provider-body", text)
                    self.assertNotIn("offline-test-secret", text)

    def test_network_errors_and_timeouts_do_not_echo_exception_secrets(self):
        for error in (requests.Timeout("offline-test-secret"), requests.ConnectionError("offline-test-secret")):
            self.post.side_effect = error
            with self.assertRaises(DeepSeekWebError) as raised:
                deepseek_search("source")
            self.assertNotIn("offline-test-secret", str(raised.exception))

    def test_invalid_responses_fail_clearly(self):
        for data in ([], {}, {"results": {}}, {"results": [None]}, {"results": [{"url": "file:///private"}]}):
            with self.subTest(data=data):
                self.post.return_value = response(data)
                with self.assertRaises(DeepSeekWebError):
                    deepseek_search("source")
        invalid_json = response({})
        invalid_json._content = b"not json"
        self.post.return_value = invalid_json
        with self.assertRaisesRegex(DeepSeekWebError, "invalid JSON"):
            deepseek_search("source")

    def test_evidence_has_hashes_status_and_redacted_urls_without_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            os.environ["PHILO_EVIDENCE_DIR"] = temporary
            deepseek_search("private-query")
            evidence = (Path(temporary) / "web_requests.jsonl").read_text(encoding="utf-8")
            event = json.loads(evidence)
            self.assertEqual(event["backend"], "deepseek")
            self.assertEqual(event["http_status"], 200)
            self.assertEqual(event["source_urls"], ["https://example.org/paper"])
            self.assertEqual(event["response_sha256"], hashlib.sha256(self.post.return_value.content).hexdigest())
            self.assertEqual(len(event["request_payload_sha256"]), 64)
            for secret in ("offline-test-secret", "private-query", "A source-backed claim.", "token=private", "section"):
                self.assertNotIn(secret, evidence)

    def test_failed_requests_also_have_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            os.environ["PHILO_EVIDENCE_DIR"] = temporary
            self.post.return_value = response({}, 401)
            with self.assertRaises(DeepSeekWebError):
                deepseek_search("source")
            event = json.loads((Path(temporary) / "web_requests.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["http_status"], 401)
            self.assertEqual(event["operation"], "search")


class WebToolDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from FlashOAgents import search_tools
        cls.tools = search_tools

    def setUp(self):
        environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "offline-test-secret", "WEB_SEARCH_PROVIDER": " DeepSeek ",
            "WEB_ACCESS_PROVIDER": " DeepSeek ",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        transport = patch("philoagent.web_tools.requests.post")
        self.post = transport.start()
        self.addCleanup(transport.stop)

    def test_actual_web_search_tool_routes_to_deepseek(self):
        self.post.return_value = response(SEARCH_RESPONSE)
        with patch.object(self.tools, "web_search_google_serper", side_effect=AssertionError("wrong backend")):
            result = self.tools.WebSearchTool().forward("source")
        self.assertIn("[Primary source](https://example.org/paper?token=private#section)", result)
        self.assertIn("A source-backed claim.", result)
        self.assertEqual(self.post.call_args.kwargs["json"]["count"], 5)

    def test_actual_crawl_tool_fetches_then_summarizes(self):
        self.post.return_value = response({"title": "Fetched page", "content": "A source-backed claim."})
        model = Mock(return_value=Mock(content="A grounded summary"))
        with patch.object(self.tools, "read_page_jina", side_effect=AssertionError("wrong backend")):
            result = self.tools.CrawlPageTool(model).forward("https://example.org/paper", "which claim?")
        self.assertEqual(result, "A grounded summary")
        self.assertIn("A source-backed claim.", model.call_args.args[0][0]["content"])
        self.assertEqual(self.post.call_args.args[0], "https://api.deepseek.com/v1/tools/fetch")

    def test_legacy_provider_defaults_still_dispatch(self):
        os.environ.pop("WEB_SEARCH_PROVIDER")
        os.environ.pop("WEB_ACCESS_PROVIDER")
        with patch.object(self.tools, "web_search_google_serper", return_value=([], "legacy search")) as serper:
            self.assertEqual(self.tools.web_search("source"), ([], "legacy search"))
            serper.assert_called_once()
        with patch.object(self.tools, "read_page_jina", return_value="legacy fetch") as jina:
            self.assertEqual(self.tools.read_page("https://example.org"), "legacy fetch")
            jina.assert_called_once()
        self.post.assert_not_called()

    def test_invalid_providers_do_not_silently_fallback(self):
        os.environ.update(WEB_SEARCH_PROVIDER="unknown", WEB_ACCESS_PROVIDER="unknown")
        with self.assertRaisesRegex(ValueError, "WEB_SEARCH_PROVIDER"):
            self.tools.web_search("source")
        with self.assertRaisesRegex(ValueError, "WEB_ACCESS_PROVIDER"):
            self.tools.read_page("https://example.org")
        self.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
