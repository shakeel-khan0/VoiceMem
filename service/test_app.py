from __future__ import annotations

import tempfile
import contextlib
import hashlib
import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from service.app import IngestRequest, VoiceMemManager, create_app


class FakeManager:
    def __init__(self):
        self.available = False
        self.error_type = None
        self.data = {"caller_a": ["Runs a dental clinic"], "caller_b": ["Runs a bakery"]}

    def initialize(self):
        self.available = True

    def close(self):
        pass

    def search(self, request):
        memories = self.data.get(request.caller_id, [])
        return {"ok": True, "memories": memories, "count": len(memories), "latency_ms": 1.25}

    def ingest(self, request):
        self.data.setdefault(request.caller_id, []).append(request.content)
        return {"ok": True, "persisted": True, "memory_count": 1}


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        self.client_context = TestClient(create_app(self.manager))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "alive": True, "voicemem_available": True, "error_type": None
        })

    def test_successful_search(self):
        response = self.client.post("/memory/search", json={
            "caller_id": "caller_a", "query": "What business do I run?"
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["memories"], ["Runs a dental clinic"])

    def test_callers_are_isolated(self):
        a = self.client.post("/memory/search", json={"caller_id": "caller_a", "query": "business"})
        b = self.client.post("/memory/search", json={"caller_id": "caller_b", "query": "business"})
        self.assertNotEqual(a.json()["memories"], b.json()["memories"])

    def test_ingest_writes_only_requested_caller_namespace(self):
        response = self.client.post("/memory/ingest", json={
            "caller_id": "caller_a", "content": "Prefers morning meetings"})
        self.assertEqual(response.status_code, 200)
        a = self.client.post(
            "/memory/search", json={"caller_id": "caller_a", "query": "preference"})
        b = self.client.post(
            "/memory/search", json={"caller_id": "caller_b", "query": "preference"})
        self.assertIn("Prefers morning meetings", a.json()["memories"])
        self.assertNotIn("Prefers morning meetings", b.json()["memories"])

    def test_manager_does_not_claim_success_without_persisted_memory(self):
        class NoWriteClient:
            def ingest(self, _content):
                print("private caller fact")
                return {"memory_ids": [], "persistent_memory_created": False}

        manager = VoiceMemManager()
        manager.available = True
        manager._clients["caller_a"] = NoWriteClient()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = manager.ingest(IngestRequest(
                caller_id="caller_a", content="I run a clinic."))
        self.assertEqual(result, {
            "ok": True, "persisted": False, "memory_count": 0})
        self.assertEqual(output.getvalue(), "")

    def test_real_manager_uses_distinct_namespaces(self):
        with tempfile.TemporaryDirectory() as root:
            manager = VoiceMemManager(root)
            manager.available = True
            manager._clients = {"caller_a": object(), "caller_b": object()}
            self.assertEqual(len(manager._clients), 2)

    def test_dynamic_callers_persist_and_remain_isolated_without_restart(self):
        stores = {}
        roots = {}

        class SearchResult:
            def __init__(self, values):
                self.result_leftbrain = list(values)
                self.result_rightbrain = []

        class PersistentClient:
            def __init__(self, config):
                self.caller_id = config["user_id"]
                self.root = config["memory_root"]
                roots[self.caller_id] = self.root
                stores.setdefault(self.root, [])

            def ingest(self, content):
                stores[self.root].append(content)
                return {"persistent_memory_created": True, "memory_ids": [content]}

            def search(self, _query, top_k=5):
                return SearchResult(stores[self.root][:top_k])

        def factory(config):
            return PersistentClient(config)

        with tempfile.TemporaryDirectory() as root, patch(
                "service.app.VoiceMem.from_config", side_effect=factory):
            manager = VoiceMemManager(root)
            manager.available = True
            self.assertTrue(manager.ingest(IngestRequest(
                caller_id="caller_a", content="Runs a bakery"))["persisted"])
            self.assertTrue(manager.ingest(IngestRequest(
                caller_id="caller_b", content="My name is Ali"))["persisted"])
            self.assertTrue(manager.ingest(IngestRequest(
                caller_id="caller_b", content="Runs a dental clinic"))["persisted"])
            self.assertEqual(
                manager.search(type("Request", (), {
                    "caller_id": "caller_a", "query": "business", "top_k": 5})())[
                    "memories"],
                ["Runs a bakery"],
            )
            self.assertEqual(
                manager.search(type("Request", (), {
                    "caller_id": "caller_b", "query": "business", "top_k": 5})())[
                    "memories"],
                ["My name is Ali", "Runs a dental clinic"],
            )

            manager.close()
            restarted = VoiceMemManager(root)
            restarted.available = True
            self.assertEqual(
                restarted.search(type("Request", (), {
                    "caller_id": "caller_b", "query": "business", "top_k": 5})())[
                    "memories"],
                ["My name is Ali", "Runs a dental clinic"],
            )
            self.assertEqual(
                roots["caller_a"].rsplit("\\", 1)[-1],
                hashlib.sha256(b"caller_a").hexdigest()[:24],
            )
            self.assertEqual(
                roots["caller_b"].rsplit("\\", 1)[-1],
                hashlib.sha256(b"caller_b").hexdigest()[:24],
            )

    def test_one_dynamic_caller_creation_failure_does_not_disable_manager(self):
        class WorkingClient:
            def ingest(self, content):
                return {"persistent_memory_created": True, "memory_ids": [content]}

        def factory(config):
            if config["user_id"] == "broken":
                raise ValueError("caller namespace failed")
            return WorkingClient()

        with tempfile.TemporaryDirectory() as root, patch(
                "service.app.VoiceMem.from_config", side_effect=factory):
            manager = VoiceMemManager(root)
            manager.available = True
            with self.assertRaises(ValueError):
                manager.ingest(IngestRequest(
                    caller_id="broken", content="My name is Broken"))
            self.assertTrue(manager.available)
            result = manager.ingest(IngestRequest(
                caller_id="healthy", content="My name is Healthy"))
            self.assertTrue(result["persisted"])
            self.assertTrue(manager.available)


if __name__ == "__main__":
    unittest.main()
