import tempfile
import unittest
from pathlib import Path

from filesystem_mcp_server import FilesystemService, McpServer, ServerConfig
from matching_agent import McpClient, ResumeMatchingAgent


class McpServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "a.txt").write_text("Python SQL", encoding="utf-8")
        self.server = McpServer(FilesystemService(ServerConfig(self.root)))

    def tearDown(self): self.temp.cleanup()

    def rpc(self, method, params=None, request_id=1):
        return self.server.dispatch({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})

    def test_discovery_and_resource_read(self):
        self.assertEqual(self.rpc("initialize")["result"]["serverInfo"]["name"], "resume-filesystem")
        self.assertIn("a.txt", str(self.rpc("resources/list")["result"]))
        self.assertEqual(self.rpc("resources/read", {"uri": "file:///a.txt"})["result"]["contents"][0]["text"], "Python SQL")

    def test_batch_and_path_protection(self):
        result = self.rpc("tools/call", {"name": "batch_process", "arguments": {"paths": ["a.txt", "missing.txt"]}})["result"]["structuredContent"]
        self.assertEqual(len(result["processed"]), 1)
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(self.rpc("tools/call", {"name": "read_file", "arguments": {"path": "../secret"}})["error"]["code"], -32003)


class AgentWorkflowTests(unittest.TestCase):
    def test_agent_uses_mcp_batch_workflow(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "ada.txt").write_text("Python engineer with SQL and Docker", encoding="utf-8")
            Path(root, "ben.txt").write_text("Graphic designer", encoding="utf-8")
            with McpClient(root) as client:
                ranked = ResumeMatchingAgent(client).match("Python SQL engineer")
            self.assertEqual(ranked[0]["resume"], "ada.txt")
            self.assertGreater(ranked[0]["score"], ranked[1]["score"])


if __name__ == "__main__": unittest.main()
