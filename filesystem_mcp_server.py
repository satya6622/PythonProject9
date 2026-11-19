#!/usr/bin/env python3
"""A small, production-oriented MCP filesystem server using stdio transport.

The server intentionally uses only the Python standard library.  Each input
line is a JSON-RPC 2.0 request and each response is written as one JSON line,
which makes it usable by MCP clients and easy to containerise.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse

JSONRPC = "2.0"
SERVER_INFO = {"name": "resume-filesystem", "version": "1.0.0"}


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code, self.message, self.data = code, message, data
        super().__init__(message)


@dataclass(frozen=True)
class ServerConfig:
    root: Path
    max_file_bytes: int = 1_000_000
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".json", ".csv", ".pdf", ".docx")


class FilesystemService:
    """Filesystem operations constrained to one configured root directory."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.root = config.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str = ".") -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise RpcError(-32003, "Path escapes the configured root") from exc
        return candidate

    def list_directory(self, path: str = ".") -> list[dict[str, Any]]:
        directory = self.resolve(path)
        if not directory.is_dir():
            raise RpcError(-32004, "Directory not found", {"path": path})
        return [
            {"path": str(item.relative_to(self.root)), "type": "directory" if item.is_dir() else "file",
             "size": item.stat().st_size, "modified": item.stat().st_mtime}
            for item in sorted(directory.iterdir(), key=lambda entry: entry.name.lower())
        ]

    def read_file(self, path: str) -> dict[str, Any]:
        file_path = self.resolve(path)
        if not file_path.is_file():
            raise RpcError(-32004, "File not found", {"path": path})
        if file_path.stat().st_size > self.config.max_file_bytes:
            raise RpcError(-32005, "File exceeds configured size limit", {"path": path})
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RpcError(-32006, "Only UTF-8 text files can be read", {"path": path}) from exc
        return {"path": str(file_path.relative_to(self.root)), "content": content,
                "mimeType": mimetypes.guess_type(file_path.name)[0] or "text/plain"}

    def metadata(self, path: str) -> dict[str, Any]:
        file_path = self.resolve(path)
        if not file_path.exists():
            raise RpcError(-32004, "Path not found", {"path": path})
        stat = file_path.stat()
        return {"path": str(file_path.relative_to(self.root)), "type": "directory" if file_path.is_dir() else "file",
                "size": stat.st_size, "modified": stat.st_mtime}

    def search_files(self, query: str, path: str = ".") -> list[dict[str, Any]]:
        base = self.resolve(path)
        if not base.is_dir():
            raise RpcError(-32004, "Directory not found", {"path": path})
        matches = []
        for item in base.rglob("*"):
            if item.is_file() and query.lower() in item.name.lower():
                matches.append(self.metadata(str(item.relative_to(self.root))))
        return matches

    def batch_process(self, paths: list[str]) -> dict[str, Any]:
        results, errors = [], []
        for path in paths:
            try:
                results.append(self.read_file(path))
            except RpcError as exc:
                errors.append({"path": path, "code": exc.code, "message": exc.message})
        return {"processed": results, "errors": errors, "total": len(paths)}

    def watch_directory(self, path: str = ".", timeout_seconds: float = 0) -> dict[str, Any]:
        """Poll once or until timeout, returning files added since this call began.

        Long-running event subscriptions do not fit request/response JSON-RPC;
        clients can poll this tool at their desired interval.
        """
        directory = self.resolve(path)
        if not directory.is_dir():
            raise RpcError(-32004, "Directory not found", {"path": path})
        before = {p.relative_to(self.root) for p in directory.rglob("*") if p.is_file()}
        deadline = time.monotonic() + max(0, min(timeout_seconds, 30))
        while time.monotonic() < deadline:
            time.sleep(0.1)
        after = {p.relative_to(self.root) for p in directory.rglob("*") if p.is_file()}
        return {"path": path, "newFiles": [str(p) for p in sorted(after - before)], "polledForSeconds": max(0, min(timeout_seconds, 30))}

    def resource_uri(self, path: str) -> str:
        return "file://" + quote(path)

    def path_from_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise RpcError(-32602, "Unsupported resource URI", {"uri": uri})
        return unquote(parsed.path).lstrip("/")

    def resources(self) -> list[dict[str, Any]]:
        resources = [{"uri": "file:///", "name": "Filesystem root", "mimeType": "inode/directory"}]
        for item in self.root.rglob("*"):
            if item.is_file() and item.stat().st_size <= self.config.max_file_bytes:
                rel = str(item.relative_to(self.root))
                resources.append({"uri": self.resource_uri(rel), "name": rel,
                                  "mimeType": mimetypes.guess_type(item.name)[0] or "text/plain"})
        return resources


TOOL_DEFINITIONS = [
    ("list_directory", "List files and folders under the configured root.", {"path": {"type": "string", "default": "."}}),
    ("read_file", "Read a UTF-8 text file.", {"path": {"type": "string"}}, ["path"]),
    ("get_file_metadata", "Get safe metadata for a path.", {"path": {"type": "string"}}, ["path"]),
    ("search_files", "Find files whose names contain a query.", {"query": {"type": "string"}, "path": {"type": "string", "default": "."}}, ["query"]),
    ("watch_directory", "Poll a directory for new files (maximum wait: 30 seconds).", {"path": {"type": "string", "default": "."}, "timeout_seconds": {"type": "number", "default": 0}}),
    ("batch_process", "Read multiple UTF-8 files and return per-file errors without failing the batch.", {"paths": {"type": "array", "items": {"type": "string"}}}, ["paths"]),
]


class McpServer:
    def __init__(self, service: FilesystemService) -> None:
        self.service = service
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "initialize": self.initialize, "resources/list": lambda _: {"resources": service.resources()},
            "resources/read": self.read_resource, "resources/templates/list": self.templates,
            "tools/list": self.list_tools, "tools/call": self.call_tool, "ping": lambda _: {},
        }

    def initialize(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"protocolVersion": "2024-11-05", "capabilities": {"resources": {"listChanged": False}, "tools": {}}, "serverInfo": SERVER_INFO}

    def templates(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"resourceTemplates": [{"uriTemplate": "file:///{path}", "name": "Filesystem file", "description": "A UTF-8 file below the configured root."}]}

    def read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        path = self.service.path_from_uri(self._required(params, "uri"))
        if not path:
            return {"contents": [{"uri": "file:///", "mimeType": "application/json", "text": json.dumps(self.service.list_directory())}]}
        data = self.service.read_file(path)
        return {"contents": [{"uri": params["uri"], "mimeType": data["mimeType"], "text": data["content"]}]}

    def list_tools(self, _: dict[str, Any]) -> dict[str, Any]:
        tools = []
        for entry in TOOL_DEFINITIONS:
            name, description, properties, *required = entry
            tools.append({"name": name, "description": description, "inputSchema": {"type": "object", "properties": properties, "required": required[0] if required else []}})
        return {"tools": tools}

    def call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name, arguments = self._required(params, "name"), params.get("arguments", {})
        calls = {"list_directory": lambda: self.service.list_directory(arguments.get("path", ".")),
                 "read_file": lambda: self.service.read_file(self._required(arguments, "path")),
                 "get_file_metadata": lambda: self.service.metadata(self._required(arguments, "path")),
                 "search_files": lambda: self.service.search_files(self._required(arguments, "query"), arguments.get("path", ".")),
                 "watch_directory": lambda: self.service.watch_directory(arguments.get("path", "."), arguments.get("timeout_seconds", 0)),
                 "batch_process": lambda: self.service.batch_process(self._required(arguments, "paths"))}
        if name not in calls:
            raise RpcError(-32601, "Unknown tool", {"name": name})
        result = calls[name]()
        return {"content": [{"type": "text", "text": json.dumps(result)}], "structuredContent": result, "isError": False}

    @staticmethod
    def _required(params: dict[str, Any], key: str) -> Any:
        if key not in params:
            raise RpcError(-32602, "Missing required parameter", {"parameter": key})
        return params[key]

    def dispatch(self, request: Any) -> dict[str, Any] | None:
        if not isinstance(request, dict) or request.get("jsonrpc") != JSONRPC or not isinstance(request.get("method"), str):
            return self.error(request.get("id") if isinstance(request, dict) else None, -32600, "Invalid Request")
        request_id, method = request.get("id"), request["method"]
        if method == "notifications/initialized":
            return None
        try:
            handler = self.handlers.get(method)
            if handler is None:
                raise RpcError(-32601, "Method not found", {"method": method})
            result = handler(request.get("params", {}))
            return None if "id" not in request else {"jsonrpc": JSONRPC, "id": request_id, "result": result}
        except RpcError as exc:
            return self.error(request_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # never leak a traceback to an MCP client
            return self.error(request_id, -32603, "Internal error", {"detail": str(exc)})

    @staticmethod
    def error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        error = {"code": code, "message": message}
        if data is not None: error["data"] = data
        return {"jsonrpc": JSONRPC, "id": request_id, "error": error}


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP filesystem server")
    parser.add_argument("--root", default=os.environ.get("MCP_FILESYSTEM_ROOT", "."))
    parser.add_argument("--max-file-bytes", type=int, default=int(os.environ.get("MCP_MAX_FILE_BYTES", "1000000")))
    args = parser.parse_args()
    server = McpServer(FilesystemService(ServerConfig(Path(args.root), args.max_file_bytes)))
    try:
        for line in sys.stdin:
            try:
                response = server.dispatch(json.loads(line))
            except json.JSONDecodeError:
                response = server.error(None, -32700, "Parse error")
            if response is not None:
                print(json.dumps(response), flush=True)
    except KeyboardInterrupt:
        # Stdio servers normally run until their client closes stdin.  Treat an
        # interactive Ctrl+C as a normal, quiet shutdown for demonstrations.
        pass


if __name__ == "__main__":
    main()
