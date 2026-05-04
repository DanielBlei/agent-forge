from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionToolParam

EXCLUDED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".idea"}


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def read_file(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read a file and return its contents.

    Args:
        path: Path to the file to read.
        limit: Optional maximum number of lines to return.

    Returns:
        dict with 'success', 'content', 'lines', and optional 'truncated' keys.
    """
    try:
        full_path = _resolve_path(path)
        with full_path.open(encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)
        if limit and total > limit:
            content = "".join(lines[:limit])
            return {
                "success": True,
                "content": content,
                "lines": limit,
                "total_lines": total,
                "truncated": True,
            }

        content = "".join(lines)
        return {
            "success": True,
            "content": content,
            "lines": total,
        }
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Error reading {path}: {e}"}


def glob_files(pattern: str, path: str | None = None) -> dict[str, Any]:
    """Find files matching a glob pattern recursively.

    Args:
        pattern: Glob pattern (e.g., '**/*.py', '*.py'). Defaults to recursive if no **.
        path: Optional directory to search in (defaults to current directory).

    Returns:
        dict with 'success', 'matches' list, and 'count'.
    """
    try:
        search_path = Path.cwd() if path is None else _resolve_path(path)
        if "**" not in pattern:
            pattern = f"**/{pattern}"
        matches = [m for m in search_path.glob(pattern) if not _is_excluded(m)]
        files = sorted([str(m.relative_to(Path.cwd())) for m in matches])
        return {
            "success": True,
            "matches": files,
            "count": len(files),
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Error with pattern '{pattern}': {e}"}


def _search_file(
    file_path: Path,
    matches: Any,
    context: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    try:
        with file_path.open(encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for i, line in enumerate(lines, 1):
            if matches(line):
                start = max(0, i - context - 1)
                end = min(len(lines), i + context)
                results.append(
                    {
                        "file": str(file_path.relative_to(Path.cwd())),
                        "line": i,
                        "content": line.rstrip(),
                        "context": "".join(lines[start:end]),
                    }
                )
    except (OSError, PermissionError):
        pass
    return results


def grep_files(
    pattern: str,
    path: str | None = None,
    include: str | None = None,
    regex: bool = True,
    context: int = 0,
) -> dict[str, Any]:
    """Search file contents for a pattern.

    Args:
        pattern: Text or regex pattern to search for.
        path: Optional file or directory path to search in.
        include: Optional glob pattern to filter files (e.g., '*.py').
        regex: If True, treat pattern as regex (default). If False, literal string.
        context: Number of context lines to include around matches.

    Returns:
        dict with 'success', 'matches' list, and 'count'.
    """
    try:
        search_path = Path.cwd() if path is None else _resolve_path(path)

        flags = re.IGNORECASE if regex else 0
        compiled = re.compile(pattern, flags) if regex else None

        results: list[dict[str, Any]] = []

        def matches(text: str) -> bool:
            if compiled:
                return bool(compiled.search(text))
            return pattern.lower() in text.lower()

        if search_path.is_file():
            files_to_search = [search_path]
        elif search_path.is_dir():
            if include:
                if "**" not in include:
                    include = f"**/{include}"
                files_to_search = (f for f in search_path.glob(include) if f.is_file() and not _is_excluded(f))
            else:
                files_to_search = (f for f in search_path.rglob("*") if f.is_file() and not _is_excluded(f))
        else:
            return {"success": False, "error": f"Invalid path: {path}"}

        for file_path in files_to_search:
            results.extend(_search_file(file_path, matches, context))

        return {
            "success": True,
            "matches": results,
            "count": len(results),
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Error searching for '{pattern}': {e}"}


def list_directory(path: str | None = None) -> dict[str, Any]:
    """List directory contents with full paths.

    Args:
        path: Optional directory path (defaults to current directory).

    Returns:
        dict with 'success', 'entries' list with 'path' and 'type' keys.
    """
    try:
        target = Path.cwd() if path is None else _resolve_path(path)
        entries = []
        for entry in sorted(os.scandir(target), key=lambda e: (not e.is_dir(), e.name)):
            full_path = str(Path(target) / entry.name)
            entries.append(
                {
                    "path": full_path,
                    "type": "dir" if entry.is_dir() else "file",
                }
            )
        return {
            "success": True,
            "entries": entries,
            "count": len(entries),
        }
    except FileNotFoundError:
        return {"success": False, "error": f"Directory not found: {path}"}
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Error listing {path}: {e}"}


TOOL_DEFINITIONS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use this to view code, config, or text files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of lines to return",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Find files matching a glob pattern. Use this to discover files by name or extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py', 'src/*.ts')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional directory to search in",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": "Search file contents for a pattern. Use this to find code, text, or specific patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional file or directory to search in",
                    },
                    "include": {
                        "type": "string",
                        "description": "Optional glob pattern to filter files (e.g., '*.py')",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Treat pattern as regex (default: true)",
                    },
                    "context": {
                        "type": "integer",
                        "description": "Number of context lines to include around matches (default: 0)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the contents of a directory with full paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list",
                    },
                },
            },
        },
    },
]

TOOL_HANDLERS: dict[str, Any] = {
    "read_file": read_file,
    "glob_files": glob_files,
    "grep_files": grep_files,
    "list_directory": list_directory,
}
