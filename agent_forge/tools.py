from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from openai.types.chat import ChatCompletionToolParam

EXCLUDED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".idea"}


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _resolve(path: str) -> Path:
    """Resolve a path to an absolute Path.

    Resolution order:
    1. Absolute paths are returned as-is.
    2. Relative paths are resolved against cwd.
    3. Bare filenames (no directory component) trigger a recursive rglob under
       cwd as a fallback.  If multiple matches exist, the shallowest one (fewest
       path parts) is used and a warning is logged so callers know the choice
       was ambiguous.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    full = Path.cwd() / p
    if full.exists():
        return full
    if len(p.parts) == 1:
        hits = sorted(
            (h for h in Path.cwd().rglob(p.name) if not _is_excluded(h)),
            key=lambda h: len(h.parts),
        )
        if hits:
            if len(hits) > 1:
                logger.warning(
                    "_resolve: '%s' matched %d files — using shallowest: %s",
                    path, len(hits), hits[0],
                )
            return hits[0]
    return full


def read_file(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read a file and return its contents.

    Args:
        path: Absolute, relative, or bare filename. Bare filenames are searched recursively.
        limit: Optional maximum number of lines to return.

    Returns:
        dict with 'success', 'content', 'lines', and optional 'truncated' keys.
    """
    try:
        full_path = _resolve(path)

        if not full_path.exists():
            return {
                "success": False,
                "error": f"File not found: {path}",
                "suggestions": [
                    "Try specifying the full absolute path",
                    "Check if the filename is correct",
                    "Use glob_files to search for similar filenames",
                    "Use list_directory to explore the directory structure",
                ],
            }

        if not full_path.is_file():
            return {
                "success": False,
                "error": f"Path is a directory, not a file: {path}",
                "suggestions": ["Use list_directory to explore its contents"],
            }

        with full_path.open(encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)
        if limit and total > limit:
            return {
                "success": True,
                "content": "".join(lines[:limit]),
                "lines": limit,
                "total_lines": total,
                "truncated": True,
            }

        return {
            "success": True,
            "content": "".join(lines),
            "lines": total,
        }
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Error reading {path}: {e}"}


def glob_files(pattern: str, path: str | None = None) -> dict[str, Any]:
    """Find files matching a glob pattern recursively.

    Args:
        pattern: Glob pattern (e.g., '**/*.py', '*.py').
        path: Optional directory to search in (defaults to current directory).

    Returns:
        dict with 'success', 'matches' list, and 'count'.
    """
    try:
        search_path = Path.cwd() if path is None else _resolve(path)
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


def _search_file(file_path: Path, matches: Any, context: int) -> list[dict[str, Any]]:
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
        search_path = Path.cwd() if path is None else _resolve(path)

        compiled = re.compile(pattern, re.IGNORECASE) if regex else None

        def matches(text: str) -> bool:
            if compiled:
                return bool(compiled.search(text))
            return pattern.lower() in text.lower()

        if search_path.is_file():
            files_to_search: Any = [search_path]
        elif search_path.is_dir():
            if include:
                glob_pattern = include if "**" in include else f"**/{include}"
                files_to_search = (f for f in search_path.glob(glob_pattern) if f.is_file() and not _is_excluded(f))
            else:
                files_to_search = (f for f in search_path.rglob("*") if f.is_file() and not _is_excluded(f))
        else:
            return {"success": False, "error": f"Invalid path: {path}"}

        results: list[dict[str, Any]] = []
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
        target_path = Path.cwd() if path is None else _resolve(path)
        entries = [
            {"path": str(Path(target_path) / ent.name), "type": "dir" if ent.is_dir() else "file"}
            for ent in sorted(os.scandir(target_path), key=lambda ent: (not ent.is_dir(), ent.name))
        ]
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
            "description": (
                "Read the contents of a file. Accepts absolute paths, relative paths, or bare filenames. "
                "Bare filenames are searched recursively under the current directory. "
                "Prefer this over glob_files when you already know the filename. "
                "Call once per file — to read multiple files, call this tool multiple times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file. Absolute, relative, or bare filename.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return from the start of the file.",
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
            "description": (
                "Find files by name or extension using a single glob pattern. Searches recursively by default. "
                "One pattern per call — to search for multiple patterns, call this tool multiple times. "
                "Use read_file instead if you already know the filename. "
                "Examples: '*.py' for all Python files, 'src/**/*.ts' for TypeScript in src/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.py', 'src/**/*.ts', 'Dockerfile*'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (defaults to current directory).",
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
            "description": (
                "Search file contents for a text or regex pattern. "
                "Use 'include' to scope the search to specific file types (e.g. '*.py'). "
                "Returns matching lines with optional surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (defaults to current directory).",
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob pattern to filter files, e.g. '*.py', '**/test_*.py'.",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Treat pattern as regex (default: true). Set false for literal search.",
                    },
                    "context": {
                        "type": "integer",
                        "description": "Number of lines of context to include around each match (default: 0).",
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
            "description": (
                "List files and subdirectories at a path. "
                "Use this to explore directory structure before reading or searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (defaults to current directory).",
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
