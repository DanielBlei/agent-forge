"""
Unit tests for the actual tools in agent_forge.tools module.
"""

import tempfile
from pathlib import Path

import pytest

from agent_forge.tools import glob_files, grep_files, list_directory, read_file


def test_read_file():
    """Test read_file tool functionality."""
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Line 1\nLine 2\nLine 3")
        temp_path = f.name

    try:
        # Test reading full file
        result = read_file(temp_path)
        assert result["success"] is True
        assert "Line 1" in result["content"]
        assert "Line 2" in result["content"]
        assert "Line 3" in result["content"]
        assert result["lines"] == 3

        # Test reading with limit
        result = read_file(temp_path, limit=2)
        assert result["success"] is True
        assert "Line 1" in result["content"]
        assert "Line 2" in result["content"]
        assert "Line 3" not in result["content"]
        assert result["lines"] == 2
        assert result["truncated"] is True

        # Test non-existent file
        result = read_file("/nonexistent/path/file.txt")
        assert result["success"] is False
        assert "File not found" in result["error"]

    finally:
        Path(temp_path).unlink()


def test_glob_files():
    """Test glob_files tool functionality."""
    # Test with current directory
    result = glob_files("**/*.py")
    assert result["success"] is True
    assert isinstance(result["matches"], list)
    assert len(result["matches"]) >= 0  # Might be 0 depending on directory
    assert result["count"] == len(result["matches"])

    # Test with specific pattern
    result = glob_files("**/test_*.py")
    assert result["success"] is True
    # Should find our test files
    assert any("test_" in f for f in result["matches"])


def test_grep_files():
    """Test grep_files tool functionality."""
    # Create temporary files in the current directory for testing
    # (grep_files works relative to BASE_DIR)
    test_files = []
    try:
        # Create test files in current directory
        test_file1 = "test_grep_1.py"
        Path(test_file1).write_text("def hello():\n    print('Hello World')\n")
        test_files.append(test_file1)

        test_file2 = "test_grep_2.py"
        Path(test_file2).write_text("def goodbye():\n    print('Goodbye')\n")
        test_files.append(test_file2)

        # Test searching for pattern
        result = grep_files("print", include="test_grep_*.py")
        assert result["success"] is True
        assert result["count"] >= 2  # Should find both print statements

        # Test with regex
        result = grep_files("def \\w+", include="test_grep_*.py", regex=True)
        assert result["success"] is True
        assert result["count"] >= 2  # Should find both function definitions

    finally:
        # Clean up test files
        for file in test_files:
            p = Path(file)
            if p.exists():
                p.unlink()


def test_list_directory():
    """Test list_directory tool functionality."""
    # Test current directory
    result = list_directory(".")
    assert result["success"] is True
    assert isinstance(result["entries"], list)
    assert len(result["entries"]) > 0
    assert result["count"] == len(result["entries"])

    # Verify entries have correct structure
    for entry in result["entries"]:
        assert "path" in entry
        assert "type" in entry
        assert entry["type"] in ["file", "dir"]

    # Test non-existent directory
    result = list_directory("/nonexistent/directory")
    assert result["success"] is False
    assert "Directory not found" in result["error"]


def test_read_file_bare_filename_resolution():
    """read_file resolves bare filenames recursively under cwd."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", dir=Path.cwd(), prefix="bare_resolve_"
    ) as f:
        f.write("found via bare name")
        temp_path = Path(f.name)

    try:
        result = read_file(temp_path.name)
        assert result["success"] is True
        assert "found via bare name" in result["content"]
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
