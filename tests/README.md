# Agent-Forge Tests

This directory contains tests for the agent-forge project.

## Directory Structure

- `tests/` - Root test directory
  - `unit/` - Unit tests (fast, isolated tests)
  - `integration/` - Integration tests (end-to-end tests)
  - `run_tests.py` - Test runner script

## Running Tests

### Using Makefile (Recommended)

The project includes a Makefile with convenient test commands:

```bash
make help              # Show available commands
make test              # Run all tests
make test-unit         # Run only unit tests
make test-integration  # Run only integration tests
make clean             # Clean up temporary files
```

### Using Python Directly

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py unit

# Run only integration tests
python tests/run_tests.py integration
```

### Using pytest Directly

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/unit/test_chat_flow.py -v

# Run tests with specific pattern
python -m pytest tests/ -k "test_chat" -v
```

## Test Organization

### Unit Tests
Unit tests focus on testing individual components in isolation:

- `test_chat_flow.py` - Tests for the Chat class and tool call handling
- `test_multi_tool.py` - Tests for multi-tool behavior and agent loop flow

### Integration Tests
Integration tests will test the complete system workflow:

- End-to-end agent loop testing
- Real API interactions (mocked)
- Complete conversation flows

## Writing New Tests

### Unit Test Example
```python
def test_some_functionality():
    """Test description."""
    # Setup
    chat = Chat(system="test")
    
    # Exercise
    chat.user("test message")
    
    # Verify
    assert len(chat) == 1
    assert chat.token_count() > 0
```

### Integration Test Example
```python
async def test_complete_workflow():
    """Test complete agent workflow."""
    # Setup with mocks
    chat = Chat(system="test")
    
    # Exercise complete flow
    # ...
    
    # Verify final state
    assert expected_outcome
```

## Test Requirements

- Python 3.12+
- pytest
- All agent-forge dependencies

Install test requirements:
```bash
pip install pytest
```
