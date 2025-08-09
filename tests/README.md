# Tests Directory

Unit tests for validating the Verilog evaluation framework using pytest.

## 🧪 Overview

These tests ensure the testbench validation functions work correctly by testing both positive and negative cases for Verilog code generation problems.

## 📁 Test Files

- **`test_wire_declare_testbench.py`** - Tests for Problem 1 (Simple wire declaration)
- **`test_wire_assign_testbench.py`** - Tests for Problem 2 (Wire assignment)

## 🚀 Running Tests

### Run all tests:
```bash
# With uv (recommended)
uv run pytest tests/

# With pytest directly
pytest tests/
```

### Run specific test file:
```bash
uv run pytest tests/test_wire_declare_testbench.py
```

### Run with verbose output:
```bash
uv run pytest -v tests/
```

## 🔍 Test Structure

Each test file typically contains:

1. **Positive Tests** (`test_check_code_with_proper_code`)
   - Validates correct Verilog solutions pass

2. **Negative Tests** (`test_check_code_with_incorrect_code`)
   - Ensures incorrect solutions are rejected

3. **Integration Tests** (`test_real_scenario`)
   - Simulates the full evaluation pipeline
   - Creates temporary files and runs actual testbench scripts

## 📋 Test Patterns

### Basic Validation Test
```python
def test_check_code_with_proper_code():
    code = """<expected_verilog_code>"""
    assert check_code(code)
```

### End-to-End Test
```python
def test_real_scenario():
    # 1. Create temporary Verilog file
    # 2. Run testbench.py script
    # 3. Verify successful execution
    # 4. Clean up temporary files
```

## ⚡ Quick Tips

- Tests import `check_code()` from actual problem testbenches
- Each test validates specific acceptance criteria
- Negative tests check common LLM mistakes
- Integration tests ensure the full pipeline works

## 🛠️ Adding New Tests

1. Create `test_<problem_name>_testbench.py`
2. Import the problem's `check_code` function
3. Add positive and negative test cases
4. Consider adding end-to-end tests for critical paths