# Evaluation Metrics

## Pass@k Metric

### Definition

Pass@k measures the probability that at least one of k generated code samples passes all verification stages.

### Three-Stage Verification

#### 1. Compilation Check (`pass@k-compile`)

- **Tool**: Icarus Verilog
- **Command**: `iverilog -o test_output solution.v`
- **Success Criteria**: No syntax errors
- **Common Failures**:
  - Missing semicolons
  - Incorrect module syntax
  - Undeclared signals

#### 2. Functional Correctness (`pass@k-func-corr`)

- **Tools**:
  - Python testbench scripts
  - Verilog testbenches with vvp
- **Success Criteria**: All test cases pass
- **Common Failures**:
  - Wrong logic implementation
  - Incorrect bit widths
  - Missing edge cases

#### 3. Synthesizability (`pass@k-sync`)

- **Tool**: Yosys
- **Command**: `yosys -p "read_verilog file.v; synth"`
- **Success Criteria**: Successful synthesis to gates
- **Common Failures**:
  - Non-synthesizable constructs
  - Timing loops
  - Unsupported primitives

## Problem Difficulty Levels

### Basic (Problems 1-4)

- Simple combinational logic
- Wire declarations
- Basic gates
- **Avg Pass Rate**: ~80%

### Intermediate (Problems 5-12)

- Sequential logic
- Counters
- Simple FSMs
- Memory blocks
- **Avg Pass Rate**: ~50%

### Advanced (Problems 13-17)

- Complex FSMs
- Arithmetic units
- Pattern detection
- **Avg Pass Rate**: ~30%

## Results Summary

### Best Performers by Difficulty

| Difficulty   | Top Model             | Overall Pass@10 Score |
| :----------- | :-------------------- | :-------------------: |
| Basic        | CodeLlama-7B-instruct |         0.93          |
| Intermediate | Deepseek-7B-instruct  |         0.83          |
| Advanced     | Deepseek-7B-instruct  |         0.86          |

### Overall Rankings

1. **CodeGen-16B-FT**: 
2. **Deepseek-coder-7B-v1.5**: 0.8518
3. **Claude Opus 4**: 0.7777
5. **CodeLlama-7B**: 0.7592
4. **Qwen-coder-2.5-7B**: 0.7407
2. **GPT-4.1**: 0.7222

## Evaluation Parameters

- **Temperature**: 0.2 (all models)
- **Max Tokens**: 1024
- **Samples per Problem**: 10
- **Total Problems**: 18
- **Timeout**: 30s per test
