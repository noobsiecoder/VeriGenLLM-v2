# Verilog Code Generation LLM Evaluation Framework

This repository contains an evaluation framework for assessing Large Language Models' (LLMs) capabilities in generating Verilog hardware description language code, based on the VeriGen research paper.

## Project Overview

This project evaluates multiple state-of-the-art LLMs on their ability to generate syntactically correct and functionally accurate Verilog code from natural language prompts. The evaluation includes both open-source models (CodeGen, DeepSeek-Coder, Qwen-Coder) and commercial models (Claude, GPT-4, Gemini).

## Directory Structure

```
dataset/
├── evals/                      # All evaluation results
│   ├── comparison_pass_5/      # Comparative analysis across models
│   │   ├── ...
│   │   └── summary.png
│   ├── comparison_pass_8/      # Comparative analysis across models
│   │   ├── ...
│   │   └── summary.png
│   ├── comparison_pass_10/     # Comparative analysis across models
│   │   ├── ...
│   │   └── summary.png
|   ├── claude-opus-4-20250514  # Scores for each n samples per model
|   └── ...                     # More models
|
├── models/                     # Raw model responses
│   ├── claude/
│   │   ├── claude-response-n10.json
│   ├── codellama/
│   │   └── codellama-response-n10.json
│   ├── deepseek-coder/
│   │   └── deepseek-coder-response.json
│   ├── gemini/
│   │   └── gemini-response-n6.json
│   ├── openai/
│   │   └── openai-response-n10.json
│   └── qwen-coder/
│   │   └── qwen-coder-response-n10.json
│   └── selected-prompts.json
│
└── test/                       # Problem set (18 Verilog challenges)
    ├── example_01/             # Wire declaration in Verilog
    ├── example_02/             # Simple wire declaration
    ├── example_03/             # 2-input AND gate
    ├── example_04/             # 3-bit priority encoder
    ├── example_05/             # 2-input multiplexer
    ├── example_06/             # Half adder
    ├── example_07/             # 1-to-12 counter
    ├── example_08/             # LFSR with taps at 3 and 5
    ├── example_09/             # FSM with two states
    ├── example_10/             # Shift left and rotate
    ├── example_11/             # Random Access Memory
    ├── example_12/             # Permutation
    ├── example_13/             # Truth table
    ├── example_14/             # Signed 8-bit adder with overflow
    ├── example_15/             # Counter with enable signal
    ├── example_16/             # FSM to recognize "101"
    ├── example_17/             # 64-bit arithmetic shift register
    ├── example_18/             # ABRO FSM
```

## Problem Set Structure

Each problem in the `test/` directory contains:
- **Verilog solution file** - The reference implementation
- **Testbench file** - For functional verification
- **metadata.json** - Contains:
  - Multiple prompt variations (different phrasings of the same problem)
  - Difficulty level (basic/intermediate/advanced)
  - Problem topic/category

### Example metadata.json:
```json
{
    "prompts": [
        "Declare a simple wire in Verilog",
        "A simple wire in Verilog",
        "Show me a basic Verilog example declaring a wire.",
        "Verilog example with just a wire.",
        "Give me a tiny Verilog module that declares a wire.",
        "Verilog declaration with one wire."
    ],
    "tags": {
        "difficulty": "basic",
        "topic": "wire declaration"
    }
}
```

## Evaluated Models

### Open-Source Models
- **CodeLlama-7B-Instruct** - Meta's code-specialized LLaMA variant
- **DeepSeek-Coder-7B-Instruct** - DeepSeek's code generation model
- **Qwen-Coder-2.5-7B-Instruct** - Alibaba's Qwen coder model

### Commercial Models
- **Claude Opus 4** - Anthropic's latest model
- **GPT-4.1** - OpenAI's flagship model
- **Gemini 2.5** - Google's multimodal model (Not used)

## Evaluation Metrics

The evaluation framework assesses:
1. **Syntactic Correctness** - Whether the generated Verilog compiles
2. **Functional Correctness** - Whether the code passes testbench verification
3. **Synthesizable** - If the verilog code is synthasizable under `Yosys` tool.

## Problem Difficulty Distribution

- **Basic (Problems 1-4)**: Simple combinational logic, basic syntax
- **Intermediate (Problems 5-12)**: Sequential logic, counters, FSMs
- **Advanced (Problems 13-17)**: Complex state machines, arithmetic units

## Usage

1. **Running Evaluations**: Each model's responses are stored in JSON format in the `models/` directory.
2. **Viewing Results**: Comparative analyses are available in `evals/multi_comparison/`
3. **Understanding Performance**: 
   - CSV files contain detailed metrics.
   - PNG visualizations show performance comparisons.
   - Diff summaries highlight model-specific strengths/weaknesses to question difficulty.

## Key Findings

Based on the VeriGen research:
- Fine-tuned models show significant improvement over base models.
- Larger models generally perform better on complex problems.
- Prompt engineering significantly impacts output quality.
- Commercial models like GPT-4 excel at advanced problems but may struggle with intermediate complexity.

## Citation

This evaluation framework is based (and extended) on the VeriGen paper:
```
@misc{https://doi.org/10.48550/arxiv.2212.11140,
  doi = {10.48550/ARXIV.2212.11140},
  url = {https://arxiv.org/abs/2212.11140},
  author = {Thakur, Shailja and Ahmad, Baleegh and Fan, Zhenxing and Pearce, Hammond and Tan, Benjamin and Karri, Ramesh and Dolan-Gavitt, Brendan and Garg, Siddharth},
  title = {Benchmarking Large Language Models for Automated Verilog RTL Code Generation},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Contributing

To add new problems or models:
1. Add problems to `test/` with corresponding `metadata.json`
2. Store model responses in `models/<model_name>/`
3. Run evaluation scripts to generate comparative analyses

## License

MIT License

## Contact

Abhishek Sriram <noobsiecoder@gmail.com>
