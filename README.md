# Enhancement: Self-Improving RLFT for Verilog Generation

This branch presents the **enhanced version** of the VeriGen project — reducing dependency on massive training data via reinforcement learning and self-improving data loops.

---

### 🎯 Goal

Train `CodeLlama-7B-Instruct` on just **20–100 handpicked Verilog examples** using **Reinforcement Learning Fine-Tuning (RLFT)** and grow the dataset online with automatically generated and verified examples.

---

### 🧪 Reinforcement Learning Setup

- **Initial Dataset**: 20–100 Verilog modules (e.g., gates, counters, FSMs)
- **Reward Function**:
  - ✅ Compiles with Icarus Verilog
  - ✅ Passes testbench
  - 📉 Shorter code / higher abstraction
- **Loop**:
  1. Fine-tune model via RL
  2. Generate new prompts
  3. Validate outputs (compile + simulate)
  4. Add verified samples to dataset
  5. Repeat

---

### Project Setup

To clone the repository:
```bash
git clone https://github.com/noobsiecoder/VeriGenLLM-v2.git
```

These tools are required to run evals for baseline LLMs (locally and in cloud):
  1. [Icarus Verilog](steveicarus.github.io/iverilog/)
  1. [Yosys](https://yosyshq.net/yosys/)

To install them:
```bash
# On MacOS, you can directly install by using homebrew:
brew install icarus-verilog
brew install yosys
```

> Note: For Linux distros and Windows OS, please check the official documentation.


This project uses [uv](https://docs.astral.sh/uv) as its Python package and project managing tool. To install it:

```bash
# On MacOS and Linux distros
curl -LsSf https://astral.sh/uv/install.sh | sh

# WindowsOS (Powershell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

### Installation

Installing all the dependencies/packages:
```bash
# With uv (recommended)
uv pip install -e .
# (Optional) For exact version sync:
uv pip sync

# With pip
# Creat virtual enviroment
python -m venv <venv-folder-name>
# Activate virtual enviroment
source <venv-folder-name>/bin/activate # Linux distros and MacOS
source <venv>\Scripts\activate.bat # Windows OS via cmd.exe
# Install all dependencies
pip install -e '.[dev]'
```

To run test, you need [pytest](https://docs.pytest.org/en/stable/) module installed. To install it:
```bash
# With uv (recommended)
uv add --dev pytest

# With pip
pip install -U pytest # install globally
# Check if installed
pytest --version
```

---

### Run

To run pass@k metric **EVALS** locally:
```bash
# With uv (recommended)
uv run src/verigenllm_v2/evals/pass-k.py

# With pip and python3 after activating virtual enviroment
python3 src/verigenllm_v2/evals/pass-k.py
```

To run **TESTS** locally:
```bash
# With uv (recommended)
uv run pytest

# With pip
pytest
```

---

### Baseline LLMs

1. [OpenAI GPT 4.1](https://platform.openai.com/docs/models/gpt-4.1)
1. [Claude Opus 4](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names)
1. [Gemini 2.5 Pro (Unavailable)](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro)

### Token Cost Summary

|   Model Name    | Input Token | Output Token | Input Token Cost | Output Token Cost | Total Credit |
| :-------------: | :---------: | :----------: | :--------------: | :---------------: | :----------: |
| Claude Opus 4\* |    8,728    |    60,536    |      $0.13       |       $4.54       |    $4.67     |
| OpenAI GPT 4.1  |     862     |    29,760    |      $0.002      |      $0.231       |    $0.233    |

### Note*

Totally 18 problem sets were used. From the `<project>/dataset/test/example-<nn>`, the each prompt from the `prompts` field in the `metadata.json` file, was chosen randomly.

Unlike OpenAI's `chat.completions.create` object, Claude API doesn't support the `n_sample` parameter in its API call (yet - noted on Jul 30th, 2025). Thus, the input-output token size of the Claude's model looks significantly higher than the OpenAI's input-output token size.

In this research, it ran in loop for *n* (here `n = 10`) times to evaluate the [Pass@k metrics](https://www.datacamp.com/tutorial/humaneval-benchmark-for-evaluating-llm-code-generation-capabilities).

---

### Contact

1. [noobsiecoder@gmail.com](mailto:noobsiecoder@gmail.com)

---