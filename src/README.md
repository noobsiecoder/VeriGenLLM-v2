# VeriGenLLM-v2 Source Code

This directory contains the core implementation for evaluating and benchmarking Large Language Models (LLMs) on Verilog code generation tasks.

## 📁 Directory Structure

```
src/
├── verigenllm_v2/
│   ├── cloud/
│   │   ├── gcp_storage.py          # Google Cloud Storage utilities
│   │   ├── read_all_questions.py   # Extract prompts from responses
│   │   └── responses_local_llm.py  # Run open-source LLMs in cloud (local)
│   ├── evals/
│   │   └── pass-k.py               # Pass@k metric evaluation pipeline
│   ├── llm_api/
│   │   ├── baselines.py            # Run commercial LLMs (Claude, OpenAI)
│   │   ├── claude_client.py        # Claude API wrapper
│   │   ├── gemini_client.py        # Gemini API wrapper
│   │   └── openai_client.py        # OpenAI API wrapper
│   └── utils/
│       └── logger.py               # Centralized logging system
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [Icarus Verilog](https://steveicarus.github.io/iverilog/) - For compilation checks
- [Yosys](https://yosyshq.net/yosys/) - For synthesis verification
- API keys for commercial LLMs (stored in `secrets/models-api.env`)

### Running Evaluations

1. **Generate responses from commercial LLMs:**
   ```bash
   uv run src/verigenllm_v2/llm_api/baselines.py
   ```

2. **Generate responses from open-source LLMs:**
   ```bash
   uv run src/verigenllm_v2/cloud/responses_local_llm.py
   ```

3. **Run Pass@k evaluation:**
   ```bash
   uv run src/verigenllm_v2/evals/pass_k.py
   ```

## 🔧 Component Overview

### LLM Clients (`llm_api/`)
- **Unified interface** for all LLM providers
- **System prompts** ensure synthesizable Verilog generation
- **Multiple sample generation** (n=10 by default)
- **Response standardization** for consistent evaluation

### Evaluation Pipeline (`evals/pass-k.py`)
Three-stage verification for each generated code:
1. **Compilation** - Icarus Verilog syntax check
2. **Functional Correctness** - Testbench simulation
3. **Synthesizability** - Yosys synthesis check

### Cloud Integration (`cloud/`)
- **GCS Upload** - Automatic backup of results and logs
- **Batch Processing** - Efficient handling of multiple prompts
- **Local LLM Support** - HuggingFace models with GPU acceleration

## 📊 Evaluation Metrics

**Pass@k**: Measures success rate across k generated samples
- `pass@k-compile`: Proportion that compile successfully
- `pass@k-func-corr`: Proportion passing functional tests  
- `pass@k-sync`: Proportion that synthesize correctly

## 🔐 Configuration

### API Keys (`secrets/models-api.env`)
```env
CLAUDE_API=your_claude_key
OPENAI_API=your_openai_key
GEMINI_API=your_gemini_key
HUGGINGFACE_TOKEN=your_hf_token
```

### GCS Credentials (`secrets/gcp-storage.json`)
Service account JSON for Google Cloud Storage access

## 📝 Logging

All components use centralized logging:
- **File logs**: `logs/{component}_{YYYY-MM-DD}.log`
- **Console output**: Real-time monitoring
- **Automatic rotation**: Daily log files

## 🧪 Supported Models

### Commercial
- Claude Opus 4
- GPT-4.1
- Gemini 2.5 Pro (not available)

### Open-Source
- CodeLlama-7B-Instruct
- DeepSeek-Coder-7B-Instruct
- Qwen2.5-Coder-7B-Instruct

## 💡 Development Tips

1. **Testing single prompts**: Modify the prompt range in baselines.py
2. **Adding new models**: Implement client in `llm_api/` following existing patterns
3. **Custom evaluation**: Extend `PassKMetrics` class in `pass-k.py`
4. **Debugging**: Check daily logs in `logs/` directory

## ⚠️ Notes

- Gemini API commented out due to issues ith API access
- Claude requires multiple API calls for n>1 samples
- Ensure sufficient GPU memory for local LLMs (~13GB per model)