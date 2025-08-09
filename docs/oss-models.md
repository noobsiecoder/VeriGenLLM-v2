# Open Source LLMs

## Evaluated Models

### 1. CodeLlama-7B-Instruct

- **Provider**: Meta
- **HuggingFace**: [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
- **Parameters**: 7B
- **Context Length**: 16,384 tokens
- **Training Data**: Code-focused dataset
- **Key Features**:
  - Instruction-tuned for code generation
  - Supports infilling capabilities
  - Good at following structured prompts
- **Verilog Performance**: Strong on basic to intermediate problems

### 2. DeepSeek-Coder-7B-Instruct-v1.5

- **Provider**: DeepSeek AI
- **HuggingFace**: [deepseek-ai/deepseek-coder-7b-instruct-v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
- **Parameters**: 7B
- **Context Length**: 16,384 tokens
- **Training Data**: 2T tokens of code & natural language
- **Key Features**:
  - Project-level code understanding
  - Repository-level context awareness
  - Optimized for multiple programming languages
- **Verilog Performance**: Good synthesis-aware generation

### 3. Qwen2.5-Coder-7B-Instruct

- **Provider**: Alibaba Cloud
- **HuggingFace**: [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- **Parameters**: 7B
- **Context Length**: 32,768 tokens
- **Training Data**: 5.5T tokens including code
- **Key Features**:
  - Latest Qwen architecture
  - Strong multilingual support
  - Extended context window
- **Verilog Performance**: Excellent on complex FSMs

### 4. CodeGen-16B (Fine-tuned)

- **Provider**: Salesforce (fine-tuned by shailja)
- **HuggingFace**: [shailja/fine-tuned-codegen-16B-Verilog](https://huggingface.co/shailja/fine-tuned-codegen-16B-Verilog)
- **Parameters**: 16B
- **Context Length**: 2,048 tokens
- **Training Data**: GitHub Verilog + textbooks
- **Key Features**:
  - Specifically fine-tuned on Verilog
  - Best Pass@k scores in evaluation
  - Handles advanced problems well
- **Verilog Performance**: Top performer across all difficulties

## Model Comparison

| Model               | Parameters | Verilog-Specific | Memory (GB) | Avg Pass@8 compile | Avg Pass@8 Func-corr | Avg Pass@8 synth |
| :------------------ | :--------: | :--------------: | :---------: | :----------------: | :------------------: | :--------------: |
| Claude Opus4 (API)  |     -      |        ❌        |      -      |       0.8333       |        0.6667        |      0.8333      |
| CodeGen-16B-FT      |    16B     |        ✅        |     ~30     |                    |                      |                  |
| CodeLlama-7B        |     7B     |        ❌        |     ~13     |       0.8864       |        0.3877        |      0.9988      |
| DeepSeek-7B         |     7B     |        ❌        |     ~13     |       0.9333       |        0.5889        |      0.9444      |
| OpenAI GPT4.1 (API) |     -      |        ❌        |      -      |       0.7778       |        0.6111        |      0.7778      |
| Qwen2.5-7B          |     7B     |        ❌        |     ~13     |       0.8321       |        0.3778        |      0.9444      |

## Usage Notes

- All models use FP16 precision for efficiency
- Requires CUDA-compatible GPU
- First run downloads model weights (~13-30GB)
- Use `device_map="auto"` for multi-GPU setups
