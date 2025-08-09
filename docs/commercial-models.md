# Commercial LLMs

## Evaluated Models

### 1. Claude Opus 4

- **Provider**: Anthropic
- **Model ID**: `claude-opus-4-20250514`
- **Context Window**: 200K tokens
- **Max Output**: 4,096 tokens
- **Pricing**:
  - Input: $15/M tokens
  - Output: $75/M tokens
- **Key Features**:
  - Superior reasoning capabilities
  - Excellent code understanding
  - Strong instruction following
- **Verilog Performance**: Excellent on all difficulty levels
- **API Limitation**: No native `n_samples` parameter

### 2. GPT-4.1

- **Provider**: OpenAI
- **Model ID**: `gpt-4.1`
- **Context Window**: 128K tokens
- **Max Output**: 4,096 tokens
- **Pricing**:
  - Input: $10/M tokens
  - Output: $30/M tokens
- **Key Features**:
  - Industry-leading performance
  - Native multi-sample generation
  - Consistent output quality
- **Verilog Performance**: Best on advanced problems

### 3. Gemini 2.5 Pro _(Currently unavailable)_

- **Provider**: Google
- **Model ID**: `gemini-2.5-pro`
- **Context Window**: 1M tokens
- **Max Output**: 8,192 tokens
- **Pricing**: Currently free (Tier 1)
- **Status**: Disabled due to API reliability issues
- **Features**:
  - Batch API support
  - Multi-modal capabilities
  - Extended context

## Model Comparison

| Model               | Parameters | Verilog-Specific | Memory (GB) | Avg Pass@8 compile | Avg Pass@8 Func-corr | Avg Pass@8 synth |
| :------------------ | :--------: | :--------------: | :---------: | :----------------: | :------------------: | :--------------: |
| Claude Opus4 (API)  |     -      |        ❌        |      -      |       0.8333       |        0.6667        |      0.8333      |
| Gemini 2.5Pro (API) |     -      |        ❌        |      -      |         -          |          -           |        -         |
| OpenAI GPT4.1 (API) |     -      |        ❌        |      -      |       0.7778       |        0.6111        |      0.7778      |

## Cost Analysis

| Model          | 18 Problems (10 samples) | Input Tokens | Output Tokens | Total Cost |
| :------------- | :----------------------: | :----------: | :-----------: | :--------: |
| Claude Opus 4  |            ✅            |    8,728     |    60,536     |   $4.67    |
| Gemini 2.5 Pro |            ❌            |      -       |       -       |   Free\*   |
| GPT-4.1        |            ✅            |     862      |    29,760     |   $0.23    |

\*Note: Claude's higher token count due to multiple API calls for n=10 samples

## API Features Comparison

| Feature              | Claude | GPT-4 | Gemini |
| :------------------- | :----: | :---: | :----: |
| Multi-sample (`n>1`) |   ❌   |  ✅   |   ✅   |
| Batch API            |   ❌   |  ❌   |   ✅   |
| Streaming            |   ✅   |  ✅   |   ✅   |
| System Messages      |   ✅   |  ✅   |   ✅   |

## Integration Notes

- All models use consistent response format
- System prompt: "You are a Verilog code generator. Output only synthesizable Verilog code."
- Temperature: 0.2 (consistent, focused generation)
- Rate limiting handled automatically
