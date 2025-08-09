# Setup Guide

## Prerequisites

### System Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 350GB+ disk space

### Required Tools

#### 1. Icarus Verilog

```bash
# macOS
brew install icarus-verilog

# Ubuntu/Debian
sudo apt-get install iverilog

# Verify installation
iverilog -v
```

#### 2. Yosys

```bash
# macOS
brew install yosys

# Ubuntu/Debian
sudo apt-get install yosys

# Verify installation
yosys -V
```

#### 3. UV Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/noobsiecoder/VeriGenLLM-v2.git
cd VeriGenLLM-v2
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -e '.[dev]'
```

### 3. Configure API Keys

Create `secrets/models-api.env`:

```env
CLAUDE_API=your_anthropic_key
OPENAI_API=your_openai_key
GEMINI_API=your_google_key
HUGGINGFACE_TOKEN=your_hf_token
```

### 4. GCS Setup (Optional)

For cloud storage backup:

1. Create GCS service account
1. Create a Google storage bucket and a VM instance
1. Download JSON key
1. Save as `secrets/gcp-storage.json`

## GPU Setup (For Local Models)

### NVIDIA Driver

```bash
# Check CUDA version
nvidia-smi

# Install CUDA toolkit if needed
# Visit: https://developer.nvidia.com/cuda-downloads
```

### PyTorch with CUDA

```bash
# Install PyTorch with CUDA support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Verification

### Test Installation

```bash
# Run tests
uv run pytest tests/

# Check Verilog tools
echo "module test; endmodule" > test.v
iverilog test.v && echo "Icarus Verilog: OK"
yosys -p "read_verilog test.v" && echo "Yosys: OK"
rm test.v
```

### Test API Connection

```bash
# Test OpenAI
uv run python -c "from openai import OpenAI; print('OpenAI: OK')"

# Test Anthropic
uv run python -c "import anthropic; print('Anthropic: OK')"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch size
   - Use smaller models
   - Enable gradient checkpointing

2. **API Rate Limits**

   - Implement exponential backoff
   - Use batch APIs where available
   - Consider local models

3. **Verilog Tool Errors**
   - Ensure tools are in PATH
   - Check version compatibility
   - Verify file permissions
