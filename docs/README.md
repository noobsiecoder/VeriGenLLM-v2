# Documentation

Comprehensive documentation for the VeriGenLLM-v2 project.

## 📚 Documentation Structure

| Document                                         | Description                               | Use When                                |
| :----------------------------------------------- | :---------------------------------------- | :-------------------------------------- |
| [setup-guide.md](./setup-guide.md)               | Complete installation and configuration   | Starting fresh or troubleshooting setup |
| [hardware.md](./hardware.md)                     | Cloud infrastructure specifications       | Deploying models or estimating costs    |
| [oss-models.md](./oss-models.md)                 | Open-source model details and comparisons | Choosing which model to use             |
| [commercial-models.md](./commercial-models.md)   | Commercial API details and pricing        | Budgeting or selecting cloud LLMs       |
| [evaluation-metrics.md](./evaluation-metrics.md) | Pass@k metrics and results explanation    | Understanding evaluation results        |

## 🚀 Quick Navigation

### Getting Started

1. Start with [setup-guide.md](./setup-guide.md) for installation
2. Review [hardware.md](./hardware.md) if using cloud GPUs
3. Choose models from [oss-models.md](./oss-models.md) or [commercial-models.md](./commercial-models.md)

### Understanding Results

1. Read [evaluation-metrics.md](./evaluation-metrics.md) for metric definitions
2. Compare model performance in the model documentation
3. Check cost analysis in [commercial-models.md](./commercial-models.md)

## 📊 Key Insights

### Model Selection Guide

| Use Case         | Recommended Model               | Rationale                    |
| :--------------- | :------------------------------ | :--------------------------- |
| Best Performance | Deepseek-coder-7B-Instruct-v1.5 | Verilog-specific fine-tuning |
| Production API   | GPT-4.1                         | Reliable, consistent         |

### Cost vs Performance

- **Budget Option**: Open-source models on own hardware
- **Balanced**: GPT-4.1 (~$0.23 per full evaluation)
- **Premium**: Claude Opus 4 (~$4.67 per full evaluation)

## 🔧 Contributing

To add new documentation:

1. Create markdown file in `docs/`
2. Use clear headings and tables
3. Keep content concise and actionable
4. Update this README with navigation entry

## 📝 Documentation Standards

- Use tables for structured data
- Include code examples with syntax highlighting
- Provide "Quick Start" sections
- Add troubleshooting for common issues
