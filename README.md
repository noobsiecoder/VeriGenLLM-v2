# Extreme Few-Shot Verilog Generation via Self-Improving RLFT

This repository contains the codebase, setup scripts, and documentation for the project **"Extreme Few-Shot Verilog Generation via Self-Improving RLFT"**, which aims to push the boundaries of hardware code generation using reinforcement learning and only **20–100 curated Verilog examples**.

### 🔬 Project Summary

This research proposes a self-improving fine-tuning strategy for the `CodeLlama-7B-Instruct` model that learns to generate Verilog HDL with strong correctness and efficiency guarantees — starting from minimal human supervision.

Inspired by and **extending** the [VeriGen project](https://dl.acm.org/doi/full/10.1145/3643681), this work aims to **match or outperform models trained on 50,000+ examples**, using only a few hundred high-quality training signals derived from compilation and simulation feedback.

---

### 🚀 Key Contributions

- **Extreme Few-Shot Learning**: Fine-tuning on just 20–100 Verilog files (e.g., FSMs, counters, gates).
- **RLFT Loop**: Using reinforcement learning fine-tuning (RLFT) with automated rewards:
  - Verilog compilation success (via Icarus Verilog)
  - Testbench correctness
  - Code length and structural metrics
- **Self-Improving Dataset**: Online generation + filtering to grow dataset from 100 to 500+ examples.
- **Two Branches**:
  - [`replication`](https://github.com/noobsiecoder/VeriGenLLM-v2/tree/replicate-v0): Faithful reproduction of the VeriGen paper.
  - [`enhancement`](https://github.com/noobsiecoder/VeriGenLLM-v2/tree/enhance-v0): Implements RLFT and dataset self-expansion using CodeLlama-7B.

---

### 🧠 Model & Infrastructure

- Model: `CodeLlama-7B-Instruct`
- Frameworks: Hugging Face Transformers, PEFT, RLHF tooling
- Infrastructure: Google Cloud Platform (V100 / A100 GPU)

---

### 📄 Branches Overview

- [`replication`](./tree/replicate-v0): Replicates baseline VeriGen results using supervised fine-tuning on ~50,000 Verilog examples.
- [`enhancement`](./tree/enhance-v0): Trains using just 100 examples, applies RLFT + online generation loop, and grows dataset dynamically.

---

### 📌 Citation

If you use or build upon this project, please cite:

```bibtex
@misc{extreme-fewshot-verilog,
  author       = {Your Name},
  title        = {Extreme Few-Shot Verilog Generation via Self-Improving RLFT},
  year         = {2025},
  howpublished = {\url{https://github.com/noobsiecoder/VeriGenLLM-v2}},
  note         = {Enhancement of VeriGen: \url{...}}
}
```

---

### 📬 Contact

For questions, please open an issue or email: `noobsiecoder@gmail.com`
