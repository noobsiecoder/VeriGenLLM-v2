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
