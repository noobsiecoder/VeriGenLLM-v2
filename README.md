# RL Fine-Tuning for Code Models

This repository contains experiments on **reinforcement learning fine-tuning (RLFT)** for code generation models. We compare **Proximal Policy Optimization (PPO)** and **Group Relative Preference Optimization (GRPO)** on the `deepseek-coder-7b-instruct` model to evaluate improvements in compilation, functional correctness, synthesis, reasoning, and code quality.

## 📌 Project Overview

* **Goal:** Enhance code generation models with RL-based techniques to improve correctness and reasoning.
* **Models:**

  * [Deepseekcoder-7b-instruct-v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
* **RL Algorithms:**

  * Proximal Policy Optimization (**PPO**)
  * Group Relative Preference Optimization (**GRPO**)

We integrate **LoRA adapters** for parameter-efficient fine-tuning and log training metrics to **Weights & Biases (W\&B)**.

## ⚙️ Setup Configuration

| **Type**            | **Configuration** |
| ------------------- | ----------------- |
| LoRA Rank           | 16                |
| Trainable Ratio     | 0.54%             |
| Clip Epsilon        | 0.2               |
| Value Coefficient   | 0.5               |
| Max Grad Norm       | 0.5               |
| Entropy Coefficient | 0.01              |

## 📊 Results

Training results are tracked in **Weights & Biases**.

* **Mean Rewards:** PPO shows higher variance but maintains stronger positive signals compared to GRPO, which stabilizes near zero.
* **Compilation Rates:** PPO maintains stability, while GRPO degrades after \~100 epochs.
* **Functional Correctness:** PPO sustains higher scores, while GRPO trends downward.
* **Reasoning Ability:** PPO improves modestly, GRPO decreases consistently.

👉 **Summary:** PPO outperforms GRPO across compilation, functional correctness, and reasoning, though with higher variance. GRPO’s strict reward shaping limits exploration and long-term stability.

## 🚀 Getting Started

### 1. Clone the repo

```bash
# Clone project
git clone https://github.com/noobsiecoder/VeriGenLLM-v2.git
cd VeriGenLLM-v2
# To use PPO and GRPO RLFT
git switch ppo-v0
# To edit code
# Create a new branch and submit pull request
git checkout -b <new-branch-name>
```

### 2. Install dependencies

```bash
# This project uses uv package manager
# Installation: https://docs.astral.sh/uv/getting-started/installation/
# After installing, sync project with all modules
uv sync
```

### 3. Run training

```bash
# To run PPO:
# Go to constants.py and change the algorithm in RLFT_TRAIN_CONFIG.rl_algorithm dict
# Note: Currently set to GRPO
# Later, to start script
uv run main.py
```

### 4. Monitor with W\&B

All metrics and plots are logged automatically to your Weights & Biases workspace.

## 🧪 Reward Function

The reward function combines multiple criteria:

$$
R = w_c \cdot \text{Compilation} + 
    w_f \cdot \text{Functional} + 
    w_s \cdot \text{Synthesise} + 
    w_r \cdot \text{Reasoning} + 
    w_q \cdot \text{Code Quality}
$$

Where $w_c, w_f, w_s, w_r, w_q$ are tunable weights.

## 📈 Example Logs

All training runs are available in [Weights & Biases](https://wandb.ai/). Example comparison plots:

* Mean Reward
* Compilation Rates
* Functional Correctness Rates
* Reasoning Rates

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss proposed changes.

## 📜 License

This project is licensed under the MIT License.
