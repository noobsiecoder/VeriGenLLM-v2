# Notebooks - Verilog LLM Evaluation Analysis

This folder contains Jupyter notebooks for analyzing and visualizing the evaluation results from the Verilog code generation experiments.

## Overview

The notebooks in this directory process the JSON evaluation data from various LLMs and generate comprehensive visualizations and summary statistics for comparative analysis.

## Main Notebook

### `eval-analysis.ipynb`

This notebook performs multi-model comparison analysis of Verilog code generation capabilities across different LLMs.

#### Features

1. **Individual Model Analysis**

   - Generates detailed tables for each model showing per-problem performance
   - Exports results as both PNG visualizations and CSV files
   - Displays metrics: k-value, compilation pass rate, functional correctness, and synchronization

2. **Comparative Summary Statistics**

   - Creates aggregate performance metrics across all models
   - Calculates average scores for each evaluation metric
   - Generates a summary comparison table

3. **Difficulty-Level Analysis**
   - Breaks down performance by problem difficulty (basic, medium, hard)
   - Creates difficulty-specific summary tables for each model
   - Helps identify model strengths/weaknesses across complexity levels

## Functions

### Core Analysis Functions

#### `load_json_data(file_path)`

Loads JSON evaluation data from specified file path.

#### `create_individual_tables(file_paths, output_dir, file_labels)`

Creates separate performance tables for each model:

- **Parameters:**
  - `file_paths`: List of JSON file paths containing evaluation results
  - `output_dir`: Directory to save output files
  - `file_labels`: Custom labels for each model
- **Outputs:**
  - PNG table visualizations
  - CSV files with raw data

#### `create_comparison_summary(file_paths, output_file, file_labels)`

Generates a summary table comparing all models:

- Shows total questions evaluated
- Average k-value (number of attempts)
- Average compilation success rate
- Average functional correctness rate
- Average synchronization success rate

#### `create_difficulty_lvl_summary(model_labels)`

Creates difficulty-based performance breakdown:

- Groups results by difficulty level (basic/medium/hard)
- Calculates average metrics per difficulty
- Generates individual summary tables for each model

#### `find_json_files(directory)`

Utility function to locate all JSON files in a specified directory.

## Input Data Format

The notebook expects JSON files with the following structure:

```json
[
  {
    "question": "Problem description",
    "difficulty": "basic|medium|hard",
    "evals": {
      "k": 10,
      "pass@k-compile": 10,
      "pass@k-func-corr": 6,
      "pass@k-sync": 5
    }
  }
]
```

## Output Files

All generated files are saved to `dataset/evals/multi_comparison/`:

### Per-Model Outputs

- `{Model_Name}_table.png` - Visual table of results
- `{Model_Name}_table.csv` - Raw data in CSV format
- `{Model_Name}_diff_summary.png` - Difficulty-level breakdown

### Comparative Outputs

- `summary.png` - Overall comparison table across all models

## Evaluated Models

The notebook analyzes the following models:

1. **Claude Opus 4** - Anthropic's latest model
2. **CodeLlama 7B Instruct** - Meta's instruction-tuned code model
3. **DeepSeek Coder 7B Instruct v1.5** - DeepSeek's code generation model
4. **OpenAI GPT-4.1** - OpenAI's flagship model
5. **Qwen Coder 2.5 7B Instruct** - Alibaba's instruction-tuned coder model

## Metrics Explained

- **k**: Number of generation attempts per problem
- **pass@k-compile**: Proportion of attempts that compile successfully
- **pass@k-func-corr**: Proportion of attempts that are functionally correct
- **pass@k-sync**: Proportion of attempts with correct synchronization behavior

## Usage

1. Ensure evaluation JSON files are in the correct directory
2. Run the notebook cells sequentially
3. Generated visualizations will be saved to `dataset/evals/multi_comparison/`

## Customization

To analyze different models or datasets:

1. Update the `file_paths` list with new JSON file locations
2. Modify the `labels` list to match your models
3. Adjust output directory paths as needed


## Visualization Examples

The notebook generates three types of visualizations:

1. **Individual Model Tables**: Detailed per-problem results with color-coded headers
2. **Summary Comparison**: Side-by-side model performance metrics
3. **Difficulty Breakdowns**: Performance stratified by problem complexity

## Notes

- Question text is truncated to 60 characters in tables for readability
- Tables use alternating row colors for better visibility
- All numerical metrics are averaged across problems for summaries
- PNG files are saved at 300 DPI for high-quality output
