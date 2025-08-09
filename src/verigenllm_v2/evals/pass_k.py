"""
This file works on evaluating baseline and fine-tuned models for VeriGen-v2 using Pass@k Metrics on compilation and functional correctness.

The evaluation pipeline tests generated Verilog code through three stages:
1. Compilation check (using Icarus Verilog)
2. Functional correctness (using testbenches)
3. Synthesizability (using Yosys)

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 1st, 2025

Note: LLM was used to generate comments
"""

import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List
from verigenllm_v2.utils.logger import Logger


# Initialize logger for tracking evaluation progress
log = Logger("pass-k_metrics").get_logger()


class PassKMetrics:
    """
    Evals for LLM on:
    1. Compilation.
    2. Functional correctness.

    This class implements the Pass@k metric for evaluating LLM-generated
    Verilog code. Pass@k measures how many of k generated samples pass
    various verification stages.
    """

    def __init__(self) -> None:
        """
        Initialize default objects

        Sets up difficulty categorization for the 18 test problems:
        - Basic: Problems 1-4 (simple combinational logic)
        - Medium: Problems 5-12 (sequential logic, FSMs)
        - Hard: Problems 13-17 (complex state machines, arithmetic)
        """
        # Categorize problem indices by difficulty
        self.basic = [i for i in range(1, 5)]  # Problems 1-4
        self.medium = [i for i in range(5, 13)]  # Problems 5-12
        self.hard = [i for i in range(13, 18)]  # Problems 13-17

        # Dictionary to store responses from different models
        self.model_responses = {}  # Format: {model_name: [responses]}

    def set_model_response(self, model: str, responses: List) -> None:
        """
        Set response for each model

        Parameters:
        -----------
        model : str
            Name of the model used (e.g., 'claude', 'gpt-4')
        responses : List
            List of response dictionaries from the LLM containing:
            - question: The prompt
            - outputs: List of generated code samples
            - config: Generation configuration

        Returns:
        --------
        None
        """
        self.model_responses[model] = responses

    def verify_compilation(self, sol_filepath: str, tb_filepath: str = None) -> bool:
        """
        Verify if Verilog code compiles using Icarus Verilog

        Parameters:
        -----------
        sol_filepath : str
            Path to the solution Verilog file
        tb_filepath : str, optional
            Path to the testbench file (if provided, compiles both together)

        Returns:
        --------
        bool
            True if compilation succeeds, False otherwise
        """
        try:
            # Case 1: Compile solution file only
            if tb_filepath == None:
                # Run iverilog compiler
                result = subprocess.run(
                    [
                        "iverilog",
                        # "-Wall", # Uncomment to display warnings
                        "-o",
                        "test_output",  # Output binary name
                        sol_filepath,
                    ],
                    capture_output=True,  # Capture stdout and stderr
                    text=True,  # Return output as string
                )

                # Check compilation result
                if result.returncode == 0:
                    # Compilation succeeded
                    if result.stderr.strip():
                        # Warnings present but compilation passed
                        log.warning("⚠ Compilation: PASS with warnings:")
                        log.info(f"  {result.stderr.strip()}")
                    else:
                        log.info("✓ Compilation: PASS")
                    return True
                else:
                    # Compilation failed
                    log.error("✗ Compilation: FAIL")
                    log.error(f"Compiler Error:\n{result.stderr.strip()}")
                    return False

            # Case 2: Compile solution with testbench
            else:
                result = subprocess.run(
                    [
                        "iverilog",
                        # "-Wall", # Uncomment to display warnings
                        "-o",
                        "test_output",
                        sol_filepath,
                        tb_filepath,  # Include testbench in compilation
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    log.info("✓ Compilation with test-cases: PASS")
                    return True
                else:
                    log.error("✗ Compilation with test-cases: FAIL")
                    log.error(f"Compiler Error:\n{result.stderr.strip()}")
                    return False

        except Exception as e:
            log.error(f"✗ Verification Failed: {e}")
            return False

    def verify_func_corr(
        self, test_py_path: str, sol_filepath: str, tb_filepath: str
    ) -> bool:
        """
        Verify functional correctness of the Verilog code

        This runs either:
        1. A Python test script (if available)
        2. Verilog testbench simulation using vvp

        Parameters:
        -----------
        test_py_path : str
            Path to Python test script
        sol_filepath : str
            Path to solution Verilog file
        tb_filepath : str
            Path to Verilog testbench file

        Returns:
        --------
        bool
            True if functional tests pass, False otherwise
        """
        try:
            # Case 1: No Verilog testbench provided
            if tb_filepath == None:
                # Check if Python test script exists
                if os.path.exists(test_py_path):
                    # Run Python test script
                    result = subprocess.run(
                        [
                            "uv",
                            "run",
                            test_py_path,
                            sol_filepath,
                        ],  # uv is a Python package runner
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        log.info("✓ Simulation: PASS")
                        return True
                    else:
                        log.error("✗ Simulation: FAIL")
                        return False
                else:
                    log.error("⚠ Test file doesn't exists")
                    return False

            # Case 2: Verilog testbench provided
            else:
                # First compile solution with testbench
                val = self.verify_compilation(
                    sol_filepath=sol_filepath, tb_filepath=tb_filepath
                )

                if val:
                    # Run simulation using vvp (Verilog VPI simulator)
                    result = subprocess.run(
                        ["vvp", "test_output"],  # Execute compiled binary
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        log.info("✓ Simulation: PASS")
                        return True
                    else:
                        log.error("✗ Simulation: FAIL")
                        return False
                else:
                    log.error("✗ Simulation: FAIL (due to compilation with test-case)")
                    return False

        except Exception as e:
            log.error(f"✗ Simulation Failed: {e}")
            return False

    def verify_synthesizablity(self, temp_filepath: str) -> bool:
        """
        Verify if the Verilog code is synthesizable using Yosys

        Yosys is an open-source synthesis tool that checks if the
        Verilog can be converted to actual hardware gates.

        Parameters:
        -----------
        temp_filepath : str
            Path to the Verilog file to synthesize

        Returns:
        --------
        bool
            True if synthesis succeeds, False otherwise
        """
        try:
            log.info("⏳ Running Yosys synthesis...")

            # Yosys script to read, synthesize, and write Verilog
            yosys_cmd = f"""
            read_verilog {temp_filepath}
            synth
            write_verilog /dev/null
            """

            # Run Yosys with the synthesis script
            yosys_result = subprocess.run(
                ["yosys", "-q", "-p", yosys_cmd],  # -q: quiet mode, -p: execute script
                capture_output=True,
                text=True,
            )

            if yosys_result.returncode == 0:
                log.info("✓ Yosys Synthesis: PASS")
                return True
            else:
                log.error("✗ Yosys Synthesis: FAIL")
                log.error(f"Yosys Error:\n{yosys_result.stderr.strip()}")
                return False

        except Exception as e:
            log.error(f"✗ Verification Failed: {e}")
            return False

    def _create_temp_verilog_files(
        self, solution_code: str, testbench_code: str
    ) -> tuple[str, str]:
        """
        Create temporary Verilog files for testing

        This function:
        1. Creates a temp file for the solution code
        2. If testbench provided, modifies it to match the solution module name
        3. Creates a temp file for the modified testbench

        Parameters:
        -----------
        solution_code : str
            Generated Verilog code from LLM
        testbench_code : str
            Testbench code (may need module name adjustment)

        Returns:
        --------
        tuple[str, str]
            Paths to temporary solution and testbench files
        """
        # Create temporary solution file
        temp_sol_filepath = None
        with tempfile.NamedTemporaryFile(
            suffix=".v",
            mode="w+",
            delete=False,  # Keep file after closing
        ) as temp_file:
            temp_file.write(solution_code)
            temp_sol_filepath = temp_file.name

        temp_tb_filepath = None
        if testbench_code != None:
            # Extract module name from solution code
            # Regex matches: module module_name #(params) (ports)
            pattern = r"module\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\#\s*\([^)]*\))?\s*(?:\([^)]*\))?"
            match = re.search(pattern, solution_code)
            module_name = None

            if match:
                log.info(f"Module name found in solution code: {match.group(1)}")
                module_name = match.group(1)

            # Find UUT (Unit Under Test) instantiation in testbench
            # Matches patterns like: module_name uut(...); or module_name DUT(...);
            uut_pattern = r"(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+(uut|dut|DUT|UUT)\s*\(((?:[^;]|\n)*?)\)\s*"
            match = re.search(uut_pattern, testbench_code)
            modified_code = None

            if match:
                log.info(f"Statement found in testbench code: {match.group(2)}")

                # Replace function to update module name in testbench
                def replace_uut(match):
                    indent = match.group(1)  # Preserve indentation
                    instance_name = match.group(3)  # uut/dut/DUT/UUT
                    port_connections = match.group(4)  # Port mappings
                    # Replace with correct module name
                    return f"{indent}{module_name} {instance_name}({port_connections})"

                # Apply replacement to testbench
                modified_code = re.sub(
                    uut_pattern, replace_uut, testbench_code, count=1
                )

            if modified_code != None:
                testbench_code = modified_code
            log.info("Testbench code modified")

            # Create temporary testbench file
            with tempfile.NamedTemporaryFile(
                suffix=".v", mode="w+", delete=False
            ) as temp_file:
                temp_file.write(testbench_code)
                temp_tb_filepath = temp_file.name

        return (temp_sol_filepath, temp_tb_filepath)

    def run_eval(self, model: str, filename: str, strict_mode=True) -> Dict:
        """
        Run complete evaluation pipeline for a model

        This is the main evaluation function that:
        1. Extracts Verilog code from LLM responses
        2. Tests compilation, functionality, and synthesizability
        3. Calculates Pass@k metrics
        4. Saves results to JSON

        Parameters:
        -----------
        model : str
            Name of the LLM model
        filename : str
            Output filename for evaluation results
        strict_mode : bool
            If True, only accepts code in ```verilog...``` blocks
            If False, accepts code in any ``` blocks

        Returns:
        --------
        Dict
            Evaluation results (also saved to file)
        """
        # Define dataset paths
        test_dataset_path = "dataset/test/"  # Contains problems and testbenches
        eval_dataset_path = "dataset/evals/"  # Where to save evaluation results

        # Get all test directories (one per problem)
        dirs = os.listdir(test_dataset_path)
        dirs.sort()  # Ensure consistent ordering
        # Remove MacOS metadata file if exists
        if ".DS_Store" in dirs:
            dirs.remove(".DS_Store")

        eval_tables = []  # Collect evaluation results
        responses = self.model_responses[model]

        # Process each problem/response
        for idx, response in enumerate(responses):
            outputs = response["outputs"]  # List of generated code samples
            question = response["question"]  # The prompt

            # Determine difficulty level based on problem index
            level = None
            if idx in range(0, 5):
                level = "basic"
            if idx in range(5, 13):
                level = "medium"
            if idx in range(13, 18):
                level = "hard"

            # Initialize evaluation metrics for this problem
            eval_table = {
                "question": question,
                "evals": {
                    "n": 0,  # Total completions
                    "compile": 0,  # Samples that compile
                    "func-corr": 0,  # Samples that pass tests
                    "synth": 0,  # Samples that synthesize
                },
                "difficulty": level,
            }

            log.info(f"For question: {question}")

            # Evaluate each generated sample
            for s_idx, output in enumerate(outputs):
                # Extract Verilog code from response
                # Strict mode: only ```verilog blocks
                # Non-strict: any ``` blocks
                pattern = r"```verilog\n(.*)```" if strict_mode else r"```(.*)```"
                match = re.search(pattern, output, re.DOTALL)

                if match:
                    verilog_code = match.group(1)
                    log.info(f"Extracted Verilog Code")

                    # Load testbench if available
                    tb_code = None
                    tb_filepath = os.path.join(
                        test_dataset_path, dirs[idx], "testbench.v"
                    )
                    if os.path.exists(tb_filepath):
                        with open(tb_filepath, "r") as fs:
                            tb_code = fs.read()

                    # Create temporary files for testing
                    sol_temp_filepath, tb_temp_filepath = (
                        self._create_temp_verilog_files(
                            solution_code=verilog_code, testbench_code=tb_code
                        )
                    )

                    # Run verification stages
                    # 1. Compilation check
                    comp_res = self.verify_compilation(sol_filepath=sol_temp_filepath)

                    # 2. Functional correctness check
                    test_py_path = os.path.join(
                        test_dataset_path, dirs[idx], "testbench.py"
                    )
                    func_corr_res = self.verify_func_corr(
                        test_py_path=test_py_path,
                        sol_filepath=sol_temp_filepath,
                        tb_filepath=tb_temp_filepath,
                    )

                    # Cleanup compiled binary if exists
                    if os.path.exists("test_output"):
                        os.remove("test_output")

                    # 3. Synthesizability check
                    syn_res = self.verify_synthesizablity(
                        temp_filepath=sol_temp_filepath
                    )

                    # Update metrics
                    eval_table["evals"]["n"] += 1  # Increment total samples

                    if comp_res:
                        log.info(
                            f"For question: {question} and solution idx: {s_idx + 1} - compiled!"
                        )
                        eval_table["evals"]["compile"] += 1

                    if func_corr_res:
                        log.info(
                            f"For question: {question} and solution idx: {s_idx + 1} - func corr!"
                        )
                        eval_table["evals"]["func-corr"] += 1

                    if syn_res:
                        log.info(
                            f"For question: {question} and solution idx: {s_idx + 1} - syn!"
                        )
                        eval_table["evals"]["synth"] += 1

                    # Cleanup temporary files
                    os.remove(sol_temp_filepath)
                    log.info(f"Temporary file:  {sol_temp_filepath} deleted.")
                    if tb_temp_filepath != None:
                        os.remove(tb_temp_filepath)
                        log.info(f"Temporary file: {tb_temp_filepath} deleted.")
                else:
                    # No valid code block found in response
                    log.error("No Verilog code block found.")

            # Add results for this problem to collection
            eval_tables.append(eval_table)

        # Save all evaluation results to JSON
        with open(os.path.join(eval_dataset_path, f"{filename}.json"), "w") as fs:
            json.dump(eval_tables, fs, indent=4)


if __name__ == "__main__":
    # Path to model response files
    models_dataset_path = "dataset/models/"

    # Initialize evaluation class
    passK = PassKMetrics()

    data = {}
    n_samples = 10  # Number of samples per problem to evaluate

    # Model configurations
    # strict_mode determines how code is extracted from responses
    models = {
        "claude": {"name": "claude-opus-4-20250514", "strict_mode": True},
        "codellama": {
            "name": "codellama-7b-instruct",
            "strict_mode": False,
        },  # CodeLlama doesn't use ```verilog
        "deepseek-coder": {
            "name": "deepseek-coder-7b-instruct-v1.5",
            "strict_mode": True,
        },
        "openai": {"name": "gpt-4.1", "strict_mode": True},
        "qwen-coder": {"name": "qwen2.5-coder-7b-instruct", "strict_mode": True},
    }

    # Process each model
    for model in models:
        # Load model responses
        response_file = os.path.join(
            models_dataset_path, model, f"{model}-response-n{n_samples}.json"
        )
        with open(response_file, "r") as fs:
            log.info(f"Read model response file from {response_file}")
            passK.set_model_response(model, responses=json.load(fs))

        # Run evaluation
        log.info(f"Running Evals for model: '{model}'")
        passK.run_eval(
            model,
            filename=models[model]["name"],
            strict_mode=models[model]["strict_mode"],
        )
