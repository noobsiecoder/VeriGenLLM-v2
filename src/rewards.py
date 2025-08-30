"""
Contains the reward function calculation algorithm

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 28th, 2025
Place:  Boston, MA
"""

import contextlib
import glob
import json
import re
import os
import subprocess
import tempfile
import textwrap
from subprocess import CompletedProcess
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from constants import Creator, REWARD_WEIGHTS_CONFIG
from src.logger import Logger
from src.models import OpenAIAPIClient


@dataclass
class RewardScores:
    """
    Object store to calculate the reward function
    """

    w_compilation: float = REWARD_WEIGHTS_CONFIG.get("compilation", 0.4)
    w_functional_correctness: float = REWARD_WEIGHTS_CONFIG.get(
        "functional_correctness", 0.5
    )
    w_synthesise: float = REWARD_WEIGHTS_CONFIG.get("synthesise", 0.3)
    w_code_quality: float = REWARD_WEIGHTS_CONFIG.get("code_quality", 0.5)
    w_reasoning: float = REWARD_WEIGHTS_CONFIG.get("reasoning", 0.6)

    def total_score(
        self,
        compilation: float,
        functional_correctness: float,
        synthesise: float,
        code_quality: float,
        reasoning: float,
    ) -> float:
        """
        Calculate total reward score
        """
        return (
            self.w_compilation * compilation
            + self.w_functional_correctness * functional_correctness
            + self.w_synthesise * synthesise
            + self.w_code_quality * code_quality
            + self.w_reasoning * reasoning
        )


class RewardFunction:
    """
    Contains all blocks to calculate reward function
    """

    def __init__(self, timeout_ms: int = 5000):
        self.log = Logger("reward-function").get_logger()
        self.verilog_pattern = re.compile(r"```verilog(.*?)```", re.DOTALL)
        self.verilog_mod_pattern = re.compile(r"(module\s.*?endmodule)`", re.DOTALL)
        self.json_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        self.timeout_ms = timeout_ms
        self.iverilog_cmd = ["iverilog", "-Wall", "-Wno-timescale", "-o", "test"]
        self.synthesis_script = textwrap.dedent("""
            # Read Verilog
            read_verilog -sv {verilog_file}

            # Check hierarchy
            hierarchy -check -auto-top

            # Run synthesis passes
            proc
            opt -full
            fsm -encoding binary
            opt -full
            memory -nomap
            opt -full
            techmap
            opt -fast

            # Check for latches (usually unintended)
            select -assert-none t:$dlatch
            select -assert-none t:$sr
            select -assert-none t:$dlatchsr

            # Generate statistics
            stat

            # Check design
            check
            """).strip()

    # ================================================================================= #
    # ============================= PUBLIC METHOD STARTS ============================== #
    # ================================================================================= #
    # ================================================================================= #
    # =========================== EXTRACT VERILOG CODE ONLY =========================== #
    # ================================================================================= #
    def extract_code(
        self, content: str, generated_by: Creator = Creator.HUMAN
    ) -> Optional[str]:
        """Extract Verilog code from content based on creator type"""
        if generated_by == Creator.HUMAN:
            return content.strip()
        elif generated_by == Creator.LLM:
            match = self.verilog_pattern.search(content)
            if match:
                return match.group(1).strip()
            else:
                # extract only by module if found
                self.log.warning("Code not found (missing ```verilog...``` block)")
                match = self.verilog_pattern.search(content)
                if match:
                    return match.group(1).strip()
                else:
                    self.log.error("Code not found (missing ```module...``` block)")
                    return "" # empty string

        else:
            self.log.error(f"Invalid Creator type: {generated_by}")
            return None

    # ================================================================================= #
    # =============================== COMPILATION SCORE =============================== #
    # ================================================================================= #
    def compilation_score(self, code_in_str: str) -> Tuple[float, Dict[str, int]]:
        """
        Compile Verilog code only (no testbench) and return score
        """
        return self._compilation_score(code_in_str, tb_code_in_str=None)

    # ================================================================================= #
    # ========================= FUNCTIONAL CORRECTNESS SCORE ========================== #
    # ================================================================================= #
    def functional_correctness_score(
        self, code_in_str: str, tb_in_str: str
    ) -> Tuple[float, Dict[str, int]]:
        """
        Compile Verilog code with testbench and run tests
        """
        if not tb_in_str:
            self.log.error("Testbench is required for functional correctness testing")
            raise ValueError("Testbench not provided")

        return self._compilation_score(code_in_str, tb_in_str)

    # ================================================================================= #
    # =============================== SYNTHESISE SCORE ================================ #
    # ================================================================================= #
    def synthesise_score(self, cd_code_in_str: str) -> float:
        """Check if code can be synthesized"""
        try:
            with self._create_temp_files_for_synthesise(cd_code_in_str) as (
                _,  # verilog path -> dummy
                script_path,
            ):
                result = self._run_yosys(script_path)

                if not result or result.returncode != 0:
                    self.log.error("Synthesis failed")
                    return -1.0

                # Check for specific issues in output
                stdout = result.stdout.decode("utf-8", errors="ignore")
                stderr = result.stderr.decode("utf-8", errors="ignore")

                # Look for critical warnings
                if "ERROR" in stdout or "ERROR" in stderr:
                    return -1.0

                return 1.0

        except Exception as e:
            self.log.error(f"Synthesis check error: {e}")
            return -1.0

    # ================================================================================= #
    # ============================== CODE QUALITY SCORE =============================== #
    # ================================================================================= #
    def code_quality_score(self, cd_code_in_str: str) -> Tuple[float, Dict[str, int]]:
        """
        Analyze code quality based on multiple metrics
        Returns: (score, metrics_dict)
        """
        if not cd_code_in_str:
            return -1.0, {}

        metrics = {
            "unused_vars": 0,
            "naming_issues": 0,
            "style_violations": 0,
            "total_lines": len(cd_code_in_str.splitlines()),
        }

        # Check for unused variables
        declared_vars = set()
        used_vars = set()

        # Remove comments first
        code_no_comments = re.sub(r"//.*$", "", cd_code_in_str, flags=re.MULTILINE)
        code_no_comments = re.sub(r"/\*.*?\*/", "", code_no_comments, flags=re.DOTALL)

        # Extract declared variables - improved patterns
        # 1. Standalone wire/reg declarations
        wire_reg_pattern = re.compile(
            r"^\s*(?:wire|reg)\s+(?:\[[^\]]+\])?\s*(\w+)", re.MULTILINE
        )
        for match in wire_reg_pattern.finditer(code_no_comments):
            declared_vars.add(match.group(1))

        # 2. Port declarations - handles "output reg [7:0] count" correctly
        port_pattern = re.compile(
            r"\b(?:input|output|inout)\s+(?:wire\s+|reg\s+)?(?:\[[^\]]+\])?\s*(\w+)"
        )
        for match in port_pattern.finditer(code_no_comments):
            declared_vars.add(match.group(1))

        # 3. Integer declarations
        int_pattern = re.compile(r"^\s*integer\s+(\w+)", re.MULTILINE)
        for match in int_pattern.finditer(code_no_comments):
            declared_vars.add(match.group(1))

        # Find used variables
        for line in code_no_comments.splitlines():
            # Skip module declarations and variable declarations
            if re.match(r"^\s*(?:module|input|output|inout|wire|reg|integer)", line):
                continue
            # Skip endmodule
            if re.match(r"^\s*endmodule", line):
                continue

            # Check each declared variable
            for var in declared_vars:
                # Use word boundaries to avoid partial matches
                if re.search(rf"\b{var}\b", line):
                    used_vars.add(var)

        metrics["unused_vars"] = len(declared_vars - used_vars)

        # Check naming conventions
        for var in declared_vars:
            if not re.match(r"^[a-z][a-z0-9_]*$", var):
                metrics["naming_issues"] += 1

        # Check line length (on original code, not comment-stripped)
        for line in cd_code_in_str.splitlines():
            if len(line) > 80:
                metrics["style_violations"] += 1

        # Calculate score
        score = 1.0
        score -= min(metrics["unused_vars"] * 0.1, 0.4)
        score -= min(metrics["naming_issues"] * 0.05, 0.3)
        score -= min(metrics["style_violations"] * 0.02, 0.2)

        return max(-1.0, score), metrics

    # ================================================================================= #
    # ================================ REASONING SCORE ================================ #
    # ================================================================================= #
    def reasoning_score(self, response: str) -> Dict:
        """
        Check if reasoning aligns with generated code
        This is done with another LLM: GPT-4o
        Returns: score [-1, 1]
        """
        prompt = f"I will pass a value, that contains both reasoning and the code generated. The reason will be enclosed in <reason>...</reason>, while the code in ```verilog...```. Score 0 to 1 on how much the reasoning matches the implementation and give your reason (max 8) as why it led to the score. If it doesn't match 1%, give -1. Keep in mind to output the data in json format like: {{ 'score': int, 'reasons': [...] }}. The value is: {response}"

        gpt_client = OpenAIAPIClient()
        gpt_client.connect()
        response = gpt_client.generate(prompt, max_tokens=600, n_samples=1)
        answer = response["outputs"][0]

        match = re.search(self.json_pattern, answer)
        if not match:
            self.log.warning("Unable to get score from GPT-LLM on reasoning")
            return 0.0

        try:
            return json.loads(match.group(1))
        except Exception as err:
            self.log.warning(f"Error in decoding data | Err: {err}")
            return 0.0

    # ================================================================================= #
    # ============================== PUBLIC METHOD ENDS =============================== #
    # ================================================================================= #
    # ================================================================================= #
    # ============================= PRIVATE METHODS STARTS ============================ #
    # ================================================================================= #
    def _compilation_score(self, cd_code_in_str: str, tb_code_in_str: Optional[str]):
        """
        Get the compilation using candidate generated code, testebench (functional correctness only)
        """

        metrics = {"warnings": 0, "errors": 0, "tests_passed": 0, "tests_total": 0}
        try:
            with self._create_temp_files_for_compilation(
                cd_code_in_str, tb_code_in_str
            ) as temp_files:
                # Compile
                compile_result = self._compile(temp_files)

                if not compile_result:
                    return -1.0, metrics

                # Parse compilation output
                metrics.update(self._parse_compile_output(compile_result))

                # If compilation failed, return early
                if (
                    compile_result.returncode != 0
                    or "I give up" in compile_result.stderr.decode("utf-8")
                ):
                    return -1.0, metrics

                # Run tests only if testbench is provided and compilation succeeded
                if tb_code_in_str:
                    test_result = self._run_tests()
                    if test_result:
                        metrics.update(self._parse_test_output(test_result))

                        # Check if test execution failed
                        if test_result.returncode != 0:
                            self.log.error(
                                f"Test execution failed: {test_result.stderr}"
                            )
                            # Still calculate score based on test results if any
                            # If no tests passed, this will result in a low score

                # Calculate score
                return self._calculate_score(
                    metrics, has_testbench=bool(tb_code_in_str)
                ), metrics

        finally:
            self._cleanup()

    @contextlib.contextmanager
    def _create_temp_files_for_synthesise(self, cd_code_in_str: str):
        """Create temporary files for synthesis"""
        verilog_fd, verilog_path = tempfile.mkstemp(suffix=".v", text=True)
        script_fd, script_path = tempfile.mkstemp(suffix=".ys", text=True)

        try:
            # Write Verilog code
            with os.fdopen(verilog_fd, "w") as f:
                f.write(cd_code_in_str)

            # Write synthesis script
            with os.fdopen(script_fd, "w") as f:
                f.write(self.synthesis_script.format(verilog_file=verilog_path))

            yield verilog_path, script_path

        finally:
            for path in [verilog_path, script_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def _run_yosys(self, script_path: str) -> Optional[CompletedProcess]:
        """Run Yosys synthesis"""
        try:
            return subprocess.run(
                ["yosys", "-s", script_path],
                capture_output=True,
                timeout=self.timeout_ms / 1000,
            )
        except subprocess.TimeoutExpired:
            self.log.error("Synthesis timeout")
            return None
        except FileNotFoundError:
            self.log.error("Yosys not found. Please install Yosys.")
            return None

    @contextlib.contextmanager
    def _create_temp_files_for_compilation(
        self, code: str, testbench: Optional[str] = None
    ):
        """Create temporary files for compilation"""
        files = []
        try:
            # Create main code file
            code_fd, code_path = tempfile.mkstemp(suffix=".v", text=True)
            with os.fdopen(code_fd, "w") as f:
                f.write(code)
            files.append(code_path)

            # Create testbench file if provided
            if testbench:
                tb_fd, tb_path = tempfile.mkstemp(suffix="_tb.v", text=True)
                with os.fdopen(tb_fd, "w") as f:
                    f.write(testbench)
                files.append(tb_path)

            yield files

        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)

    def _compile(self, file_paths: List[str]) -> Optional[CompletedProcess]:
        """Compile Verilog files"""
        cmd = self.iverilog_cmd + file_paths
        self.log.info(f"Running: {' '.join(cmd)}")

        try:
            return subprocess.run(
                cmd, capture_output=True, timeout=self.timeout_ms / 1000
            )
        except subprocess.TimeoutExpired:
            self.log.error("Compilation timeout")
            return None

    def _run_tests(self) -> Optional[CompletedProcess]:
        """Run compiled tests"""
        try:
            return subprocess.run(
                ["vvp", "test"], capture_output=True, timeout=self.timeout_ms / 1000
            )
        except subprocess.TimeoutExpired:
            self.log.error("Test execution timeout")
            return None

    def _parse_compile_output(self, result: CompletedProcess) -> Dict[str, int]:
        """Parse compilation output for warnings and errors"""
        metrics = {"warnings": 0, "errors": 0}
        stderr = result.stderr.decode("utf-8", errors="ignore")

        for line in stderr.splitlines():
            if "warning" in line.lower():
                metrics["warnings"] += 1
            elif "error" in line.lower():
                metrics["errors"] += 1

        return metrics

    def _parse_test_output(self, result: CompletedProcess) -> Dict[str, int]:
        """Parse test output for pass/fail counts"""
        metrics = {"tests_passed": 0, "tests_total": 0}
        stdout = result.stdout.decode("utf-8", errors="ignore")

        # Look for test summary
        pattern = r"Test Summary:\s*(\d+)/(\d+)\s*tests passed"
        match = re.search(pattern, stdout)
        if match:
            metrics["tests_passed"] = int(match.group(1))
            metrics["tests_total"] = int(match.group(2))

        return metrics

    def _calculate_score(
        self, metrics: Dict[str, int], has_testbench: bool = False
    ) -> float:
        """Calculate score based on metrics"""
        # If there are compilation errors and no testbench, return -1.0
        if metrics["errors"] > 0 and not has_testbench:
            return -1.0

        # If we have a testbench but compilation had errors and no tests ran
        if has_testbench and metrics["errors"] > 0 and metrics["tests_total"] == 0:
            return -1.0

        # Base score calculation
        if has_testbench and metrics["tests_total"] > 0:
            # For functional correctness, base score on test results
            score = metrics["tests_passed"] / metrics["tests_total"]
        else:
            # For compilation only, perfect score if no errors
            score = 1.0 if metrics["errors"] == 0 else 0.0

        # Apply penalties
        score -= min(metrics["warnings"] * 0.04, 0.4)
        score -= min(metrics["errors"] * 0.08, 0.8)

        return max(-1.0, score)

    def _cleanup(self):
        """Clean up generated files"""
        if os.path.exists("test"):
            os.remove("test")
        for vcd_file in glob.glob("*.vcd"):
            os.remove(vcd_file)

    # ================================================================================= #
    # ============================== PRIVATE METHODS ENDS ============================= #
    # ================================================================================= #
