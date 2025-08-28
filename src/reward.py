"""
Common reward function for all RL Policy

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 27th, 2025
Place:  Boston, MA
"""

import os
import re
import glob
import tempfile
import contextlib
import subprocess
from typing import Dict, List, Tuple, Optional
from subprocess import CompletedProcess
import difflib
from src.constants import Creator


class VerilogCodeAnalyzer:
    """Handles Verilog code analysis and scoring"""

    def __init__(self, log):
        self.log = log
        self.verilog_pattern = re.compile(r"```verilog\n(.*?)```", re.DOTALL)
        self.reasoning_pattern = re.compile(r"<reason>\n(.*?)</reason>", re.DOTALL)

    def extract_code(
        self, content: str, generated_by: Creator = Creator.Human
    ) -> Optional[str]:
        """Extract Verilog code from content based on creator type"""
        if generated_by == Creator.Human:
            return content.strip()

        if generated_by == Creator.LLM:
            match = self.verilog_pattern.search(content)
            if match:
                return match.group(1).strip()
            self.log.error("Code not found (missing ```verilog...``` block)")
            return None

        self.log.error(f"Invalid Creator type: {generated_by}")
        return None

    def extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning from LLM response"""
        match = self.reasoning_pattern.search(response)
        if match:
            return match.group(1).strip()
        self.log.warning("Reasoning not found (missing <reason>...</reason> tags)")
        return None

    def analyze_code_quality(self, code: str) -> Tuple[float, Dict[str, int]]:
        """
        Analyze code quality based on multiple metrics
        Returns: (score, metrics_dict)
        """
        if not code:
            return -1.0, {}

        metrics = {
            "unused_vars": 0,
            "naming_issues": 0,
            "style_violations": 0,
            "total_lines": len(code.splitlines()),
        }

        # Check for unused variables
        declared_vars = set()
        used_vars = set()

        # Remove comments first
        code_no_comments = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
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
        for line in code.splitlines():
            if len(line) > 80:
                metrics["style_violations"] += 1

        # Calculate score
        score = 1.0
        score -= min(metrics["unused_vars"] * 0.1, 0.4)
        score -= min(metrics["naming_issues"] * 0.05, 0.3)
        score -= min(metrics["style_violations"] * 0.02, 0.2)

        return max(-1.0, score), metrics


class VerilogCompiler:
    """Handles Verilog compilation and testing"""

    def __init__(self, log, timeout_ms: int = 5000):
        self.log = log
        self.timeout_ms = timeout_ms
        self.iverilog_cmd = ["iverilog", "-Wall", "-Wno-timescale", "-o", "test"]

    def compilation_score(self, code: str) -> Tuple[float, Dict[str, int]]:
        """
        Compile Verilog code only (no testbench) and return score
        """
        return self.compile_and_score(code, testbench=None)

    def functional_correctness_score(
        self, code: str, testbench: str
    ) -> Tuple[float, Dict[str, int]]:
        """
        Compile Verilog code with testbench and run tests
        """
        if not testbench:
            self.log.error("Testbench is required for functional correctness testing")
            return -1.0, {"error": "No testbench provided"}

        return self.compile_and_score(code, testbench)

    def compile_and_score(
        self, code: str, testbench: Optional[str] = None
    ) -> Tuple[float, Dict[str, int]]:
        """
        Compile Verilog code and return score with detailed metrics
        Internal method used by both compilation_score and functional_correctness_score
        """
        metrics = {"warnings": 0, "errors": 0, "tests_passed": 0, "tests_total": 0}

        try:
            with self._create_temp_files(code, testbench) as temp_files:
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
                if testbench:
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
                    metrics, has_testbench=bool(testbench)
                ), metrics

        finally:
            self._cleanup()

    @contextlib.contextmanager
    def _create_temp_files(self, code: str, testbench: Optional[str] = None):
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


class SynthesisChecker:
    """Handles synthesis checking with Yosys"""

    def __init__(self, log, timeout_ms: int = 10000):
        self.log = log
        self.timeout_ms = timeout_ms
        self.synthesis_script = """
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
"""

    def check_synthesis(self, code: str) -> float:
        """Check if code can be synthesized"""
        try:
            with self._create_temp_files(code) as (verilog_path, script_path):
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

    @contextlib.contextmanager
    def _create_temp_files(self, code: str):
        """Create temporary files for synthesis"""
        verilog_fd, verilog_path = tempfile.mkstemp(suffix=".v", text=True)
        script_fd, script_path = tempfile.mkstemp(suffix=".ys", text=True)

        try:
            # Write Verilog code
            with os.fdopen(verilog_fd, "w") as f:
                f.write(code)

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


class CodeSimilarityChecker:
    """Check code similarity using AST-like analysis"""

    def __init__(self, log):
        self.log = log

    def calculate_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate similarity between two Verilog code pieces
        Returns: similarity score [0, 1] where 0 is identical and 1 is completely different
        """
        if not code1 or not code2:
            return 1.0

        # Normalize code for comparison
        norm_code1 = self._normalize_code(code1)
        norm_code2 = self._normalize_code(code2)

        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, norm_code1, norm_code2)
        similarity_ratio = matcher.ratio()

        # Convert to dissimilarity score (0 = same, 1 = different)
        return 1.0 - similarity_ratio

    def _normalize_code(self, code: str) -> str:
        """Normalize Verilog code for comparison"""
        # Remove comments
        code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

        # Normalize whitespace
        code = " ".join(code.split())

        # Extract structural elements
        elements = []

        # Extract module declarations
        modules = re.findall(r"module\s+(\w+)", code)
        elements.extend(f"module:{m}" for m in modules)

        # Extract input/output declarations
        io_pattern = r"(input|output|inout)\s+(?:\[[^\]]+\])?\s*(\w+)"
        for match in re.finditer(io_pattern, code):
            elements.append(f"{match.group(1)}:{match.group(2)}")

        # Extract always blocks
        always_blocks = re.findall(r"always\s*@\s*\([^)]+\)", code)
        elements.extend(f"always:{block}" for block in always_blocks)

        return "\n".join(sorted(elements))


class ReasoningChecker:
    """Check if reasoning matches generated code"""

    def __init__(self, log):
        self.log = log

    def check_reasoning_alignment(self, reasoning: str, code: str) -> float:
        """
        Check if reasoning aligns with generated code
        Returns: score [-1, 1]
        """
        if not reasoning or not code:
            return -1.0

        # Extract key concepts from reasoning
        reasoning_concepts = self._extract_concepts(reasoning)

        # Check if concepts appear in code
        matches = 0
        for concept in reasoning_concepts:
            if self._concept_in_code(concept, code):
                matches += 1

        if not reasoning_concepts:
            return 0.0

        # Calculate score
        match_ratio = matches / len(reasoning_concepts)

        # Convert to [-1, 1] scale
        return (match_ratio * 2) - 1

    def _extract_concepts(self, reasoning: str) -> List[str]:
        """Extract key concepts from reasoning text"""
        concepts = []

        # Common Verilog concepts to look for
        concept_patterns = [
            r"\b(always block|always @)\b",
            r"\b(posedge|negedge)\b",
            r"\b(state machine|FSM)\b",
            r"\b(counter|timer)\b",
            r"\b(register|flip-flop)\b",
            r"\b(multiplexer|mux)\b",
            r"\b(decoder|encoder)\b",
            r"\b(module|interface)\b",
            r"\b(input|output|wire|reg)\b",
        ]

        reasoning_lower = reasoning.lower()
        for pattern in concept_patterns:
            if re.search(pattern, reasoning_lower, re.IGNORECASE):
                concepts.append(pattern.strip(r"\b()"))

        return concepts

    def _concept_in_code(self, concept: str, code: str) -> bool:
        """Check if a concept appears in the code"""
        # Normalize concept
        concept_words = concept.lower().split("|")
        code_lower = code.lower()

        for word in concept_words:
            if word in code_lower:
                return True

        return False
