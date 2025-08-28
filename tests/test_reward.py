"""
Testbench to check RLFT training

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 26th, 2025
Place:  Boston, MA
"""

import glob
import os
import random
import re
import tempfile
import subprocess
import pytest
from unittest.mock import Mock, patch
from typing import Iterator
from src.constants import Creator, RLPolicy, RewardScores
from src.rlft import RLFineTuner
from src.reward import (
    VerilogCodeAnalyzer,
    VerilogCompiler,
    SynthesisChecker,
    CodeSimilarityChecker,
    ReasoningChecker,
)

# Common point of dataset
dataset_path = "dataset/testbench/hdlbits/"


def group_dataset():
    # Load dataset order (Easy -> Medium -> Hard)
    dataset_files_order = []
    with open(os.path.join(dataset_path, "orders.txt"), "r") as fs:
        content = fs.read()
        dataset_files_order = [
            line.strip() for line in content.splitlines() if line.strip()
        ]

    return dataset_files_order


def test_dataset():
    """
    Test if dataset retrieval is successful
    and doesn't hit any error/raises error
    """
    TOTAL = 43
    dataset_files_order = group_dataset()

    assert (len(dataset_files_order)) == TOTAL


def test_code_reading_in_batches():
    """
    Mock test if file reading in batches happens as anticipated
    """

    def _create_batches(data, size) -> Iterator[str]:
        for idx in range(0, len(data), size):
            yield data[idx : idx + size]

    def _correct_filename(pattern: str) -> str:
        matching_files = glob.glob(pattern)

        if matching_files:
            return matching_files[0]  # Get the first (and presumably only) match
        else:
            # Handle case where no file is found
            raise FileNotFoundError(f"Incorrect path: {pattern}")

    def random_idx() -> int:
        return random.randint(1, 3)

    dataset_files_order = group_dataset()
    SIZE = 2  # TEST
    EPOCH = 4  # TEST -> Suppose MAX 5 digits (0..4)
    for idx, batch in enumerate(
        _create_batches(dataset_files_order, SIZE)
    ):  # batch is a tuple
        paths_tb_code = [
            _correct_filename(os.path.join(dataset_path, dirname, "tb_*.v"))
            for dirname in batch
        ]
        paths_gt_code = [
            _correct_filename(os.path.join(dataset_path, dirname, "answer_*.v"))
            for dirname in batch
        ]

        paths_pm = [
            _correct_filename(
                os.path.join(
                    dataset_path,
                    dirname,
                    f"prompt{3 - EPOCH if EPOCH < 3 else random_idx()}_*.v",
                )
            )
            for dirname in batch
        ]

        if idx == len(dataset_files_order) / SIZE:
            assert len(paths_tb_code) == SIZE
            assert len(paths_gt_code) == SIZE
            assert len(paths_pm) == SIZE


def test_pattern_regex():
    """
    Mock unit test of Regex pattern matching for code
    """

    pattern = r"(?<=verilog\n)[\s\S]*?endmodule"
    content = """```verilog
// This is a 7458 microcontroller problem
module microcontroller_7458 ( 
    input p1a, p1b, p1c, p1d, p1e, p1f,
    output p1y,
    input p2a, p2b, p2c, p2d,
    output p2y
);

    wire W, X, Y, Z; // assign output signals

    assign W = p1a & p1b & p1c;
    assign X = p2a & p2b;

    assign Y = p1d & p1e & p1f;
    assign Z = p2c & p2d;

    assign p1y = W | Y; // output signal at P1
    assign p2y = X | Z; // output signal at P2

endmodule
```

```verilog
// This is a 7458 microcontroller problem
module abcd_efg ( 
    input p1a, p1b, p1c, p1d, p1e, p1f,
    output p1y,
);

    // NOTHING

endmodule
```"""

    match = re.search(pattern, content, re.DOTALL)
    if match:
        matched_str = """// This is a 7458 microcontroller problem
module microcontroller_7458 ( 
    input p1a, p1b, p1c, p1d, p1e, p1f,
    output p1y,
    input p2a, p2b, p2c, p2d,
    output p2y
);

    wire W, X, Y, Z; // assign output signals

    assign W = p1a & p1b & p1c;
    assign X = p2a & p2b;

    assign Y = p1d & p1e & p1f;
    assign Z = p2c & p2d;

    assign p1y = W | Y; // output signal at P1
    assign p2y = X | Z; // output signal at P2

endmodule"""
        assert match.group(0) == matched_str


def test_verilog_code_analyzer():
    """Test VerilogCodeAnalyzer functionality"""
    mock_log = Mock()
    analyzer = VerilogCodeAnalyzer(mock_log)

    # Test extract_code for HUMAN creator
    human_code = "module test(); endmodule"
    assert analyzer.extract_code(human_code, Creator.Human) == human_code.strip()

    # Test extract_code for LLM creator
    llm_response = """Here's the code:
```verilog
module counter(
    input clk,
    output reg [7:0] count
);
    always @(posedge clk)
        count <= count + 1;
endmodule
```"""
    extracted = analyzer.extract_code(llm_response, Creator.LLM)
    assert "module counter" in extracted
    assert "endmodule" in extracted

    # Test extract_code with missing verilog block
    bad_response = "Here's some text without code blocks"
    assert analyzer.extract_code(bad_response, Creator.LLM) is None

    # Test extract_reasoning
    response_with_reasoning = """
<reason>
I need to create a counter that increments on each clock edge.
The counter should be 8 bits wide.
</reason>

```verilog
module counter();
endmodule
```"""
    reasoning = analyzer.extract_reasoning(response_with_reasoning)
    assert "create a counter" in reasoning
    assert "8 bits wide" in reasoning

    # Test extract_reasoning with missing tags
    assert analyzer.extract_reasoning("No reasoning tags here") is None


def test_code_quality_analysis():
    """Test code quality scoring"""
    mock_log = Mock()
    analyzer = VerilogCodeAnalyzer(mock_log)

    # Test perfect code
    good_code = """module counter(
    input clk,
    input rst,
    output reg [7:0] count
);
    always @(posedge clk) begin
        if (rst)
            count <= 8'b0;
        else
            count <= count + 1;
    end
endmodule"""
    score, metrics = analyzer.analyze_code_quality(good_code)
    assert score > 0.8  # Should be high quality
    assert metrics["unused_vars"] == 0

    # Test code with unused variables
    bad_code = """module test(
    input a,
    output b
);
    wire unused_wire;
    wire another_unused;
    assign b = a;
endmodule"""
    score, metrics = analyzer.analyze_code_quality(bad_code)
    assert score <= 0.8  # Should be lower due to unused vars
    assert metrics["unused_vars"] > 0

    # Test code with long lines
    long_line_code = """module test();
    assign very_long_signal_name = this_is_a_very_long_line_that_exceeds_eighty_characters_and_should_be_flagged_as_a_style_violation;
endmodule"""
    score, metrics = analyzer.analyze_code_quality(long_line_code)
    assert metrics["style_violations"] > 0

    # Test empty code
    score, metrics = analyzer.analyze_code_quality("")
    assert score == -1.0


def test_compilation_with_mocking():
    """Test VerilogCompiler with mocked subprocess"""
    mock_log = Mock()
    compiler = VerilogCompiler(mock_log)

    good_code = """module test(input a, output b);
    assign b = a;
endmodule"""

    # Mock successful compilation
    with patch("subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = b""
        mock_result.stdout = b"Test Summary: 10/10 tests passed"
        mock_run.return_value = mock_result

        score, metrics = compiler.compile_and_score(good_code, "testbench code")
        assert score == 1.0
        assert metrics["tests_passed"] == 10
        assert metrics["tests_total"] == 10

    # Mock compilation with warnings
    with patch("subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = (
            b"warning: implicit wire declaration\nwarning: unused variable"
        )
        mock_result.stdout = b"Test Summary: 10/10 tests passed"
        mock_run.return_value = mock_result

        score, metrics = compiler.compile_and_score(good_code)
        assert score < 1.0  # Should be penalized for warnings
        assert metrics["warnings"] == 2

    # Mock compilation failure
    with patch("subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = b"error: syntax error"
        mock_run.return_value = mock_result

        score, metrics = compiler.compile_and_score(good_code)
        assert score == -1.0
        assert metrics["errors"] == 1


def test_synthesis_checker():
    """Test SynthesisChecker functionality"""
    mock_log = Mock()
    synth = SynthesisChecker(mock_log)

    good_code = """module counter(
    input clk,
    output reg [7:0] count
);
    always @(posedge clk)
        count <= count + 1;
endmodule"""

    # Mock successful synthesis
    with patch("subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = b"Successfully synthesized"
        mock_result.stderr = b""
        mock_run.return_value = mock_result

        score = synth.check_synthesis(good_code)
        assert score == 1.0

    # Mock synthesis failure
    with patch("subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = b"ERROR: Cannot synthesize"
        mock_result.stderr = b""
        mock_run.return_value = mock_result

        score = synth.check_synthesis(good_code)
        assert score == -1.0

    # Test with missing yosys
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        score = synth.check_synthesis(good_code)
        assert score == -1.0


def test_code_similarity_checker():
    """Test CodeSimilarityChecker functionality"""
    mock_log = Mock()
    similarity = CodeSimilarityChecker(mock_log)

    # Test identical code
    code1 = """module test(input a, output b);
    assign b = a;
endmodule"""
    score = similarity.calculate_similarity(code1, code1)
    assert score == 0.0  # Identical = 0 (no difference)

    # Test completely different code
    code2 = """module different(input x, input y, output z);
    always @(*) z = x & y;
endmodule"""
    score = similarity.calculate_similarity(code1, code2)
    assert score > 0.40  # Should be quite different

    # Test similar structure, different names
    code3 = """module test(input c, output d);
    assign d = c;
endmodule"""
    score = similarity.calculate_similarity(code1, code3)
    assert 0 < score < 0.5  # Similar but not identical

    # Test with empty code
    assert similarity.calculate_similarity("", "something") == 1.0
    assert similarity.calculate_similarity("something", "") == 1.0


def test_reasoning_checker():
    """Test ReasoningChecker functionality"""
    mock_log = Mock()
    reasoning = ReasoningChecker(mock_log)

    # Test good alignment
    reasoning_text = """
    I need to create a counter with an always block that triggers on posedge clk.
    The counter should use a register to store the count value.
    """
    code = """module counter(
    input clk,
    output reg [7:0] count
);
    always @(posedge clk)
        count <= count + 1;
endmodule"""

    score = reasoning.check_reasoning_alignment(reasoning_text, code)
    assert score > 0  # Should have positive alignment

    # Test poor alignment
    reasoning_text = "I will create a multiplexer with select lines"
    code = """module adder(input a, input b, output sum);
    assign sum = a + b;
endmodule"""

    score = reasoning.check_reasoning_alignment(reasoning_text, code)
    assert score < 0  # Should have negative alignment

    # Test with empty inputs
    assert reasoning.check_reasoning_alignment("", "code") == -1.0
    assert reasoning.check_reasoning_alignment("reasoning", "") == -1.0


def test_reward_scores_dataclass():
    """Test RewardScores functionality"""
    scores = RewardScores()

    # Test initial values
    assert scores.code_quality == 0.0
    assert scores.compilation == 0.0
    assert scores.total_reward == 0.0

    # Test with some scores
    scores.code_quality = 0.8
    scores.compilation = 1.0
    scores.functional_correctness = 0.9
    scores.synthesis = 1.0
    scores.similarity = 0.3
    scores.reasoning = 0.7

    # Check total reward calculation
    total = scores.total_reward
    assert 0 < total < 1  # Should be weighted average

    # Verify weights sum to 1.0 (approximately)
    weights_sum = 0.15 + 0.20 + 0.25 + 0.15 + 0.10 + 0.15
    assert abs(weights_sum - 1.0) < 0.001


def test_rlft_initialization():
    """Test RLFineTuner initialization"""
    model_info = {"name": "test-model", "id": "v1"}

    # Test with PPO policy
    with patch("src.rlft.Logger") as mock_logger:
        mock_logger.return_value.get_logger.return_value = Mock()
        tuner = RLFineTuner(model_info, RLPolicy.PPO)
        assert tuner.model_info == model_info
        assert hasattr(tuner, "policy")
        assert hasattr(tuner, "analyzer")
        assert hasattr(tuner, "compiler")

    # Test with GRPO policy
    with patch("src.rlft.Logger") as mock_logger:
        mock_logger.return_value.get_logger.return_value = Mock()
        tuner = RLFineTuner(model_info, RLPolicy.GRPO)
        assert tuner.model_info == model_info


def test_calculate_reward_integration():
    """Test full reward calculation with mocked components"""
    model_info = {"name": "test-model", "id": "v1"}

    with patch("src.rlft.Logger") as mock_logger:
        mock_logger.return_value.get_logger.return_value = Mock()
        tuner = RLFineTuner(model_info)

        # Mock all component methods
        tuner.analyzer.extract_code = Mock(return_value="module test(); endmodule")
        tuner.analyzer.extract_reasoning = Mock(return_value="Test reasoning")
        tuner.analyzer.analyze_code_quality = Mock(return_value=(0.9, {}))
        tuner.compiler.compile_and_score = Mock(return_value=(1.0, {}))
        tuner.synthesizer.check_synthesis = Mock(return_value=1.0)
        tuner.similarity_checker.calculate_similarity = Mock(return_value=0.2)
        tuner.reasoning_checker.check_reasoning_alignment = Mock(return_value=0.8)

        candidate = """
<reason>
Test reasoning
</reason>

```verilog
module test(); endmodule
```"""

        ground_truth = "module test(); endmodule"
        testbench = "testbench code"

        scores, metrics = tuner.calculate_reward(candidate, ground_truth, testbench)

        # Verify all components were called
        assert tuner.analyzer.extract_code.called
        assert (
            tuner.compiler.compile_and_score.call_count == 2
        )  # compilation + functional
        assert tuner.synthesizer.check_synthesis.called
        assert tuner.similarity_checker.calculate_similarity.called
        assert tuner.reasoning_checker.check_reasoning_alignment.called

        # Verify scores
        assert scores.code_quality == 0.9
        assert scores.compilation == 1.0
        assert scores.functional_correctness == 1.0
        assert scores.synthesis == 1.0
        assert scores.similarity == 0.2
        assert scores.reasoning == 0.8
        assert scores.total_reward > 0


def test_train_step():
    """Test training step functionality"""
    model_info = {"name": "test-model", "id": "v1"}

    with patch("src.rlft.Logger") as mock_logger:
        mock_logger.return_value.get_logger.return_value = Mock()
        tuner = RLFineTuner(model_info)

        # Mock calculate_reward
        mock_scores = RewardScores(
            code_quality=0.9,
            compilation=1.0,
            functional_correctness=0.8,
            synthesis=1.0,
            similarity=0.3,
            reasoning=0.7,
        )
        tuner.calculate_reward = Mock(return_value=(mock_scores, {}))

        batch_data = [
            {
                "prompt": "Create a counter",
                "response": "```verilog\nmodule counter(); endmodule\n```",
                "ground_truth": "module counter(); endmodule",
                "testbench": "testbench code",
            },
            {
                "prompt": "Create an adder",
                "response": "```verilog\nmodule adder(); endmodule\n```",
                "ground_truth": "module adder(); endmodule",
            },
        ]

        result = tuner.train_step(batch_data)

        assert "mean_reward" in result
        assert "individual_metrics" in result
        assert len(result["individual_metrics"]) == 2
        assert result["mean_reward"] > 0


def test_tempfile_cleanup():
    """Test that temporary files are properly cleaned up"""
    mock_log = Mock()
    compiler = VerilogCompiler(mock_log)

    code = "module test(); endmodule"

    # Track created files
    created_files = []
    original_mkstemp = tempfile.mkstemp

    def track_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        created_files.append(path)
        return fd, path

    with patch("tempfile.mkstemp", side_effect=track_mkstemp):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr=b"", stdout=b"")

            # This should create and clean up temp files
            compiler.compile_and_score(code, "testbench")

    # Verify all temp files were cleaned up
    for path in created_files:
        assert not os.path.exists(path)


def test_error_handling():
    """Test error handling in various scenarios"""
    mock_log = Mock()

    # Test analyzer with invalid Creator
    analyzer = VerilogCodeAnalyzer(mock_log)
    result = analyzer.extract_code("code", "invalid_creator")
    assert result is None

    # Test compiler with timeout
    compiler = VerilogCompiler(mock_log, timeout_ms=1)
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 0.001)):
        score, metrics = compiler.compile_and_score("code")
        assert score == -1.0

    # Test synthesis with exception
    synth = SynthesisChecker(mock_log)
    with patch("subprocess.run", side_effect=Exception("Test error")):
        score = synth.check_synthesis("code")
        assert score == -1.0


# ===================== PYTEST CONFIGURATION =====================


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


@pytest.fixture
def mock_logger():
    """Fixture to provide a mock logger"""
    mock_log = Mock()
    mock_log.info = Mock()
    mock_log.error = Mock()
    mock_log.warning = Mock()
    mock_log.critical = Mock()
    return mock_log


@pytest.fixture
def sample_verilog_code():
    """Fixture providing sample Verilog code"""
    return """module counter(
    input clk,
    input rst,
    output reg [7:0] count
);
    always @(posedge clk) begin
        if (rst)
            count <= 8'b0;
        else
            count <= count + 1;
    end
endmodule"""


@pytest.fixture
def sample_testbench():
    """Fixture providing sample testbench code"""
    return """module counter_tb;
    reg clk, rst;
    wire [7:0] count;
    
    counter uut(
        .clk(clk),
        .rst(rst),
        .count(count)
    );
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        rst = 1;
        #20 rst = 0;
        #100 $finish;
    end
endmodule"""


# Run specific test categories
if __name__ == "__main__":
    import sys

    # Run all tests
    pytest.main([__file__] + sys.argv[1:])
