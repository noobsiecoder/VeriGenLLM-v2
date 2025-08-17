"""
Testbench to check evals of all Model

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 16th, 2025
Place:  Boston, MA
"""

import math
import json
import os
import re
import subprocess
import tempfile
from src.evals import Evals, ResponseEvals

evals = Evals()  # Instantiate globally
response_evals = ResponseEvals()  # Instantiate globally


def test_per_question_metrics():
    """
    Test eval metric for model's response per 1 question
    """
    question_response = None
    with open("dataset/results/baselines/evals/evals-code-llama.json", "r") as fs:
        question_response = json.load(fs)
    eval = evals.per_question_eval(question_response[0])
    eval["question"] = question_response[0]["question"]
    assert eval["first_correct_idx"] == -1
    assert math.isclose(eval["pass_k_metric"]["compilation"], 1.0)
    assert math.isclose(eval["pass_k_metric"]["functional_correctness"], 0.0)
    assert math.isclose(eval["pass_k_metric"]["synthesisability"], 1.0)
    assert math.isclose(round(eval["pass_k_metric"]["overall"], 2), 0.67)


def test_response_evals():
    """
    Mock test of running the Response Evals on results/prompts-*.json file
    """
    # Consider "prompts-all.json" wasn't generated
    response_path = "dataset/results/baselines"
    json_files = os.listdir(response_path)
    json_files.remove("prompts-all.json") if "prompts-all.json" in json_files else None
    for json_file in json_files:
        with open(os.path.join(response_path, json_file), "r") as fs:
            responses = json.load(fs)
            # prelim checks
            assert type(responses) == list
            assert len(responses) == 17

            # tight checks
            for response in responses:
                data = {
                    "model": response["response"]["config"]["model"],
                    "temperature": response["response"]["config"]["temperature"],
                    "max_tokens": response["response"]["config"]["max_tokens"],
                    "output": response["response"]["outputs"][0],  # -> iter here
                    "testbench": response["testbench"],
                }
                eval = response_evals.evaluate_response(data)
                assert eval["compilation"]["status"]
                assert eval["functional_correctness"]["status"]
                break


def test_llm_answer_eval():
    """
    Check how many outputs are correct for 'n' samples generated and more data.
    """
    # Check claude model's response
    response = {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.2,
        "max_tokens": 512,
        "output": "```verilog\nmodule adv_fsm(\ninput clk,\ninput reset,\ninput x,\noutput z ); \n\nreg [1:0] present_state, next_state;\nparameter IDLE=0, S1=1, S10=2, S101=3;\n\n// Output logic\nassign z = (present_state == S101);\n\n// State register\nalways @(posedge clk or posedge reset) begin\n    if (reset)\n        present_state <= IDLE;\n    else\n        present_state <= next_state;\nend\n\n// Next state logic\nalways @(*) begin\n    case (present_state)\n        IDLE: begin\n            if (x == 1)\n                next_state = S1;\n            else\n                next_state = IDLE;\n        end\n        \n        S1: begin\n            if (x == 0)\n                next_state = S10;\n            else\n                next_state = IDLE;\n        end\n        \n        S10: begin\n            if (x == 1)\n                next_state = S101;\n            else\n                next_state = IDLE;\n        end\n        \n        S101: begin\n            next_state = IDLE;\n        end\n        \n        default: next_state = IDLE;\n    endcase\nend\n\nendmodule\n```",
        "testbench": '\n`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps\n\nmodule tb_adv_fsm;\n\n    reg clk, reset, x;\n    wire z;\n\n    // duration for each bit = 20 * timescale = 20 * 1 ns  = 20ns\n    localparam period = 20;  \n    adv_fsm UUT (.clk(clk), .reset(reset), .x(x), .z(z) );\n\n    initial // Clock generation\n        begin\n        clk = 0;\n        forever begin\n        #(period/2);\n        clk = ~clk;\n        end\n    end\n\n    initial begin\n\n        #2;\n        \n        // check reset\n        reset = 1; x = 0;\n        #period;\n        // goes to state IDLE\n        if(z!==0) begin\n            $display("test 1 failed");\n            $finish;\n        end\n        else $display("x=%b, z=%b",x,z);\n\n        // start fsm\n        reset = 0; \n        x = 0;\n        #period;\n        // stays in state IDLE\n        if(z!==0) begin\n            $display("test 2 failed");\n            $finish;\n        end\n        else $display("x=%b, z=%b",x,z);\n\n        // start pattern\n        x = 1;\n        #period;\n        // goes to S1\n        if(z!==0) begin\n            $display("test 3 failed");\n            $finish;\n        end\n        else $display("x=%b, z=%b",x,z);\n        x = 0;\n        #period;\n        // goes to S2\n        if(z!==0) begin\n            $display("test 4 failed");\n            $finish;\n        end\n        else $display("x=%b, z=%b",x,z);\n        x = 1;\n        #period;\n        // goes to S3\n        if(z!==1) begin\n            $display("test 5 failed");\n            $finish;\n        end\n        else $display("x=%b, z=%b",x,z);\n\n        #period;\n        // goes to S1\n        if(z!==0) begin\n            $display("test 6 failed");\n            $finish;\n        end\n        else $display("x=%b, z=%b",x,z);\n\n        $display("all tests passed");\n        $finish;\n\n    end\n\nendmodule',
    }
    response_eval = response_evals.evaluate_response(response)
    assert response_eval["meta"]["model"] == response["model"]
    assert response_eval["code_analysis"]["code"]["lines"] == 53
    assert response_eval["code_analysis"]["code"]["comments"] == 3
    assert response_eval["code_analysis"]["code"]["completed"]
    assert response_eval["code_analysis"]["misc"]["md_style"]
    assert response_eval["compilation"]["status"]
    assert response_eval["compilation"]["error"]["count"] == None
    assert response_eval["functional_correctness"]["status"]
    assert response_eval["synthesisability"]["status"]

    # Check codellama model
    response = {
        "model": "meta-llama/CodeLlama-7b-Instruct-hf",
        "temperature": 0.2,
        "max_tokens": 512,
        "output": " module signed_adder(input [7:0] a, input [7:0] b, output [7:0] s, output overflow );\n\n  wire [15:0] a_ext;\n  wire [15:0] b_ext;\n  wire [15:0] s_ext;\n  wire [15:0] sum;\n  wire [15:0] diff;\n  wire [15:0] diff_ext;\n  wire [15:0] diff_abs;\n  wire [15:0] diff_abs_ext;\n  wire [15:0] diff_abs_ext_2;\n  wire [15:0] diff_abs_ext_3;\n  wire [15:0] diff_abs_ext_4;\n  wire [15:0] diff_abs_ext_5;\n  wire [15:0] diff_abs_ext_6;\n  wire [15:0] diff_abs_ext_7;\n  wire [15:0] diff_abs_ext_8;\n  wire [15:0] diff_abs_ext_9;\n  wire [15:0] diff_abs_ext_10;\n  wire [15:0] diff_abs_ext_11;\n  wire [15:0] diff_abs_ext_12;\n  wire [15:0] diff_abs_ext_13;\n  wire [15:0] diff_abs_ext_14;\n  wire [15:0] diff_abs_ext_15;\n  wire [15:0] diff_abs_ext_16;\n  wire [15:0] diff_abs_ext_17;\n  wire [15:0] diff_abs_ext_18;\n  wire [15:0] diff_abs_ext_19;\n  wire [15:0] diff_abs_ext_20;\n  wire [15:0] diff_abs_ext_21;\n  wire [15:0] diff_abs_ext_22;\n  wire [1",
        "testbench": "",
    }
    response_eval = response_evals.evaluate_response(response)
    assert response_eval["meta"]["model"] == response["model"]
    assert response_eval["code_analysis"]["code"]["lines"] == 32
    assert response_eval["code_analysis"]["code"]["comments"] == 0
    assert response_eval["code_analysis"]["code"]["attempted"]
    assert not response_eval["code_analysis"]["code"]["completed"]
    assert not response_eval["code_analysis"]["misc"]["md_style"]
    assert not response_eval["compilation"]["status"]
    assert response_eval["compilation"]["error"]["count"] == None

    # Check dummy response
    response = {
        "model": "dummy-LLM",
        "temperature": 0.2,
        "max_tokens": 512,
        "output": "Hello, World",
        "testbench": "",
    }
    response_eval = response_evals.evaluate_response(response)
    assert response_eval["meta"]["model"] == response["model"]
    assert response_eval["code_analysis"]["code"]["lines"] == 0
    assert response_eval["code_analysis"]["code"]["comments"] == 0
    assert not response_eval["code_analysis"]["code"]["attempted"]
    assert not response_eval["code_analysis"]["code"]["completed"]
    assert not response_eval["code_analysis"]["misc"]["md_style"]
    assert not response_eval["compilation"]["status"]
    assert response_eval["compilation"]["error"]["count"] == None

    # Check dummy response -> error code
    # Re-initialized input [7:0] a
    response = {
        "model": "dummy-LLM",
        "temperature": 0.2,
        "max_tokens": 512,
        "output": "```verilog\nmodule signed_adder(input [7:0] a, input [7:0] b, output [7:0] s, output overflow);\n\ninput [7:0] a\nassign s = a + b;\nassign overflow = (a[7] == b[7]) && (s[7] != a[7]);\n\nendmodule\n```",
        "testbench": '\n`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps\n\nmodule tb_signed_adder;\n\n    reg [7:0] a,b;\n    wire [7:0] s;\n    wire overflow;\n\n    integer increment_a = 0;\n    \n    // duration for each bit = 2 * timescale = 2 * 1 ns  = 2ns\n    localparam period = 2;  \n\n    signed_adder UUT (.a(a), .b(b), .s(s), .overflow(overflow) );\n    \n    initial // initial block executes only once\n        begin\n\n            // values for inputs\n            a = 0; b= 0;\n            while (b<8\'b1111_1111) begin\n\n                #period; // wait for period \n                if (s!==a+b || overflow!==(a[7]&&b[7]&&(~s[7]) || (~a[7])&&(~b[7])&&(s[7])) ) begin\n                    $display("test failed");\n                    $display("  a = %b , b = %b, sum=%b, overflow = %b", a,b,s,overflow);\n                    $finish;\n                end\n                $display("  a = %b , b = %b, sum=%b, overflow = %b", a,b,s,overflow);\n\n                if (increment_a) a = a+1;\n                else b = b+1;\n\n                if (increment_a) increment_a = 0;\n                else increment_a=1; \n            end\n        \n            $display("all tests passed");\n            $finish;\n\n        end\n\nendmodule',
        "testbench": "",
    }
    response_eval = response_evals.evaluate_response(response)
    assert response_eval["meta"]["model"] == response["model"]
    assert response_eval["code_analysis"]["code"]["lines"] == 7
    assert response_eval["code_analysis"]["code"]["comments"] == 0
    assert response_eval["code_analysis"]["code"]["attempted"]
    assert response_eval["code_analysis"]["code"]["completed"]
    assert response_eval["code_analysis"]["misc"]["md_style"]
    assert not response_eval["compilation"]["status"]
    assert response_eval["compilation"]["error"]["count"] == 1
    assert not response_eval["functional_correctness"]["status"]
    # assert response_eval["synthesizability"]["status"]


def test_mock_code_analysis():
    """
    Mock test for code analysis
    """
    # claude code
    code_str = "```verilog\nmodule signed_adder(input [7:0] a, input [7:0] b, output [7:0] s, output overflow);\n\nassign s = a + b;\nassign overflow = (a[7] == b[7]) && (s[7] != a[7]);\n\nendmodule\n```"
    pattern = r"```verilog\n(.*)```"
    match = re.search(pattern, code_str, re.DOTALL)
    assert match
    if match:
        code = match.group(1)
        # Check for lines
        newline_pattern = r"\n"
        expression = re.compile(newline_pattern)
        assert len(expression.findall(code)) + 1 == 7

    # claude code with comments
    code_str = "```verilog\nmodule adv_fsm(\ninput clk,\ninput reset,\ninput x,\noutput z ); \n\nreg [1:0] present_state, next_state;\nparameter IDLE=0, S1=1, S10=2, S101=3;\n\n// Output logic\nassign z = (present_state == S101);\n\n// State register\nalways @(posedge clk or posedge reset) begin\n    if (reset)\n        present_state <= IDLE;\n    else\n        present_state <= next_state;\nend\n\n// Next state logic\nalways @(*) begin\n    case (present_state)\n        IDLE: begin\n            if (x == 1)\n                next_state = S1;\n            else\n                next_state = IDLE;\n        end\n        \n        S1: begin\n            if (x == 0)\n                next_state = S10;\n            else\n                next_state = IDLE;\n        end\n        \n        S10: begin\n            if (x == 1)\n                next_state = S101;\n            else\n                next_state = IDLE;\n        end\n        \n        S101: begin\n            next_state = IDLE;\n        end\n        \n        default: next_state = IDLE;\n    endcase\nend\n\nendmodule\n```"
    pattern = r"(module.*\nendmodule)"
    match = re.search(pattern, code_str, re.DOTALL)
    assert match
    if match:
        code = match.group(1)
        # Check for comments
        comments_pattern = r"\/\/"
        expression = re.compile(comments_pattern)
        assert len(expression.findall(code)) == 3

        # Check if MD style is followed
        md_pattern = r"```verilog(.*)```"
        new_match = re.search(md_pattern, code_str, re.DOTALL)
        assert new_match
        if new_match:
            assert code.strip() == new_match.group(1).strip()

    # codellama code with explanation
    code_str = " Here is the Verilog code for the finite state machine you described:\n```\nmodule adv_fsm(\n    input clk,\n    input reset,\n    input x,\n    output z\n);\n\nreg [1:0] present_state, next_state;\nparameter IDLE=0, S1=1, S10=2, S101=3;\n\nalways @(posedge clk or posedge reset) begin\n    if (reset) begin\n        present_state <= IDLE;\n    end else begin\n        present_state <= next_state;\n    end\nend\n\nalways @(*) begin\n    case (present_state)\n        IDLE: begin\n            if (x) begin\n                next_state <= S1;\n            end else begin\n                next_state <= IDLE;\n            end\n        end\n        S1: begin\n            if (x) begin\n                next_state <= S10;\n            end else begin\n                next_state <= IDLE;\n            end\n        end\n        S10: begin\n            if (x) begin\n                next_state <= S101;\n            end else begin\n                next_state <= IDLE;\n            end\n        end\n        S101: begin\n            next_state <= IDLE;\n        end\n    endcase\nend\n\nassign z = (present_state == S101);\n\nendmodule\n```\nThis code defines a module `adv_fsm` that takes an input signal `x` and an output signal `z`. The module has a finite state machine that recognizes the sequence 101 on the input signal `x`. The state machine has four states: `IDLE`, `S1`, `S10`, and `S101`. The output signal `z` is asserted to 1 when the state machine is in the `S101` state. The state machine is reset to the `IDLE` state when the reset signal is high. Otherwise, it is assigned the next state based on the current state and the input signal `x`."
    # The best pattern selector -> get module and endmodule
    # If no module/endmodule -> omit operation
    pattern = r"(module.*\nendmodule)"
    match = re.search(pattern, code_str, re.DOTALL)
    assert match
    if match:
        code = match.group(1)
        # Check for lines
        newline_pattern = r"\n"
        expression = re.compile(newline_pattern)
        assert len(expression.findall(code)) + 1 == 50
        # Check for comments
        comments_pattern = r"\/\/"
        expression = re.compile(comments_pattern)
        assert len(expression.findall(code)) == 0


def test_mock_warn_error_grouping():
    """
    Mock test to check if Warnings and Errors are grouped properly
    """
    warnings = []
    w_idx = -1
    errors = []
    stderr_data = [
        "warning: Some modules have no timescale. This may cause",
        "       : confusing timing results.\tAffected modules are:",
        "       :   -- module left_rotate declared here: /var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:1",
        "warning: Some modules have no timescale. This may cause",
        "       : confusing timing results.\tAffected modules are:",
        "       :   -- module left_rotate declared here: /var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:1",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17: error: A reference to a wire or reg (`amount') is not allowed in a constant expression.",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17: error: Part select expressions must be constant.",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17:      : This msb expression violates the rule: (amount)-('sd1)",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17: error: A reference to a wire or reg (`amount') is not allowed in a constant expression.",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17: error: Part select expressions must be constant.",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17:      : This lsb expression violates the rule: amount",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17: error: A reference to a wire or reg (`amount') is not allowed in a constant expression.",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17: error: Part select expressions must be constant.",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17:      : This msb expression violates the rule: (amount)-('sd1)",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17: error: A reference to a wire or reg (`amount') is not allowed in a constant expression.",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17: error: Part select expressions must be constant.",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:17:      : This lsb expression violates the rule: amount",
        "/var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpiklbtymb.v:12: warning: Instantiating module left_rotate with dangling input port 2 (reset) floating.",
        "8 error(s) during elaboration.",
    ]
    idx = 0
    while idx < len(stderr_data):
        if "warning" in stderr_data[idx]:
            warnings.append(stderr_data[idx])
            w_idx += 1
        elif "error" in stderr_data[idx]:
            idx += 1
            break
        else:
            warnings[w_idx] += stderr_data[idx]
        idx += 1
    for i in range(idx, len(stderr_data)):
        errors.append(stderr_data[i])

    assert len(warnings) == 2
    assert warnings == [
        "warning: Some modules have no timescale. This may cause       : confusing timing results.\tAffected modules are:       :   -- module left_rotate declared here: /var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:1",
        "warning: Some modules have no timescale. This may cause       : confusing timing results.\tAffected modules are:       :   -- module left_rotate declared here: /var/folders/rd/p1qd4kln5sndfk_3g_db5tqh0000gn/T/tmpcm368483.v:1",
    ]
    assert len(errors) == 13


def test_mock_code_compilation():
    """
    Mock test for code compilation analysis
    """
    # claude code with comments
    # This code will fail compilation -> string not cleaned
    code_str = "```verilog\nmodule adv_fsm(\ninput clk,\ninput reset,\ninput x,\noutput z ); \n\nreg [1:0] present_state, next_state;\nparameter IDLE=0, S1=1, S10=2, S101=3;\n\n// Output logic\nassign z = (present_state == S101);\n\n// State register\nalways @(posedge clk or posedge reset) begin\n    if (reset)\n        present_state <= IDLE;\n    else\n        present_state <= next_state;\nend\n\n// Next state logic\nalways @(*) begin\n    case (present_state)\n        IDLE: begin\n            if (x == 1)\n                next_state = S1;\n            else\n                next_state = IDLE;\n        end\n        \n        S1: begin\n            if (x == 0)\n                next_state = S10;\n            else\n                next_state = IDLE;\n        end\n        \n        S10: begin\n            if (x == 1)\n                next_state = S101;\n            else\n                next_state = IDLE;\n        end\n        \n        S101: begin\n            next_state = IDLE;\n        end\n        \n        default: next_state = IDLE;\n    endcase\nend\n\nendmodule\n```"
    temp_sol_filepath = None
    with tempfile.NamedTemporaryFile(
        suffix=".v",
        mode="w+",
        delete=False,  # Keep file after closing
    ) as temp_file:
        temp_file.write(code_str)
        temp_sol_filepath = temp_file.name

    result = subprocess.run(
        [
            "iverilog",
            "-Wall",  # Uncomment to display warnings
            "-o",
            "test_output",  # Output binary name
            temp_sol_filepath,
        ],
        capture_output=True,  # Capture stdout and stderr
        text=True,  # Return output as string
    )

    # Remove temp file
    os.remove(temp_sol_filepath)
    # Remove output file
    if os.path.exists("test_output"):
        os.remove("test_output")
    # Check compilation result
    assert result.returncode == 2
    # Check for errors and count
    errors = result.stderr.splitlines()
    if "I give up." in errors:
        errors.remove("I give up.")
    assert len(errors) == 2

    # claude code
    # This code will produce warning -> added unused variable
    code_str = "module signed_adder(input [7:0] a, input [7:0] b, output [7:0] s, output overflow);\n\nassign undeclared_wire = 1'b1;\nassign s = a + b;\nassign overflow = (a[7] == b[7]) && (s[7] != a[7]);\n\nendmodule"
    temp_sol_filepath = None
    with tempfile.NamedTemporaryFile(
        suffix=".v",
        mode="w+",
        delete=False,  # Keep file after closing
    ) as temp_file:
        temp_file.write(code_str)
        temp_sol_filepath = temp_file.name

    result = subprocess.run(
        [
            "iverilog",
            "-Wall",
            "-Winfloop",  # Warn about infinite loops
            "-Wanachronisms",  # Warn about old constructs
            "-Wsensitivity-entire-array",  # Array sensitivity warnings
            "-o",
            "test_output",  # Output binary name
            temp_sol_filepath,
        ],
        capture_output=True,  # Capture stdout and stderr
        text=True,  # Return output as string
    )

    # Remove temp file
    os.remove(temp_sol_filepath)
    # Remove output file
    if os.path.exists("test_output"):
        os.remove("test_output")
    # Check compilation result
    assert result.returncode == 0
    # Check for warnings
    if result.stderr:
        warnings = result.stderr.splitlines()
        assert len(warnings) == 1


def test_mock_code_functional_correctness():
    """
    Mock test for code functional correctness analysis
    """
    # claude code -> modified
    code_str = "module counter( \ninput clk,\ninput reset,\noutput reg [3:0] q\n); \n\nalways @(posedge clk or posedge reset) begin\n    if (reset) begin\n        q <= 4'd0;\n    end else begin\n        if (q >= 4'd11) begin\n            q <= 4'd0;\n        end else begin\n            q <= q + 2;\n        end\n    end\nend\n\nendmodule"
    testbench_code_str = '`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps\n\nmodule tb_counter;\n\n    reg clk, reset;\n    wire [3:0] q;\n\n    // duration for each bit = 20 * timescale = 20 * 1 ns  = 20ns\n    localparam period = 20;  \n    counter UUT (.clk(clk), .reset(reset), .q(q) );\n\n    initial // Clock generation\n        begin\n        clk = 0;\n        forever begin\n        #(period/2);\n        clk = ~clk;\n        end\n    end\n\n    initial begin\n\n        #2;\n        \n        // check reset\n        reset = 1;\n        #period;\n        if(q!==1) begin\n            $display("test 1 failed");\n            $finish;\n        end\n        else $display("clk=%b, reset=%b, q=%b",clk,reset, q);\n\n        // check value does not change during reset\n        #period;\n        if(q!==1) begin\n            $display("test 1a failed");\n            $finish;\n        end\n        else $display("clk=%b, reset=%b, q=%b",clk,reset, q);\n\n        // start counter\n        reset = 0;\n        #period;\n        if(q!==2) begin\n            $display("test 2 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==3) begin\n            $display("test 3 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==4) begin\n            $display("test 4 failed");\n            //$finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==5) begin\n            $display("test 5 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==6) begin\n            $display("test 6 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==7) begin\n            $display("test 7 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==8) begin\n            $display("test 8 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==9) begin\n            $display("test 9 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==10) begin\n            $display("test 10 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==11) begin\n            $display("test 11 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==12) begin\n            $display("test 12 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        // counter should go back to 1\n        #period;\n        if(q!==1) begin\n            $display("test 13 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n        \n\n        // check reset after a few cycles\n        #period;\n        if(q!==2) begin\n            $display("test 14 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==3) begin\n            $display("test 15 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==4) begin\n            $display("test 16 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        #period;\n        if(q!==5) begin\n            $display("test 17 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n        reset = 1;\n        #period;\n        if(q!==1) begin\n            $display("test 18 failed");\n            $finish;\n        end\n        else $display("q=%b",q);\n\n\n        $display("all tests passed");\n        $finish;\n\n    end\n\nendmodule'
    temp_sol_filepath = None
    with tempfile.NamedTemporaryFile(
        suffix=".v",
        mode="w+",
        delete=False,  # Keep file after closing
    ) as temp_file:
        temp_file.write(code_str)
        temp_sol_filepath = temp_file.name

    temp_tb_filepath = None
    with tempfile.NamedTemporaryFile(
        suffix=".v",
        mode="w+",
        delete=False,  # Keep file after closing
    ) as temp_file:
        temp_file.write(testbench_code_str)
        temp_tb_filepath = temp_file.name

    result = subprocess.run(
        [
            "iverilog",
            "-Wall",
            "-Winfloop",  # Warn about infinite loops
            "-Wanachronisms",  # Warn about old constructs
            "-Wsensitivity-entire-array",  # Array sensitivity warnings
            "-o",
            "test_output",  # Output binary name
            temp_sol_filepath,
            temp_tb_filepath,
        ],
        capture_output=True,  # Capture stdout and stderr
        text=True,  # Return output as string
    )
    # Check compilation result
    assert result.returncode == 0
    # Check for warnings and errors
    assert result.stderr  # produces warning

    os.remove(temp_sol_filepath)  # remove temp file from local machine
    os.remove(temp_tb_filepath)  # remove temp file from local machine

    if os.path.exists("test_output"):
        func_result = subprocess.run(
            ["vvp", "test_output"],
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Return output as string
        )
        os.remove("test_output")  # remove output file from local machine
        assert func_result.returncode == 0
        if func_result.stdout:
            stdout = func_result.stdout
            assert "all tests passed" not in stdout

            fail_pattern = r"(test.*failed)"
            expression = re.compile(fail_pattern)
            assert expression.findall(stdout) == ["test 1 failed"]
