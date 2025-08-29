"""
Run tests on PPO in CPU (or even GPU if found)
Primarily focuses on mock tests for PPO

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 28th, 2025
Place:  Boston, MA
"""

import textwrap
from unittest.mock import patch
from src.rewards import RewardFunction
from constants import ENVLoader


ENVLoader()  # load ENV globally


def test_reward_function(monkeypatch):
    """
    Mock test reward function
    """
    reward_func = RewardFunction()

    # Compilation success = 1.00
    code = textwrap.dedent("""
        module top_module( input in, output out );
            assign out = in;
        endmodule          
        """)
    assert reward_func.compilation_score(code_in_str=code)[0] == 1.0
    # Compilation majorly success <= 0.95
    code = textwrap.dedent("""
        module top_module( input in, output out );
            // Create an implicit net
            assign implicit_net = 1'b0;
                           
            // Create another implicit net
            assign another_implicit_net = 1'b0;
        endmodule       
        """)
    score = reward_func.compilation_score(code_in_str=code)[0]
    assert 0.50 < score and score <= 0.95
    # Compilation partially success <= 0.6
    code = textwrap.dedent("""
        module top_module( input in, output out );
            // Create an implicit net
            assign implicit_net = 1'b0;
                           
            // Create more implicit net
            assign another_implicit_net2 = 1'b0;
            assign another_implicit_net3 = 1'b0;
            assign another_implicit_net4 = 1'b0;
            assign another_implicit_net5 = 1'b0;
            assign another_implicit_net6 = 1'b0;
            assign another_implicit_net7 = 1'b0;
            assign another_implicit_net8 = 1'b0;
            assign another_implicit_net9 = 1'b0;
            assign another_implicit_net0 = 1'b0;
            
            // Create a combinational loop warning
            wire a, b;
            assign a = b;
            assign b = a;
            
            // Unused parameter
            parameter UNUSED_PARAM = 8;
            
            // Create a situation with potential issues
            wire [3:0] wide_sig = 4'b1010;
            wire narrow_sig;
            assign narrow_sig = wide_sig;  // Width mismatch
            
            // Another implicit net usage
            assign another_implicit = in & implicit_net;
        endmodule
        """)
    score = reward_func.compilation_score(code_in_str=code)[0]
    assert 0 < score and score <= 0.6
    # Compilation failure = -1.00
    code = textwrap.dedent("""
        module top_module( input in, output out );
            assign in = out // missing semi-colon
        endmodule          
        """)
    assert reward_func.compilation_score(code_in_str=code)[0] == -1.0

    # Synthesise success = 1.00
    code = textwrap.dedent("""
        module top_module( input in, output out );
            assign out = in;
        endmodule          
        """)
    assert reward_func.sythesise_score(cd_code_in_str=code) == 1.0
    # Synthesise failure = -1.00
    code = textwrap.dedent("""
        module top_module( input in, output out );
            assign out = in
        endmodule          
        """)
    assert reward_func.sythesise_score(cd_code_in_str=code) == -1.0

    # Reasoning bad == 0.00
    fake_answer = "```json\n{\n  \"score\": 0,\n  \"reasons\": [\n    \"The reasoning states that the output 'out' should be assigned to the input 'in'.\",\n    \"The code incorrectly assigns 'in' to 'out', which is the reverse of what the reasoning describes.\",\n    \"The assignment statement in the code is 'assign in = out', but it should be 'assign out = in' to match the reasoning.\",\n    \"The code does not implement the described behavior of assigning the output to the input.\"\n  ]\n}\n```"

    # monkeypatch the OpenAI client so it never calls real API
    class DummyClient:
        def connect(self):
            pass

        def generate(self, prompt, max_tokens, n_samples):
            return {"outputs": [fake_answer]}

    monkeypatch.setattr("src.rewards.OpenAIAPIClient", lambda: DummyClient())
    response = textwrap.dedent("""
        <reason>
        Here's a verilog module
        </reason>
        module top_module( input in, output out );
            assign in = out
        endmodule
    """)
    response = reward_func.reasoning_score(response)
    assert response["score"] == 0.0

    # Code quality success = 1.00
    code = textwrap.dedent("""
        module top_module( input in, output out );
            assign out = in;
        endmodule          
        """)
    quality = reward_func.code_quality_score(cd_code_in_str=code)[0]
    assert quality == 1.0
    # Code quality majorly success <= 0.8
    code = textwrap.dedent("""
        module top_module( input in, output out );
        endmodule          
        """)
    quality = reward_func.code_quality_score(cd_code_in_str=code)[0]
    assert 0.5 < quality and quality <= 0.8
