"""
This is a verilog testbench written in Python
Checks verilog code if it has `wire ..` statement

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 4th, 2025
"""

import re
import sys
from verigenllm_v2.utils.logger import Logger

log = Logger("func_test").get_logger()


def check_code(solution_code: str) -> bool:
    """
    Run code checker -> uses regex to check if statement is found.

    Parameters:
    -----------
    solution_code: str
        string parsed from verilog file

    Returns:
    --------
    Boolean value if statement is present
    """
    pattern = r"wire\s+([a-zA-Z_].*)"
    occurence = 0
    for line in solution_code.strip().split("\n"):
        match = re.search(pattern, line)
        if match:
            occurence += 1
            log.info(
                f"Wire declared in solution code: {match.group(1)} and occurence = {occurence}"
            )

    if occurence == 1:
        log.info("✓ Simulation: PASS")
        return True
    else:
        log.error(f"✗ Simulation Failed: occurence count: {occurence}")
        return False


if __name__ == "__main__":
    filepath = sys.argv[1]
    with open(filepath, "r") as fs:
        solution_code = fs.read()
        check_code(solution_code)
