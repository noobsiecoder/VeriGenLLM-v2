import tempfile
import subprocess
import os


def comprehensive_synthesis_check(verilog_code, top_module=None):
    """
    Comprehensive synthesis check with detailed analysis
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".v", delete=False) as f:
        f.write(verilog_code)
        temp_verilog_file = f.name

    try:
        # Enhanced Yosys script with more checks
        yosys_script = f"""
# Read Verilog
read_verilog -sv {temp_verilog_file}

# Check hierarchy
hierarchy -check {f"-top {top_module}" if top_module else "-auto-top"}

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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ys", delete=False) as f:
            f.write(yosys_script)
            temp_script_file = f.name

        result = subprocess.run(
            ["yosys", "-s", temp_script_file, "-q"],  # -q for quiet mode
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Parse results
        synthesis_info = {
            "synthesizable": result.returncode == 0,
            "has_latches": "$dlatch" in result.stdout or "$sr" in result.stdout,
            "warnings": [],
            "errors": [],
            "stats": {},
        }

        # Extract warnings and errors
        for line in result.stdout.split("\n"):
            if "Warning:" in line:
                synthesis_info["warnings"].append(line)
            elif "ERROR:" in line:
                synthesis_info["errors"].append(line)
                synthesis_info["synthesizable"] = False
            elif "Number of cells:" in line:
                synthesis_info["stats"]["cells"] = line.strip()
            elif "Number of wires:" in line:
                synthesis_info["stats"]["wires"] = line.strip()

        return synthesis_info

    except Exception as e:
        return {"synthesizable": False, "errors": [str(e)], "warnings": [], "stats": {}}
    finally:
        if os.path.exists(temp_verilog_file):
            os.unlink(temp_verilog_file)
        if "temp_script_file" in locals() and os.path.exists(temp_script_file):
            os.unlink(temp_script_file)


verilog_code = None
with open("answer_bcd-fadd.v", "r") as fs:
    verilog_code = fs.read()

print(comprehensive_synthesis_check(verilog_code))
