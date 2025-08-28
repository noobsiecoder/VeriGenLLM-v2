# Read Verilog
read_verilog -sv answer_vector100.v

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
