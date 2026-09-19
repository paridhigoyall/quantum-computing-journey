from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Create circuit
qc = QuantumCircuit(2,1)

# Prepare output qubit
qc.x(1)

# Superposition
qc.h([0,1])

# -------------------
# Constant Oracle
# f(x)=0
# Do nothing
# -------------------

# Interference
qc.h(0)

# Measure
qc.measure(0,0)

print(qc)

# Simulator
simulator = AerSimulator()

compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)