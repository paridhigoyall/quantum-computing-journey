from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# 3 input qubits + 1 output qubit
qc = QuantumCircuit(4, 3)

# Prepare output qubit
qc.x(3)

# Apply Hadamard
qc.h(range(4))

# -------------------------
# Constant Oracle
# f(x)=0
# (Do Nothing)
# -------------------------

# Apply Hadamard again to input qubits
qc.h(range(3))

# Measure
qc.measure(range(3), range(3))

print(qc)

sim = AerSimulator()

compiled = transpile(qc, sim)

result = sim.run(compiled, shots=1000).result()

counts = result.get_counts()

print(counts)