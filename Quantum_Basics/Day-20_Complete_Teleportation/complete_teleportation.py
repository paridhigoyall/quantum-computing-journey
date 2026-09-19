from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile

# Create circuit
qc = QuantumCircuit(3, 2)

# ----------------------------
# Prepare Alice's qubit
# ----------------------------
qc.h(0)

# ----------------------------
# Create Bell Pair
# ----------------------------
qc.h(1)
qc.cx(1, 2)

# ----------------------------
# Bell Measurement
# ----------------------------
qc.cx(0, 1)
qc.h(0)

# ----------------------------
# Measure Alice
# ----------------------------
qc.measure([0,1],[0,1])

print(qc)

simulator = AerSimulator()

compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=1024)

result = job.result()

counts = result.get_counts()

print(counts)