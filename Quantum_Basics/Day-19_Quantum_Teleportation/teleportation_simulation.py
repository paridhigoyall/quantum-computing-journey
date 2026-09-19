from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qc = QuantumCircuit(3, 3)

# Alice's qubit
qc.h(0)

# Bell pair
qc.h(1)
qc.cx(1, 2)

# Teleportation operations
qc.cx(0, 1)
qc.h(0)

# Measure Alice's qubits
qc.measure([0, 1], [0, 1])

simulator = AerSimulator()

compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=1024)

result = job.result()

counts = result.get_counts()

print(counts)