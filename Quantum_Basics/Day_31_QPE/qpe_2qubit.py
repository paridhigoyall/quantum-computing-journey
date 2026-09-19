from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

# 2 phase qubits + 1 target qubit
qc = QuantumCircuit(3, 2)

# -------------------------
# Prepare target eigenstate |1>
# -------------------------
qc.x(2)

# -------------------------
# Put phase register into superposition
# -------------------------
qc.h(0)
qc.h(1)

# -------------------------
# Controlled-U
# U = S
# Eigenvalue on |1> = i
# -------------------------
qc.cp(np.pi / 2, 0, 2)

# -------------------------
# Controlled-U²
# S² = Z
# -------------------------
qc.cz(1, 2)

# -------------------------
# Inverse QFT
# -------------------------
qc.h(1)
qc.cp(-np.pi / 2, 1, 0)
qc.h(0)

qc.swap(0, 1)

# -------------------------
# Measurement
# -------------------------
qc.measure(0, 0)
qc.measure(1, 1)

print("QPE Circuit:")
print(qc)

# -------------------------
# Simulation
# -------------------------
simulator = AerSimulator()

compiled = transpile(qc, simulator)

result = simulator.run(
    compiled,
    shots=1000
).result()

counts = result.get_counts()

print("\nMeasurement Results:")
print(counts)