from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

qc = QuantumCircuit(3, 3)

# Prepare a non-zero input state: |101>
qc.x(0)
qc.x(2)

# -------------------------
# 3-qubit QFT
# -------------------------

qc.h(0)
qc.cp(np.pi / 2, 1, 0)
qc.cp(np.pi / 4, 2, 0)

qc.h(1)
qc.cp(np.pi / 2, 2, 1)

qc.h(2)

# Reverse qubit order
qc.swap(0, 2)

# Measure
qc.measure(range(3), range(3))

print(qc)

# -------------------------
# Simulation
# -------------------------

simulator = AerSimulator()

compiled = transpile(qc, simulator)

result = simulator.run(
    compiled,
    shots=1024
).result()

counts = result.get_counts()

print("\nMeasurement results:")
print(counts)