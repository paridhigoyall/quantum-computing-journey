from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

# --------------------------------
# Create circuit
# --------------------------------

qc = QuantumCircuit(1, 1)

# Prepare amplitude a = 1/4
# theta = pi/6
# Ry(2*theta) = Ry(pi/3)

qc.ry(np.pi / 3, 0)

# Measure
qc.measure(0, 0)

print("Amplitude preparation circuit:")
print(qc)

# --------------------------------
# Simulate
# --------------------------------

simulator = AerSimulator()

compiled = transpile(qc, simulator)

result = simulator.run(
    compiled,
    shots=1000
).result()

counts = result.get_counts()

print("\nMeasurement results:")
print(counts)