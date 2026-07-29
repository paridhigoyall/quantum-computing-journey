from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create circuit
qc = QuantumCircuit(1, 1)

# Put qubit in superposition

qc.x(0)

# Measure
qc.measure(0, 0)

# Simulator
simulator = AerSimulator()

compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=10000)

result = job.result()

counts = result.get_counts()

print(counts)

# Plot
plot_histogram(counts)

plt.show()