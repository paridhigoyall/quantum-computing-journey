from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

qc = QuantumCircuit(4,3)

qc.x(3)

qc.h(range(4))

# Balanced Oracle
qc.cx(0,3)
qc.cx(1,3)
qc.cx(2,3)

qc.h(range(3))

qc.measure(range(3), range(3))

sim = AerSimulator()

compiled = transpile(qc, sim)

result = sim.run(compiled, shots=1000).result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)

plt.title("Deutsch–Jozsa Algorithm")

plt.show()