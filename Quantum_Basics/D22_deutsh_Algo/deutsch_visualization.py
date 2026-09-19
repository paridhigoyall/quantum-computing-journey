from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

qc = QuantumCircuit(2,1)

qc.x(1)

qc.h([0,1])

# Balanced Oracle
qc.cx(0,1)

qc.h(0)

qc.measure(0,0)

simulator = AerSimulator()

compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=1000)

counts = job.result().get_counts()

print(counts)

plot_histogram(counts)

plt.show()