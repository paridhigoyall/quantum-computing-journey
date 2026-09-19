from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

qc = QuantumCircuit(4,4)

qc.h(range(4))

qc.measure(range(4), range(4))

simulator = AerSimulator()

compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=4096)

counts = job.result().get_counts()

print(counts)

plot_histogram(counts)

plt.show()