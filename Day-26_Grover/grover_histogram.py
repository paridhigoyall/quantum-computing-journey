from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

from oracle import oracle
from diffuser import diffuser

qc = QuantumCircuit(2, 2)

qc.h([0, 1])

qc.compose(oracle(), inplace=True)

qc.compose(diffuser(), inplace=True)

qc.measure([0, 1], [0, 1])

sim = AerSimulator()

compiled = transpile(qc, sim)

job = sim.run(compiled, shots=1000)

counts = job.result().get_counts()

print(counts)

plot_histogram(counts)

plt.title("Grover Search (Target = |11⟩)")

plt.show()