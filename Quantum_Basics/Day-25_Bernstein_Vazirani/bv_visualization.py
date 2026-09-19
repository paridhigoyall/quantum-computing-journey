from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

secret = "101"

n = len(secret)

qc = QuantumCircuit(n + 1, n)

qc.x(n)
qc.h(range(n + 1))

for i, bit in enumerate(secret):
    if bit == "1":
        qc.cx(i, n)

qc.h(range(n))

qc.measure(range(n), range(n))

sim = AerSimulator()

compiled = transpile(qc, sim)

result = sim.run(compiled, shots=1000).result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)

plt.title("Bernstein–Vazirani Algorithm")

plt.show()