from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create circuit
qc = QuantumCircuit(2, 2)

# Create Bell state
#qc.h(0)
qc.cx(0, 1)
#for experiment comment both gates alternatively and then save them and run to know the differnece 
# Measure both qubits
qc.measure([0, 1], [0, 1])

# Simulator
simulator = AerSimulator()

compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)

# Plot histogram
plot_histogram(counts)

plt.show()