from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# ----------------------------------------
# Create Quantum Circuit
# ----------------------------------------
qc = QuantumCircuit(3, 3)

# ----------------------------------------
# Step 1 : Prepare Alice's qubit
# (Using Hadamard just for demonstration)
# ----------------------------------------
qc.h(0)

# ----------------------------------------
# Step 2 : Create Bell Pair
# ----------------------------------------
qc.h(1)
qc.cx(1, 2)

# ----------------------------------------
# Step 3 : Alice's Operations
# ----------------------------------------
qc.cx(0, 1)
qc.h(0)

# ----------------------------------------
# Step 4 : Measure Alice's qubits
# ----------------------------------------
qc.measure([0, 1], [0, 1])

# ----------------------------------------
# Run Simulation
# ----------------------------------------
simulator = AerSimulator()

compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=1024)

result = job.result()

counts = result.get_counts()

print("Measurement Counts:")
print(counts)

# ----------------------------------------
# Plot Histogram
# ----------------------------------------
plot_histogram(counts)

plt.title("Quantum Teleportation - Measurement Results")

plt.show()
