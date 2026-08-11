from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# --------------------------------
# Quantum Phase Estimation
# Phase = 1/2
# --------------------------------

qc = QuantumCircuit(2, 1)

# Prepare eigenstate |1>
qc.x(1)

# Put phase/control qubit into superposition
qc.h(0)

# Controlled-Z
qc.cz(0, 1)

# Inverse QFT for one qubit
# For one qubit, inverse QFT = Hadamard
qc.h(0)

# Measure the phase qubit
qc.measure(0, 0)

print("QPE Circuit:")
print(qc)

# --------------------------------
# Simulation
# --------------------------------

simulator = AerSimulator()

compiled_circuit = transpile(qc, simulator)

result = simulator.run(
    compiled_circuit,
    shots=1000
).result()

counts = result.get_counts()

print("\nMeasurement Results:")
print(counts)