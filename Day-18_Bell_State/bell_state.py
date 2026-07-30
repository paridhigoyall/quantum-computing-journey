from qiskit import QuantumCircuit

# Create a circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Step 1: Create superposition
qc.h(0)

# Step 2: Create entanglement
qc.cx(0, 1)

# Measure both qubits
qc.measure([0, 1], [0, 1])

# Draw the circuit
print(qc)