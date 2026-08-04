from qiskit import QuantumCircuit

qc = QuantumCircuit(4)

# Apply Hadamard to every qubit
qc.h(range(4))

print(qc)