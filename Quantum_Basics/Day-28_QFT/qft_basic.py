from qiskit import QuantumCircuit

qc = QuantumCircuit(2)

# First Hadamard
qc.h(0)

# Controlled Phase
qc.cp(3.14159/2, 1, 0)

# Second Hadamard
qc.h(1)

# Swap
qc.swap(0,1)

print(qc)