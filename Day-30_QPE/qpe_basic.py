from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 1)

# Target eigenstate |1>
qc.x(1)

# Control superposition
qc.h(0)

# Controlled-Z
qc.cz(0, 1)

# Hadamard / inverse-QFT for one qubit
qc.h(0)

# Measure phase
qc.measure(0, 0)

print(qc)