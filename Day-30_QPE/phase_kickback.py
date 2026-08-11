from qiskit import QuantumCircuit

qc = QuantumCircuit(2)

# Put control into superposition
qc.h(0)

# Prepare target |1>
qc.x(1)

# Controlled-Z
qc.cz(0, 1)

print(qc)
