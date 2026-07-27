from qiskit import QuantumCircuit

qc = QuantumCircuit(1)

qc.h(0)

print(qc)

#2 gates together
qc = QuantumCircuit(1)

qc.h(0)
qc.x(0)

print(qc)