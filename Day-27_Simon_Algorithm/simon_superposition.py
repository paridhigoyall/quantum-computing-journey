from qiskit import QuantumCircuit

n = 3

qc = QuantumCircuit(n)

qc.h(range(n))

print(qc)