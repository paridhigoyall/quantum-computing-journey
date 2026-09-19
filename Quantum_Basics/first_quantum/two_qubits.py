qc = QuantumCircuit(2)

qc.h(0)
qc.x(1)

print(qc)

#creating entanglement
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)

qc.h(0)

qc.cx(0,1)

print(qc)
qc.draw("mpl")