from qiskit import QuantumCircuit

qc = QuantumCircuit(1)

qc.h(0)

qc.s(0)

qc.t(0)

print(qc)


#S rotates the phase by 90°
#T rotates the phase by 45°