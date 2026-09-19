from qiskit import QuantumCircuit
import numpy as np

qc = QuantumCircuit(2)

qc.h(0)

qc.cp(np.pi / 2, 1, 0)

print(qc)