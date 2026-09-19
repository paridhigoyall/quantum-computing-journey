from qiskit import QuantumCircuit

from oracle import oracle
from diffuser import diffuser


qc = QuantumCircuit(2, 2)

# ------------------------
# Superposition
# ------------------------

qc.h([0, 1])

# ------------------------
# Oracle
# ------------------------

qc.compose(oracle(), inplace=True)

# ------------------------
# Diffuser
# ------------------------

qc.compose(diffuser(), inplace=True)

# ------------------------
# Measurement
# ------------------------

qc.measure([0, 1], [0, 1])

print(qc)