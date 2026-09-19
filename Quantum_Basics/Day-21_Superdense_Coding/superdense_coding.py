from qiskit import QuantumCircuit


# 2 qubits + 2 classical bits
qc = QuantumCircuit(2, 2)


# -------------------------
# Step 1: Create Bell Pair
# -------------------------

qc.h(0)
qc.cx(0, 1)


# -------------------------
# Step 2: Alice encodes 11
# -------------------------

qc.z(0)
qc.x(0)


# -------------------------
# Step 3: Bob decodes
# -------------------------

qc.cx(0, 1)
qc.h(0)


# -------------------------
# Measurement
# -------------------------

qc.measure([0,1], [0,1])


print(qc)