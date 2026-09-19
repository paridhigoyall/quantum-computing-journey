from qiskit import QuantumCircuit

# 3 qubits, 3 classical bits
qc = QuantumCircuit(3, 3)

# ---------------------------------
# Step 1: Prepare Alice's qubit
# ---------------------------------
qc.h(0)

# ---------------------------------
# Step 2: Create Bell Pair
# ---------------------------------
qc.h(1)
qc.cx(1, 2)

# ---------------------------------
# Step 3: Alice's Operations
# ---------------------------------
qc.cx(0, 1)
qc.h(0)

# ---------------------------------
# Step 4: Measure Alice's qubits
# ---------------------------------
qc.measure([0, 1], [0, 1])

print(qc)