from qiskit import QuantumCircuit

secret = "101"

n = len(secret)

qc = QuantumCircuit(n + 1, n)

# Output qubit
qc.x(n)

# Hadamard
qc.h(range(n + 1))

# Oracle
for i, bit in enumerate(secret):
    if bit == "1":
        qc.cx(i, n)

# Hadamard
qc.h(range(n))

# Measurement
qc.measure(range(n), range(n))

print(qc)