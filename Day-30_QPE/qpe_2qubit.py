from qiskit import QuantumCircuit

qc = QuantumCircuit(3, 2)

# -------------------------
# Prepare target eigenstate |1>
# -------------------------

qc.x(2)

# -------------------------
# Phase register
# -------------------------

qc.h(0)
qc.h(1)

# -------------------------
# Controlled-U
# U = S
# -------------------------

qc.cp(3.141592653589793 / 2, 0, 2)

# -------------------------
# Controlled-U²
# U² = Z
# -------------------------

qc.cz(1, 2)

# -------------------------
# Inverse QFT on phase register
# -------------------------

#qc.swap(0, 1)

#qc.h(0)

#qc.cp(-3.141592653589793 / 2, 1, 0)

#qc.h(1)


# Inverse QFT
qc.h(1)
qc.cp(-3.141592653589793 / 2, 1, 0)
qc.h(0)
qc.swap(0, 1)
# -------------------------
# Measurement
# -------------------------

qc.measure(0, 0)
qc.measure(1, 1)

print(qc)


from qiskit import transpile
from qiskit_aer import AerSimulator

simulator = AerSimulator()

compiled = transpile(qc, simulator)

result = simulator.run(
    compiled,
    shots=1000
).result()

counts = result.get_counts()

print("\nMeasurement Results:")
print(counts)