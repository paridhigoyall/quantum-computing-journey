import numpy as np
from qiskit import QuantumCircuit


# --------------------------------
# Amplitude
# --------------------------------

a = 0.25

theta = np.arcsin(np.sqrt(a))

print("Amplitude a:", a)
print("Theta:", theta)
print("2 * theta:", 2 * theta)


# --------------------------------
# State preparation A
# --------------------------------

A = QuantumCircuit(1)

A.ry(2 * theta, 0)

print("\nState preparation A:")
print(A)


# --------------------------------
# Simplified Grover operator Q
# --------------------------------

Q = QuantumCircuit(1)

Q.ry(4 * theta, 0)

print("\nSimplified Grover operator Q:")
print(Q)


# --------------------------------
# Apply A followed by Q
# --------------------------------

test = QuantumCircuit(1)

test.compose(A, inplace=True)
test.compose(Q, inplace=True)

print("\nA followed by Q:")
print(test)


# --------------------------------
# Show the rotation
# --------------------------------

print("\nExpected phase information:")
print("theta =", theta)
print("2 * theta =", 2 * theta)
print("2 * theta / pi =", (2 * theta) / np.pi)