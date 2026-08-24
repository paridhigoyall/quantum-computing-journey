import numpy as np

from qiskit.quantum_info import Pauli


# ============================================================
# Day 39 - Jordan-Wigner Transformation
#
# Goal:
# Understand how fermionic operators are represented
# using qubit Pauli operators.
# ============================================================


print("Jordan-Wigner Transformation")
print("============================")
print()


# ============================================================
# Jordan-Wigner idea
# ============================================================
#
# For fermionic mode j:
#
#     a_j^† = (Z_0 Z_1 ... Z_{j-1}) (X_j - iY_j) / 2
#
#     a_j   = (Z_0 Z_1 ... Z_{j-1}) (X_j + iY_j) / 2
#
# The string of Z operators keeps track of fermionic
# anticommutation.
#
# For mode 0:
#
#     a_0^† = (X_0 - iY_0) / 2
#
# For mode 1:
#
#     a_1^† = Z_0 (X_1 - iY_1) / 2
#
# ============================================================


print("Jordan-Wigner mapping:")
print()

print(
    "a0† = (X0 - iY0) / 2"
)

print(
    "a0  = (X0 + iY0) / 2"
)

print()

print(
    "a1† = Z0 (X1 - iY1) / 2"
)

print(
    "a1  = Z0 (X1 + iY1) / 2"
)

print()


# ============================================================
# One-fermionic-mode number operator
# ============================================================
#
# n = a†a
#
# Under Jordan-Wigner:
#
# n = (I - Z) / 2
#
# This is extremely important.
#
# A fermionic occupation number:
#
#     0 -> qubit |0>
#     1 -> qubit |1>
#
# ============================================================


I = np.eye(2)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)


number_operator = (
    I - Z
) / 2


print("One-mode number operator:")
print(number_operator)

print()


# ============================================================
# Verify occupation states
# ============================================================

zero_state = np.array([
    [1],
    [0]
], dtype=complex)

one_state = np.array([
    [0],
    [1]
], dtype=complex)


zero_occupation = (
    zero_state.conj().T
    @ number_operator
    @ zero_state
).item()


one_occupation = (
    one_state.conj().T
    @ number_operator
    @ one_state
).item()


print(
    f"Occupation of |0> = "
    f"{zero_occupation.real:.1f}"
)

print(
    f"Occupation of |1> = "
    f"{one_occupation.real:.1f}"
)

print()


# ============================================================
# Two-mode number operators
# ============================================================
#
# For two fermionic modes:
#
# n0 = (I - Z0) / 2
#
# n1 = (I - Z1) / 2
#
# ============================================================


I2 = np.eye(2)

Z0 = np.kron(Z, I2)
Z1 = np.kron(I2, Z)


n0 = (
    np.eye(4) - Z0
) / 2


n1 = (
    np.eye(4) - Z1
) / 2


print("Two-mode number operators:")
print()

print("n0 =")
print(n0)

print()

print("n1 =")
print(n1)

print()


# ============================================================
# Occupation basis
# ============================================================
#
# The computational basis now represents fermionic
# occupation states:
#
# |00> -> no particles
# |01> -> particle in mode 0
# |10> -> particle in mode 1
# |11> -> particles in both modes
#
# ============================================================


basis_states = [
    np.array([[1], [0], [0], [0]], dtype=complex),
    np.array([[0], [1], [0], [0]], dtype=complex),
    np.array([[0], [0], [1], [0]], dtype=complex),
    np.array([[0], [0], [0], [1]], dtype=complex),
]


labels = [
    "|00>",
    "|01>",
    "|10>",
    "|11>",
]


print("Fermionic occupation states:")
print()


for label, state in zip(labels, basis_states):

    occupation0 = (
        state.conj().T
        @ n0
        @ state
    ).item().real

    occupation1 = (
        state.conj().T
        @ n1
        @ state
    ).item().real

    print(
        f"{label} -> "
        f"n0={occupation0:.0f}, "
        f"n1={occupation1:.0f}"
    )


print()


# ============================================================
# Total particle number
# ============================================================


total_number = n0 + n1


print("Total particle-number operator:")
print(total_number)

print()


# ============================================================
# Verify total particle number
# ============================================================


print("Total particle number:")

for label, state in zip(labels, basis_states):

    particles = (
        state.conj().T
        @ total_number
        @ state
    ).item().real

    print(
        f"{label} -> "
        f"N={particles:.0f}"
    )


print()


# ============================================================
# Connection to VQE
# ============================================================


print("Connection to VQE")
print("==================")

print()

print(
    "Fermionic Hamiltonian"
)

print(
    "        ↓"
)

print(
    "Jordan-Wigner transformation"
)

print(
    "        ↓"
)

print(
    "Pauli Hamiltonian"
)

print(
    "        ↓"
)

print(
    "Qubit Hamiltonian"
)

print(
    "        ↓"
)

print(
    "VQE"
)

print()

print("Day 39 foundation complete.")

# ============================================================
# Fermionic Creation and Annihilation Operators
# ============================================================

I = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)


# Single-mode operators
a_dagger = (X - 1j * Y) / 2
a = (X + 1j * Y) / 2


print("\nSingle-mode fermionic operators:")
print("================================")

print("\na† =")
print(a_dagger)

print("\na =")
print(a)

# ============================================================
# Verify creation and annihilation
# ============================================================

zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)

print("\nOperator action:")
print("================")

print("a† |0> =", a_dagger @ zero)
print("a  |1> =", a @ one)

print("a† |1> =", a_dagger @ one)
print("a  |0> =", a @ zero)

# ============================================================
# Fermionic Anticommutation Relation
# ============================================================

print("\nFermionic anticommutation:")
print("==========================")

anticommutator = (
    a @ a_dagger
    + a_dagger @ a
)

identity = np.eye(2, dtype=complex)

print("\n{a, a†} =")
print(anticommutator)

print("\nIdentity I =")
print(identity)

print(
    "\nDoes {a, a†} = I ?",
    np.allclose(anticommutator, identity)
)