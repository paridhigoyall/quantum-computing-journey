import numpy as np


# ============================================================
# Jordan-Wigner Transformation
# ============================================================

print("Jordan-Wigner Transformation")
print("============================")
print()

print("Jordan-Wigner mapping:")
print()

print("a0† = (X0 - iY0) / 2")
print("a0  = (X0 + iY0) / 2")
print()

print("a1† = Z0 (X1 - iY1) / 2")
print("a1  = Z0 (X1 + iY1) / 2")


# ============================================================
# Pauli matrices
# ============================================================

I2 = np.eye(2, dtype=complex)

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


# ============================================================
# Single-mode fermionic operators
# ============================================================

a_dagger = np.array([
    [0, 0],
    [1, 0]
], dtype=complex)

a = np.array([
    [0, 1],
    [0, 0]
], dtype=complex)

# Number operator
number_operator = a_dagger @ a

zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)


print()
print("One-mode number operator:")
print(number_operator)

print()
print(
    "Occupation of |0> =",
    np.vdot(zero, number_operator @ zero).real
)

print(
    "Occupation of |1> =",
    np.vdot(one, number_operator @ one).real
)


# ============================================================
# Two-mode Jordan-Wigner operators
#
# Basis ordering:
#
# |00>
# |01>
# |10>
# |11>
# ============================================================

a0_dagger = np.kron(a_dagger, I2)
a0 = np.kron(a, I2)

a1_dagger = np.kron(Z, a_dagger)
a1 = np.kron(Z, a)


# ============================================================
# Two-mode number operators
# ============================================================

n0 = a0_dagger @ a0
n1 = a1_dagger @ a1

print()
print("Two-mode number operators:")

print()
print("n0 =")
print(n0)

print()
print("n1 =")
print(n1)


# ============================================================
# Fermionic occupation states
# ============================================================

basis_states = {

    "|00>": np.array(
        [1, 0, 0, 0],
        dtype=complex
    ),

    "|01>": np.array(
        [0, 1, 0, 0],
        dtype=complex
    ),

    "|10>": np.array(
        [0, 0, 1, 0],
        dtype=complex
    ),

    "|11>": np.array(
        [0, 0, 0, 1],
        dtype=complex
    )
}


print()
print("Fermionic occupation states:")
print()

for label, state in basis_states.items():

    occupation0 = np.vdot(
        state,
        n0 @ state
    ).real

    occupation1 = np.vdot(
        state,
        n1 @ state
    ).real

    print(
        f"{label} -> "
        f"n0={int(round(occupation0))}, "
        f"n1={int(round(occupation1))}"
    )


# ============================================================
# Total particle-number operator
# ============================================================

N = n0 + n1

print()
print("Total particle-number operator:")
print(N)

print()
print("Total particle number:")

for label, state in basis_states.items():

    total_number = np.vdot(
        state,
        N @ state
    ).real

    print(
        f"{label} -> "
        f"N={int(round(total_number))}"
    )


# ============================================================
# Connection to VQE
# ============================================================

print()
print("Connection to VQE")
print("==================")

print()
print("Fermionic Hamiltonian")
print("        ↓")
print("Jordan-Wigner transformation")
print("        ↓")
print("Pauli Hamiltonian")
print("        ↓")
print("Qubit Hamiltonian")
print("        ↓")
print("VQE")

print()
print("Day 39 foundation complete.")


# ============================================================
# Single-mode fermionic operators
# ============================================================

print()
print("Single-mode fermionic operators:")
print("================================")

print()
print("a† =")
print(a_dagger)

print()
print("a =")
print(a)


# ============================================================
# Operator action
# ============================================================

print()
print("Operator action:")
print("================")

print(
    "a† |0> =",
    a_dagger @ zero
)

print(
    "a  |1> =",
    a @ one
)

print(
    "a† |1> =",
    a_dagger @ one
)

print(
    "a  |0> =",
    a @ zero
)


# ============================================================
# Fermionic anticommutation
#
# {a, a†} = a a† + a† a
#
# Expected:
#
# {a, a†} = I
# ============================================================

anticommutator = (
    a @ a_dagger
    +
    a_dagger @ a
)

print()
print("Fermionic anticommutation:")
print("==========================")

print()
print("{a, a†} =")
print(anticommutator)

print()
print("Identity I =")
print(I2)

print()
print(
    "Does {a, a†} = I ?",
    np.allclose(
        anticommutator,
        I2
    )
)


# ============================================================
# Two-mode fermionic operators
# ============================================================

print()
print("Two-mode fermionic operators:")
print("=============================")

print()
print("a0† =")
print(a0_dagger)

print()
print("a0 =")
print(a0)

print()
print("a1† =")
print(a1_dagger)

print()
print("a1 =")
print(a1)


# ============================================================
# Cross-mode anticommutation
#
# {a0, a1†} = 0
#
# {a0, a1} = 0
# ============================================================

cross_dagger = (
    a0 @ a1_dagger
    +
    a1_dagger @ a0
)

cross_annihilation = (
    a0 @ a1
    +
    a1 @ a0
)

zero4 = np.zeros(
    (4, 4),
    dtype=complex
)


print()
print("Cross-mode anticommutation:")
print("============================")

print()
print("{a0, a1†} =")
print(cross_dagger)

print()
print(
    "Does {a0, a1†} = 0 ?",
    np.allclose(
        cross_dagger,
        zero4
    )
)

print()
print("{a0, a1} =")
print(cross_annihilation)

print()
print(
    "Does {a0, a1} = 0 ?",
    np.allclose(
        cross_annihilation,
        zero4
    )
)


# ============================================================
# Why the Jordan-Wigner Z-string is necessary
# ============================================================

wrong_a1_dagger = np.kron(
    I2,
    a_dagger
)

wrong_cross = (
    a0 @ wrong_a1_dagger
    +
    wrong_a1_dagger @ a0
)

print()
print("Without the Jordan-Wigner Z-string:")
print("===================================")

print()
print("{a0, wrong_a1†} =")
print(wrong_cross)

print()
print(
    "Does {a0, wrong_a1†} = 0 ?",
    np.allclose(
        wrong_cross,
        zero4
    )
)


# ============================================================
# PART 1
#
# Number operator -> Pauli Z
#
# n = a†a
#
# Jordan-Wigner result:
#
# n = (I - Z) / 2
# ============================================================

print()
print("Number Operator -> Pauli Z")
print("===========================")

number_from_z = (
    I2 - Z
) / 2


print()
print("Identity I:")
print(I2)

print()
print("Pauli Z:")
print(Z)

print()
print("(I - Z) / 2:")
print(number_from_z)

print()
print("Original number operator:")
print(number_operator)

print()
print(
    "Does (I - Z) / 2 equal n ?",
    np.allclose(
        number_from_z,
        number_operator
    )
)

print()
print("Mathematical result:")
print("n = (I - Z) / 2")
# ============================================================
# Fermionic hopping term -> Pauli operators
#
# Hopping term:
#
#     a0† a1 + a1† a0
#
# Under Jordan-Wigner:
#
#     a0† a1 + a1† a0
#
# becomes:
#
#     (X0 X1 + Y0 Y1) / 2
# ============================================================

print()
print("Fermionic Hopping -> Pauli Operators")
print("=====================================")

# Fermionic hopping operator
hopping_fermionic = (
    a0_dagger @ a1
    +
    a1_dagger @ a0
)

# Two-qubit Pauli operators
X0X1 = np.kron(X, X)
Y0Y1 = np.kron(Y, Y)

# Jordan-Wigner Pauli representation
hopping_pauli = (
    X0X1 + Y0Y1
) / 2


print()
print("Fermionic hopping operator:")
print("a0† a1 + a1† a0 =")
print(hopping_fermionic)

print()
print("Pauli representation:")
print("(X0 X1 + Y0 Y1) / 2 =")
print(hopping_pauli)


# ============================================================
# Verify the mapping
# ============================================================

print()
print("Verifying Jordan-Wigner hopping mapping:")
print("=========================================")

print()
print(
    "Does fermionic hopping equal Pauli hopping?",
    np.allclose(
        hopping_fermionic,
        hopping_pauli
    )
)


# ============================================================
# Individual Pauli products
# ============================================================

print()
print("Individual Pauli products:")
print("==========================")

print()
print("X0 X1 =")
print(X0X1)

print()
print("Y0 Y1 =")
print(Y0Y1)


# ============================================================
# Hopping action on occupation states
# ============================================================

print()
print("Hopping action on occupation states:")
print("=====================================")

for label, state in basis_states.items():

    result = hopping_fermionic @ state

    print()
    print(f"{label} ->")
    print(result)


# ============================================================
# Physical interpretation
# ============================================================

print()
print("Physical interpretation:")
print("========================")

print()
print("|00> -> no particle available to hop")

print("|11> -> both modes occupied; no allowed single-particle hop")

print("|01> <-> |10>")

print("A particle moves between mode 0 and mode 1.")

print()
print("Jordan-Wigner hopping rule:")
print()
print("a0† a1 + a1† a0")
print("        ↓")
print("(X0 X1 + Y0 Y1) / 2")
# ============================================================
# Two-Mode Fermionic Hamiltonian
#
# H_f =
#     eps0 n0
#   + eps1 n1
#   + t (a0† a1 + a1† a0)
#   + U n0 n1
#
# We will construct it in two ways:
#
# 1. Directly with fermionic operators
# 2. After Jordan-Wigner transformation
#
# Then compare their matrices and eigenvalues.
# ============================================================

print()
print("Two-Mode Fermionic Hamiltonian")
print("==============================")

# Model parameters
eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8

print()
print("Hamiltonian parameters:")
print(f"eps0 = {eps0}")
print(f"eps1 = {eps1}")
print(f"t    = {t}")
print(f"U    = {U}")


# ============================================================
# Fermionic Hamiltonian
# ============================================================

H_fermionic = (
    eps0 * n0
    + eps1 * n1
    + t * (
        a0_dagger @ a1
        + a1_dagger @ a0
    )
    + U * (
        n0 @ n1
    )
)


print()
print("Fermionic Hamiltonian:")
print("H_f =")
print(H_fermionic)


# ============================================================
# Jordan-Wigner Pauli representation
#
# n0 = (I - Z0) / 2
# n1 = (I - Z1) / 2
#
# hopping =
# (X0X1 + Y0Y1) / 2
# ============================================================

I4 = np.kron(I2, I2)

Z0 = np.kron(Z, I2)
Z1 = np.kron(I2, Z)

X0X1 = np.kron(X, X)
Y0Y1 = np.kron(Y, Y)

n0_pauli = (
    I4 - Z0
) / 2

n1_pauli = (
    I4 - Z1
) / 2

hopping_pauli = (
    X0X1 + Y0Y1
) / 2

interaction_pauli = (
    n0_pauli @ n1_pauli
)


# ============================================================
# Qubit Hamiltonian
# ============================================================

H_qubit = (
    eps0 * n0_pauli
    + eps1 * n1_pauli
    + t * hopping_pauli
    + U * interaction_pauli
)


print()
print("Jordan-Wigner Pauli Hamiltonian:")
print("H_qubit =")
print(H_qubit)


# ============================================================
# Verify matrix equality
# ============================================================

print()
print("Verifying fermionic -> qubit mapping:")
print("======================================")

print()
print(
    "Does H_fermionic = H_qubit ?",
    np.allclose(
        H_fermionic,
        H_qubit
    )
)


# ============================================================
# Eigenvalues
# ============================================================

fermionic_eigenvalues = np.linalg.eigvalsh(
    H_fermionic
)

qubit_eigenvalues = np.linalg.eigvalsh(
    H_qubit
)

print()
print("Fermionic Hamiltonian eigenvalues:")
print(fermionic_eigenvalues)

print()
print("Qubit Hamiltonian eigenvalues:")
print(qubit_eigenvalues)


print()
print(
    "Do the eigenvalues match?",
    np.allclose(
        fermionic_eigenvalues,
        qubit_eigenvalues
    )
)


# ============================================================
# Ground-state energy
# ============================================================

fermionic_ground_energy = np.min(
    fermionic_eigenvalues
)

qubit_ground_energy = np.min(
    qubit_eigenvalues
)

print()
print("Ground-state energy:")
print(
    f"Fermionic = {fermionic_ground_energy:.10f}"
)

print(
    f"Qubit     = {qubit_ground_energy:.10f}"
)


# ============================================================
# Final summary
# ============================================================

print()
print("==========================================")
print("FERMIONIC -> JORDAN-WIGNER -> QUBIT")
print("==========================================")

print()
print("Occupation:")
print("n_i = (I - Z_i) / 2")

print()
print("Hopping:")
print(
    "a0†a1 + a1†a0 = "
    "(X0X1 + Y0Y1) / 2"
)

print()
print("Interaction:")
print(
    "n0 n1 = "
    "[(I-Z0)/2][(I-Z1)/2]"
)

print()
print(
    "Hamiltonian matrices identical?",
    np.allclose(H_fermionic, H_qubit)
)

print(
    "Spectra identical?",
    np.allclose(
        fermionic_eigenvalues,
        qubit_eigenvalues
    )
)

print()
print("Jordan-Wigner Hamiltonian construction complete.")