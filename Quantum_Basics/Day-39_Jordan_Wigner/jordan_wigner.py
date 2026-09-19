import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ============================================================
# BASIC MATRICES
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


def kron(a, b):
    return np.kron(a, b)


def close(a, b, tol=1e-10):
    return np.allclose(a, b, atol=tol)


def anticommutator(a, b):
    return a @ b + b @ a


# ============================================================
# JORDAN-WIGNER TRANSFORMATION
# ============================================================

print("Jordan-Wigner Transformation")
print("============================\n")

print("Jordan-Wigner mapping:\n")

print("a0† = (X0 - iY0) / 2")
print("a0  = (X0 + iY0) / 2")
print()

print("a1† = Z0 (X1 - iY1) / 2")
print("a1  = Z0 (X1 + iY1) / 2")
print()


# ============================================================
# SINGLE-MODE FERMIONIC OPERATORS
# ============================================================

a_dag = np.array([
    [0, 0],
    [1, 0]
], dtype=complex)

a = np.array([
    [0, 1],
    [0, 0]
], dtype=complex)


# ============================================================
# ONE-MODE NUMBER OPERATOR
# ============================================================

n = a_dag @ a

print("One-mode number operator:")
print(n)

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

print(
    "\nOccupation of |0> =",
    np.real(ket0.conj() @ n @ ket0)
)

print(
    "Occupation of |1> =",
    np.real(ket1.conj() @ n @ ket1)
)


# ============================================================
# TWO-MODE FERMIONIC OPERATORS
# ============================================================

a0_dag = kron(a_dag, I)
a0 = kron(a, I)

a1_dag = kron(Z, a_dag)
a1 = kron(Z, a)


# ============================================================
# TWO-MODE NUMBER OPERATORS
# ============================================================

n0 = a0_dag @ a0
n1 = a1_dag @ a1

print("\nTwo-mode number operators:")

print("\nn0 =")
print(n0)

print("\nn1 =")
print(n1)


# ============================================================
# FERMIONIC OCCUPATION STATES
# ============================================================

basis_states = {
    "|00>": np.array([1, 0, 0, 0], dtype=complex),
    "|01>": np.array([0, 1, 0, 0], dtype=complex),
    "|10>": np.array([0, 0, 1, 0], dtype=complex),
    "|11>": np.array([0, 0, 0, 1], dtype=complex),
}

print("\nFermionic occupation states:\n")

for name, state in basis_states.items():

    occupation_0 = np.real(
        state.conj() @ n0 @ state
    )

    occupation_1 = np.real(
        state.conj() @ n1 @ state
    )

    print(
        f"{name} -> "
        f"n0={int(round(occupation_0))}, "
        f"n1={int(round(occupation_1))}"
    )


# ============================================================
# TOTAL PARTICLE NUMBER
# ============================================================

N_total = n0 + n1

print("\nTotal particle-number operator:")
print(N_total)

print("\nTotal particle number:")

for name, state in basis_states.items():

    particle_number = np.real(
        state.conj() @ N_total @ state
    )

    print(
        f"{name} -> N={int(round(particle_number))}"
    )


# ============================================================
# CONNECTION TO VQE
# ============================================================

print("""
Connection to VQE
==================

Fermionic Hamiltonian
        ↓
Jordan-Wigner transformation
        ↓
Pauli Hamiltonian
        ↓
Qubit Hamiltonian
        ↓
VQE
""")


# ============================================================
# SINGLE-MODE OPERATOR ACTION
# ============================================================

print("Single-mode fermionic operators:")
print("================================")

print("\na† =")
print(a_dag)

print("\na =")
print(a)

print("\nOperator action:")
print("================")

print("a† |0> =", a_dag @ ket0)
print("a  |1> =", a @ ket1)
print("a† |1> =", a_dag @ ket1)
print("a  |0> =", a @ ket0)


# ============================================================
# FERMIONIC ANTICOMMUTATION
# ============================================================

single_anticommutator = anticommutator(
    a,
    a_dag
)

print("\nFermionic anticommutation:")
print("==========================")

print("\n{a, a†} =")
print(single_anticommutator)

print("\nIdentity I =")
print(I)

print(
    "\nDoes {a, a†} = I ?",
    close(single_anticommutator, I)
)


# ============================================================
# TWO-MODE FERMIONIC OPERATORS
# ============================================================

print("\nTwo-mode fermionic operators:")
print("=============================")

print("\na0† =")
print(a0_dag)

print("\na0 =")
print(a0)

print("\na1† =")
print(a1_dag)

print("\na1 =")
print(a1)


# ============================================================
# CROSS-MODE ANTICOMMUTATION
# ============================================================

anti_a0_a1dag = anticommutator(
    a0,
    a1_dag
)

anti_a0_a1 = anticommutator(
    a0,
    a1
)

print("\nCross-mode anticommutation:")
print("============================")

print("\n{a0, a1†} =")
print(anti_a0_a1dag)

print(
    "\nDoes {a0, a1†} = 0 ?",
    close(
        anti_a0_a1dag,
        np.zeros((4, 4), dtype=complex)
    )
)

print("\n{a0, a1} =")
print(anti_a0_a1)

print(
    "\nDoes {a0, a1} = 0 ?",
    close(
        anti_a0_a1,
        np.zeros((4, 4), dtype=complex)
    )
)


# ============================================================
# WHY THE Z-STRING IS NECESSARY
# ============================================================

wrong_a1_dag = kron(I, a_dag)

wrong_anticommutator = anticommutator(
    a0,
    wrong_a1_dag
)

print("\nWithout the Jordan-Wigner Z-string:")
print("===================================")

print("\n{a0, wrong_a1†} =")
print(wrong_anticommutator)

print(
    "\nDoes {a0, wrong_a1†} = 0 ?",
    close(
        wrong_anticommutator,
        np.zeros((4, 4), dtype=complex)
    )
)


# ============================================================
# NUMBER OPERATOR -> PAULI Z
# ============================================================

n_from_z = (I - Z) / 2

print("\nNumber Operator -> Pauli Z")
print("===========================")

print("\nIdentity I:")
print(I)

print("\nPauli Z:")
print(Z)

print("\n(I - Z) / 2:")
print(n_from_z)

print("\nOriginal number operator:")
print(n)

print(
    "\nDoes (I - Z) / 2 equal n ?",
    close(n_from_z, n)
)

print("\nMathematical result:")
print("n = (I - Z) / 2")


# ============================================================
# TWO-MODE NUMBER OPERATORS IN PAULI FORM
# ============================================================

identity_4 = kron(I, I)

Z0 = kron(Z, I)
Z1 = kron(I, Z)

n0_pauli = (
    identity_4 - Z0
) / 2

n1_pauli = (
    identity_4 - Z1
) / 2

print("\nTwo-mode number operators in Pauli form:")
print("=========================================")

print("\nn0 = (I - Z0) / 2")
print(n0_pauli)

print("\nn1 = (I - Z1) / 2")
print(n1_pauli)

print(
    "\nDoes Pauli n0 equal fermionic n0 ?",
    close(n0_pauli, n0)
)

print(
    "Does Pauli n1 equal fermionic n1 ?",
    close(n1_pauli, n1)
)


# ============================================================
# FERMIONIC HOPPING -> PAULI OPERATORS
# ============================================================

hopping_fermionic = (
    a0_dag @ a1
    +
    a1_dag @ a0
)

X0X1 = kron(X, X)
Y0Y1 = kron(Y, Y)

hopping_pauli = (
    X0X1 + Y0Y1
) / 2

print("\nFermionic Hopping -> Pauli Operators")
print("=====================================")

print("\nFermionic hopping operator:")
print("a0† a1 + a1† a0 =")
print(hopping_fermionic)

print("\nPauli representation:")
print("(X0 X1 + Y0 Y1) / 2 =")
print(hopping_pauli)

print("\nVerifying Jordan-Wigner hopping mapping:")
print("=========================================")

print(
    "Does fermionic hopping equal Pauli hopping?",
    close(
        hopping_fermionic,
        hopping_pauli
    )
)


# ============================================================
# INDIVIDUAL PAULI PRODUCTS
# ============================================================

print("\nIndividual Pauli products:")
print("==========================")

print("\nX0 X1 =")
print(X0X1)

print("\nY0 Y1 =")
print(Y0Y1)


# ============================================================
# HOPPING ACTION ON OCCUPATION STATES
# ============================================================

print("\nHopping action on occupation states:")
print("=====================================")

for name, state in basis_states.items():

    result = hopping_fermionic @ state

    print(f"\n{name} ->")
    print(result)


print("""
Physical interpretation:
========================

|00> -> no particle available to hop
|11> -> both modes occupied; no allowed single-particle hop
|01> <-> |10>
A particle moves between mode 0 and mode 1.

Jordan-Wigner hopping rule:

a0† a1 + a1† a0
        ↓
(X0 X1 + Y0 Y1) / 2
""")


# ============================================================
# TWO-MODE FERMIONIC HAMILTONIAN
# ============================================================

print("\nTwo-Mode Fermionic Hamiltonian")
print("==============================")

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8

print("\nHamiltonian parameters:")
print(f"eps0 = {eps0}")
print(f"eps1 = {eps1}")
print(f"t    = {t}")
print(f"U    = {U}")


# H = eps0*n0 + eps1*n1
#     + t*(a0†a1 + a1†a0)
#     + U*n0*n1

H_fermionic = (
    eps0 * n0
    +
    eps1 * n1
    +
    t * hopping_fermionic
    +
    U * (n0 @ n1)
)


# ============================================================
# JORDAN-WIGNER QUBIT HAMILTONIAN
# ============================================================

ZZ = kron(Z, Z)

interaction_pauli = (
    identity_4
    - Z0
    - Z1
    + ZZ
) / 4

H_qubit = (
    eps0 * n0_pauli
    +
    eps1 * n1_pauli
    +
    t * hopping_pauli
    +
    U * interaction_pauli
)


print("\nFermionic Hamiltonian:")
print("H_f =")
print(H_fermionic)

print("\nJordan-Wigner Pauli Hamiltonian:")
print("H_qubit =")
print(H_qubit)


# ============================================================
# VERIFY FERMIONIC -> QUBIT MAPPING
# ============================================================

print("\nVerifying fermionic -> qubit mapping:")
print("======================================")

print(
    "Does H_fermionic = H_qubit ?",
    close(
        H_fermionic,
        H_qubit
    )
)


# ============================================================
# HAMILTONIAN EIGENVALUES
# ============================================================

fermionic_eigenvalues = np.linalg.eigvalsh(
    H_fermionic
)

qubit_eigenvalues = np.linalg.eigvalsh(
    H_qubit
)

print("\nFermionic Hamiltonian eigenvalues:")
print(fermionic_eigenvalues)

print("\nQubit Hamiltonian eigenvalues:")
print(qubit_eigenvalues)

print(
    "\nDo the eigenvalues match?",
    np.allclose(
        fermionic_eigenvalues,
        qubit_eigenvalues,
        atol=1e-10
    )
)


# ============================================================
# PARTICLE-NUMBER SECTORS
# ============================================================

print("\nParticle-number sectors:")
print("========================")

print("\nN = 0:")
print("  |00>")

print("\nN = 1:")
print("  |01>")
print("  |10>")

print("\nN = 2:")
print("  |11>")


# ============================================================
# ONE-PARTICLE SECTOR
# ============================================================

# Basis ordering:
#
# |00>
# |01>
# |10>
# |11>
#
# N=1 sector:
#
# |01>
# |10>

one_particle_indices = [1, 2]

H_one_particle = H_qubit[
    np.ix_(
        one_particle_indices,
        one_particle_indices
    )
]

print("\nOne-particle sector:")
print("====================")

print("Basis = {|01>, |10>}")

print("\nRestricted Hamiltonian:")
print(H_one_particle)


# ============================================================
# ONE-PARTICLE EIGENVALUES
# ============================================================

one_particle_eigenvalues, one_particle_vectors = (
    np.linalg.eigh(H_one_particle)
)

print("\nOne-particle-sector eigenvalues:")
print(one_particle_eigenvalues)

print(
    "\nLowest energy in N=1 sector = "
    f"{one_particle_eigenvalues[0]:.10f}"
)


# ============================================================
# ONE-PARTICLE GROUND STATE
# ============================================================

ground_vector = one_particle_vectors[:, 0]

print("\nOne-particle ground-state vector:")
print(ground_vector)

print("\nInterpretation:")
print("|psi> = c0 |01> + c1 |10>")

print(
    f"c0 = {ground_vector[0]:.10f}"
)

print(
    f"c1 = {ground_vector[1]:.10f}"
)


# ============================================================
# FULL HAMILTONIAN GROUND STATE
# ============================================================

full_eigenvalues, full_eigenvectors = np.linalg.eigh(
    H_qubit
)

full_ground_energy = full_eigenvalues[0]
full_ground_state = full_eigenvectors[:, 0]

print("\nFull Hamiltonian ground state:")
print("==============================")

print(
    f"Ground-state energy = "
    f"{full_ground_energy:.10f}"
)

print(
    "\nGround-state vector in "
    "{|00>, |01>, |10>, |11>}:"
)

print(full_ground_state)


# ============================================================
# GROUND-STATE PARTICLE NUMBER
# ============================================================

ground_particle_number = np.real(
    full_ground_state.conj()
    @ N_total
    @ full_ground_state
)

print(
    "\nParticle number expectation of "
    f"full ground state = "
    f"{ground_particle_number:.10f}"
)


# ============================================================
# EXPANDED PAULI HAMILTONIAN
# ============================================================

H_pauli_expanded = (
    eps0 * (identity_4 - Z0) / 2
    +
    eps1 * (identity_4 - Z1) / 2
    +
    t * (X0X1 + Y0Y1) / 2
    +
    U * (
        identity_4
        - Z0
        - Z1
        + ZZ
    ) / 4
)

print("\nExpanded Pauli Hamiltonian:")
print("===========================")

print(
    "H = eps0*(I-Z0)/2"
    " + eps1*(I-Z1)/2"
    " + t*(X0X1+Y0Y1)/2"
    " + U*(I-Z0-Z1+Z0Z1)/4"
)

print("\nExpanded matrix:")
print(H_pauli_expanded)

print(
    "\nDoes expanded Pauli Hamiltonian "
    "equal H_qubit ?",
    close(
        H_pauli_expanded,
        H_qubit
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("""
==========================================
FERMIONIC -> JORDAN-WIGNER -> QUBIT -> VQE
==========================================

What we established:

1. Fermions use creation/annihilation operators.

2. Fermionic operators obey anticommutation rules.

3. Jordan-Wigner adds the required Z-string.

4. Number operator:
       n_i = (I - Z_i) / 2

5. Hopping:
       a0†a1 + a1†a0
       = (X0X1 + Y0Y1) / 2

6. Interaction:
       n0n1
       = (I-Z0-Z1+Z0Z1) / 4

7. Fermionic and qubit Hamiltonians are identical.

8. Therefore their spectra are identical.

9. Particle-number sectors can be identified.

10. The physical N=1 sector can be isolated.

Practical pipeline:

Physical fermion problem
        ↓
Fermionic Hamiltonian
        ↓
Jordan-Wigner transformation
        ↓
Pauli / qubit Hamiltonian
        ↓
Choose physical particle-number sector
        ↓
VQE
        ↓
Ground-state energy
""")