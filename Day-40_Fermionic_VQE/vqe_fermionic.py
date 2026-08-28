import numpy as np
from scipy.optimize import minimize


# ============================================================
# DAY 40 - FERMIONIC HAMILTONIAN -> JORDAN-WIGNER -> VQE
# ============================================================

np.set_printoptions(precision=6, suppress=True)


# ============================================================
# 1. BASIC PAULI MATRICES
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


# Two-qubit operators
I4 = kron(I, I)

X0 = kron(X, I)
X1 = kron(I, X)

Y0 = kron(Y, I)
Y1 = kron(I, Y)

Z0 = kron(Z, I)
Z1 = kron(I, Z)

X0X1 = kron(X, X)
Y0Y1 = kron(Y, Y)
Z0Z1 = kron(Z, Z)


# ============================================================
# 2. PHYSICAL FERMIONIC MODEL
# ============================================================

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8

print("=" * 60)
print("DAY 40 - FERMIONIC VQE")
print("=" * 60)

print("\nPhysical parameters:")
print(f"eps0 = {eps0}")
print(f"eps1 = {eps1}")
print(f"t    = {t}")
print(f"U    = {U}")


# ============================================================
# 3. JORDAN-WIGNER MAPPED HAMILTONIAN
#
# n0 = (I - Z0) / 2
# n1 = (I - Z1) / 2
#
# hopping =
# (X0X1 + Y0Y1) / 2
#
# n0*n1 =
# (I - Z0 - Z1 + Z0Z1) / 4
# ============================================================

n0 = (I4 - Z0) / 2
n1 = (I4 - Z1) / 2

hopping = (X0X1 + Y0Y1) / 2

interaction = (
    I4 - Z0 - Z1 + Z0Z1
) / 4


H = (
    eps0 * n0
    + eps1 * n1
    + t * hopping
    + U * interaction
)


# ============================================================
# 4. DISPLAY PAULI HAMILTONIAN
# ============================================================

print("\n" + "=" * 60)
print("JORDAN-WIGNER QUBIT HAMILTONIAN")
print("=" * 60)

print("""
H =
eps0 * (I - Z0)/2
+ eps1 * (I - Z1)/2
+ t * (X0X1 + Y0Y1)/2
+ U * (I - Z0 - Z1 + Z0Z1)/4
""")

print("Numerical Pauli coefficients:")

identity_coeff = (
    eps0 / 2
    + eps1 / 2
    + U / 4
)

z0_coeff = (
    -eps0 / 2
    - U / 4
)

z1_coeff = (
    -eps1 / 2
    - U / 4
)

zz_coeff = U / 4

xx_coeff = t / 2
yy_coeff = t / 2

print(f"I      : {identity_coeff:+.6f}")
print(f"Z0     : {z0_coeff:+.6f}")
print(f"Z1     : {z1_coeff:+.6f}")
print(f"Z0Z1   : {zz_coeff:+.6f}")
print(f"X0X1   : {xx_coeff:+.6f}")
print(f"Y0Y1   : {yy_coeff:+.6f}")

print("\nHamiltonian matrix:")
print(H)


# ============================================================
# 5. EXACT DIAGONALIZATION
#
# We calculate the exact answer only for verification.
# VQE itself will not use this value during optimization.
# ============================================================

exact_eigenvalues, exact_eigenvectors = np.linalg.eigh(H)

exact_ground_energy = exact_eigenvalues[0]
exact_ground_state = exact_eigenvectors[:, 0]

print("\n" + "=" * 60)
print("EXACT SOLUTION")
print("=" * 60)

print("\nEigenvalues:")
print(exact_eigenvalues)

print(
    f"\nExact ground-state energy = "
    f"{exact_ground_energy:.10f}"
)

print("\nExact ground state:")
print(exact_ground_state)


# ============================================================
# 6. PARTICLE NUMBER
#
# N = n0 + n1
#
# For the one-particle problem:
#
# |01>
# |10>
#
# are the physically relevant states.
# ============================================================

N = n0 + n1

basis_states = {
    "|00>": np.array([1, 0, 0, 0], dtype=complex),
    "|01>": np.array([0, 1, 0, 0], dtype=complex),
    "|10>": np.array([0, 0, 1, 0], dtype=complex),
    "|11>": np.array([0, 0, 0, 1], dtype=complex),
}

print("\n" + "=" * 60)
print("PARTICLE NUMBER")
print("=" * 60)

for name, state in basis_states.items():

    particle_number = np.real(
        state.conj() @ N @ state
    )

    print(
        f"{name} -> N = "
        f"{int(round(particle_number))}"
    )


# ============================================================
# 7. ONE-PARTICLE SUBSPACE
#
# We keep:
#
# |01>
# |10>
#
# because both contain exactly one particle.
# ============================================================

one_particle_indices = [1, 2]

H_one_particle = H[
    np.ix_(
        one_particle_indices,
        one_particle_indices
    )
]

print("\n" + "=" * 60)
print("ONE-PARTICLE SECTOR")
print("=" * 60)

print("\nBasis:")
print("|01>, |10>")

print("\nRestricted Hamiltonian:")
print(H_one_particle)

sector_eigenvalues, sector_eigenvectors = np.linalg.eigh(
    H_one_particle
)

sector_ground_energy = sector_eigenvalues[0]

print("\nOne-particle eigenvalues:")
print(sector_eigenvalues)

print(
    f"\nOne-particle ground energy = "
    f"{sector_ground_energy:.10f}"
)


# ============================================================
# 8. VQE ANSATZ
#
# We need a state that stays inside the one-particle sector.
#
# Starting state:
#
# |01>
#
# Then use an XX+YY rotation to mix:
#
# |01> <-> |10>
#
# The ansatz therefore has the form:
#
# |psi(theta)> =
# cos(theta/2)|01>
# - i sin(theta/2)|10>
#
# The exact phase convention is not important for the energy.
# ============================================================

def ansatz_state(theta):

    state = np.zeros(4, dtype=complex)

    state[1] = np.cos(theta / 2)
    state[2] = -1j * np.sin(theta / 2)

    return state


# ============================================================
# 9. EXPECTATION VALUE
# ============================================================

def expectation(state, operator):

    return np.real(
        state.conj() @ operator @ state
    )


# ============================================================
# 10. VQE ENERGY FUNCTION
# ============================================================

def vqe_energy(parameters):

    theta = parameters[0]

    state = ansatz_state(theta)

    energy = expectation(
        state,
        H
    )

    return energy


# ============================================================
# 11. INITIAL VQE GUESS
# ============================================================

initial_theta = 0.5

initial_state = ansatz_state(initial_theta)

initial_energy = vqe_energy(
    [initial_theta]
)

print("\n" + "=" * 60)
print("INITIAL VQE STATE")
print("=" * 60)

print(
    f"\nInitial theta = "
    f"{initial_theta:.6f}"
)

print("\nInitial state:")
print(initial_state)

print(
    f"\nInitial VQE energy = "
    f"{initial_energy:.10f}"
)


# ============================================================
# 12. VQE OPTIMIZATION
# ============================================================

print("\n" + "=" * 60)
print("VQE OPTIMIZATION")
print("=" * 60)

evaluation_count = 0


def objective(parameters):

    global evaluation_count

    evaluation_count += 1

    energy = vqe_energy(parameters)

    print(
        f"Step {evaluation_count:03d}: "
        f"theta={parameters[0]: .8f} "
        f"| E={energy: .10f}"
    )

    return energy


result = minimize(
    objective,
    x0=[initial_theta],
    method="COBYLA",
    options={
        "maxiter": 100,
        "rhobeg": 0.5,
        "tol": 1e-10
    }
)


# ============================================================
# 13. OPTIMIZATION RESULT
# ============================================================

best_theta = result.x[0]
optimized_energy = result.fun

optimized_state = ansatz_state(
    best_theta
)

print("\n" + "=" * 60)
print("VQE RESULT")
print("=" * 60)

print(
    f"\nBest theta = "
    f"{best_theta:.10f}"
)

print(
    f"Optimized VQE energy = "
    f"{optimized_energy:.10f}"
)

print(
    f"Exact ground energy  = "
    f"{exact_ground_energy:.10f}"
)


# ============================================================
# 14. OPTIMIZED STATE
# ============================================================

print("\nOptimized state:")
print(optimized_state)


# ============================================================
# 15. EXPECTATION VALUES OF IMPORTANT OPERATORS
# ============================================================

z0_expectation = expectation(
    optimized_state,
    Z0
)

z1_expectation = expectation(
    optimized_state,
    Z1
)

zz_expectation = expectation(
    optimized_state,
    Z0Z1
)

xx_expectation = expectation(
    optimized_state,
    X0X1
)

yy_expectation = expectation(
    optimized_state,
    Y0Y1
)

particle_number_expectation = expectation(
    optimized_state,
    N
)

print("\n" + "=" * 60)
print("OPTIMIZED EXPECTATION VALUES")
print("=" * 60)

print(
    f"\n<Z0>   = {z0_expectation:.10f}"
)

print(
    f"<Z1>   = {z1_expectation:.10f}"
)

print(
    f"<Z0Z1> = {zz_expectation:.10f}"
)

print(
    f"<X0X1> = {xx_expectation:.10f}"
)

print(
    f"<Y0Y1> = {yy_expectation:.10f}"
)

print(
    f"<N>    = "
    f"{particle_number_expectation:.10f}"
)


# ============================================================
# 16. ENERGY RECONSTRUCTION FROM PAULI EXPECTATIONS
#
# This is the form VQE actually uses:
#
# E =
# cI <I>
# + cZ0 <Z0>
# + cZ1 <Z1>
# + cZZ <Z0Z1>
# + cXX <X0X1>
# + cYY <Y0Y1>
# ============================================================

reconstructed_energy = (
    identity_coeff
    + z0_coeff * z0_expectation
    + z1_coeff * z1_expectation
    + zz_coeff * zz_expectation
    + xx_coeff * xx_expectation
    + yy_coeff * yy_expectation
)

print("\n" + "=" * 60)
print("PAULI ENERGY RECONSTRUCTION")
print("=" * 60)

print("""
E =
cI <I>
+ cZ0 <Z0>
+ cZ1 <Z1>
+ cZZ <Z0Z1>
+ cXX <X0X1>
+ cYY <Y0Y1>
""")

print(
    f"Reconstructed energy = "
    f"{reconstructed_energy:.10f}"
)


# ============================================================
# 17. VQE ERROR
# ============================================================

vqe_error = abs(
    optimized_energy
    - exact_ground_energy
)

print("\n" + "=" * 60)
print("VQE ACCURACY")
print("=" * 60)

print(
    f"\nVQE error = "
    f"{vqe_error:.10e}"
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 40 SUMMARY")
print("=" * 60)

print("""
Physical fermionic Hamiltonian
        ↓
Jordan-Wigner transformation
        ↓
Pauli Hamiltonian
        ↓
Choose one-particle sector
        ↓
Prepare initial state |01>
        ↓
Apply parameterized ansatz
        ↓
Calculate <H>
        ↓
COBYLA changes theta
        ↓
Energy decreases
        ↓
Best theta found
        ↓
Compare with exact ground energy
""")

print(
    f"Exact ground energy : "
    f"{exact_ground_energy:.10f}"
)

print(
    f"VQE energy          : "
    f"{optimized_energy:.10f}"
)

print(
    f"Absolute error      : "
    f"{vqe_error:.10e}"
)

if vqe_error < 1e-6:
    print("\nSUCCESS: VQE reached the exact ground-state energy.")
else:
    print("\nVQE completed, but the error is above the target.")


print("\nDone.")