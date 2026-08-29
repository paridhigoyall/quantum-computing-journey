import numpy as np
from scipy.optimize import minimize


# ============================================================
# DAY 40 - FERMIONIC VQE
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


# ============================================================
# 2. TWO-QUBIT OPERATORS
# ============================================================

I4 = np.kron(I, I)

X0 = np.kron(X, I)
X1 = np.kron(I, X)

Y0 = np.kron(Y, I)
Y1 = np.kron(I, Y)

Z0 = np.kron(Z, I)
Z1 = np.kron(I, Z)

X0X1 = np.kron(X, X)
Y0Y1 = np.kron(Y, Y)
Z0Z1 = np.kron(Z, Z)


# ============================================================
# 3. PHYSICAL FERMIONIC MODEL
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
# 4. JORDAN-WIGNER MAPPING
# ============================================================

n0 = (I4 - Z0) / 2
n1 = (I4 - Z1) / 2

hopping = (X0X1 + Y0Y1) / 2

interaction = (
    I4
    - Z0
    - Z1
    + Z0Z1
) / 4


# ============================================================
# 5. QUBIT HAMILTONIAN
# ============================================================

H = (
    eps0 * n0
    + eps1 * n1
    + t * hopping
    + U * interaction
)


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


# ============================================================
# 6. PAULI COEFFICIENTS
# ============================================================

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


print("Numerical Pauli coefficients:")

print(f"I      : {identity_coeff:+.6f}")
print(f"Z0     : {z0_coeff:+.6f}")
print(f"Z1     : {z1_coeff:+.6f}")
print(f"Z0Z1   : {zz_coeff:+.6f}")
print(f"X0X1   : {xx_coeff:+.6f}")
print(f"Y0Y1   : {yy_coeff:+.6f}")

print("\nHamiltonian matrix:")
print(H)


# ============================================================
# 7. EXACT FULL HAMILTONIAN SOLUTION
#
# This includes every particle-number sector.
#
# IMPORTANT:
# The absolute ground state of this Hamiltonian is |00>
# with energy 0.
#
# Our VQE below intentionally works in the ONE-PARTICLE
# sector because that is the sector containing |01> and |10>.
# ============================================================

exact_eigenvalues, exact_eigenvectors = np.linalg.eigh(H)

full_ground_energy = exact_eigenvalues[0]
full_ground_state = exact_eigenvectors[:, 0]

print("\n" + "=" * 60)
print("EXACT FULL HAMILTONIAN SOLUTION")
print("=" * 60)

print("\nEigenvalues:")
print(exact_eigenvalues)

print(
    f"\nFull ground-state energy = "
    f"{full_ground_energy:.10f}"
)

print("\nFull ground state:")
print(full_ground_state)


# ============================================================
# 8. PARTICLE NUMBER OPERATOR
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
# 9. ONE-PARTICLE SECTOR
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

one_particle_ground_energy = sector_eigenvalues[0]

one_particle_ground_vector = sector_eigenvectors[:, 0]


print("\nOne-particle eigenvalues:")
print(sector_eigenvalues)

print(
    f"\nOne-particle ground energy = "
    f"{one_particle_ground_energy:.10f}"
)

print("\nExact one-particle ground state coefficients:")
print(one_particle_ground_vector)


# ============================================================
# 10. VQE ANSATZ
#
# We need the ansatz to actually MIX |01> and |10>.
#
# The previous ansatz used:
#
#     |01> + (-i)|10>
#
# which made the real hopping expectation vanish.
#
# Here we deliberately use a REAL superposition:
#
#     |psi(theta)>
#
#       = cos(theta/2)|01>
#       + sin(theta/2)|10>
#
# This allows <X0X1 + Y0Y1> to contribute.
#
# theta controls how much amplitude is placed in each state.
# ============================================================

def ansatz_state(theta):

    state = np.zeros(4, dtype=complex)

    state[1] = np.cos(theta / 2)
    state[2] = np.sin(theta / 2)

    return state


# ============================================================
# 11. EXPECTATION VALUE
# ============================================================

def expectation(state, operator):

    return np.real(
        state.conj() @ operator @ state
    )


# ============================================================
# 12. VQE ENERGY
# ============================================================

def vqe_energy(parameters):

    theta = parameters[0]

    state = ansatz_state(theta)

    return expectation(
        state,
        H
    )


# ============================================================
# 13. INITIAL STATE
# ============================================================

initial_theta = 0.5

initial_state = ansatz_state(
    initial_theta
)

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
# 14. VQE OPTIMIZATION
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
# 15. OPTIMIZATION RESULT
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
    f"Exact one-particle energy = "
    f"{one_particle_ground_energy:.10f}"
)

print("\nOptimized state:")
print(optimized_state)


# ============================================================
# 16. EXPECTATION VALUES
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
    f"\n<Z0>   = "
    f"{z0_expectation:.10f}"
)

print(
    f"<Z1>   = "
    f"{z1_expectation:.10f}"
)

print(
    f"<Z0Z1> = "
    f"{zz_expectation:.10f}"
)

print(
    f"<X0X1> = "
    f"{xx_expectation:.10f}"
)

print(
    f"<Y0Y1> = "
    f"{yy_expectation:.10f}"
)

print(
    f"<N>    = "
    f"{particle_number_expectation:.10f}"
)


# ============================================================
# 17. PAULI ENERGY RECONSTRUCTION
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
# 18. VQE ERROR
#
# IMPORTANT:
# Compare against the ground energy of the SAME sector.
# ============================================================

vqe_error = abs(
    optimized_energy
    - one_particle_ground_energy
)


print("\n" + "=" * 60)
print("VQE ACCURACY")
print("=" * 60)

print(
    f"\nOne-particle exact energy = "
    f"{one_particle_ground_energy:.10f}"
)

print(
    f"VQE energy               = "
    f"{optimized_energy:.10f}"
)

print(
    f"VQE error                = "
    f"{vqe_error:.10e}"
)


# ============================================================
# 19. Z-BASIS SHOT MEASUREMENT
#
# Measuring in the computational basis gives:
#
# 00 -> Z0=+1, Z1=+1, Z0Z1=+1
# 01 -> Z0=+1, Z1=-1, Z0Z1=-1
# 10 -> Z0=-1, Z1=+1, Z0Z1=-1
# 11 -> Z0=-1, Z1=-1, Z0Z1=+1
# ============================================================

def measure_z_basis(state, shots=2048, seed=None):

    rng = np.random.default_rng(seed)

    probabilities = np.abs(state) ** 2

    probabilities = (
        probabilities
        / np.sum(probabilities)
    )

    basis_labels = [
        "00",
        "01",
        "10",
        "11"
    ]

    samples = rng.choice(
        basis_labels,
        size=shots,
        p=probabilities
    )

    counts = {
        "00": int(np.sum(samples == "00")),
        "01": int(np.sum(samples == "01")),
        "10": int(np.sum(samples == "10")),
        "11": int(np.sum(samples == "11"))
    }

    return counts


# ============================================================
# 20. EXPECTATIONS FROM Z COUNTS
# ============================================================

def z_expectations_from_counts(counts):

    shots = sum(counts.values())

    n00 = counts.get("00", 0)
    n01 = counts.get("01", 0)
    n10 = counts.get("10", 0)
    n11 = counts.get("11", 0)

    z0 = (
        n00
        + n01
        - n10
        - n11
    ) / shots

    z1 = (
        n00
        - n01
        + n10
        - n11
    ) / shots

    zz = (
        n00
        - n01
        - n10
        + n11
    ) / shots

    return z0, z1, zz


# ============================================================
# 21. TEST Z-BASIS MEASUREMENT
# ============================================================

print("\n" + "=" * 60)
print("Z-BASIS SHOT MEASUREMENT")
print("=" * 60)

shots = 2048

z_counts = measure_z_basis(
    optimized_state,
    shots=shots,
    seed=42
)

z0_measured, z1_measured, zz_measured = (
    z_expectations_from_counts(
        z_counts
    )
)


print("\nMeasurement shots:")
print(shots)

print("\nZ-basis counts:")
print(z_counts)

print("\nMeasured expectation values:")

print(
    f"<Z0>   = "
    f"{z0_measured:.6f}"
)

print(
    f"<Z1>   = "
    f"{z1_measured:.6f}"
)

print(
    f"<Z0Z1> = "
    f"{zz_measured:.6f}"
)


# ============================================================
# 22. IDEAL VS MEASURED
# ============================================================

print("\nIdeal vs measured:")

print(
    f"<Z0>   : "
    f"ideal={z0_expectation:.6f}, "
    f"measured={z0_measured:.6f}"
)

print(
    f"<Z1>   : "
    f"ideal={z1_expectation:.6f}, "
    f"measured={z1_measured:.6f}"
)

print(
    f"<Z0Z1> : "
    f"ideal={zz_expectation:.6f}, "
    f"measured={zz_measured:.6f}"
)


# ============================================================
# 23. FINAL SUMMARY
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
Select one-particle sector
        ↓
Prepare |01>
        ↓
Parameterized superposition
        ↓
Calculate <H>
        ↓
COBYLA changes theta
        ↓
Energy decreases
        ↓
Find one-particle ground state
        ↓
Measure in Z basis
        ↓
Estimate expectation values
""")

print(
    f"Full Hamiltonian ground : "
    f"{full_ground_energy:.10f}"
)

print(
    f"One-particle exact       : "
    f"{one_particle_ground_energy:.10f}"
)

print(
    f"VQE energy               : "
    f"{optimized_energy:.10f}"
)

print(
    f"VQE error                : "
    f"{vqe_error:.10e}"
)

print(
    f"Measured <Z0>            : "
    f"{z0_measured:.6f}"
)

print(
    f"Measured <Z1>            : "
    f"{z1_measured:.6f}"
)

print(
    f"Measured <Z0Z1>          : "
    f"{zz_measured:.6f}"
)


if vqe_error < 1e-6:

    print(
        "\nSUCCESS: "
        "VQE reached the one-particle ground-state energy."
    )

else:

    print(
        "\nVQE completed, "
        "but the error is above the target."
    )


print("\nDone.")