import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# DAY 42 - SHOT-BASED FERMIONIC VQE
# ============================================================

print("=" * 60)
print("DAY 42 - SHOT-BASED FERMIONIC VQE")
print("=" * 60)

# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8

SHOTS = 2048

print("\nPhysical parameters:")
print(f"eps0 = {eps0}")
print(f"eps1 = {eps1}")
print(f"t    = {t}")
print(f"U    = {U}")
print(f"Shots per Pauli measurement = {SHOTS}")


# ============================================================
# PAULI MATRICES
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


I2 = kron(I, I)
X0X1 = kron(X, X)
Y0Y1 = kron(Y, Y)
Z0 = kron(Z, I)
Z1 = kron(I, Z)
Z0Z1 = kron(Z, Z)


# ============================================================
# JORDAN-WIGNER PAULI HAMILTONIAN
# ============================================================

cI = (eps0 + eps1) / 2 + U / 4
cZ0 = -eps0 / 2 - U / 4
cZ1 = -eps1 / 2 - U / 4
cZZ = U / 4
cXX = t / 2
cYY = t / 2

H = (
    cI * I2
    + cZ0 * Z0
    + cZ1 * Z1
    + cZZ * Z0Z1
    + cXX * X0X1
    + cYY * Y0Y1
)

print("\n" + "=" * 60)
print("PAULI HAMILTONIAN")
print("=" * 60)

print(f"I      : {cI:+.6f}")
print(f"Z0     : {cZ0:+.6f}")
print(f"Z1     : {cZ1:+.6f}")
print(f"Z0Z1   : {cZZ:+.6f}")
print(f"X0X1   : {cXX:+.6f}")
print(f"Y0Y1   : {cYY:+.6f}")


# ============================================================
# EXACT ONE-PARTICLE SOLUTION
# ============================================================

H_one = np.array([
    [eps1, t],
    [t, eps0]
], dtype=complex)

one_eigenvalues, one_eigenvectors = np.linalg.eigh(H_one)

exact_energy = one_eigenvalues[0]

print("\n" + "=" * 60)
print("EXACT ONE-PARTICLE SOLUTION")
print("=" * 60)

print("\nOne-particle Hamiltonian:")
print(H_one)

print("\nEigenvalues:")
print(one_eigenvalues)

print(f"\nExact one-particle ground energy = {exact_energy:.10f}")


# ============================================================
# VQE ANSATZ
# ============================================================

def prepare_state(theta):
    """
    One-particle ansatz:

        |psi(theta)> =
            cos(theta/2)|01>
          + sin(theta/2)|10>

    This automatically keeps N = 1.
    """

    state = np.array([
        0,
        np.cos(theta / 2),
        np.sin(theta / 2),
        0
    ], dtype=complex)

    return state / np.linalg.norm(state)


# ============================================================
# IDEAL EXPECTATION VALUE
# ============================================================

def expectation(state, operator):
    return float(np.real(np.vdot(state, operator @ state)))


def ideal_energy(theta):
    state = prepare_state(theta)
    return expectation(state, H)


# ============================================================
# BASIS ROTATIONS
# ============================================================

H_gate = (X + Z) / np.sqrt(2)

S_dagger = np.array([
    [1, 0],
    [0, -1j]
], dtype=complex)


def rotate_for_basis(state, basis):
    """
    Convert X/Y measurements into Z measurements.

    Z:
        no rotation

    X:
        H

    Y:
        S-dagger followed by H
    """

    if basis == "Z":
        return state

    if basis == "X":
        rotation = kron(H_gate, H_gate)
        return rotation @ state

    if basis == "Y":
        single_rotation = H_gate @ S_dagger
        rotation = kron(single_rotation, single_rotation)
        return rotation @ state

    raise ValueError("Unknown measurement basis")


# ============================================================
# SHOT SAMPLING
# ============================================================

rng = np.random.default_rng()


def measure_counts(state, basis, shots=SHOTS):

    rotated_state = rotate_for_basis(state, basis)

    probabilities = np.abs(rotated_state) ** 2
    probabilities = probabilities / np.sum(probabilities)

    outcomes = ["00", "01", "10", "11"]

    samples = rng.choice(
        4,
        size=shots,
        p=probabilities
    )

    counts = {
        "00": int(np.sum(samples == 0)),
        "01": int(np.sum(samples == 1)),
        "10": int(np.sum(samples == 2)),
        "11": int(np.sum(samples == 3))
    }

    return counts


# ============================================================
# EXPECTATION FROM COUNTS
# ============================================================

def expectation_from_counts(counts, observable):

    total = sum(counts.values())

    expectation_value = 0.0

    for bitstring, count in counts.items():

        b0 = int(bitstring[0])
        b1 = int(bitstring[1])

        z0 = 1 if b0 == 0 else -1
        z1 = 1 if b1 == 0 else -1

        if observable == "Z0":
            eigenvalue = z0

        elif observable == "Z1":
            eigenvalue = z1

        elif observable == "Z0Z1":
            eigenvalue = z0 * z1

        elif observable == "X0X1":
            eigenvalue = z0 * z1

        elif observable == "Y0Y1":
            eigenvalue = z0 * z1

        else:
            raise ValueError("Unknown observable")

        expectation_value += eigenvalue * count

    return expectation_value / total


# ============================================================
# SHOT-BASED ENERGY
# ============================================================

def measured_energy(theta, verbose=False):

    state = prepare_state(theta)

    # --------------------------------------------------------
    # Z measurement
    # --------------------------------------------------------

    z_counts = measure_counts(
        state,
        "Z",
        SHOTS
    )

    z0 = expectation_from_counts(
        z_counts,
        "Z0"
    )

    z1 = expectation_from_counts(
        z_counts,
        "Z1"
    )

    zz = expectation_from_counts(
        z_counts,
        "Z0Z1"
    )

    # --------------------------------------------------------
    # X measurement
    # --------------------------------------------------------

    x_counts = measure_counts(
        state,
        "X",
        SHOTS
    )

    xx = expectation_from_counts(
        x_counts,
        "X0X1"
    )

    # --------------------------------------------------------
    # Y measurement
    # --------------------------------------------------------

    y_counts = measure_counts(
        state,
        "Y",
        SHOTS
    )

    yy = expectation_from_counts(
        y_counts,
        "Y0Y1"
    )

    # --------------------------------------------------------
    # Hamiltonian reconstruction
    # --------------------------------------------------------

    energy = (
        cI
        + cZ0 * z0
        + cZ1 * z1
        + cZZ * zz
        + cXX * xx
        + cYY * yy
    )

    if verbose:

        print("\nMeasurement details:")
        print(f"Z counts = {z_counts}")
        print(f"X counts = {x_counts}")
        print(f"Y counts = {y_counts}")

        print("\nMeasured expectation values:")
        print(f"<Z0>   = {z0:.6f}")
        print(f"<Z1>   = {z1:.6f}")
        print(f"<Z0Z1> = {zz:.6f}")
        print(f"<X0X1> = {xx:.6f}")
        print(f"<Y0Y1> = {yy:.6f}")

    return energy


# ============================================================
# INITIAL STATE
# ============================================================

initial_theta = 0.5

print("\n" + "=" * 60)
print("INITIAL STATE")
print("=" * 60)

print(f"\nInitial theta = {initial_theta:.6f}")

initial_state = prepare_state(initial_theta)

print("\nInitial state:")
print(initial_state)

print(
    f"\nInitial ideal energy = "
    f"{ideal_energy(initial_theta):.10f}"
)


# ============================================================
# SHOT-BASED OPTIMIZATION
# ============================================================

print("\n" + "=" * 60)
print("SHOT-BASED VQE OPTIMIZATION")
print("=" * 60)

print("""
Important:

The optimizer does NOT receive the ideal energy.

Instead:

theta
  ↓
prepare state
  ↓
measure Z
measure X
measure Y
  ↓
estimate expectation values
  ↓
reconstruct energy
  ↓
COBYLA receives measured energy
""")

# We use scipy's COBYLA exactly as in our previous VQE work.
from scipy.optimize import minimize


history = []


def objective(x):

    theta = float(x[0])

    energy = measured_energy(theta)

    history.append((theta, energy))

    print(
        f"Step {len(history):03d}: "
        f"theta={theta: .8f} | "
        f"Measured E={energy: .10f}"
    )

    return energy


result = minimize(
    objective,
    x0=np.array([initial_theta]),
    method="COBYLA",
    options={
        "maxiter": 60,
        "rhobeg": 0.5,
        "tol": 1e-5
    }
)


# ============================================================
# OPTIMIZATION RESULT
# ============================================================

best_theta = float(result.x[0])

best_state = prepare_state(best_theta)

best_ideal_energy = ideal_energy(best_theta)

# Perform one fresh final measurement
final_measured_energy = measured_energy(
    best_theta,
    verbose=True
)


print("\n" + "=" * 60)
print("VQE RESULT")
print("=" * 60)

print(f"\nBest theta = {best_theta:.10f}")

print(f"Ideal energy at best theta = "
      f"{best_ideal_energy:.10f}")

print(f"Final measured energy = "
      f"{final_measured_energy:.10f}")

print(f"Exact one-particle energy = "
      f"{exact_energy:.10f}")

print("\nOptimized state:")
print(best_state)


# ============================================================
# FINAL IDEAL EXPECTATION VALUES
# ============================================================

ideal_z0 = expectation(best_state, Z0)
ideal_z1 = expectation(best_state, Z1)
ideal_zz = expectation(best_state, Z0Z1)
ideal_xx = expectation(best_state, X0X1)
ideal_yy = expectation(best_state, Y0Y1)

print("\n" + "=" * 60)
print("FINAL IDEAL EXPECTATION VALUES")
print("=" * 60)

print(f"\n<Z0>   = {ideal_z0:.10f}")
print(f"<Z1>   = {ideal_z1:.10f}")
print(f"<Z0Z1> = {ideal_zz:.10f}")
print(f"<X0X1> = {ideal_xx:.10f}")
print(f"<Y0Y1> = {ideal_yy:.10f}")


# ============================================================
# FINAL ACCURACY
# ============================================================

ideal_error = abs(
    best_ideal_energy - exact_energy
)

measurement_error = abs(
    final_measured_energy - best_ideal_energy
)

total_error = abs(
    final_measured_energy - exact_energy
)


print("\n" + "=" * 60)
print("FINAL ACCURACY")
print("=" * 60)

print(
    f"\nExact energy       = "
    f"{exact_energy:.10f}"
)

print(
    f"Ideal VQE energy   = "
    f"{best_ideal_energy:.10f}"
)

print(
    f"Measured VQE energy = "
    f"{final_measured_energy:.10f}"
)

print(
    f"\nIdeal VQE error    = "
    f"{ideal_error:.10e}"
)

print(
    f"Measurement error  = "
    f"{measurement_error:.10e}"
)

print(
    f"Total error        = "
    f"{total_error:.10e}"
)


# ============================================================
# DAY 42 SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 42 SUMMARY")
print("=" * 60)

print("""
Day 41:
    Optimize using ideal state-vector energy.

Day 42:
    Optimize using shot-based measured energy.

Pipeline:

theta
   ↓
Parameterized quantum state
   ↓
Z / X / Y basis measurements
   ↓
Finite measurement shots
   ↓
Expectation values
   ↓
Hamiltonian energy
   ↓
COBYLA
   ↓
New theta
   ↓
Repeat
""")

print(f"Exact energy          : {exact_energy:.10f}")
print(f"Ideal optimized       : {best_ideal_energy:.10f}")
print(f"Measured final energy : {final_measured_energy:.10f}")

print(f"\nShots per measurement : {SHOTS}")

if total_error < 0.05:
    print("\nSUCCESS: Shot-based VQE reached an energy close to")
    print("the exact one-particle ground-state energy.")
else:
    print("\nShot noise is significant.")
    print("Increasing the number of shots should improve accuracy.")

print("\nDay 42 shot-based VQE complete.")