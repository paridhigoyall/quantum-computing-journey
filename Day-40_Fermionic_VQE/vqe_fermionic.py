import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# DAY 43 - SHOT BUDGET AND VQE ACCURACY
# ============================================================

print("=" * 60)
print("DAY 43 - SHOT BUDGET AND VQE ACCURACY")
print("=" * 60)


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8

SHOT_BUDGETS = [512, 2048, 8192]

print("\nPhysical parameters:")
print(f"eps0 = {eps0}")
print(f"eps1 = {eps1}")
print(f"t    = {t}")
print(f"U    = {U}")

print("\nShot budgets:")
for shots in SHOT_BUDGETS:
    print(f"  {shots} shots")


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
print("JORDAN-WIGNER PAULI HAMILTONIAN")
print("=" * 60)

print("\nPauli coefficients:")
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

eigenvalues, eigenvectors = np.linalg.eigh(H_one)

exact_energy = float(eigenvalues[0])

print("\n" + "=" * 60)
print("EXACT ONE-PARTICLE SOLUTION")
print("=" * 60)

print("\nOne-particle Hamiltonian:")
print(H_one)

print("\nEigenvalues:")
print(eigenvalues)

print(f"\nExact one-particle ground energy = "
      f"{exact_energy:.10f}")


# ============================================================
# VQE ANSATZ
# ============================================================

def prepare_state(theta):
    """
    One-particle ansatz:

        |psi(theta)>
        =
        cos(theta/2)|01>
        +
        sin(theta/2)|10>

    Therefore the state always remains in N = 1 sector.
    """

    state = np.array([
        0,
        np.cos(theta / 2),
        np.sin(theta / 2),
        0
    ], dtype=complex)

    return state / np.linalg.norm(state)


# ============================================================
# EXPECTATION VALUE
# ============================================================

def expectation(state, operator):
    return float(
        np.real(
            np.vdot(state, operator @ state)
        )
    )


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

    if basis == "Z":
        return state

    if basis == "X":

        rotation = kron(
            H_gate,
            H_gate
        )

        return rotation @ state

    if basis == "Y":

        single_rotation = H_gate @ S_dagger

        rotation = kron(
            single_rotation,
            single_rotation
        )

        return rotation @ state

    raise ValueError("Unknown measurement basis")


# ============================================================
# SHOT MEASUREMENT
# ============================================================

def measure_counts(state, basis, shots, rng):

    rotated_state = rotate_for_basis(
        state,
        basis
    )

    probabilities = np.abs(rotated_state) ** 2

    probabilities = (
        probabilities /
        np.sum(probabilities)
    )

    samples = rng.choice(
        4,
        size=shots,
        p=probabilities
    )

    return {
        "00": int(np.sum(samples == 0)),
        "01": int(np.sum(samples == 1)),
        "10": int(np.sum(samples == 2)),
        "11": int(np.sum(samples == 3))
    }


# ============================================================
# EXPECTATION FROM MEASUREMENT COUNTS
# ============================================================

def expectation_from_counts(
    counts,
    observable
):

    total = sum(counts.values())

    value = 0.0

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

            raise ValueError(
                "Unknown observable"
            )

        value += eigenvalue * count

    return value / total


# ============================================================
# SHOT-BASED ENERGY
# ============================================================

def measured_energy(
    theta,
    shots,
    rng,
    return_details=False
):

    state = prepare_state(theta)

    # --------------------------------------------------------
    # Z BASIS
    # --------------------------------------------------------

    z_counts = measure_counts(
        state,
        "Z",
        shots,
        rng
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
    # X BASIS
    # --------------------------------------------------------

    x_counts = measure_counts(
        state,
        "X",
        shots,
        rng
    )

    xx = expectation_from_counts(
        x_counts,
        "X0X1"
    )

    # --------------------------------------------------------
    # Y BASIS
    # --------------------------------------------------------

    y_counts = measure_counts(
        state,
        "Y",
        shots,
        rng
    )

    yy = expectation_from_counts(
        y_counts,
        "Y0Y1"
    )

    # --------------------------------------------------------
    # ENERGY RECONSTRUCTION
    # --------------------------------------------------------

    energy = (
        cI
        + cZ0 * z0
        + cZ1 * z1
        + cZZ * zz
        + cXX * xx
        + cYY * yy
    )

    if return_details:

        return energy, {
            "Z": z_counts,
            "X": x_counts,
            "Y": y_counts,
            "Z0": z0,
            "Z1": z1,
            "Z0Z1": zz,
            "X0X1": xx,
            "Y0Y1": yy
        }

    return energy


# ============================================================
# INITIAL STATE
# ============================================================

initial_theta = 0.5

print("\n" + "=" * 60)
print("INITIAL STATE")
print("=" * 60)

print(f"\nInitial theta = {initial_theta:.6f}")

initial_state = prepare_state(
    initial_theta
)

print("\nInitial state:")
print(initial_state)

print(
    f"\nInitial ideal energy = "
    f"{ideal_energy(initial_theta):.10f}"
)


# ============================================================
# RUN VQE FOR DIFFERENT SHOT BUDGETS
# ============================================================

results = []


for shots in SHOT_BUDGETS:

    print("\n")
    print("=" * 60)
    print(f"VQE WITH {shots} SHOTS")
    print("=" * 60)

    # --------------------------------------------------------
    # New RNG for each experiment
    # --------------------------------------------------------

    rng = np.random.default_rng(
        1000 + shots
    )

    history = []

    def objective(x):

        theta = float(x[0])

        energy = measured_energy(
            theta,
            shots,
            rng
        )

        history.append(
            (theta, energy)
        )

        print(
            f"Step {len(history):03d}: "
            f"theta={theta: .8f} | "
            f"Measured E={energy: .10f}"
        )

        return energy

    # --------------------------------------------------------
    # COBYLA
    # --------------------------------------------------------

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

    best_theta = float(
        result.x[0]
    )

    state = prepare_state(
        best_theta
    )

    ideal_vqe_energy = ideal_energy(
        best_theta
    )

    final_measured_energy, details = measured_energy(
        best_theta,
        shots,
        rng,
        return_details=True
    )

    ideal_error = abs(
        ideal_vqe_energy -
        exact_energy
    )

    measurement_error = abs(
        final_measured_energy -
        ideal_vqe_energy
    )

    total_error = abs(
        final_measured_energy -
        exact_energy
    )

    results.append({
        "shots": shots,
        "theta": best_theta,
        "ideal_energy": ideal_vqe_energy,
        "measured_energy": final_measured_energy,
        "ideal_error": ideal_error,
        "measurement_error": measurement_error,
        "total_error": total_error
    })

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print(f"RESULT FOR {shots} SHOTS")
    print("-" * 60)

    print(
        f"Best theta              = "
        f"{best_theta:.10f}"
    )

    print(
        f"Ideal VQE energy        = "
        f"{ideal_vqe_energy:.10f}"
    )

    print(
        f"Measured VQE energy     = "
        f"{final_measured_energy:.10f}"
    )

    print(
        f"Exact energy            = "
        f"{exact_energy:.10f}"
    )

    print(
        f"Ideal VQE error         = "
        f"{ideal_error:.10e}"
    )

    print(
        f"Measurement error       = "
        f"{measurement_error:.10e}"
    )

    print(
        f"Total measured error    = "
        f"{total_error:.10e}"
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("SHOT BUDGET COMPARISON")
print("=" * 60)

print(
    "\n"
    f"{'Shots':>8} "
    f"{'Theta':>14} "
    f"{'Ideal E':>14} "
    f"{'Measured E':>14} "
    f"{'Total Error':>14}"
)

print("-" * 68)

for item in results:

    print(
        f"{item['shots']:>8} "
        f"{item['theta']:>14.8f} "
        f"{item['ideal_energy']:>14.8f} "
        f"{item['measured_energy']:>14.8f} "
        f"{item['total_error']:>14.8f}"
    )


# ============================================================
# EXPECTATION VALUES FOR HIGHEST SHOT BUDGET
# ============================================================

best_result = results[-1]

best_theta = best_result["theta"]

best_state = prepare_state(
    best_theta
)

ideal_z0 = expectation(
    best_state,
    Z0
)

ideal_z1 = expectation(
    best_state,
    Z1
)

ideal_zz = expectation(
    best_state,
    Z0Z1
)

ideal_xx = expectation(
    best_state,
    X0X1
)

ideal_yy = expectation(
    best_state,
    Y0Y1
)

print("\n" + "=" * 60)
print("FINAL IDEAL EXPECTATION VALUES")
print("=" * 60)

print(f"\n<Z0>   = {ideal_z0:.10f}")
print(f"<Z1>   = {ideal_z1:.10f}")
print(f"<Z0Z1> = {ideal_zz:.10f}")
print(f"<X0X1> = {ideal_xx:.10f}")
print(f"<Y0Y1> = {ideal_yy:.10f}")


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("DAY 43 SUMMARY")
print("=" * 60)

print("""
Today we changed only the measurement budget.

The same VQE was run with:

    512 shots
        ↓
    2048 shots
        ↓
    8192 shots

Pipeline:

theta
   ↓
Parameterized state
   ↓
Finite-shot measurement
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

The important idea:

More shots
   ↓
Better statistical estimate
   ↓
Less measurement noise
   ↓
More reliable VQE energy
""")


print("\nExact one-particle energy:")
print(
    f"{exact_energy:.10f}"
)

print("\nResults:")

for item in results:

    print(
        f"{item['shots']:5d} shots -> "
        f"measured energy = "
        f"{item['measured_energy']:.10f}, "
        f"error = "
        f"{item['total_error']:.10e}"
    )


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 60)

if results[-1]["total_error"] < 0.02:

    print(
        "SUCCESS: Higher shot budget gives a "
        "reliable VQE energy estimate."
    )

else:

    print(
        "Measurement noise is still significant. "
        "More shots may be required."
    )

print("=" * 60)

print("\nDay 43 shot-budget experiment complete.")