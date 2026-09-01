import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# DAY 44 - VQE WITH READOUT NOISE
# ============================================================

print("=" * 60)
print("DAY 44 - VQE WITH READOUT NOISE")
print("=" * 60)


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8

SHOTS = 2048

# Readout error probabilities
NOISE_LEVELS = [0.00, 0.02, 0.05]

print("\nPhysical parameters:")
print(f"eps0 = {eps0}")
print(f"eps1 = {eps1}")
print(f"t    = {t}")
print(f"U    = {U}")

print(f"\nShots per measurement = {SHOTS}")

print("\nReadout noise levels:")
for noise in NOISE_LEVELS:
    print(f"  {noise * 100:.0f}% readout error")


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

    The state therefore remains in the N = 1 sector.
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

    return float(
        np.real(
            np.vdot(
                state,
                operator @ state
            )
        )
    )


def ideal_energy(theta):

    state = prepare_state(theta)

    return expectation(
        state,
        H
    )


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
# IDEAL SHOT GENERATION
# ============================================================

def generate_ideal_samples(
    state,
    basis,
    shots,
    rng
):

    rotated_state = rotate_for_basis(
        state,
        basis
    )

    probabilities = np.abs(
        rotated_state
    ) ** 2

    probabilities /= np.sum(
        probabilities
    )

    samples = rng.choice(
        4,
        size=shots,
        p=probabilities
    )

    return samples


# ============================================================
# READOUT NOISE
# ============================================================

def apply_readout_noise(
    samples,
    noise_probability,
    rng
):
    """
    Simulate measurement/readout errors.

    Each measured qubit has an independent probability
    of having its classical bit flipped.

    Example:

        actual 0
           ↓
        2% error
           ↓
        reported 1

    and:

        actual 1
           ↓
        2% error
           ↓
        reported 0
    """

    noisy_samples = samples.copy()

    if noise_probability <= 0:
        return noisy_samples

    for i in range(len(noisy_samples)):

        # Convert integer outcome to two bits
        b0 = (noisy_samples[i] >> 1) & 1
        b1 = noisy_samples[i] & 1

        # Qubit 0 readout error
        if rng.random() < noise_probability:
            b0 = 1 - b0

        # Qubit 1 readout error
        if rng.random() < noise_probability:
            b1 = 1 - b1

        # Convert bits back to integer
        noisy_samples[i] = (
            (b0 << 1) | b1
        )

    return noisy_samples


# ============================================================
# COUNTS
# ============================================================

def samples_to_counts(samples):

    return {
        "00": int(np.sum(samples == 0)),
        "01": int(np.sum(samples == 1)),
        "10": int(np.sum(samples == 2)),
        "11": int(np.sum(samples == 3))
    }


# ============================================================
# EXPECTATION FROM COUNTS
# ============================================================

def expectation_from_counts(
    counts,
    observable
):

    total = sum(
        counts.values()
    )

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

        value += (
            eigenvalue * count
        )

    return value / total


# ============================================================
# NOISY MEASUREMENT
# ============================================================

def measure_counts(
    state,
    basis,
    shots,
    noise_probability,
    rng
):

    ideal_samples = generate_ideal_samples(
        state,
        basis,
        shots,
        rng
    )

    noisy_samples = apply_readout_noise(
        ideal_samples,
        noise_probability,
        rng
    )

    return samples_to_counts(
        noisy_samples
    )


# ============================================================
# SHOT-BASED ENERGY
# ============================================================

def measured_energy(
    theta,
    shots,
    noise_probability,
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
        noise_probability,
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
        noise_probability,
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
        noise_probability,
        rng
    )

    yy = expectation_from_counts(
        y_counts,
        "Y0Y1"
    )

    # --------------------------------------------------------
    # RECONSTRUCT ENERGY
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

print(
    f"\nInitial theta = "
    f"{initial_theta:.6f}"
)

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
# VQE FOR EACH NOISE LEVEL
# ============================================================

results = []


for noise in NOISE_LEVELS:

    print("\n")
    print("=" * 60)
    print(
        f"VQE WITH "
        f"{noise * 100:.0f}% READOUT NOISE"
    )
    print("=" * 60)

    # Separate random generator for each experiment
    rng = np.random.default_rng(
        44000 + int(noise * 100)
    )

    history = []

    def objective(x):

        theta = float(x[0])

        energy = measured_energy(
            theta,
            SHOTS,
            noise,
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

    best_state = prepare_state(
        best_theta
    )

    ideal_vqe_energy = ideal_energy(
        best_theta
    )

    final_measured_energy, details = measured_energy(
        best_theta,
        SHOTS,
        noise,
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
        "noise": noise,
        "theta": best_theta,
        "ideal_energy": ideal_vqe_energy,
        "measured_energy": final_measured_energy,
        "ideal_error": ideal_error,
        "measurement_error": measurement_error,
        "total_error": total_error
    })

    # --------------------------------------------------------
    # FINAL DETAILS
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print(
        f"RESULT FOR "
        f"{noise * 100:.0f}% READOUT NOISE"
    )
    print("-" * 60)

    print(
        f"Best theta           = "
        f"{best_theta:.10f}"
    )

    print(
        f"Ideal VQE energy     = "
        f"{ideal_vqe_energy:.10f}"
    )

    print(
        f"Measured VQE energy  = "
        f"{final_measured_energy:.10f}"
    )

    print(
        f"Exact energy         = "
        f"{exact_energy:.10f}"
    )

    print(
        f"Ideal VQE error      = "
        f"{ideal_error:.10e}"
    )

    print(
        f"Measurement error    = "
        f"{measurement_error:.10e}"
    )

    print(
        f"Total measured error = "
        f"{total_error:.10e}"
    )


# ============================================================
# NOISE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("READOUT NOISE COMPARISON")
print("=" * 60)

print(
    "\n"
    f"{'Noise':>8} "
    f"{'Theta':>14} "
    f"{'Ideal E':>14} "
    f"{'Measured E':>14} "
    f"{'Total Error':>14}"
)

print("-" * 68)

for item in results:

    print(
        f"{item['noise'] * 100:>7.0f}% "
        f"{item['theta']:>14.8f} "
        f"{item['ideal_energy']:>14.8f} "
        f"{item['measured_energy']:>14.8f} "
        f"{item['total_error']:>14.8f}"
    )


# ============================================================
# EXPECTATION VALUES FOR 0% NOISE
# ============================================================

zero_noise_result = results[0]

zero_noise_theta = (
    zero_noise_result["theta"]
)

zero_noise_state = prepare_state(
    zero_noise_theta
)

print("\n" + "=" * 60)
print("IDEAL EXPECTATION VALUES")
print("=" * 60)

print(
    f"\n<Z0>   = "
    f"{expectation(zero_noise_state, Z0):.10f}"
)

print(
    f"<Z1>   = "
    f"{expectation(zero_noise_state, Z1):.10f}"
)

print(
    f"<Z0Z1> = "
    f"{expectation(zero_noise_state, Z0Z1):.10f}"
)

print(
    f"<X0X1> = "
    f"{expectation(zero_noise_state, X0X1):.10f}"
)

print(
    f"<Y0Y1> = "
    f"{expectation(zero_noise_state, Y0Y1):.10f}"
)


# ============================================================
# DAY 44 SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 44 SUMMARY")
print("=" * 60)

print("""
Day 43:
    Studied finite-shot measurement noise.

Day 44:
    Added readout noise to the measurement process.

Pipeline:

Parameterized state
        ↓
Quantum measurement
        ↓
Readout error
        ↓
Measured bitstring
        ↓
Expectation value
        ↓
Hamiltonian energy
        ↓
COBYLA
        ↓
New theta
        ↓
Repeat

Noise levels tested:

0%
 ↓
2%
 ↓
5%

The goal is to observe how imperfect
readout affects VQE accuracy.
""")

print(
    f"\nExact one-particle energy = "
    f"{exact_energy:.10f}"
)

print("\nFinal comparison:")

for item in results:

    print(
        f"{item['noise'] * 100:>5.0f}% noise -> "
        f"Measured E = "
        f"{item['measured_energy']:.10f}, "
        f"Error = "
        f"{item['total_error']:.10e}"
    )


# ============================================================
# SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 60)

print(
    "SUCCESS: VQE was tested under "
    "different readout-noise levels."
)

print("=" * 60)

print("\nDay 44 readout-noise experiment complete.")