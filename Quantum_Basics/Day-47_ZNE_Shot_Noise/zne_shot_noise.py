import numpy as np
from scipy.optimize import minimize


# ============================================================
# DAY 47
# ZERO-NOISE EXTRAPOLATION WITH SHOT NOISE
# ============================================================


# ------------------------------------------------------------
# 1. Physical Hamiltonian parameters
# ------------------------------------------------------------

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8


# ------------------------------------------------------------
# 2. One-qubit matrices
# ------------------------------------------------------------

I1 = np.eye(2)

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


# ------------------------------------------------------------
# 3. Two-qubit identity
# ------------------------------------------------------------

I2 = np.kron(
    I1,
    I1
)


# ------------------------------------------------------------
# 4. Kronecker product helper
# ------------------------------------------------------------

def kron(a, b):
    return np.kron(a, b)


# ------------------------------------------------------------
# 5. Two-qubit operators
# ------------------------------------------------------------

X0 = kron(X, I1)
X1 = kron(I1, X)

Y0 = kron(Y, I1)
Y1 = kron(I1, Y)

Z0 = kron(Z, I1)
Z1 = kron(I1, Z)

X0X1 = kron(X, X)
Y0Y1 = kron(Y, Y)

Z0Z1 = kron(Z, Z)


# ------------------------------------------------------------
# 6. Number operators
# ------------------------------------------------------------

n0_operator = (
    I2 - Z0
) / 2

n1_operator = (
    I2 - Z1
) / 2


# ------------------------------------------------------------
# 7. Fermionic Hamiltonian
# ------------------------------------------------------------

H = (
    eps0 * n0_operator
    + eps1 * n1_operator
    + t * (X0X1 + Y0Y1) / 2
    + U * (n0_operator @ n1_operator)
)


# ------------------------------------------------------------
# 8. One-particle sector
# ------------------------------------------------------------

one_particle_indices = [
    1,
    2
]

H_one = H[
    np.ix_(
        one_particle_indices,
        one_particle_indices
    )
]

exact_eigenvalues = np.linalg.eigvalsh(
    H_one
)

exact_energy = exact_eigenvalues[0]


# ============================================================
# VQE
# ============================================================


# ------------------------------------------------------------
# 9. VQE ansatz
# ------------------------------------------------------------

def state(theta):
    """
    One-particle VQE ansatz:

        |psi(theta)>
        =
        cos(theta/2)|01>
        +
        sin(theta/2)|10>
    """

    psi = np.zeros(
        4,
        dtype=complex
    )

    psi[1] = np.cos(
        theta / 2
    )

    psi[2] = np.sin(
        theta / 2
    )

    return psi


# ------------------------------------------------------------
# 10. Expectation value
# ------------------------------------------------------------

def expectation(
    psi,
    operator
):

    return np.real(
        np.vdot(
            psi,
            operator @ psi
        )
    )


# ------------------------------------------------------------
# 11. Ideal VQE energy
# ------------------------------------------------------------

def ideal_energy(theta):

    psi = state(theta)

    return expectation(
        psi,
        H
    )


# ------------------------------------------------------------
# 12. Find ideal VQE optimum
# ------------------------------------------------------------

result = minimize(
    lambda x: ideal_energy(x[0]),
    x0=[1.0],
    method="COBYLA",
    options={
        "maxiter": 500,
        "rhobeg": 0.5,
        "tol": 1e-10
    }
)

best_theta = result.x[0]

psi_opt = state(
    best_theta
)

ideal_vqe_energy = ideal_energy(
    best_theta
)


# ============================================================
# NOISE MODEL
# ============================================================


# ------------------------------------------------------------
# 13. Depolarizing noise
# ------------------------------------------------------------

def apply_depolarizing_noise(
    psi,
    noise_strength
):
    """
    Apply a simple depolarizing noise model.

    rho_noisy =
        (1-p) rho
        +
        p I/d
    """

    # Convert statevector to density matrix.

    rho = np.outer(
        psi,
        np.conjugate(psi)
    )

    dimension = len(psi)

    maximally_mixed_state = (
        np.eye(dimension)
        / dimension
    )

    noisy_rho = (
        (1 - noise_strength) * rho
        + noise_strength * maximally_mixed_state
    )

    return noisy_rho


# ------------------------------------------------------------
# 14. Density matrix expectation
# ------------------------------------------------------------

def density_matrix_expectation(
    rho,
    operator
):

    value = np.trace(
        rho @ operator
    )

    return np.real(value)


# ============================================================
# MEASUREMENT
# ============================================================


# ------------------------------------------------------------
# 15. Measurement probabilities
# ------------------------------------------------------------

def measurement_probabilities(
    rho,
    basis_operator
):
    """
    Calculate probabilities of the +1 and -1
    outcomes for a Pauli observable.

    For a Pauli operator P:

        P(+1) = (1 + <P>) / 2
        P(-1) = (1 - <P>) / 2
    """

    expectation_value = density_matrix_expectation(
        rho,
        basis_operator
    )

    p_plus = (
        1 + expectation_value
    ) / 2

    p_minus = (
        1 - expectation_value
    ) / 2

    # Protect against tiny floating-point errors.

    p_plus = np.clip(
        p_plus,
        0,
        1
    )

    p_minus = np.clip(
        p_minus,
        0,
        1
    )

    return np.array([
        p_plus,
        p_minus
    ])


# ------------------------------------------------------------
# 16. Sample Pauli measurement
# ------------------------------------------------------------

def sample_pauli_measurement(
    rho,
    operator,
    shots
):
    """
    Simulate finite-shot measurement.

    Each shot returns either:

        +1
        -1

    The expectation value is estimated from
    the average measurement outcome.
    """

    probabilities = measurement_probabilities(
        rho,
        operator
    )

    outcomes = np.random.choice(
        [1, -1],
        size=shots,
        p=probabilities
    )

    measured_expectation = np.mean(
        outcomes
    )

    return measured_expectation


# ============================================================
# NOISY ENERGY MEASUREMENT
# ============================================================


# ------------------------------------------------------------
# 17. Measure noisy energy using finite shots
# ------------------------------------------------------------

def noisy_energy_with_shots(
    theta,
    noise_strength,
    shots
):
    """
    Prepare the VQE state, add quantum noise,
    and estimate the Hamiltonian energy using
    finite measurement shots.
    """

    psi = state(theta)

    rho = apply_depolarizing_noise(
        psi,
        noise_strength
    )

    # --------------------------------------------------------
    # Measure Z0
    # --------------------------------------------------------

    measured_z0 = sample_pauli_measurement(
        rho,
        Z0,
        shots
    )

    # --------------------------------------------------------
    # Measure Z1
    # --------------------------------------------------------

    measured_z1 = sample_pauli_measurement(
        rho,
        Z1,
        shots
    )

    # --------------------------------------------------------
    # Measure Z0Z1
    # --------------------------------------------------------

    measured_z0z1 = sample_pauli_measurement(
        rho,
        Z0Z1,
        shots
    )

    # --------------------------------------------------------
    # Measure X0X1
    # --------------------------------------------------------

    measured_xx = sample_pauli_measurement(
        rho,
        X0X1,
        shots
    )

    # --------------------------------------------------------
    # Measure Y0Y1
    # --------------------------------------------------------

    measured_yy = sample_pauli_measurement(
        rho,
        Y0Y1,
        shots
    )

    # --------------------------------------------------------
    # Reconstruct energy
    # --------------------------------------------------------

    n0 = (
        1 - measured_z0
    ) / 2

    n1 = (
        1 - measured_z1
    ) / 2

    interaction = (
        1
        - measured_z0
        - measured_z1
        + measured_z0z1
    ) / 4

    energy = (
        eps0 * n0
        + eps1 * n1
        + t * (
            measured_xx
            + measured_yy
        ) / 2
        + U * interaction
    )

    return energy


# ============================================================
# ZERO-NOISE EXTRAPOLATION
# ============================================================


# ------------------------------------------------------------
# 18. Experimental settings
# ------------------------------------------------------------

shots = 2048

base_noise = 0.05

noise_scales = np.array([
    1.0,
    2.0,
    3.0
])


# ------------------------------------------------------------
# 19. Run ZNE experiment
# ------------------------------------------------------------

measured_energies = []

actual_noise_levels = []

for scale in noise_scales:

    current_noise = (
        base_noise * scale
    )

    current_noise = min(
        current_noise,
        1.0
    )

    energy = noisy_energy_with_shots(
        best_theta,
        current_noise,
        shots
    )

    measured_energies.append(
        energy
    )

    actual_noise_levels.append(
        current_noise
    )


measured_energies = np.array(
    measured_energies
)

actual_noise_levels = np.array(
    actual_noise_levels
)


# ------------------------------------------------------------
# 20. Linear extrapolation
# ------------------------------------------------------------

# Fit:
#
# E(lambda) = a*lambda + b
#
# At lambda = 0:
#
# E(0) = b
#
# Therefore the intercept b is our
# zero-noise estimate.

coefficients = np.polyfit(
    noise_scales,
    measured_energies,
    1
)

slope = coefficients[0]

intercept = coefficients[1]

zne_energy = intercept


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(
        "DAY 47 — ZNE WITH SHOT NOISE"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Exact result
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("EXACT RESULT")
    print("-" * 70)

    print(
        "\nOne-particle Hamiltonian:"
    )

    print(H_one)

    print(
        "\nExact eigenvalues:"
    )

    print(exact_eigenvalues)

    print(
        "\nExact ground-state energy:"
    )

    print(
        f"{exact_energy:.10f}"
    )


    # --------------------------------------------------------
    # Ideal VQE
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("IDEAL VQE")
    print("-" * 70)

    print(
        f"\nOptimal theta:       "
        f"{best_theta:.10f}"
    )

    print(
        f"Ideal VQE energy:    "
        f"{ideal_vqe_energy:.10f}"
    )


    # --------------------------------------------------------
    # Experiment settings
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("EXPERIMENT SETTINGS")
    print("-" * 70)

    print(
        f"\nShots per observable: "
        f"{shots}"
    )

    print(
        f"Base noise:            "
        f"{base_noise:.4f}"
    )

    print(
        "\nNoise scaling factors:"
    )

    print("1x")
    print("2x")
    print("3x")


    # --------------------------------------------------------
    # Noisy measurements
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("SHOT-BASED NOISY MEASUREMENTS")
    print("-" * 70)

    for scale, noise, energy in zip(
        noise_scales,
        actual_noise_levels,
        measured_energies
    ):

        print(
            f"\nNoise scale:     {scale:.1f}x"
        )

        print(
            f"Noise strength:  {noise:.4f}"
        )

        print(
            f"Measured energy: {energy:.10f}"
        )


    # --------------------------------------------------------
    # Extrapolation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ZERO-NOISE EXTRAPOLATION")
    print("-" * 70)

    print(
        f"\nSlope:     {slope:.10f}"
    )

    print(
        f"Intercept: {intercept:.10f}"
    )

    print(
        "\nEstimated zero-noise energy:"
    )

    print(
        f"{zne_energy:.10f}"
    )


    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL ENERGY COMPARISON")
    print("=" * 70)

    print(
        f"\nExact energy:       "
        f"{exact_energy:.10f}"
    )

    print(
        f"Ideal VQE energy:   "
        f"{ideal_vqe_energy:.10f}"
    )

    print(
        f"1x noisy energy:    "
        f"{measured_energies[0]:.10f}"
    )

    print(
        f"2x noisy energy:    "
        f"{measured_energies[1]:.10f}"
    )

    print(
        f"3x noisy energy:    "
        f"{measured_energies[2]:.10f}"
    )

    print(
        f"ZNE energy:         "
        f"{zne_energy:.10f}"
    )


    # --------------------------------------------------------
    # Error comparison
    # --------------------------------------------------------

    ideal_error = abs(
        ideal_vqe_energy
        - exact_energy
    )

    noisy_error = abs(
        measured_energies[0]
        - exact_energy
    )

    zne_error = abs(
        zne_energy
        - exact_energy
    )


    print("\n" + "-" * 70)
    print("ERROR COMPARISON")
    print("-" * 70)

    print(
        f"\nIdeal VQE error: "
        f"{ideal_error:.10f}"
    )

    print(
        f"1x noisy error:  "
        f"{noisy_error:.10f}"
    )

    print(
        f"ZNE error:       "
        f"{zne_error:.10f}"
    )


    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    if noisy_error > 0:

        improvement = (
            (noisy_error - zne_error)
            / noisy_error
        ) * 100

        print(
            f"\nZNE error improvement: "
            f"{improvement:.2f}%"
        )


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    print(
        "\nThe energy was measured at multiple"
    )

    print(
        "noise levels using finite measurement shots."
    )

    print(
        "\nBecause the measurements are sampled,"
    )

    print(
        "the energy values fluctuate statistically."
    )

    print(
        "\nZNE fits the noisy measurements and"
    )

    print(
        "extrapolates the fitted curve toward"
    )

    print(
        "zero noise."
    )


    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "DAY 47 ZNE + SHOT NOISE EXPERIMENT COMPLETE"
    )

    print("=" * 70)