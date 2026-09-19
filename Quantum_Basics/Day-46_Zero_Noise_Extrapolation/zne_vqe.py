import numpy as np
from scipy.optimize import minimize


# ============================================================
# DAY 46
# ZERO-NOISE EXTRAPOLATION (ZNE)
# ============================================================


# ------------------------------------------------------------
# 1. Physical Hamiltonian parameters
# ------------------------------------------------------------

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8


# ------------------------------------------------------------
# 2. One-qubit Pauli matrices
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

I2 = np.kron(I1, I1)


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
# 8. One-particle Hamiltonian
# ------------------------------------------------------------

one_particle_indices = [1, 2]

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
    One-particle ansatz:

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
# 11. Ideal energy
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
    Simulate simple quantum noise.

    The noisy density matrix is:

        rho_noisy
        =
        (1-p) rho
        +
        p I/d

    where:

        p = noise_strength
        d = dimension of the Hilbert space

    This represents the state becoming partially mixed.
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
# 14. Expectation from density matrix
# ------------------------------------------------------------

def density_matrix_expectation(
    rho,
    operator
):

    value = np.trace(
        rho @ operator
    )

    return np.real(value)


# ------------------------------------------------------------
# 15. Noisy energy
# ------------------------------------------------------------

def noisy_energy(
    theta,
    noise_strength
):
    """
    Calculate energy after applying
    the noise model.
    """

    psi = state(theta)

    noisy_rho = apply_depolarizing_noise(
        psi,
        noise_strength
    )

    energy = density_matrix_expectation(
        noisy_rho,
        H
    )

    return energy


# ============================================================
# ZERO-NOISE EXTRAPOLATION
# ============================================================


# ------------------------------------------------------------
# 16. Noise scaling factors
# ------------------------------------------------------------

noise_scales = np.array([
    1.0,
    2.0,
    3.0
])


# ------------------------------------------------------------
# 17. Base noise level
# ------------------------------------------------------------

base_noise = 0.05


# ------------------------------------------------------------
# 18. Measure energy at different noise levels
# ------------------------------------------------------------

measured_energies = []

for scale in noise_scales:

    current_noise = (
        base_noise * scale
    )

    # Prevent probability-like noise parameter
    # from exceeding 1.

    current_noise = min(
        current_noise,
        1.0
    )

    energy = noisy_energy(
        best_theta,
        current_noise
    )

    measured_energies.append(
        energy
    )


measured_energies = np.array(
    measured_energies
)


# ------------------------------------------------------------
# 19. Linear extrapolation
# ------------------------------------------------------------

# We assume approximately:
#
# E(lambda) = a * lambda + b
#
# where:
#
# lambda = noise scaling factor
#
# b = estimated zero-noise energy
#
# At lambda = 0:
#
# E(0) = b


coefficients = np.polyfit(
    noise_scales,
    measured_energies,
    1
)

slope = coefficients[0]

intercept = coefficients[1]


# Zero-noise estimate.

zne_energy = intercept


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("DAY 46 — ZERO-NOISE EXTRAPOLATION")
    print("=" * 65)


    # --------------------------------------------------------
    # Exact result
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("EXACT RESULT")
    print("-" * 65)

    print("\nOne-particle Hamiltonian:")

    print(H_one)

    print("\nExact eigenvalues:")

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

    print("\n" + "-" * 65)
    print("IDEAL VQE")
    print("-" * 65)

    print(
        f"\nOptimal theta:       "
        f"{best_theta:.10f}"
    )

    print(
        f"Ideal VQE energy:    "
        f"{ideal_vqe_energy:.10f}"
    )


    # --------------------------------------------------------
    # Noise model
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("NOISE MODEL")
    print("-" * 65)

    print(
        f"\nBase noise strength: "
        f"{base_noise:.4f}"
    )

    print(
        "\nWe evaluate the same VQE state at:"
    )

    print("1× noise")
    print("2× noise")
    print("3× noise")


    # --------------------------------------------------------
    # Noise measurements
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("NOISY ENERGY MEASUREMENTS")
    print("-" * 65)

    for scale, energy in zip(
        noise_scales,
        measured_energies
    ):

        current_noise = (
            base_noise * scale
        )

        print(
            f"\nNoise scale: {scale:.1f}x"
        )

        print(
            f"Noise strength: {current_noise:.4f}"
        )

        print(
            f"Energy: {energy:.10f}"
        )


    # --------------------------------------------------------
    # Linear model
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("EXTRAPOLATION MODEL")
    print("-" * 65)

    print(
        f"\nSlope:     {slope:.10f}"
    )

    print(
        f"Intercept: {intercept:.10f}"
    )

    print(
        "\nThe intercept represents the estimated"
    )

    print(
        "energy at zero noise."
    )


    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ZERO-NOISE EXTRAPOLATION RESULT")
    print("=" * 65)

    print(
        f"\nExact energy:          "
        f"{exact_energy:.10f}"
    )

    print(
        f"Ideal VQE energy:      "
        f"{ideal_vqe_energy:.10f}"
    )

    print(
        f"1x noisy energy:       "
        f"{measured_energies[0]:.10f}"
    )

    print(
        f"2x noisy energy:       "
        f"{measured_energies[1]:.10f}"
    )

    print(
        f"3x noisy energy:       "
        f"{measured_energies[2]:.10f}"
    )

    print(
        f"ZNE energy:            "
        f"{zne_energy:.10f}"
    )


    # --------------------------------------------------------
    # Errors
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


    print("\n" + "-" * 65)
    print("ERROR COMPARISON")
    print("-" * 65)

    print(
        f"\nIdeal VQE error:       "
        f"{ideal_error:.10f}"
    )

    print(
        f"1x noisy error:        "
        f"{noisy_error:.10f}"
    )

    print(
        f"ZNE error:             "
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
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "DAY 46 ZERO-NOISE EXTRAPOLATION "
        "EXPERIMENT COMPLETE"
    )

    print("=" * 65)