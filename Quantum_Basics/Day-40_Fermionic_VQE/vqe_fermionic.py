import numpy as np
from scipy.optimize import minimize


# ============================================================
# DAY 45
# FERMIONIC VQE — READOUT ERROR MITIGATION
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
# 5. Two-qubit Pauli operators
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
# 6. Fermionic Hamiltonian
#    after Jordan-Wigner transformation
# ------------------------------------------------------------

# Hamiltonian:
#
# H =
#
# eps0 * n0
# + eps1 * n1
# + t * (X0X1 + Y0Y1)/2
# + U * n0*n1
#
# where
#
# n0 = (I2 - Z0)/2
# n1 = (I2 - Z1)/2

n0_operator = (
    I2 - Z0
) / 2

n1_operator = (
    I2 - Z1
) / 2


H = (
    eps0 * n0_operator
    + eps1 * n1_operator
    + t * (X0X1 + Y0Y1) / 2
    + U * (n0_operator @ n1_operator)
)


# ------------------------------------------------------------
# 7. One-particle sector
# ------------------------------------------------------------

# Computational basis:
#
# |00> -> index 0 -> 0 particles
# |01> -> index 1 -> 1 particle
# |10> -> index 2 -> 1 particle
# |11> -> index 3 -> 2 particles
#
# Therefore we only use indices 1 and 2.

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


# ------------------------------------------------------------
# 8. VQE ansatz
# ------------------------------------------------------------

def state(theta):
    """
    Prepare the one-particle VQE state:

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
# 9. Expectation value
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
# 10. Ideal VQE energy
# ------------------------------------------------------------

def ideal_energy(theta):

    psi = state(theta)

    return expectation(
        psi,
        H
    )


# ------------------------------------------------------------
# 11. Find ideal VQE optimum
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


# ------------------------------------------------------------
# 12. Convert statevector to probabilities
# ------------------------------------------------------------

def probabilities_from_state(psi):

    probabilities = np.abs(psi) ** 2

    return probabilities


# ------------------------------------------------------------
# 13. Sample measurement counts
# ------------------------------------------------------------

def sample_counts(
    probabilities,
    shots
):

    outcomes = np.random.choice(
        4,
        size=shots,
        p=probabilities
    )

    counts = np.bincount(
        outcomes,
        minlength=4
    )

    return counts


# ------------------------------------------------------------
# 14. Apply classical readout noise
# ------------------------------------------------------------

def apply_readout_noise(
    counts,
    error_rate
):
    """
    Each measured qubit has probability p
    of being reported incorrectly.
    """

    noisy_counts = np.zeros(
        4,
        dtype=int
    )

    for state_index, count in enumerate(counts):

        for _ in range(count):

            # Convert index to two bits.
            #
            # 00 -> 0
            # 01 -> 1
            # 10 -> 2
            # 11 -> 3

            b0 = (
                state_index >> 1
            ) & 1

            b1 = (
                state_index
                & 1
            )

            # Flip first bit.

            if np.random.random() < error_rate:
                b0 = 1 - b0

            # Flip second bit.

            if np.random.random() < error_rate:
                b1 = 1 - b1

            # Convert bits back to index.

            noisy_index = (
                2 * b0 + b1
            )

            noisy_counts[
                noisy_index
            ] += 1

    return noisy_counts


# ------------------------------------------------------------
# 15. Create readout confusion matrix
# ------------------------------------------------------------

def create_confusion_matrix(
    error_rate
):
    """
    Single-qubit confusion matrix:

                    Reported
                    0       1

        Actual 0   1-p      p
        Actual 1    p      1-p
    """

    p = error_rate

    single_qubit_matrix = np.array([
        [1 - p, p],
        [p, 1 - p]
    ])

    # Independent errors on both qubits.

    two_qubit_matrix = np.kron(
        single_qubit_matrix,
        single_qubit_matrix
    )

    return two_qubit_matrix


# ------------------------------------------------------------
# 16. Readout-error mitigation
# ------------------------------------------------------------

def mitigate_counts(
    noisy_counts,
    error_rate
):
    """
    Correct noisy probabilities using
    the inverse of the known confusion matrix.
    """

    shots = np.sum(
        noisy_counts
    )

    measured_probabilities = (
        noisy_counts / shots
    )

    confusion_matrix = (
        create_confusion_matrix(
            error_rate
        )
    )

    # Measurement model:
    #
    # p_measured = C @ p_true
    #
    # Therefore:
    #
    # p_true = C^(-1) @ p_measured
    #
    # solve() is numerically preferable to
    # explicitly calculating the inverse.

    corrected_probabilities = np.linalg.solve(
        confusion_matrix,
        measured_probabilities
    )

    # Matrix inversion combined with finite-shot
    # statistics can produce small negative values.

    corrected_probabilities = np.clip(
        corrected_probabilities,
        0,
        None
    )

    # Renormalize.

    total = np.sum(
        corrected_probabilities
    )

    if total > 0:

        corrected_probabilities /= total

    return corrected_probabilities


# ------------------------------------------------------------
# 17. Expectation from computational-basis probabilities
# ------------------------------------------------------------

def expectation_from_probabilities(
    probabilities,
    operator
):

    eigenvalues = np.real(
        np.diag(operator)
    )

    return np.sum(
        probabilities * eigenvalues
    )


# ------------------------------------------------------------
# 18. Z-basis measurements
# ------------------------------------------------------------

def measure_z_expectations(
    psi,
    shots,
    readout_error
):

    # True probabilities.

    ideal_probabilities = (
        probabilities_from_state(
            psi
        )
    )

    # True measurement counts.

    true_counts = sample_counts(
        ideal_probabilities,
        shots
    )

    # Add readout noise.

    noisy_counts = apply_readout_noise(
        true_counts,
        readout_error
    )

    # Convert noisy counts to probabilities.

    noisy_probabilities = (
        noisy_counts
        / np.sum(noisy_counts)
    )

    # Correct noisy probabilities.

    corrected_probabilities = (
        mitigate_counts(
            noisy_counts,
            readout_error
        )
    )

    # --------------------------------------------------------
    # Raw expectation values
    # --------------------------------------------------------

    raw_z0 = expectation_from_probabilities(
        noisy_probabilities,
        Z0
    )

    raw_z1 = expectation_from_probabilities(
        noisy_probabilities,
        Z1
    )

    raw_z0z1 = expectation_from_probabilities(
        noisy_probabilities,
        Z0Z1
    )

    # --------------------------------------------------------
    # Mitigated expectation values
    # --------------------------------------------------------

    corrected_z0 = expectation_from_probabilities(
        corrected_probabilities,
        Z0
    )

    corrected_z1 = expectation_from_probabilities(
        corrected_probabilities,
        Z1
    )

    corrected_z0z1 = expectation_from_probabilities(
        corrected_probabilities,
        Z0Z1
    )

    return (
        true_counts,
        noisy_counts,
        noisy_probabilities,
        corrected_probabilities,

        raw_z0,
        raw_z1,
        raw_z0z1,

        corrected_z0,
        corrected_z1,
        corrected_z0z1
    )


# ------------------------------------------------------------
# 19. Rotation for X-basis measurement
# ------------------------------------------------------------

def rotate_for_x(psi):

    H_gate = (
        1 / np.sqrt(2)
    ) * np.array([
        [1, 1],
        [1, -1]
    ], dtype=complex)

    rotation = kron(
        H_gate,
        H_gate
    )

    return rotation @ psi


# ------------------------------------------------------------
# 20. Rotation for Y-basis measurement
# ------------------------------------------------------------

def rotate_for_y(psi):

    S_dagger = np.array([
        [1, 0],
        [0, -1j]
    ], dtype=complex)

    H_gate = (
        1 / np.sqrt(2)
    ) * np.array([
        [1, 1],
        [1, -1]
    ], dtype=complex)

    single_rotation = (
        H_gate @ S_dagger
    )

    rotation = kron(
        single_rotation,
        single_rotation
    )

    return rotation @ psi


# ------------------------------------------------------------
# 21. Measure XX or YY
# ------------------------------------------------------------

def measure_two_qubit_pauli(
    rotated_state,
    shots,
    readout_error
):

    probabilities = (
        probabilities_from_state(
            rotated_state
        )
    )

    # True counts.

    true_counts = sample_counts(
        probabilities,
        shots
    )

    # Apply readout noise.

    noisy_counts = apply_readout_noise(
        true_counts,
        readout_error
    )

    # Noisy probabilities.

    noisy_probabilities = (
        noisy_counts
        / np.sum(noisy_counts)
    )

    # Correct probabilities.

    corrected_probabilities = (
        mitigate_counts(
            noisy_counts,
            readout_error
        )
    )

    # For XX and YY:
    #
    # 00 -> +1
    # 01 -> -1
    # 10 -> -1
    # 11 -> +1

    eigenvalues = np.array([
        1,
        -1,
        -1,
        1
    ])

    raw_expectation = np.sum(
        noisy_probabilities
        * eigenvalues
    )

    corrected_expectation = np.sum(
        corrected_probabilities
        * eigenvalues
    )

    return (
        true_counts,
        noisy_counts,
        raw_expectation,
        corrected_expectation
    )


# ------------------------------------------------------------
# 22. Complete energy measurement
# ------------------------------------------------------------

def measure_energy(
    psi,
    shots,
    readout_error
):

    # ========================================================
    # Z measurements
    # ========================================================

    (
        true_z_counts,
        noisy_z_counts,
        noisy_z_probabilities,
        corrected_z_probabilities,

        raw_z0,
        raw_z1,
        raw_z0z1,

        corrected_z0,
        corrected_z1,
        corrected_z0z1

    ) = measure_z_expectations(
        psi,
        shots,
        readout_error
    )


    # ========================================================
    # X measurement
    # ========================================================

    psi_x = rotate_for_x(
        psi
    )

    (
        true_x_counts,
        noisy_x_counts,
        raw_xx,
        corrected_xx

    ) = measure_two_qubit_pauli(
        psi_x,
        shots,
        readout_error
    )


    # ========================================================
    # Y measurement
    # ========================================================

    psi_y = rotate_for_y(
        psi
    )

    (
        true_y_counts,
        noisy_y_counts,
        raw_yy,
        corrected_yy

    ) = measure_two_qubit_pauli(
        psi_y,
        shots,
        readout_error
    )


    # ========================================================
    # Reconstruct energy
    # ========================================================

    def energy_from_expectations(
        z0,
        z1,
        z0z1,
        xx,
        yy
    ):

        # Number operators.

        n0 = (
            1 - z0
        ) / 2

        n1 = (
            1 - z1
        ) / 2

        # Interaction term:
        #
        # n0*n1
        #
        # = (1 - Z0 - Z1 + Z0Z1)/4

        interaction = (
            1
            - z0
            - z1
            + z0z1
        ) / 4

        energy = (
            eps0 * n0
            + eps1 * n1
            + t * (xx + yy) / 2
            + U * interaction
        )

        return energy


    # Raw noisy energy.

    raw_energy = energy_from_expectations(
        raw_z0,
        raw_z1,
        raw_z0z1,
        raw_xx,
        raw_yy
    )


    # Mitigated energy.

    mitigated_energy = energy_from_expectations(
        corrected_z0,
        corrected_z1,
        corrected_z0z1,
        corrected_xx,
        corrected_yy
    )


    return (
        raw_energy,
        mitigated_energy,

        {
            "z0": raw_z0,
            "z1": raw_z1,
            "z0z1": raw_z0z1,
            "xx": raw_xx,
            "yy": raw_yy
        },

        {
            "z0": corrected_z0,
            "z1": corrected_z1,
            "z0z1": corrected_z0z1,
            "xx": corrected_xx,
            "yy": corrected_yy
        },

        {
            "true_z": true_z_counts,
            "noisy_z": noisy_z_counts,

            "true_x": true_x_counts,
            "noisy_x": noisy_x_counts,

            "true_y": true_y_counts,
            "noisy_y": noisy_y_counts
        }
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Experiment settings
    # --------------------------------------------------------

    shots = 2048

    # Day 44 used 5% readout noise.

    readout_error = 0.05


    # ========================================================
    # EXACT RESULTS
    # ========================================================

    print("\n" + "=" * 65)
    print("DAY 45 — READOUT ERROR MITIGATION")
    print("=" * 65)

    print("\nOne-particle Hamiltonian:")

    print(H_one)

    print("\nExact one-particle eigenvalues:")

    print(exact_eigenvalues)

    print("\nExact ground-state energy:")

    print(
        f"{exact_energy:.10f}"
    )


    # ========================================================
    # IDEAL VQE
    # ========================================================

    print("\n" + "-" * 65)
    print("IDEAL VQE")
    print("-" * 65)

    print(
        f"Optimal theta:       "
        f"{best_theta:.10f}"
    )

    print(
        f"Ideal VQE energy:    "
        f"{ideal_vqe_energy:.10f}"
    )

    print("\nOptimized state:")

    print(psi_opt)


    # ========================================================
    # IDEAL EXPECTATION VALUES
    # ========================================================

    ideal_z0 = expectation(
        psi_opt,
        Z0
    )

    ideal_z1 = expectation(
        psi_opt,
        Z1
    )

    ideal_z0z1 = expectation(
        psi_opt,
        Z0Z1
    )

    ideal_xx = expectation(
        psi_opt,
        X0X1
    )

    ideal_yy = expectation(
        psi_opt,
        Y0Y1
    )

    print("\nIdeal expectation values:")

    print(
        f"<Z0>   = {ideal_z0:.10f}"
    )

    print(
        f"<Z1>   = {ideal_z1:.10f}"
    )

    print(
        f"<Z0Z1> = {ideal_z0z1:.10f}"
    )

    print(
        f"<XX>   = {ideal_xx:.10f}"
    )

    print(
        f"<YY>   = {ideal_yy:.10f}"
    )


    # ========================================================
    # READOUT ERROR MODEL
    # ========================================================

    confusion_matrix = (
        create_confusion_matrix(
            readout_error
        )
    )

    print("\n" + "-" * 65)
    print("READOUT ERROR MODEL")
    print("-" * 65)

    print(
        f"\nReadout error per qubit: "
        f"{readout_error * 100:.1f}%"
    )

    print("\nTwo-qubit confusion matrix:")

    print(confusion_matrix)


    # ========================================================
    # NOISY MEASUREMENT + MITIGATION
    # ========================================================

    (
        raw_energy,
        mitigated_energy,

        raw_expectations,
        corrected_expectations,

        counts

    ) = measure_energy(
        psi_opt,
        shots,
        readout_error
    )


    # ========================================================
    # MEASUREMENT RESULTS
    # ========================================================

    print("\n" + "-" * 65)
    print("MEASUREMENT RESULTS")
    print("-" * 65)

    print(
        f"\nShots: {shots}"
    )


    print("\nRaw noisy expectation values:")

    for name, value in raw_expectations.items():

        print(
            f"<{name}> = {value:.10f}"
        )


    print("\nMitigated expectation values:")

    for name, value in corrected_expectations.items():

        print(
            f"<{name}> = {value:.10f}"
        )


    # ========================================================
    # ENERGY COMPARISON
    # ========================================================

    print("\n" + "=" * 65)
    print("ENERGY COMPARISON")
    print("=" * 65)


    raw_error = abs(
        raw_energy
        - exact_energy
    )


    mitigated_error = abs(
        mitigated_energy
        - exact_energy
    )


    ideal_error = abs(
        ideal_vqe_energy
        - exact_energy
    )


    print(
        f"\nExact energy:       "
        f"{exact_energy:.10f}"
    )

    print(
        f"Ideal VQE energy:   "
        f"{ideal_vqe_energy:.10f}"
    )

    print(
        f"Raw noisy energy:   "
        f"{raw_energy:.10f}"
    )

    print(
        f"Mitigated energy:   "
        f"{mitigated_energy:.10f}"
    )


    print("\nErrors:")

    print(
        f"Ideal VQE error:    "
        f"{ideal_error:.10f}"
    )

    print(
        f"Raw noisy error:    "
        f"{raw_error:.10f}"
    )

    print(
        f"Mitigated error:    "
        f"{mitigated_error:.10f}"
    )


    # ========================================================
    # MITIGATION IMPROVEMENT
    # ========================================================

    if raw_error > 0:

        improvement = (
            (raw_error - mitigated_error)
            / raw_error
        ) * 100

        print(
            f"\nError improvement from mitigation: "
            f"{improvement:.2f}%"
        )


    # ========================================================
    # Z-BASIS COUNTS
    # ========================================================

    print("\n" + "-" * 65)
    print("Z-BASIS COUNTS")
    print("-" * 65)


    print("\nTrue counts:")

    print(
        f"00 = {counts['true_z'][0]}, "
        f"01 = {counts['true_z'][1]}, "
        f"10 = {counts['true_z'][2]}, "
        f"11 = {counts['true_z'][3]}"
    )


    print("\nNoisy reported counts:")

    print(
        f"00 = {counts['noisy_z'][0]}, "
        f"01 = {counts['noisy_z'][1]}, "
        f"10 = {counts['noisy_z'][2]}, "
        f"11 = {counts['noisy_z'][3]}"
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 65)
    print("DAY 45 READOUT-MITIGATION EXPERIMENT COMPLETE")
    print("=" * 65)