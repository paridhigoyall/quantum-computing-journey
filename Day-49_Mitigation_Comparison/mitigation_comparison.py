import numpy as np


# ============================================================
# DAY 49
# ERROR MITIGATION COMPARISON
#
# Compare:
#   1. Raw noisy VQE
#   2. Readout Error Mitigation (REM)
#   3. Zero-Noise Extrapolation (ZNE)
#   4. REM + ZNE
# ============================================================


# ------------------------------------------------------------
# 1. Basic matrices
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

I2 = np.kron(I1, I1)

Z0 = np.kron(Z, I1)
Z1 = np.kron(I1, Z)

Z0Z1 = np.kron(Z, Z)

XX = np.kron(X, X)
YY = np.kron(Y, Y)


# ------------------------------------------------------------
# 2. Physical parameters
# ------------------------------------------------------------

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8


# ------------------------------------------------------------
# 3. Full Hamiltonian
# ------------------------------------------------------------

H = (
    eps0 * (I2 - Z0) / 2
    + eps1 * (I2 - Z1) / 2
    + t * (XX + YY) / 2
    + U * ((I2 - Z0) / 2) @ ((I2 - Z1) / 2)
)


# ------------------------------------------------------------
# 4. Exact energy
#
# IMPORTANT:
# The VQE ansatz is restricted to the one-particle sector.
# Therefore we must NOT use the global minimum of the
# complete 4x4 Hamiltonian.
# ------------------------------------------------------------

one_particle_hamiltonian = np.array([
    [eps0, t],
    [t, eps1]
])

one_particle_eigenvalues = np.linalg.eigvalsh(
    one_particle_hamiltonian
)

exact_energy = one_particle_eigenvalues[0]


# ------------------------------------------------------------
# 5. Optimized VQE state
# ------------------------------------------------------------

theta = 2.03444393

state = np.array([
    0,
    np.cos(theta / 2),
    np.sin(theta / 2),
    0
], dtype=complex)

state = state / np.linalg.norm(state)

rho_ideal = np.outer(
    state,
    state.conj()
)


# ------------------------------------------------------------
# 6. Expectation value
# ------------------------------------------------------------

def expectation(rho, operator):
    return np.real(
        np.trace(rho @ operator)
    )


# ------------------------------------------------------------
# 7. Quantum noise
#
# Educational depolarizing model:
#
# rho_noisy =
#       (1-p)rho + p(I/d)
# ------------------------------------------------------------

def apply_quantum_noise(
    rho,
    noise_strength
):

    dimension = rho.shape[0]

    mixed_state = (
        np.eye(dimension)
        / dimension
    )

    rho_noisy = (
        (1 - noise_strength) * rho
        + noise_strength * mixed_state
    )

    return rho_noisy


# ------------------------------------------------------------
# 8. Readout confusion matrix
# ------------------------------------------------------------

def create_confusion_matrix(
    readout_error
):

    single_qubit = np.array([
        [1 - readout_error, readout_error],
        [readout_error, 1 - readout_error]
    ])

    two_qubit = np.kron(
        single_qubit,
        single_qubit
    )

    return two_qubit


# ------------------------------------------------------------
# 9. Computational-basis measurement
# ------------------------------------------------------------

def measure_z_basis(
    rho,
    shots,
    readout_error
):

    probabilities = np.real(
        np.diag(rho)
    )

    probabilities = np.clip(
        probabilities,
        0,
        None
    )

    probabilities /= np.sum(
        probabilities
    )

    # Ideal quantum measurement
    true_counts = np.random.multinomial(
        shots,
        probabilities
    )

    # Apply readout noise
    noisy_counts = np.zeros(
        4,
        dtype=int
    )

    for state_index, count in enumerate(
        true_counts
    ):

        if count == 0:
            continue

        q0 = (state_index >> 1) & 1
        q1 = state_index & 1

        for _ in range(count):

            measured_q0 = q0
            measured_q1 = q1

            # Readout error on qubit 0
            if np.random.random() < readout_error:
                measured_q0 ^= 1

            # Readout error on qubit 1
            if np.random.random() < readout_error:
                measured_q1 ^= 1

            measured_index = (
                measured_q0 * 2
                + measured_q1
            )

            noisy_counts[
                measured_index
            ] += 1

    return noisy_counts


# ------------------------------------------------------------
# 10. Convert probabilities into Z expectations
# ------------------------------------------------------------

def calculate_z_expectations(
    probabilities
):

    # Basis:
    #
    # 00
    # 01
    # 10
    # 11

    z0_values = np.array([
        1,
        1,
        -1,
        -1
    ])

    z1_values = np.array([
        1,
        -1,
        1,
        -1
    ])

    z0z1_values = np.array([
        1,
        -1,
        -1,
        1
    ])

    exp_z0 = np.dot(
        probabilities,
        z0_values
    )

    exp_z1 = np.dot(
        probabilities,
        z1_values
    )

    exp_z0z1 = np.dot(
        probabilities,
        z0z1_values
    )

    return (
        exp_z0,
        exp_z1,
        exp_z0z1
    )


# ------------------------------------------------------------
# 11. Readout Error Mitigation
# ------------------------------------------------------------

def mitigate_readout(
    counts,
    shots,
    readout_error
):

    measured_probabilities = (
        counts / shots
    )

    confusion_matrix = (
        create_confusion_matrix(
            readout_error
        )
    )

    # Solve:
    #
    # p_measured = C p_true
    #
    # for p_true

    corrected_probabilities = np.linalg.solve(
        confusion_matrix,
        measured_probabilities
    )

    # Remove tiny negative values caused
    # by finite-shot statistics/numerical effects
    corrected_probabilities = np.clip(
        corrected_probabilities,
        0,
        None
    )

    # Normalize
    total = np.sum(
        corrected_probabilities
    )

    if total > 0:
        corrected_probabilities /= total

    return corrected_probabilities


# ------------------------------------------------------------
# 12. Measure a general Pauli expectation
#
# Used for XX and YY.
# ------------------------------------------------------------

def measure_pauli(
    true_expectation,
    shots,
    readout_error
):

    # Probability of +1 outcome
    p_plus = (
        1 + true_expectation
    ) / 2

    # Quantum shot noise
    plus_count = np.random.binomial(
        shots,
        p_plus
    )

    minus_count = shots - plus_count

    # For two qubits, parity changes if
    # exactly one bit is flipped.
    parity_flip_probability = (
        2
        * readout_error
        * (1 - readout_error)
    )

    observed_plus_from_plus = (
        np.random.binomial(
            plus_count,
            1 - parity_flip_probability
        )
    )

    observed_plus_from_minus = (
        np.random.binomial(
            minus_count,
            parity_flip_probability
        )
    )

    observed_plus = (
        observed_plus_from_plus
        + observed_plus_from_minus
    )

    measured_expectation = (
        2 * observed_plus / shots
        - 1
    )

    return measured_expectation


# ------------------------------------------------------------
# 13. Mitigate XX/YY readout error
# ------------------------------------------------------------

def mitigate_pauli(
    measured_expectation,
    readout_error
):

    # Readout error scales a two-qubit
    # parity expectation by:
    #
    # (1 - 2p)^2

    scale_factor = (
        1 - 2 * readout_error
    ) ** 2

    if abs(scale_factor) < 1e-12:
        return measured_expectation

    corrected = (
        measured_expectation
        / scale_factor
    )

    corrected = np.clip(
        corrected,
        -1,
        1
    )

    return corrected


# ------------------------------------------------------------
# 14. Calculate VQE energy
# ------------------------------------------------------------

def calculate_energy(
    exp_z0,
    exp_z1,
    exp_z0z1,
    exp_xx,
    exp_yy
):

    n0 = (
        1 - exp_z0
    ) / 2

    n1 = (
        1 - exp_z1
    ) / 2

    interaction = (
        1
        - exp_z0
        - exp_z1
        + exp_z0z1
    ) / 4

    hopping = (
        t
        * (exp_xx + exp_yy)
        / 2
    )

    energy = (
        eps0 * n0
        + eps1 * n1
        + hopping
        + U * interaction
    )

    return energy


# ------------------------------------------------------------
# 15. Run one noisy experiment
# ------------------------------------------------------------

def run_experiment(
    noise_strength,
    shots,
    readout_error
):

    # ----------------------------------------
    # Apply quantum noise
    # ----------------------------------------

    rho_noisy = apply_quantum_noise(
        rho_ideal,
        noise_strength
    )

    # ----------------------------------------
    # Z-basis measurement
    # ----------------------------------------

    counts = measure_z_basis(
        rho_noisy,
        shots,
        readout_error
    )

    raw_probabilities = (
        counts / shots
    )

    # Raw Z expectations
    raw_z0, raw_z1, raw_z0z1 = (
        calculate_z_expectations(
            raw_probabilities
        )
    )

    # ----------------------------------------
    # Readout mitigation
    # ----------------------------------------

    corrected_probabilities = (
        mitigate_readout(
            counts,
            shots,
            readout_error
        )
    )

    corrected_z0, corrected_z1, corrected_z0z1 = (
        calculate_z_expectations(
            corrected_probabilities
        )
    )

    # ----------------------------------------
    # XX measurement
    # ----------------------------------------

    true_xx = expectation(
        rho_noisy,
        XX
    )

    raw_xx = measure_pauli(
        true_xx,
        shots,
        readout_error
    )

    corrected_xx = mitigate_pauli(
        raw_xx,
        readout_error
    )

    # ----------------------------------------
    # YY measurement
    # ----------------------------------------

    true_yy = expectation(
        rho_noisy,
        YY
    )

    raw_yy = measure_pauli(
        true_yy,
        shots,
        readout_error
    )

    corrected_yy = mitigate_pauli(
        raw_yy,
        readout_error
    )

    # ----------------------------------------
    # Raw noisy energy
    # ----------------------------------------

    raw_energy = calculate_energy(
        raw_z0,
        raw_z1,
        raw_z0z1,
        raw_xx,
        raw_yy
    )

    # ----------------------------------------
    # REM energy
    # ----------------------------------------

    rem_energy = calculate_energy(
        corrected_z0,
        corrected_z1,
        corrected_z0z1,
        corrected_xx,
        corrected_yy
    )

    return (
        raw_energy,
        rem_energy
    )


# ============================================================
# MAIN
# ============================================================

shots = 2048

base_noise = 0.05

readout_error = 0.05

noise_scales = np.array([
    1,
    2,
    3
], dtype=float)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

print("=" * 70)
print("DAY 49 - ERROR MITIGATION COMPARISON")
print("=" * 70)

print()

print(
    f"Exact one-particle energy : "
    f"{exact_energy:.10f}"
)

print(
    f"Ideal VQE energy          : "
    f"{expectation(rho_ideal, H):.10f}"
)

print(
    f"Shots                     : "
    f"{shots}"
)

print(
    f"Base quantum noise        : "
    f"{base_noise}"
)

print(
    f"Readout error             : "
    f"{readout_error}"
)

print()


# ------------------------------------------------------------
# Ideal VQE error
# ------------------------------------------------------------

ideal_vqe_energy = expectation(
    rho_ideal,
    H
)

ideal_error = abs(
    ideal_vqe_energy
    - exact_energy
)


# ------------------------------------------------------------
# Run experiments
# ------------------------------------------------------------

raw_energies = []

rem_energies = []

for scale in noise_scales:

    noise_strength = (
        base_noise * scale
    )

    raw_energy, rem_energy = (
        run_experiment(
            noise_strength,
            shots,
            readout_error
        )
    )

    raw_energies.append(
        raw_energy
    )

    rem_energies.append(
        rem_energy
    )

    print("-" * 70)

    print(
        f"Noise scale = {int(scale)}x"
    )

    print(
        f"Noise strength = "
        f"{noise_strength:.4f}"
    )

    print(
        f"Raw noisy energy : "
        f"{raw_energy:.10f}"
    )

    print(
        f"REM energy       : "
        f"{rem_energy:.10f}"
    )

    print(
        f"Raw error        : "
        f"{abs(raw_energy - exact_energy):.10f}"
    )

    print(
        f"REM error        : "
        f"{abs(rem_energy - exact_energy):.10f}"
    )

    print()


# ------------------------------------------------------------
# ZNE using RAW energies
# ------------------------------------------------------------

raw_energies = np.array(
    raw_energies
)

raw_slope, raw_intercept = np.polyfit(
    noise_scales,
    raw_energies,
    1
)

raw_zne_energy = raw_intercept


# ------------------------------------------------------------
# ZNE using REM energies
# ------------------------------------------------------------

rem_energies = np.array(
    rem_energies
)

rem_slope, rem_intercept = np.polyfit(
    noise_scales,
    rem_energies,
    1
)

rem_zne_energy = rem_intercept


# ------------------------------------------------------------
# Errors
# ------------------------------------------------------------

raw_1x_error = abs(
    raw_energies[0]
    - exact_energy
)

rem_1x_error = abs(
    rem_energies[0]
    - exact_energy
)

zne_error = abs(
    raw_zne_energy
    - exact_energy
)

combined_error = abs(
    rem_zne_energy
    - exact_energy
)


# ------------------------------------------------------------
# Improvement percentages
# ------------------------------------------------------------

def calculate_improvement(
    original_error,
    new_error
):

    if original_error == 0:
        return 0.0

    return (
        (original_error - new_error)
        / original_error
        * 100
    )


rem_improvement = calculate_improvement(
    raw_1x_error,
    rem_1x_error
)

zne_improvement = calculate_improvement(
    raw_1x_error,
    zne_error
)

combined_improvement = calculate_improvement(
    raw_1x_error,
    combined_error
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print()

print(
    f"Exact energy                : "
    f"{exact_energy:.10f}"
)

print(
    f"Ideal VQE energy            : "
    f"{ideal_vqe_energy:.10f}"
)

print()

print(
    f"Raw 1x energy               : "
    f"{raw_energies[0]:.10f}"
)

print(
    f"REM 1x energy               : "
    f"{rem_energies[0]:.10f}"
)

print(
    f"Raw ZNE energy              : "
    f"{raw_zne_energy:.10f}"
)

print(
    f"REM + ZNE energy            : "
    f"{rem_zne_energy:.10f}"
)

print()

print("-" * 70)
print("ERRORS")
print("-" * 70)

print()

print(
    f"Ideal VQE error             : "
    f"{ideal_error:.10f}"
)

print(
    f"Raw 1x error                : "
    f"{raw_1x_error:.10f}"
)

print(
    f"REM 1x error                : "
    f"{rem_1x_error:.10f}"
)

print(
    f"Raw ZNE error               : "
    f"{zne_error:.10f}"
)

print(
    f"REM + ZNE error             : "
    f"{combined_error:.10f}"
)

print()

print("-" * 70)
print("ERROR REDUCTION")
print("-" * 70)

print()

print(
    f"REM improvement             : "
    f"{rem_improvement:.2f}%"
)

print(
    f"ZNE improvement             : "
    f"{zne_improvement:.2f}%"
)

print(
    f"REM + ZNE improvement       : "
    f"{combined_improvement:.2f}%"
)

print()

print("-" * 70)
print("ZNE MODELS")
print("-" * 70)

print()

print(
    "Raw ZNE model:"
)

print(
    f"E(lambda) = "
    f"{raw_slope:.10f} * lambda "
    f"+ {raw_intercept:.10f}"
)

print()

print(
    "REM + ZNE model:"
)

print(
    f"E(lambda) = "
    f"{rem_slope:.10f} * lambda "
    f"+ {rem_intercept:.10f}"
)

print()

print("=" * 70)
print("MITIGATION PIPELINE")
print("=" * 70)

print()

print(
    "Ideal VQE"
    " -> Quantum noise"
    " -> Readout noise"
    " -> REM"
    " -> ZNE"
    " -> Final estimate"
)

print()

print("=" * 70)
print("DAY 49 COMPLETE")
print("=" * 70)