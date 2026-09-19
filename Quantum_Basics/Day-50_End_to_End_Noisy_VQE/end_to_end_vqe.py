import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DAY 50
# END-TO-END NOISY VQE BENCHMARK
#
# Compare:
#   1. Exact energy
#   2. Ideal VQE
#   3. Raw noisy VQE
#   4. Readout Error Mitigation (REM)
#   5. Zero-Noise Extrapolation (ZNE)
#   6. REM + ZNE
#
# Noise model:
#   - Depolarizing quantum noise
#   - Readout bit-flip noise
#   - Finite-shot measurement
# ============================================================


# ============================================================
# 1. BASIC MATRICES
# ============================================================

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


# ============================================================
# 2. PHYSICAL PARAMETERS
# ============================================================

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8


# ============================================================
# 3. ONE-PARTICLE HAMILTONIAN
#
# Our VQE ansatz is restricted to:
#
# |01>
# |10>
#
# Therefore the relevant Hamiltonian is:
#
#       | eps0    t   |
# H =   |            |
#       |  t     eps1 |
# ============================================================

one_particle_hamiltonian = np.array([
    [eps0, t],
    [t, eps1]
])

one_particle_eigenvalues = np.linalg.eigvalsh(
    one_particle_hamiltonian
)

exact_energy = one_particle_eigenvalues[0]


# ============================================================
# 4. OPTIMIZED VQE STATE
# ============================================================

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


# ============================================================
# 5. EXPECTATION VALUE
# ============================================================

def expectation(rho, operator):

    return np.real(
        np.trace(rho @ operator)
    )


# ============================================================
# 6. HAMILTONIAN ENERGY
# ============================================================

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


# ============================================================
# 7. IDEAL VQE ENERGY
# ============================================================

ideal_z0 = expectation(
    rho_ideal,
    Z0
)

ideal_z1 = expectation(
    rho_ideal,
    Z1
)

ideal_z0z1 = expectation(
    rho_ideal,
    Z0Z1
)

ideal_xx = expectation(
    rho_ideal,
    XX
)

ideal_yy = expectation(
    rho_ideal,
    YY
)

ideal_energy = calculate_energy(
    ideal_z0,
    ideal_z1,
    ideal_z0z1,
    ideal_xx,
    ideal_yy
)


# ============================================================
# 8. QUANTUM NOISE
#
# Educational depolarizing model:
#
# rho_noisy =
#       (1-p) rho + p I/d
# ============================================================

def apply_quantum_noise(
    rho,
    noise_strength
):

    dimension = rho.shape[0]

    maximally_mixed = (
        np.eye(dimension)
        / dimension
    )

    rho_noisy = (
        (1 - noise_strength) * rho
        + noise_strength * maximally_mixed
    )

    return rho_noisy


# ============================================================
# 9. READOUT CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    readout_error
):

    single_qubit = np.array([
        [1 - readout_error, readout_error],
        [readout_error, 1 - readout_error]
    ])

    return np.kron(
        single_qubit,
        single_qubit
    )


# ============================================================
# 10. Z-BASIS MEASUREMENT
# ============================================================

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

    # Quantum measurement / shot noise
    true_counts = np.random.multinomial(
        shots,
        probabilities
    )

    # Apply readout bit-flip noise
    noisy_counts = np.zeros(
        4,
        dtype=int
    )

    for state_index, count in enumerate(
        true_counts
    ):

        if count == 0:
            continue

        # Computational basis:
        #
        # 00 -> 0
        # 01 -> 1
        # 10 -> 2
        # 11 -> 3

        q0 = (state_index >> 1) & 1
        q1 = state_index & 1

        for _ in range(count):

            measured_q0 = q0
            measured_q1 = q1

            if np.random.random() < readout_error:
                measured_q0 ^= 1

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


# ============================================================
# 11. Z EXPECTATIONS FROM PROBABILITIES
# ============================================================

def calculate_z_expectations(
    probabilities
):

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


# ============================================================
# 12. READOUT ERROR MITIGATION
#
# p_measured = C p_true
#
# Solve for:
#
# p_true = C^(-1) p_measured
# ============================================================

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

    corrected_probabilities = np.linalg.solve(
        confusion_matrix,
        measured_probabilities
    )

    # Numerical cleanup
    corrected_probabilities = np.clip(
        corrected_probabilities,
        0,
        None
    )

    total = np.sum(
        corrected_probabilities
    )

    if total > 0:

        corrected_probabilities /= total

    return corrected_probabilities


# ============================================================
# 13. GENERAL PAULI MEASUREMENT
#
# Used for XX and YY.
# ============================================================

def measure_pauli(
    true_expectation,
    shots,
    readout_error
):

    # Probability of +1
    p_plus = (
        1 + true_expectation
    ) / 2

    # Finite-shot sampling
    plus_count = np.random.binomial(
        shots,
        p_plus
    )

    minus_count = (
        shots - plus_count
    )

    # For two qubits:
    # parity flips when exactly one
    # of the two bits flips.
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


# ============================================================
# 14. MITIGATE XX / YY
# ============================================================

def mitigate_pauli(
    measured_expectation,
    readout_error
):

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


# ============================================================
# 15. RUN ONE NOISY EXPERIMENT
# ============================================================

def run_noisy_experiment(
    noise_strength,
    shots,
    readout_error
):

    # --------------------------------------------------------
    # Apply quantum noise
    # --------------------------------------------------------

    rho_noisy = apply_quantum_noise(
        rho_ideal,
        noise_strength
    )

    # --------------------------------------------------------
    # Z-basis measurement
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Readout mitigation for Z measurements
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # XX
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # YY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Raw energy
    # --------------------------------------------------------

    raw_energy = calculate_energy(
        raw_z0,
        raw_z1,
        raw_z0z1,
        raw_xx,
        raw_yy
    )

    # --------------------------------------------------------
    # REM energy
    # --------------------------------------------------------

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
# 16. EXPERIMENT PARAMETERS
# ============================================================

shots = 2048

base_noise = 0.05

readout_error = 0.05

noise_scales = np.array([
    1,
    2,
    3
], dtype=float)


# ============================================================
# 17. HEADER
# ============================================================

print("=" * 75)
print("DAY 50 - END-TO-END NOISY VQE BENCHMARK")
print("=" * 75)

print()

print(
    f"Exact energy       : "
    f"{exact_energy:.10f}"
)

print(
    f"Ideal VQE energy   : "
    f"{ideal_energy:.10f}"
)

print(
    f"Shots              : "
    f"{shots}"
)

print(
    f"Quantum noise      : "
    f"{base_noise}"
)

print(
    f"Readout error      : "
    f"{readout_error}"
)

print()


# ============================================================
# 18. RUN 1x, 2x, 3x
# ============================================================

raw_energies = []

rem_energies = []


for scale in noise_scales:

    noise_strength = (
        base_noise * scale
    )

    raw_energy, rem_energy = (
        run_noisy_experiment(
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

    print("-" * 75)

    print(
        f"Noise scale       : "
        f"{int(scale)}x"
    )

    print(
        f"Noise strength    : "
        f"{noise_strength:.4f}"
    )

    print(
        f"Raw energy        : "
        f"{raw_energy:.10f}"
    )

    print(
        f"REM energy        : "
        f"{rem_energy:.10f}"
    )

    print(
        f"Raw error         : "
        f"{abs(raw_energy - exact_energy):.10f}"
    )

    print(
        f"REM error         : "
        f"{abs(rem_energy - exact_energy):.10f}"
    )

    print()


# Convert to NumPy arrays
raw_energies = np.array(
    raw_energies
)

rem_energies = np.array(
    rem_energies
)


# ============================================================
# 19. ZNE ON RAW ENERGIES
# ============================================================

raw_slope, raw_intercept = np.polyfit(
    noise_scales,
    raw_energies,
    1
)

raw_zne_energy = raw_intercept


# ============================================================
# 20. ZNE AFTER READOUT MITIGATION
#
# This is the complete:
#
# REM + ZNE
# ============================================================

rem_slope, rem_intercept = np.polyfit(
    noise_scales,
    rem_energies,
    1
)

rem_zne_energy = rem_intercept


# ============================================================
# 21. ERRORS
# ============================================================

ideal_error = abs(
    ideal_energy
    - exact_energy
)

raw_error = abs(
    raw_energies[0]
    - exact_energy
)

rem_error = abs(
    rem_energies[0]
    - exact_energy
)

raw_zne_error = abs(
    raw_zne_energy
    - exact_energy
)

rem_zne_error = abs(
    rem_zne_energy
    - exact_energy
)


# ============================================================
# 22. ERROR REDUCTION
# ============================================================

def error_reduction(
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


rem_improvement = error_reduction(
    raw_error,
    rem_error
)

zne_improvement = error_reduction(
    raw_error,
    raw_zne_error
)

combined_improvement = error_reduction(
    raw_error,
    rem_zne_error
)


# ============================================================
# 23. PERCENTAGE ERRORS
# ============================================================

raw_percentage_error = (
    raw_error
    / abs(exact_energy)
    * 100
)

rem_percentage_error = (
    rem_error
    / abs(exact_energy)
    * 100
)

raw_zne_percentage_error = (
    raw_zne_error
    / abs(exact_energy)
    * 100
)

rem_zne_percentage_error = (
    rem_zne_error
    / abs(exact_energy)
    * 100
)


# ============================================================
# 24. FINAL RESULTS
# ============================================================

print("=" * 75)
print("FINAL BENCHMARK")
print("=" * 75)

print()

print(
    f"Exact energy                : "
    f"{exact_energy:.10f}"
)

print(
    f"Ideal VQE energy            : "
    f"{ideal_energy:.10f}"
)

print()

print(
    f"Raw noisy 1x energy         : "
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


# ============================================================
# 25. ERROR TABLE
# ============================================================

print("-" * 75)
print("ERROR ANALYSIS")
print("-" * 75)

print()

print(
    f"Ideal VQE error             : "
    f"{ideal_error:.10f}"
)

print(
    f"Raw noisy error             : "
    f"{raw_error:.10f}"
)

print(
    f"REM error                   : "
    f"{rem_error:.10f}"
)

print(
    f"Raw ZNE error               : "
    f"{raw_zne_error:.10f}"
)

print(
    f"REM + ZNE error             : "
    f"{rem_zne_error:.10f}"
)

print()


# ============================================================
# 26. PERCENTAGE ERROR
# ============================================================

print("-" * 75)
print("PERCENTAGE ERROR")
print("-" * 75)

print()

print(
    f"Raw noisy                   : "
    f"{raw_percentage_error:.4f}%"
)

print(
    f"REM                        : "
    f"{rem_percentage_error:.4f}%"
)

print(
    f"Raw ZNE                    : "
    f"{raw_zne_percentage_error:.4f}%"
)

print(
    f"REM + ZNE                  : "
    f"{rem_zne_percentage_error:.4f}%"
)

print()


# ============================================================
# 27. ERROR REDUCTION
# ============================================================

print("-" * 75)
print("ERROR REDUCTION")
print("-" * 75)

print()

print(
    f"REM improvement            : "
    f"{rem_improvement:.2f}%"
)

print(
    f"ZNE improvement            : "
    f"{zne_improvement:.2f}%"
)

print(
    f"REM + ZNE improvement      : "
    f"{combined_improvement:.2f}%"
)

print()


# ============================================================
# 28. ZNE MODELS
# ============================================================

print("-" * 75)
print("ZNE MODELS")
print("-" * 75)

print()

print("Raw ZNE:")

print(
    f"E(lambda) = "
    f"{raw_slope:.10f} * lambda "
    f"+ {raw_intercept:.10f}"
)

print()

print("REM + ZNE:")

print(
    f"E(lambda) = "
    f"{rem_slope:.10f} * lambda "
    f"+ {rem_intercept:.10f}"
)

print()


# ============================================================
# 29. FINAL PIPELINE
# ============================================================

print("=" * 75)
print("END-TO-END PIPELINE")
print("=" * 75)

print()

print(
    "Ideal VQE"
    " -> Quantum noise"
    " -> Measurement"
    " -> Readout noise"
    " -> REM"
    " -> ZNE"
    " -> Final estimate"
)

print()

print("=" * 75)
print("DAY 50 COMPUTATION COMPLETE")
print("=" * 75)


# ============================================================
# 30. VISUALIZATION
# ============================================================

methods = [
    "Exact",
    "Ideal VQE",
    "Raw Noisy",
    "REM",
    "Raw ZNE",
    "REM + ZNE"
]

energies = [
    exact_energy,
    ideal_energy,
    raw_energies[0],
    rem_energies[0],
    raw_zne_energy,
    rem_zne_energy
]

errors = [
    0,
    ideal_error,
    raw_error,
    rem_error,
    raw_zne_error,
    rem_zne_error
]


# ------------------------------------------------------------
# Energy comparison
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    energies
)

plt.axhline(
    exact_energy,
    linestyle="--",
    label="Exact Energy"
)

plt.ylabel("Energy")

plt.title(
    "Day 50 - Noisy VQE Energy Comparison"
)

plt.xticks(
    rotation=20
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "day50_energy_comparison.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Error comparison
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    errors
)

plt.ylabel("Absolute Error")

plt.title(
    "Day 50 - Error Comparison"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "day50_error_comparison.png",
    dpi=300
)

plt.show()


print()
print("Graphs saved as:")
print("  day50_energy_comparison.png")
print("  day50_error_comparison.png")