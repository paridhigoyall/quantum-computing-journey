import numpy as np


# ============================================================
# DAY 48
# Combined Error Mitigation:
# Readout Error Mitigation + Zero-Noise Extrapolation
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


# Two-qubit operators
Z0 = np.kron(Z, I1)
Z1 = np.kron(I1, Z)

Z0Z1 = np.kron(Z, Z)

XX = np.kron(X, X)
YY = np.kron(Y, Y)


# ------------------------------------------------------------
# 2. Fermionic Hamiltonian
# ------------------------------------------------------------

eps0 = 0.7
eps1 = 1.1
t = -0.4
U = 0.8

H = (
    eps0 * (I2 - Z0) / 2
    + eps1 * (I2 - Z1) / 2
    + t * (XX + YY) / 2
    + U * ((I2 - Z0) / 2) @ ((I2 - Z1) / 2)
)


# ------------------------------------------------------------
# 3. Exact ground-state energy
# ------------------------------------------------------------

eigenvalues = np.linalg.eigvalsh(H)

# Our VQE ansatz is restricted to the one-particle sector.
# Therefore, use the lowest energy in that sector.
one_particle_hamiltonian = np.array([
    [eps0, t],
    [t, eps1]
])

one_particle_eigenvalues = np.linalg.eigvalsh(
    one_particle_hamiltonian
)

exact_energy = one_particle_eigenvalues[0]

# ------------------------------------------------------------
# 4. VQE optimized state
# ------------------------------------------------------------

theta = 2.03444393

state = np.array([
    0,
    np.cos(theta / 2),
    np.sin(theta / 2),
    0
], dtype=complex)

# Normalize
state = state / np.linalg.norm(state)

rho_ideal = np.outer(state, state.conj())


# ------------------------------------------------------------
# 5. Expectation value helper
# ------------------------------------------------------------

def exact_expectation(rho, operator):
    return np.real(np.trace(rho @ operator))


# ------------------------------------------------------------
# 6. Apply quantum noise
#
# Educational depolarizing model:
#
# rho_noisy = (1-p) rho + p I/d
# ------------------------------------------------------------

def apply_quantum_noise(rho, noise_strength):
    dimension = rho.shape[0]

    mixed_state = np.eye(dimension, dtype=complex) / dimension

    rho_noisy = (
        (1 - noise_strength) * rho
        + noise_strength * mixed_state
    )

    return rho_noisy


# ------------------------------------------------------------
# 7. Readout-noise confusion matrix
# ------------------------------------------------------------

def create_confusion_matrix(readout_error):
    single_qubit = np.array([
        [1 - readout_error, readout_error],
        [readout_error, 1 - readout_error]
    ])

    two_qubit = np.kron(single_qubit, single_qubit)

    return two_qubit


# ------------------------------------------------------------
# 8. Sample computational-basis measurements
# ------------------------------------------------------------

def sample_computational_basis(rho, shots, readout_error):
    probabilities = np.real(np.diag(rho))

    # Numerical cleanup
    probabilities = np.clip(probabilities, 0, None)
    probabilities = probabilities / np.sum(probabilities)

    # Ideal measurement sampling
    counts = np.random.multinomial(
        shots,
        probabilities
    )

    # Apply readout error to every measured bit
    noisy_counts = np.zeros(4, dtype=int)

    for state_index, count in enumerate(counts):

        if count == 0:
            continue

        # Convert index to two-bit state
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

            noisy_counts[measured_index] += 1

    return noisy_counts


# ------------------------------------------------------------
# 9. Readout-error mitigation
#
# p_measured = C p_true
#
# Therefore:
#
# p_true = C^-1 p_measured
# ------------------------------------------------------------

def mitigate_readout(counts, shots, readout_error):

    measured_probabilities = counts / shots

    confusion_matrix = create_confusion_matrix(
        readout_error
    )

    # Solve instead of explicitly calculating inverse
    mitigated_probabilities = np.linalg.solve(
        confusion_matrix,
        measured_probabilities
    )

    # Remove tiny numerical negative values
    mitigated_probabilities = np.clip(
        mitigated_probabilities,
        0,
        None
    )

    # Normalize
    total = np.sum(mitigated_probabilities)

    if total > 0:
        mitigated_probabilities /= total

    return mitigated_probabilities


# ------------------------------------------------------------
# 10. Convert probabilities to Z expectations
# ------------------------------------------------------------

def z_expectations_from_probabilities(probabilities):

    # Basis:
    # 00 -> index 0
    # 01 -> index 1
    # 10 -> index 2
    # 11 -> index 3

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

    return exp_z0, exp_z1, exp_z0z1


# ------------------------------------------------------------
# 11. Sample ±1 outcomes for a Pauli observable
#
# Used for XX and YY measurements.
# ------------------------------------------------------------

def sample_pauli_expectation(
    ideal_expectation,
    shots,
    readout_error
):

    # First sample the quantum measurement result.
    p_plus = (1 + ideal_expectation) / 2

    plus_count = np.random.binomial(
        shots,
        p_plus
    )

    minus_count = shots - plus_count

    # For a two-qubit Pauli measurement,
    # readout errors can flip the parity.
    #
    # Probability that parity remains correct:
    # (1-p)^2 + p^2
    #
    # Probability that parity flips:
    # 2p(1-p)

    parity_flip_probability = (
        2
        * readout_error
        * (1 - readout_error)
    )

    observed_plus = np.random.binomial(
        plus_count,
        1 - parity_flip_probability
    )

    observed_minus = np.random.binomial(
        minus_count,
        parity_flip_probability
    )

    measured_plus = (
        observed_plus
        + observed_minus
    )

    measured_expectation = (
        2 * measured_plus / shots
        - 1
    )

    return measured_expectation


# ------------------------------------------------------------
# 12. Mitigate a Pauli expectation value
# ------------------------------------------------------------

def mitigate_pauli_expectation(
    measured_expectation,
    readout_error
):

    parity_factor = (
        1 - 2 * readout_error
    ) ** 2

    if abs(parity_factor) < 1e-12:
        return measured_expectation

    mitigated = (
        measured_expectation
        / parity_factor
    )

    # Physical expectation values are between -1 and +1
    mitigated = np.clip(
        mitigated,
        -1,
        1
    )

    return mitigated


# ------------------------------------------------------------
# 13. Calculate energy from expectations
# ------------------------------------------------------------

def calculate_energy(
    exp_z0,
    exp_z1,
    exp_z0z1,
    exp_xx,
    exp_yy
):

    # Number operators:
    #
    # n0 = (I-Z0)/2
    # n1 = (I-Z1)/2

    n0 = (1 - exp_z0) / 2
    n1 = (1 - exp_z1) / 2

    interaction = (
        (1 - exp_z0 - exp_z1 + exp_z0z1)
        / 4
    )

    hopping = (
        t * (exp_xx + exp_yy) / 2
    )

    energy = (
        eps0 * n0
        + eps1 * n1
        + hopping
        + U * interaction
    )

    return energy


# ------------------------------------------------------------
# 14. Main experiment
# ------------------------------------------------------------

shots = 2048

base_noise = 0.05

noise_scales = [1, 2, 3]

readout_error = 0.05


print("=" * 65)
print("DAY 48 - COMBINED ERROR MITIGATION")
print("Readout Error Mitigation + Zero-Noise Extrapolation")
print("=" * 65)

print()

print(f"Exact ground-state energy : {exact_energy:.10f}")
print(f"VQE optimized theta       : {theta:.10f}")
print(f"Shots per measurement     : {shots}")
print(f"Base quantum noise        : {base_noise}")
print(f"Readout error             : {readout_error}")

print()


# ------------------------------------------------------------
# Ideal expectations
# ------------------------------------------------------------

ideal_z0 = exact_expectation(
    rho_ideal,
    Z0
)

ideal_z1 = exact_expectation(
    rho_ideal,
    Z1
)

ideal_z0z1 = exact_expectation(
    rho_ideal,
    Z0Z1
)

ideal_xx = exact_expectation(
    rho_ideal,
    XX
)

ideal_yy = exact_expectation(
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

print("-" * 65)
print("IDEAL VQE")
print("-" * 65)

print(f"Ideal <Z0>      : {ideal_z0:.10f}")
print(f"Ideal <Z1>      : {ideal_z1:.10f}")
print(f"Ideal <Z0Z1>    : {ideal_z0z1:.10f}")
print(f"Ideal <XX>      : {ideal_xx:.10f}")
print(f"Ideal <YY>      : {ideal_yy:.10f}")
print(f"Ideal energy    : {ideal_energy:.10f}")

print()


# ------------------------------------------------------------
# Store energies for ZNE
# ------------------------------------------------------------

raw_energies = []

mitigated_energies = []


# ------------------------------------------------------------
# Run at 1×, 2× and 3× noise
# ------------------------------------------------------------

for scale in noise_scales:

    noise_strength = base_noise * scale

    # Prevent invalid noise strength
    noise_strength = min(
        noise_strength,
        0.99
    )

    print("-" * 65)
    print(f"NOISE SCALE = {scale}x")
    print(f"Quantum noise = {noise_strength:.4f}")
    print("-" * 65)

    # --------------------------------------------
    # Apply quantum noise
    # --------------------------------------------

    rho_noisy = apply_quantum_noise(
        rho_ideal,
        noise_strength
    )

    # --------------------------------------------
    # Z-basis measurements
    # --------------------------------------------

    counts = sample_computational_basis(
        rho_noisy,
        shots,
        readout_error
    )

    # Raw probabilities
    raw_probabilities = counts / shots

    # Raw Z expectations
    raw_z0, raw_z1, raw_z0z1 = (
        z_expectations_from_probabilities(
            raw_probabilities
        )
    )

    # --------------------------------------------
    # Readout mitigation
    # --------------------------------------------

    mitigated_probabilities = mitigate_readout(
        counts,
        shots,
        readout_error
    )

    mitigated_z0, mitigated_z1, mitigated_z0z1 = (
        z_expectations_from_probabilities(
            mitigated_probabilities
        )
    )

    # --------------------------------------------
    # XX measurement
    # --------------------------------------------

    noisy_xx_true = exact_expectation(
        rho_noisy,
        XX
    )

    raw_xx = sample_pauli_expectation(
        noisy_xx_true,
        shots,
        readout_error
    )

    mitigated_xx = mitigate_pauli_expectation(
        raw_xx,
        readout_error
    )

    # --------------------------------------------
    # YY measurement
    # --------------------------------------------

    noisy_yy_true = exact_expectation(
        rho_noisy,
        YY
    )

    raw_yy = sample_pauli_expectation(
        noisy_yy_true,
        shots,
        readout_error
    )

    mitigated_yy = mitigate_pauli_expectation(
        raw_yy,
        readout_error
    )

    # --------------------------------------------
    # Raw noisy energy
    # --------------------------------------------

    raw_energy = calculate_energy(
        raw_z0,
        raw_z1,
        raw_z0z1,
        raw_xx,
        raw_yy
    )

    # --------------------------------------------
    # Readout-mitigated energy
    # --------------------------------------------

    mitigated_energy = calculate_energy(
        mitigated_z0,
        mitigated_z1,
        mitigated_z0z1,
        mitigated_xx,
        mitigated_yy
    )

    raw_energies.append(raw_energy)

    mitigated_energies.append(
        mitigated_energy
    )

    print()

    print("Raw measured expectations:")
    print(f"  <Z0>       = {raw_z0:.6f}")
    print(f"  <Z1>       = {raw_z1:.6f}")
    print(f"  <Z0Z1>     = {raw_z0z1:.6f}")
    print(f"  <XX>       = {raw_xx:.6f}")
    print(f"  <YY>       = {raw_yy:.6f}")

    print()

    print(f"Raw noisy energy      = {raw_energy:.10f}")
    print(
        f"After readout mitigation = "
        f"{mitigated_energy:.10f}"
    )

    print()


# ------------------------------------------------------------
# 15. Zero-Noise Extrapolation
# ------------------------------------------------------------

noise_scales_array = np.array(
    noise_scales,
    dtype=float
)

mitigated_energies_array = np.array(
    mitigated_energies
)

# Fit:
#
# E(lambda) = slope * lambda + intercept

slope, intercept = np.polyfit(
    noise_scales_array,
    mitigated_energies_array,
    1
)

zne_energy = intercept


# ------------------------------------------------------------
# 16. Error analysis
# ------------------------------------------------------------

raw_1x_error = abs(
    raw_energies[0]
    - exact_energy
)

mitigated_1x_error = abs(
    mitigated_energies[0]
    - exact_energy
)

zne_error = abs(
    zne_energy
    - exact_energy
)


if raw_1x_error > 0:
    combined_improvement = (
        (raw_1x_error - zne_error)
        / raw_1x_error
        * 100
    )
else:
    combined_improvement = 0


if raw_1x_error > 0:
    readout_improvement = (
        (raw_1x_error - mitigated_1x_error)
        / raw_1x_error
        * 100
    )
else:
    readout_improvement = 0


# ------------------------------------------------------------
# 17. Final results
# ------------------------------------------------------------

print("=" * 65)
print("FINAL RESULTS")
print("=" * 65)

print()

print(f"Exact energy                 : {exact_energy:.10f}")
print(f"Ideal VQE energy             : {ideal_energy:.10f}")

print()

print("Raw energies:")
for scale, energy in zip(
    noise_scales,
    raw_energies
):
    print(
        f"  {scale}x noise : "
        f"{energy:.10f}"
    )

print()

print("Readout-mitigated energies:")
for scale, energy in zip(
    noise_scales,
    mitigated_energies
):
    print(
        f"  {scale}x noise : "
        f"{energy:.10f}"
    )

print()

print(f"ZNE slope                    : {slope:.10f}")
print(f"ZNE final energy             : {zne_energy:.10f}")

print()

print(f"Raw 1x error                 : {raw_1x_error:.10f}")
print(
    f"Readout-mitigated 1x error  : "
    f"{mitigated_1x_error:.10f}"
)
print(f"Final ZNE error              : {zne_error:.10f}")

print()

print(
    f"Readout mitigation improvement : "
    f"{readout_improvement:.2f}%"
)

print(
    f"Combined error reduction       : "
    f"{combined_improvement:.2f}%"
)

print()

print("=" * 65)
print("PIPELINE")
print("=" * 65)

print(
    "VQE state"
    " -> Quantum noise"
    " -> Measurement"
    " -> Readout noise"
    " -> Readout mitigation"
    " -> ZNE"
    " -> Final energy"
)

print("=" * 65)