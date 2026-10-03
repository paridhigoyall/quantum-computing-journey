import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error
)


# ============================================================
# CONFIGURATION
# ============================================================

np.random.seed(42)

N_QUBITS = 2
LAYERS = 3

LEARNING_RATE = 0.15
ITERATIONS = 150

EPSILON = 1e-12

TARGET = np.array([
    0.10,
    0.20,
    0.60,
    0.10
])

STATES = [
    "00",
    "01",
    "10",
    "11"
]


# ============================================================
# GENERATOR
# ============================================================

def generator_circuit(theta):

    qc = QuantumCircuit(N_QUBITS)

    index = 0

    for _ in range(LAYERS):

        # Trainable rotations
        for q in range(N_QUBITS):

            qc.ry(theta[index], q)
            index += 1

            qc.rz(theta[index], q)
            index += 1

        # Entanglement
        for q in range(N_QUBITS - 1):

            qc.cx(q, q + 1)

    return qc


# ============================================================
# EXACT STATEVECTOR DISTRIBUTION
# ============================================================

def exact_distribution(theta):

    qc = generator_circuit(theta)

    state = Statevector.from_instruction(qc)

    probabilities = np.abs(
        state.data
    ) ** 2

    return probabilities


# ============================================================
# SHOT-BASED DISTRIBUTION
# ============================================================

def shot_distribution(
    theta,
    shots=1000,
    noise_model=None
):

    qc = generator_circuit(theta)

    # Add classical measurement
    qc.measure_all()

    simulator = AerSimulator(
        noise_model=noise_model
    )

    result = simulator.run(
        qc,
        shots=shots
    ).result()

    counts = result.get_counts()

    probabilities = np.zeros(
        2 ** N_QUBITS
    )

    for bitstring, count in counts.items():

        # Qiskit count keys are returned
        # in classical-bit order.
        index = int(
            bitstring,
            2
        )

        probabilities[index] = (
            count / shots
        )

    return probabilities


# ============================================================
# LOSS
# ============================================================

def mse_loss(P, Q):

    return np.mean(
        (P - Q) ** 2
    )


# ============================================================
# KL DIVERGENCE
# ============================================================

def kl_divergence(P, Q):

    P = np.clip(
        P,
        EPSILON,
        None
    )

    Q = np.clip(
        Q,
        EPSILON,
        None
    )

    return np.sum(
        P * np.log(P / Q)
    )


# ============================================================
# JS DIVERGENCE
# ============================================================

def js_divergence(P, Q):

    P = np.clip(
        P,
        EPSILON,
        None
    )

    Q = np.clip(
        Q,
        EPSILON,
        None
    )

    M = (P + Q) / 2

    return (
        0.5 * kl_divergence(P, M)
        +
        0.5 * kl_divergence(Q, M)
    )


# ============================================================
# WASSERSTEIN DISTANCE
# ============================================================

def wasserstein_distance(P, Q):

    cumulative_P = np.cumsum(P)
    cumulative_Q = np.cumsum(Q)

    return np.sum(
        np.abs(
            cumulative_P -
            cumulative_Q
        )
    )


# ============================================================
# TARGET STATE
# ============================================================

def target_statevector():

    amplitudes = np.sqrt(TARGET)

    return Statevector(
        amplitudes
    )


# ============================================================
# FIDELITY
# ============================================================

def fidelity(
    target_state,
    generated_state
):

    overlap = np.vdot(
        target_state.data,
        generated_state.data
    )

    return float(
        np.abs(overlap) ** 2
    )


# ============================================================
# MODE COVERAGE
# ============================================================

def mode_coverage(
    target,
    generated,
    threshold=0.10
):

    target_modes = (
        target >= threshold
    )

    generated_modes = (
        generated >= threshold
    )

    total_modes = np.sum(
        target_modes
    )

    if total_modes == 0:

        return 0.0

    covered = np.sum(
        target_modes &
        generated_modes
    )

    return covered / total_modes


# ============================================================
# CREATE DEPOLARIZING NOISE
# ============================================================

def create_depolarizing_noise(
    probability
):

    noise_model = NoiseModel()

    # One-qubit depolarizing noise
    one_qubit_error = (
        depolarizing_error(
            probability,
            1
        )
    )

    # Two-qubit depolarizing noise
    two_qubit_error = (
        depolarizing_error(
            probability,
            2
        )
    )

    # Apply to rotation gates
    noise_model.add_all_qubit_quantum_error(
        one_qubit_error,
        ["ry", "rz"]
    )

    # Apply stronger two-qubit error
    noise_model.add_all_qubit_quantum_error(
        two_qubit_error,
        ["cx"]
    )

    return noise_model


# ============================================================
# CREATE BIT-FLIP NOISE
# ============================================================

def create_bitflip_noise(
    probability
):

    noise_model = NoiseModel()

    bitflip = pauli_error([
        ("X", probability),
        ("I", 1 - probability)
    ])

    noise_model.add_all_qubit_quantum_error(
        bitflip,
        ["ry", "rz"]
    )

    return noise_model


# ============================================================
# RANDOM INITIAL PARAMETERS
# ============================================================

NUM_PARAMETERS = (
    LAYERS *
    N_QUBITS *
    2
)

theta = np.random.uniform(
    -np.pi,
    np.pi,
    NUM_PARAMETERS
)


# ============================================================
# TRAIN USING IDEAL STATEVECTOR
# ============================================================

print("=" * 70)
print("TRAINING IDEAL QUANTUM GENERATOR")
print("=" * 70)


for iteration in range(
    ITERATIONS
):

    current_distribution = (
        exact_distribution(theta)
    )

    loss = mse_loss(
        TARGET,
        current_distribution
    )

    gradients = np.zeros_like(theta)

    shift = np.pi / 2

    # Parameter-shift
    for i in range(
        len(theta)
    ):

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[i] += shift
        theta_minus[i] -= shift

        loss_plus = mse_loss(
            TARGET,
            exact_distribution(
                theta_plus
            )
        )

        loss_minus = mse_loss(
            TARGET,
            exact_distribution(
                theta_minus
            )
        )

        gradients[i] = (
            loss_plus -
            loss_minus
        ) / 2

    theta -= (
        LEARNING_RATE *
        gradients
    )

    if iteration % 25 == 0:

        print(
            f"Iteration {iteration:3d} | "
            f"Loss = {loss:.8f}"
        )


print("\nTraining complete.")


# ============================================================
# IDEAL DISTRIBUTION
# ============================================================

ideal_distribution = (
    exact_distribution(theta)
)


print("\nIDEAL DISTRIBUTION")
print("=" * 70)

for state, probability in zip(
    STATES,
    ideal_distribution
):

    print(
        f"|{state}> : "
        f"{probability:.6f}"
    )


# ============================================================
# TARGET STATE
# ============================================================

target_state = (
    target_statevector()
)


# ============================================================
# SHOT EXPERIMENT
# ============================================================

shot_values = [
    100,
    1000,
    10000
]

shot_results = {}


print("\n\nFINITE-SHOT EXPERIMENT")
print("=" * 70)


for shots in shot_values:

    distribution = (
        shot_distribution(
            theta,
            shots=shots
        )
    )

    shot_results[shots] = (
        distribution
    )

    print(
        f"\nShots = {shots}"
    )

    for state, probability in zip(
        STATES,
        distribution
    ):

        print(
            f"|{state}> : "
            f"{probability:.6f}"
        )


# ============================================================
# DEPOLARIZING NOISE EXPERIMENT
# ============================================================

noise_levels = [
    0.01,
    0.05,
    0.10
]

noise_results = {}


print("\n\nDEPOLARIZING NOISE EXPERIMENT")
print("=" * 70)


for probability in noise_levels:

    noise_model = (
        create_depolarizing_noise(
            probability
        )
    )

    distribution = (
        shot_distribution(
            theta,
            shots=10000,
            noise_model=noise_model
        )
    )

    noise_results[probability] = (
        distribution
    )

    print(
        f"\nNoise probability = "
        f"{probability}"
    )

    for state, p in zip(
        STATES,
        distribution
    ):

        print(
            f"|{state}> : "
            f"{p:.6f}"
        )


# ============================================================
# BIT-FLIP EXPERIMENT
# ============================================================

bitflip_probability = 0.05

bitflip_model = (
    create_bitflip_noise(
        bitflip_probability
    )
)

bitflip_distribution = (
    shot_distribution(
        theta,
        shots=10000,
        noise_model=bitflip_model
    )
)


print("\n\nBIT-FLIP NOISE")
print("=" * 70)

for state, probability in zip(
    STATES,
    bitflip_distribution
):

    print(
        f"|{state}> : "
        f"{probability:.6f}"
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_distribution(
    name,
    distribution
):

    mse = mse_loss(
        TARGET,
        distribution
    )

    kl = kl_divergence(
        TARGET,
        distribution
    )

    js = js_divergence(
        TARGET,
        distribution
    )

    wasserstein = (
        wasserstein_distance(
            TARGET,
            distribution
        )
    )

    coverage = mode_coverage(
        TARGET,
        distribution
    )

    print(
        f"{name:<30}"
        f"MSE={mse:.6f} | "
        f"KL={kl:.6f} | "
        f"JS={js:.6f} | "
        f"W={wasserstein:.6f} | "
        f"Coverage={coverage:.2%}"
    )

    return {
        "MSE": mse,
        "KL": kl,
        "JS": js,
        "Wasserstein": wasserstein,
        "Coverage": coverage
    }


# ============================================================
# EVALUATE EVERYTHING
# ============================================================

print("\n\nFINAL EVALUATION")
print("=" * 100)

evaluation_results = {}

evaluation_results["Ideal"] = (
    evaluate_distribution(
        "Ideal",
        ideal_distribution
    )
)


for shots, distribution in (
    shot_results.items()
):

    evaluation_results[
        f"{shots} shots"
    ] = evaluate_distribution(
        f"{shots} shots",
        distribution
    )


for probability, distribution in (
    noise_results.items()
):

    evaluation_results[
        f"Noise {probability}"
    ] = evaluate_distribution(
        f"Depolarizing {probability}",
        distribution
    )


evaluation_results[
    "Bit-flip 0.05"
] = evaluate_distribution(
    "Bit-flip 0.05",
    bitflip_distribution
)


# ============================================================
# PLOT 1
# IDEAL VS TARGET
# ============================================================

x = np.arange(
    len(STATES)
)

width = 0.35

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - width / 2,
    TARGET,
    width,
    label="Target"
)

plt.bar(
    x + width / 2,
    ideal_distribution,
    width,
    label="Ideal Generator"
)

plt.xticks(
    x,
    STATES
)

plt.xlabel(
    "Quantum State"
)

plt.ylabel(
    "Probability"
)

plt.title(
    "Target vs Ideal Quantum Generator"
)

plt.legend()

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# PLOT 2
# SHOT NOISE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for shots, distribution in (
    shot_results.items()
):

    plt.plot(
        STATES,
        distribution,
        marker="o",
        label=f"{shots} shots"
    )

plt.plot(
    STATES,
    TARGET,
    marker="x",
    linewidth=2,
    label="Target"
)

plt.xlabel(
    "Quantum State"
)

plt.ylabel(
    "Probability"
)

plt.title(
    "Effect of Finite Measurement Shots"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# PLOT 3
# DEPOLARIZING NOISE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    STATES,
    TARGET,
    marker="x",
    linewidth=2,
    label="Target"
)

plt.plot(
    STATES,
    ideal_distribution,
    marker="o",
    label="Ideal"
)

for probability, distribution in (
    noise_results.items()
):

    plt.plot(
        STATES,
        distribution,
        marker="o",
        label=f"Noise p={probability}"
    )

plt.xlabel(
    "Quantum State"
)

plt.ylabel(
    "Probability"
)

plt.title(
    "Effect of Depolarizing Noise"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# PLOT 4
# METRIC DEGRADATION
# ============================================================

noise_x = [
    0.0,
    0.01,
    0.05,
    0.10
]

noise_distributions = [
    ideal_distribution,
    noise_results[0.01],
    noise_results[0.05],
    noise_results[0.10]
]

mse_values = [
    mse_loss(
        TARGET,
        distribution
    )
    for distribution in
    noise_distributions
]

js_values = [
    js_divergence(
        TARGET,
        distribution
    )
    for distribution in
    noise_distributions
]

wasserstein_values = [
    wasserstein_distance(
        TARGET,
        distribution
    )
    for distribution in
    noise_distributions
]

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    noise_x,
    mse_values,
    marker="o",
    label="MSE"
)

plt.plot(
    noise_x,
    js_values,
    marker="o",
    label="JS Divergence"
)

plt.plot(
    noise_x,
    wasserstein_values,
    marker="o",
    label="Wasserstein"
)

plt.xlabel(
    "Noise Probability"
)

plt.ylabel(
    "Metric Value"
)

plt.title(
    "Generator Quality vs Noise"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("DAY 74 COMPLETE")
print("=" * 70)

print("""
Key observations:

1. Exact Statevector simulation gives ideal probabilities.

2. Finite shots introduce statistical sampling error.

3. Increasing the number of shots generally reduces
   sampling fluctuations.

4. Quantum noise changes the distribution itself.

5. Increasing noise generally reduces the quality of
   the generated distribution.

6. Hardware-aware QML must evaluate both:
      - algorithmic performance
      - hardware robustness

7. A good QML model should not only learn the target
   distribution in an ideal simulator.

   It should remain useful under realistic execution.
""")