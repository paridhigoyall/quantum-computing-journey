import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


# ============================================================
# CONFIGURATION
# ============================================================

np.random.seed(42)

N_QUBITS = 2
LAYERS = 3

SHOTS = 10000

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
# QUANTUM GENERATOR
# ============================================================

def generator_circuit(theta):

    qc = QuantumCircuit(N_QUBITS)

    index = 0

    for _ in range(LAYERS):

        for q in range(N_QUBITS):

            qc.ry(theta[index], q)
            index += 1

            qc.rz(theta[index], q)
            index += 1

        qc.cx(0, 1)

    return qc


# ============================================================
# EXACT DISTRIBUTION
# ============================================================

def exact_distribution(theta):

    qc = generator_circuit(theta)

    state = Statevector.from_instruction(qc)

    return np.abs(state.data) ** 2


# ============================================================
# TRAINING
# ============================================================

def mse_loss(P, Q):

    return np.mean(
        (P - Q) ** 2
    )


def train_generator():

    num_parameters = (
        LAYERS *
        N_QUBITS *
        2
    )

    theta = np.random.uniform(
        -np.pi,
        np.pi,
        num_parameters
    )

    learning_rate = 0.15
    iterations = 150

    for iteration in range(iterations):

        current = exact_distribution(
            theta
        )

        gradients = np.zeros_like(
            theta
        )

        shift = np.pi / 2

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
            learning_rate *
            gradients
        )

    return theta


# ============================================================
# BASIC BIT-FLIP NOISE
# ============================================================

def create_measurement_like_noise(
    probability
):

    noise_model = NoiseModel()

    error = pauli_error([
        ("X", probability),
        ("I", 1 - probability)
    ])

    noise_model.add_all_qubit_quantum_error(
        error,
        ["ry", "rz"]
    )

    return noise_model


# ============================================================
# RUN CIRCUIT
# ============================================================

def run_circuit(
    theta,
    shots=10000,
    noise_model=None
):

    qc = generator_circuit(theta)

    qc.measure_all()

    simulator = AerSimulator(
        noise_model=noise_model
    )

    result = simulator.run(
        qc,
        shots=shots
    ).result()

    counts = result.get_counts()

    distribution = np.zeros(
        2 ** N_QUBITS
    )

    for bitstring, count in counts.items():

        index = int(
            bitstring,
            2
        )

        distribution[index] = (
            count / shots
        )

    return distribution


# ============================================================
# MEASUREMENT ERROR MODEL
# ============================================================

def measurement_error_matrix(
    bitflip_probability
):

    p = bitflip_probability

    single_qubit = np.array([
        [1 - p, p],
        [p, 1 - p]
    ])

    # Tensor product for two qubits
    matrix = np.kron(
        single_qubit,
        single_qubit
    )

    return matrix


# ============================================================
# APPLY MEASUREMENT ERROR
# ============================================================

def apply_measurement_error(
    distribution,
    bitflip_probability
):

    matrix = measurement_error_matrix(
        bitflip_probability
    )

    observed = (
        matrix @ distribution
    )

    return observed


# ============================================================
# MITIGATION BY MATRIX INVERSION
# ============================================================

def mitigate_measurement_error(
    observed,
    calibration_matrix
):

    inverse_matrix = np.linalg.inv(
        calibration_matrix
    )

    corrected = (
        inverse_matrix @ observed
    )

    # Remove small negative probabilities
    corrected = np.clip(
        corrected,
        0,
        None
    )

    # Renormalize
    total = np.sum(
        corrected
    )

    if total > 0:

        corrected /= total

    return corrected


# ============================================================
# METRICS
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


def js_divergence(P, Q):

    M = (
        P + Q
    ) / 2

    return (
        0.5 *
        kl_divergence(P, M)
        +
        0.5 *
        kl_divergence(Q, M)
    )


def wasserstein_distance(P, Q):

    cumulative_P = np.cumsum(P)
    cumulative_Q = np.cumsum(Q)

    return np.sum(
        np.abs(
            cumulative_P -
            cumulative_Q
        )
    )


def evaluate(P):

    mse = mse_loss(
        TARGET,
        P
    )

    kl = kl_divergence(
        TARGET,
        P
    )

    js = js_divergence(
        TARGET,
        P
    )

    wasserstein = (
        wasserstein_distance(
            TARGET,
            P
        )
    )

    return {
        "MSE": mse,
        "KL": kl,
        "JS": js,
        "Wasserstein": wasserstein
    }


# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("TRAINING GENERATOR")
print("=" * 70)

theta = train_generator()

ideal_distribution = (
    exact_distribution(theta)
)


# ============================================================
# IDEAL DISTRIBUTION
# ============================================================

print("\nIDEAL DISTRIBUTION")

for state, probability in zip(
    STATES,
    ideal_distribution
):

    print(
        f"|{state}> : "
        f"{probability:.6f}"
    )


# ============================================================
# MEASUREMENT ERROR
# ============================================================

measurement_probability = 0.05

calibration_matrix = (
    measurement_error_matrix(
        measurement_probability
    )
)


# Simulated observed distribution
observed_distribution = (
    apply_measurement_error(
        ideal_distribution,
        measurement_probability
    )
)


print("\nOBSERVED DISTRIBUTION")

for state, probability in zip(
    STATES,
    observed_distribution
):

    print(
        f"|{state}> : "
        f"{probability:.6f}"
    )


# ============================================================
# MITIGATION
# ============================================================

mitigated_distribution = (
    mitigate_measurement_error(
        observed_distribution,
        calibration_matrix
    )
)


print("\nMITIGATED DISTRIBUTION")

for state, probability in zip(
    STATES,
    mitigated_distribution
):

    print(
        f"|{state}> : "
        f"{probability:.6f}"
    )


# ============================================================
# EVALUATE IDEAL / OBSERVED / MITIGATED
# ============================================================

ideal_metrics = evaluate(
    ideal_distribution
)

observed_metrics = evaluate(
    observed_distribution
)

mitigated_metrics = evaluate(
    mitigated_distribution
)


print("\n")
print("=" * 90)
print("MEASUREMENT ERROR MITIGATION")
print("=" * 90)

print(
    f"{'Distribution':<20}"
    f"{'MSE':<15}"
    f"{'KL':<15}"
    f"{'JS':<15}"
    f"{'Wasserstein':<15}"
)

print("-" * 90)

for name, metrics in [
    ("Ideal", ideal_metrics),
    ("Observed", observed_metrics),
    ("Mitigated", mitigated_metrics)
]:

    print(
        f"{name:<20}"
        f"{metrics['MSE']:<15.6f}"
        f"{metrics['KL']:<15.6f}"
        f"{metrics['JS']:<15.6f}"
        f"{metrics['Wasserstein']:<15.6f}"
    )


# ============================================================
# ZERO-NOISE EXTRAPOLATION
# ============================================================

print("\n")
print("=" * 70)
print("ZERO-NOISE EXTRAPOLATION")
print("=" * 70)


noise_scales = np.array([
    1.0,
    2.0,
    3.0
])

base_noise = 0.02

noise_distributions = []


for scale in noise_scales:

    probability = (
        base_noise *
        scale
    )

    distribution = (
        apply_measurement_error(
            ideal_distribution,
            probability
        )
    )

    noise_distributions.append(
        distribution
    )

    print(
        f"\nNoise scale = {scale}"
    )

    for state, p in zip(
        STATES,
        distribution
    ):

        print(
            f"|{state}> : "
            f"{p:.6f}"
        )


noise_distributions = np.array(
    noise_distributions
)


# ============================================================
# LINEAR EXTRAPOLATION TO ZERO NOISE
# ============================================================

zne_distribution = np.zeros(
    len(STATES)
)


for state_index in range(
    len(STATES)
):

    observed_values = (
        noise_distributions[
            :,
            state_index
        ]
    )

    coefficients = np.polyfit(
        noise_scales,
        observed_values,
        1
    )

    intercept = coefficients[1]

    zne_distribution[
        state_index
    ] = intercept


# Clip and normalize
zne_distribution = np.clip(
    zne_distribution,
    0,
    None
)

zne_distribution /= np.sum(
    zne_distribution
)


print("\nZNE ESTIMATE")

for state, probability in zip(
    STATES,
    zne_distribution
):

    print(
        f"|{state}> : "
        f"{probability:.6f}"
    )


# ============================================================
# ZNE METRICS
# ============================================================

zne_metrics = evaluate(
    zne_distribution
)


print("\nZNE METRICS")

print(
    f"MSE         : "
    f"{zne_metrics['MSE']:.8f}"
)

print(
    f"KL          : "
    f"{zne_metrics['KL']:.8f}"
)

print(
    f"JS          : "
    f"{zne_metrics['JS']:.8f}"
)

print(
    f"Wasserstein : "
    f"{zne_metrics['Wasserstein']:.8f}"
)


# ============================================================
# DISTRIBUTION PLOT
# ============================================================

x = np.arange(
    len(STATES)
)

width = 0.2

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    x - 1.5 * width,
    TARGET,
    width,
    label="Target"
)

plt.bar(
    x - 0.5 * width,
    ideal_distribution,
    width,
    label="Ideal"
)

plt.bar(
    x + 0.5 * width,
    observed_distribution,
    width,
    label="Observed"
)

plt.bar(
    x + 1.5 * width,
    mitigated_distribution,
    width,
    label="Mitigated"
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
    "Measurement Error Mitigation"
)

plt.legend()

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# ZNE PLOT
# ============================================================

plt.figure(
    figsize=(11, 6)
)

for state_index, state in enumerate(
    STATES
):

    values = (
        noise_distributions[
            :,
            state_index
        ]
    )

    plt.plot(
        noise_scales,
        values,
        marker="o",
        label=f"|{state}>"
    )

    plt.scatter(
        0,
        zne_distribution[
            state_index
        ],
        s=100
    )


plt.xlabel(
    "Noise Scale"
)

plt.ylabel(
    "Probability"
)

plt.title(
    "Zero-Noise Extrapolation"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("DAY 75 COMPLETE")
print("=" * 70)

print("""
Measurement-error mitigation:

Observed distribution
        ↓
Calibration matrix
        ↓
Matrix inversion
        ↓
Corrected distribution

Zero-noise extrapolation:

Run at several noise levels
        ↓
Measure observable/distribution
        ↓
Fit noise dependence
        ↓
Extrapolate to noise = 0

Important:
Error mitigation does NOT magically restore
the original quantum state.

It estimates a better answer using
additional computation and assumptions.
""")