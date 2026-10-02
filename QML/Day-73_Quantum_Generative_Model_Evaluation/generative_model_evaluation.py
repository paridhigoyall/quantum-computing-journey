import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


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
# GENERATOR ARCHITECTURES
# ============================================================

def hardware_efficient_circuit(theta):

    qc = QuantumCircuit(N_QUBITS)

    index = 0

    for _ in range(LAYERS):

        for q in range(N_QUBITS):

            qc.ry(theta[index], q)
            index += 1

            qc.rz(theta[index], q)
            index += 1

        for q in range(N_QUBITS - 1):

            qc.cx(q, q + 1)

    return qc


def entangling_circuit(theta):

    qc = QuantumCircuit(N_QUBITS)

    index = 0

    for _ in range(LAYERS):

        for q in range(N_QUBITS):

            qc.ry(theta[index], q)
            index += 1

            qc.rz(theta[index], q)
            index += 1

        qc.cx(0, 1)
        qc.cx(1, 0)

    return qc


def data_reuploading_circuit(theta):

    qc = QuantumCircuit(N_QUBITS)

    latent = np.array([
        np.pi / 3,
        np.pi / 5
    ])

    index = 0

    for _ in range(LAYERS):

        # Data encoding
        for q in range(N_QUBITS):

            qc.ry(latent[q], q)

        # Trainable layer
        for q in range(N_QUBITS):

            qc.ry(theta[index], q)
            index += 1

            qc.rz(theta[index], q)
            index += 1

        # Entanglement
        qc.cx(0, 1)

    return qc


# ============================================================
# QUANTUM DISTRIBUTION
# ============================================================

def get_statevector(circuit_function, theta):

    qc = circuit_function(theta)

    return Statevector.from_instruction(qc)


def get_distribution(statevector):

    return np.abs(statevector.data) ** 2


# ============================================================
# TRAINING LOSS
# ============================================================

def mse_loss(P, Q):

    return np.mean((P - Q) ** 2)


# ============================================================
# PARAMETER-SHIFT GRADIENT
# ============================================================

def parameter_shift_gradient(
    circuit_function,
    theta,
    target
):

    gradients = np.zeros_like(theta)

    shift = np.pi / 2

    for i in range(len(theta)):

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[i] += shift
        theta_minus[i] -= shift

        state_plus = get_statevector(
            circuit_function,
            theta_plus
        )

        state_minus = get_statevector(
            circuit_function,
            theta_minus
        )

        P_plus = get_distribution(state_plus)
        P_minus = get_distribution(state_minus)

        loss_plus = mse_loss(
            P_plus,
            target
        )

        loss_minus = mse_loss(
            P_minus,
            target
        )

        gradients[i] = (
            loss_plus - loss_minus
        ) / 2

    return gradients


# ============================================================
# TRAIN GENERATOR
# ============================================================

def train_generator(
    circuit_function,
    initial_theta,
    target
):

    theta = initial_theta.copy()

    loss_history = []

    for iteration in range(ITERATIONS):

        state = get_statevector(
            circuit_function,
            theta
        )

        probabilities = get_distribution(
            state
        )

        loss = mse_loss(
            probabilities,
            target
        )

        loss_history.append(loss)

        gradients = parameter_shift_gradient(
            circuit_function,
            theta,
            target
        )

        theta -= LEARNING_RATE * gradients

    final_state = get_statevector(
        circuit_function,
        theta
    )

    final_distribution = get_distribution(
        final_state
    )

    return (
        theta,
        final_state,
        final_distribution,
        loss_history
    )


# ============================================================
# METRIC 1: MSE
# ============================================================

def calculate_mse(P, Q):

    return np.mean(
        (P - Q) ** 2
    )


# ============================================================
# METRIC 2: KL DIVERGENCE
# ============================================================

def calculate_kl(P, Q):

    P_safe = np.clip(
        P,
        EPSILON,
        None
    )

    Q_safe = np.clip(
        Q,
        EPSILON,
        None
    )

    return np.sum(
        P_safe *
        np.log(P_safe / Q_safe)
    )


# ============================================================
# METRIC 3: JS DIVERGENCE
# ============================================================

def calculate_js(P, Q):

    P_safe = np.clip(
        P,
        EPSILON,
        None
    )

    Q_safe = np.clip(
        Q,
        EPSILON,
        None
    )

    M = (P_safe + Q_safe) / 2

    kl_pm = np.sum(
        P_safe *
        np.log(P_safe / M)
    )

    kl_qm = np.sum(
        Q_safe *
        np.log(Q_safe / M)
    )

    return 0.5 * (
        kl_pm + kl_qm
    )


# ============================================================
# METRIC 4: WASSERSTEIN DISTANCE
# ============================================================

def calculate_wasserstein(P, Q):

    """
    For equally spaced 1D states,
    Wasserstein-1 distance can be computed
    using cumulative distributions.
    """

    cumulative_P = np.cumsum(P)
    cumulative_Q = np.cumsum(Q)

    return np.sum(
        np.abs(
            cumulative_P -
            cumulative_Q
        )
    )


# ============================================================
# METRIC 5: QUANTUM FIDELITY
# ============================================================

def calculate_fidelity(
    target_state,
    generated_state
):

    overlap = np.vdot(
        target_state.data,
        generated_state.data
    )

    fidelity = np.abs(
        overlap
    ) ** 2

    return float(
        np.real(fidelity)
    )


# ============================================================
# TARGET STATE
# ============================================================

def create_target_state():

    amplitudes = np.sqrt(TARGET)

    return Statevector(
        amplitudes
    )


# ============================================================
# METRIC 6: MODE COVERAGE
# ============================================================

def calculate_mode_coverage(
    target,
    generated,
    threshold=0.10
):

    important_modes = (
        target >= threshold
    )

    learned_modes = (
        generated >= threshold
    )

    target_mode_count = np.sum(
        important_modes
    )

    if target_mode_count == 0:

        return 0.0

    covered_modes = np.sum(
        important_modes &
        learned_modes
    )

    return covered_modes / target_mode_count


# ============================================================
# INITIAL PARAMETERS
# ============================================================

NUM_PARAMETERS = (
    LAYERS *
    N_QUBITS *
    2
)

initial_theta = np.random.uniform(
    -np.pi,
    np.pi,
    NUM_PARAMETERS
)


# ============================================================
# TRAIN ALL THREE GENERATORS
# ============================================================

print("=" * 70)
print("TRAINING QUANTUM GENERATORS")
print("=" * 70)


theta_hardware, state_hardware, dist_hardware, loss_hardware = (
    train_generator(
        hardware_efficient_circuit,
        initial_theta,
        TARGET
    )
)


theta_entangling, state_entangling, dist_entangling, loss_entangling = (
    train_generator(
        entangling_circuit,
        initial_theta,
        TARGET
    )
)


theta_reupload, state_reupload, dist_reupload, loss_reupload = (
    train_generator(
        data_reuploading_circuit,
        initial_theta,
        TARGET
    )
)


# ============================================================
# TARGET STATE
# ============================================================

target_state = create_target_state()


# ============================================================
# EVALUATE MODELS
# ============================================================

models = {

    "Hardware Efficient": (
        dist_hardware,
        state_hardware
    ),

    "Strongly Entangling": (
        dist_entangling,
        state_entangling
    ),

    "Data Re-uploading": (
        dist_reupload,
        state_reupload
    )
}


results = {}


for name, (
    distribution,
    state
) in models.items():

    mse = calculate_mse(
        TARGET,
        distribution
    )

    kl = calculate_kl(
        TARGET,
        distribution
    )

    js = calculate_js(
        TARGET,
        distribution
    )

    wasserstein = calculate_wasserstein(
        TARGET,
        distribution
    )

    fidelity = calculate_fidelity(
        target_state,
        state
    )

    coverage = calculate_mode_coverage(
        TARGET,
        distribution
    )

    results[name] = {

        "MSE": mse,

        "KL": kl,

        "JS": js,

        "Wasserstein": wasserstein,

        "Fidelity": fidelity,

        "Mode Coverage": coverage
    }


# ============================================================
# PRINT TARGET
# ============================================================

print("\nTARGET DISTRIBUTION")
print("=" * 70)

for state, probability in zip(
    STATES,
    TARGET
):

    print(
        f"|{state}> : "
        f"{probability:.6f}"
    )


# ============================================================
# PRINT GENERATED DISTRIBUTIONS
# ============================================================

for name, (
    distribution,
    _
) in models.items():

    print("\n")
    print(name.upper())
    print("=" * 70)

    for state, probability in zip(
        STATES,
        distribution
    ):

        print(
            f"|{state}> : "
            f"{probability:.6f}"
        )


# ============================================================
# PRINT METRICS
# ============================================================

print("\n\nEVALUATION RESULTS")
print("=" * 100)

print(
    f"{'Model':<25}"
    f"{'MSE':<15}"
    f"{'KL':<15}"
    f"{'JS':<15}"
    f"{'Wasserstein':<18}"
    f"{'Fidelity':<15}"
    f"{'Coverage':<15}"
)

print("-" * 100)


for name, metrics in results.items():

    print(
        f"{name:<25}"
        f"{metrics['MSE']:<15.6f}"
        f"{metrics['KL']:<15.6f}"
        f"{metrics['JS']:<15.6f}"
        f"{metrics['Wasserstein']:<18.6f}"
        f"{metrics['Fidelity']:<15.6f}"
        f"{metrics['Mode Coverage']:<15.2%}"
    )


# ============================================================
# PLOT 1: TRAINING LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    loss_hardware,
    label="Hardware Efficient"
)

plt.plot(
    loss_entangling,
    label="Strongly Entangling"
)

plt.plot(
    loss_reupload,
    label="Data Re-uploading"
)

plt.xlabel("Iteration")

plt.ylabel("MSE Loss")

plt.title(
    "Quantum Generator Training"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# PLOT 2: DISTRIBUTION COMPARISON
# ============================================================

x = np.arange(
    len(STATES)
)

width = 0.2

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - 1.5 * width,
    TARGET,
    width,
    label="Target"
)

plt.bar(
    x - 0.5 * width,
    dist_hardware,
    width,
    label="Hardware Efficient"
)

plt.bar(
    x + 0.5 * width,
    dist_entangling,
    width,
    label="Strongly Entangling"
)

plt.bar(
    x + 1.5 * width,
    dist_reupload,
    width,
    label="Data Re-uploading"
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
    "Target vs Generated Distributions"
)

plt.legend()

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# PLOT 3: METRIC COMPARISON
# ============================================================

metric_names = [
    "MSE",
    "KL",
    "JS",
    "Wasserstein"
]

model_names = list(
    results.keys()
)

x = np.arange(
    len(metric_names)
)

width = 0.25

plt.figure(
    figsize=(12, 6)
)

for i, model in enumerate(
    model_names
):

    values = [
        results[model][metric]
        for metric in metric_names
    ]

    plt.bar(
        x + i * width,
        values,
        width,
        label=model
    )

plt.xticks(
    x + width,
    metric_names
)

plt.ylabel(
    "Metric Value"
)

plt.title(
    "Generative Model Evaluation Metrics"
)

plt.legend()

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
MSE:
Measures element-wise probability error.

KL:
Measures directional information divergence.

JS:
Measures symmetric distribution divergence.

Wasserstein:
Measures probability transport distance.

Fidelity:
Measures similarity between the complete
target and generated quantum states.

Mode Coverage:
Measures how many important target modes
were successfully represented.

Important:
Do NOT select a generator using only one metric.

A strong generative model should ideally show:

- low MSE
- low KL
- low JS
- low Wasserstein distance
- high fidelity
- high mode coverage
""")