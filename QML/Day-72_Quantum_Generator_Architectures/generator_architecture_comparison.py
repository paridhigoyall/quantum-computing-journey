import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

np.random.seed(SEED)

N_QUBITS = 2
N_PARAMS_PER_LAYER = 2 * N_QUBITS

LAYERS = 3
LEARNING_RATE = 0.15
ITERATIONS = 150

TARGET = np.array([
    0.10,   # |00>
    0.20,   # |01>
    0.60,   # |10>
    0.10    # |11>
])


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_distribution(qc):
    """
    Execute circuit using exact Statevector simulation
    and return computational-basis probabilities.
    """

    state = Statevector.from_instruction(qc)

    probabilities = np.abs(state.data) ** 2

    return probabilities


def mse_loss(probabilities, target):
    """
    Mean squared error between generated and target distribution.
    """

    return np.mean((probabilities - target) ** 2)


# ============================================================
# ARCHITECTURE 1
# HARDWARE-EFFICIENT GENERATOR
# ============================================================

def hardware_efficient_circuit(theta):
    """
    Hardware-efficient ansatz.

    Each layer:
        RY on every qubit
        RZ on every qubit
        CNOT chain
    """

    qc = QuantumCircuit(N_QUBITS)

    index = 0

    for layer in range(LAYERS):

        # Single-qubit rotations
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
# ARCHITECTURE 2
# STRONGLY ENTANGLING GENERATOR
# ============================================================

def entangling_circuit(theta):
    """
    Stronger entangling generator.

    Each layer:
        RY
        RZ
        CNOT 0 -> 1
        CNOT 1 -> 0
    """

    qc = QuantumCircuit(N_QUBITS)

    index = 0

    for layer in range(LAYERS):

        for q in range(N_QUBITS):

            qc.ry(theta[index], q)
            index += 1

            qc.rz(theta[index], q)
            index += 1

        # Bidirectional interaction
        qc.cx(0, 1)
        qc.cx(1, 0)

    return qc


# ============================================================
# ARCHITECTURE 3
# DATA RE-UPLOADING GENERATOR
# ============================================================

def data_reuploading_circuit(theta):
    """
    Data-reuploading style generator.

    We use fixed latent values and repeatedly inject
    them into the circuit between trainable layers.
    """

    qc = QuantumCircuit(N_QUBITS)

    latent = np.array([
        np.pi / 3,
        np.pi / 5
    ])

    index = 0

    for layer in range(LAYERS):

        # -------------------------
        # Data encoding
        # -------------------------

        for q in range(N_QUBITS):

            qc.ry(latent[q], q)

        # -------------------------
        # Trainable transformation
        # -------------------------

        for q in range(N_QUBITS):

            qc.ry(theta[index], q)
            index += 1

            qc.rz(theta[index], q)
            index += 1

        # -------------------------
        # Entanglement
        # -------------------------

        qc.cx(0, 1)

    return qc


# ============================================================
# PARAMETER SHIFT
# ============================================================

def parameter_shift_gradient(
    circuit_function,
    theta,
    target
):
    """
    Calculate gradient using parameter-shift rule.

    dL/dtheta_i =
        [L(theta_i + pi/2) -
         L(theta_i - pi/2)] / 2
    """

    gradients = np.zeros_like(theta)

    shift = np.pi / 2

    for i in range(len(theta)):

        theta_forward = theta.copy()
        theta_backward = theta.copy()

        theta_forward[i] += shift
        theta_backward[i] -= shift

        qc_forward = circuit_function(theta_forward)
        qc_backward = circuit_function(theta_backward)

        p_forward = get_distribution(qc_forward)
        p_backward = get_distribution(qc_backward)

        loss_forward = mse_loss(
            p_forward,
            target
        )

        loss_backward = mse_loss(
            p_backward,
            target
        )

        gradients[i] = (
            loss_forward - loss_backward
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

    gradient_history = []

    for iteration in range(ITERATIONS):

        # ----------------------------------
        # Forward pass
        # ----------------------------------

        qc = circuit_function(theta)

        probabilities = get_distribution(qc)

        loss = mse_loss(
            probabilities,
            target
        )

        loss_history.append(loss)

        # ----------------------------------
        # Compute gradient
        # ----------------------------------

        gradients = parameter_shift_gradient(
            circuit_function,
            theta,
            target
        )

        gradient_norm = np.linalg.norm(
            gradients
        )

        gradient_history.append(
            gradient_norm
        )

        # ----------------------------------
        # Gradient descent
        # ----------------------------------

        theta -= LEARNING_RATE * gradients

        # ----------------------------------
        # Logging
        # ----------------------------------

        if iteration % 25 == 0:

            print(
                f"Iteration {iteration:3d} | "
                f"Loss = {loss:.8f} | "
                f"Gradient = {gradient_norm:.6f}"
            )

    final_qc = circuit_function(theta)

    final_distribution = get_distribution(
        final_qc
    )

    return (
        theta,
        final_distribution,
        loss_history,
        gradient_history
    )


# ============================================================
# INITIAL PARAMETERS
# ============================================================

num_parameters = (
    LAYERS *
    N_PARAMS_PER_LAYER
)

initial_parameters = (
    np.random.uniform(
        -np.pi,
        np.pi,
        num_parameters
    )
)


# ============================================================
# TRAIN ALL ARCHITECTURES
# ============================================================

print("\n" + "=" * 60)
print("HARDWARE-EFFICIENT GENERATOR")
print("=" * 60)

theta_hardware = initial_parameters.copy()

(
    trained_hardware,
    distribution_hardware,
    loss_hardware,
    gradient_hardware
) = train_generator(
    hardware_efficient_circuit,
    theta_hardware,
    TARGET
)


print("\n" + "=" * 60)
print("STRONGLY ENTANGLING GENERATOR")
print("=" * 60)

theta_entangling = initial_parameters.copy()

(
    trained_entangling,
    distribution_entangling,
    loss_entangling,
    gradient_entangling
) = train_generator(
    entangling_circuit,
    theta_entangling,
    TARGET
)


print("\n" + "=" * 60)
print("DATA RE-UPLOADING GENERATOR")
print("=" * 60)

theta_reupload = initial_parameters.copy()

(
    trained_reupload,
    distribution_reupload,
    loss_reupload,
    gradient_reupload
) = train_generator(
    data_reuploading_circuit,
    theta_reupload,
    TARGET
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

states = [
    "00",
    "01",
    "10",
    "11"
]


print("\n\nFINAL DISTRIBUTIONS")
print("=" * 60)

print("\nTarget:")
for state, probability in zip(states, TARGET):
    print(
        f"|{state}> : {probability:.4f}"
    )


print("\nHardware Efficient:")
for state, probability in zip(
    states,
    distribution_hardware
):
    print(
        f"|{state}> : {probability:.4f}"
    )


print("\nStrongly Entangling:")
for state, probability in zip(
    states,
    distribution_entangling
):
    print(
        f"|{state}> : {probability:.4f}"
    )


print("\nData Re-uploading:")
for state, probability in zip(
    states,
    distribution_reupload
):
    print(
        f"|{state}> : {probability:.4f}"
    )


# ============================================================
# FINAL LOSSES
# ============================================================

final_loss_hardware = mse_loss(
    distribution_hardware,
    TARGET
)

final_loss_entangling = mse_loss(
    distribution_entangling,
    TARGET
)

final_loss_reupload = mse_loss(
    distribution_reupload,
    TARGET
)


print("\nFINAL LOSSES")
print("=" * 60)

print(
    f"Hardware Efficient : "
    f"{final_loss_hardware:.8f}"
)

print(
    f"Strongly Entangling: "
    f"{final_loss_entangling:.8f}"
)

print(
    f"Data Re-uploading  : "
    f"{final_loss_reupload:.8f}"
)


# ============================================================
# CIRCUIT VISUALIZATION
# ============================================================

print("\n\nHARDWARE-EFFICIENT CIRCUIT")
print("=" * 60)

print(
    hardware_efficient_circuit(
        trained_hardware
    ).draw("text")
)


print("\n\nSTRONGLY ENTANGLING CIRCUIT")
print("=" * 60)

print(
    entangling_circuit(
        trained_entangling
    ).draw("text")
)


print("\n\nDATA RE-UPLOADING CIRCUIT")
print("=" * 60)

print(
    data_reuploading_circuit(
        trained_reupload
    ).draw("text")
)


# ============================================================
# PLOT 1
# TRAINING LOSS
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
    "Quantum Generator Training Loss"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# PLOT 2
# FINAL DISTRIBUTION
# ============================================================

x = np.arange(len(states))

width = 0.2

plt.figure(figsize=(10, 6))

plt.bar(
    x - 1.5 * width,
    TARGET,
    width,
    label="Target"
)

plt.bar(
    x - 0.5 * width,
    distribution_hardware,
    width,
    label="Hardware Efficient"
)

plt.bar(
    x + 0.5 * width,
    distribution_entangling,
    width,
    label="Strongly Entangling"
)

plt.bar(
    x + 1.5 * width,
    distribution_reupload,
    width,
    label="Data Re-uploading"
)

plt.xticks(
    x,
    states
)

plt.xlabel("Quantum State")

plt.ylabel("Probability")

plt.title(
    "Target vs Generated Probability Distributions"
)

plt.legend()

plt.grid(axis="y")

plt.show()


# ============================================================
# PLOT 3
# GRADIENT NORMS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    gradient_hardware,
    label="Hardware Efficient"
)

plt.plot(
    gradient_entangling,
    label="Strongly Entangling"
)

plt.plot(
    gradient_reupload,
    label="Data Re-uploading"
)

plt.xlabel("Iteration")

plt.ylabel("Gradient Norm")

plt.title(
    "Gradient Behavior During Generator Training"
)

plt.legend()

plt.grid(True)

plt.show()