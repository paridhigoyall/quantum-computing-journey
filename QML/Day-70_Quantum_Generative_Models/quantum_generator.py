import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_QUBITS = 2

EPOCHS = 100

LEARNING_RATE = 0.2


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

# Basis states:
#
# |00> -> 0.10
# |01> -> 0.20
# |10> -> 0.60
# |11> -> 0.10

target_distribution = np.array([
    0.10,
    0.20,
    0.60,
    0.10
])


# ============================================================
# QUANTUM GENERATOR
# ============================================================

def quantum_generator(theta):

    qc = QuantumCircuit(
        NUM_QUBITS
    )

    # --------------------------------------------------------
    # Trainable rotations
    # --------------------------------------------------------

    qc.ry(
        theta[0],
        0
    )

    qc.ry(
        theta[1],
        1
    )

    # --------------------------------------------------------
    # Entanglement
    # --------------------------------------------------------

    qc.cx(
        0,
        1
    )

    return qc


# ============================================================
# GENERATED PROBABILITIES
# ============================================================

def generated_distribution(theta):

    qc = quantum_generator(
        theta
    )

    state = Statevector.from_instruction(
        qc
    )

    probabilities = np.abs(
        state.data
    ) ** 2

    return probabilities


# ============================================================
# LOSS
# ============================================================

def distribution_loss(
    theta
):

    generated = generated_distribution(
        theta
    )

    return np.mean(
        (
            generated
            - target_distribution
        ) ** 2
    )


# ============================================================
# PARAMETER-SHIFT GRADIENT
# ============================================================

def parameter_shift_gradient(
    theta,
    parameter_index
):

    shift = np.pi / 2

    theta_plus = theta.copy()

    theta_minus = theta.copy()

    theta_plus[
        parameter_index
    ] += shift

    theta_minus[
        parameter_index
    ] -= shift

    loss_plus = distribution_loss(
        theta_plus
    )

    loss_minus = distribution_loss(
        theta_minus
    )

    return (
        loss_plus
        - loss_minus
    ) / 2


# ============================================================
# FULL GRADIENT
# ============================================================

def compute_gradients(
    theta
):

    gradients = np.zeros_like(
        theta
    )

    for i in range(
        len(theta)
    ):

        gradients[i] = (
            parameter_shift_gradient(
                theta,
                i
            )
        )

    return gradients


# ============================================================
# TRAINING
# ============================================================

def train():

    rng = np.random.default_rng(
        SEED
    )

    # Random initialization
    theta = rng.uniform(
        -0.5,
        0.5,
        2
    )

    loss_history = []

    print("\n" + "=" * 65)
    print("QUANTUM GENERATIVE MODEL")
    print("=" * 65)

    for epoch in range(
        EPOCHS
    ):

        # ----------------------------------------------------
        # Current loss
        # ----------------------------------------------------

        loss = distribution_loss(
            theta
        )

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        gradients = compute_gradients(
            theta
        )

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        theta -= (
            LEARNING_RATE
            * gradients
        )

        loss_history.append(
            loss
        )

        if (
            epoch % 10 == 0
            or epoch == EPOCHS - 1
        ):

            print(
                f"Epoch {epoch + 1:03d} | "
                f"Loss = {loss:.8f} | "
                f"Theta = {theta}"
            )

    return (
        theta,
        loss_history
    )


# ============================================================
# TRAIN
# ============================================================

theta_final, loss_history = train()


# ============================================================
# FINAL DISTRIBUTION
# ============================================================

generated = generated_distribution(
    theta_final
)

print("\n" + "=" * 65)
print("FINAL DISTRIBUTION")
print("=" * 65)

states = [
    "00",
    "01",
    "10",
    "11"
]

print(
    f"{'State':<10}"
    f"{'Target':<15}"
    f"{'Generated':<15}"
)

print("-" * 40)

for state, target, prediction in zip(
    states,
    target_distribution,
    generated
):

    print(
        f"{state:<10}"
        f"{target:<15.4f}"
        f"{prediction:<15.4f}"
    )


# ============================================================
# FINAL LOSS
# ============================================================

final_loss = distribution_loss(
    theta_final
)

print(
    f"\nFinal MSE Loss: "
    f"{final_loss:.8f}"
)


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    loss_history
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE Distribution Loss"
)

plt.title(
    "Quantum Generator Training"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# DISTRIBUTION COMPARISON
# ============================================================

x = np.arange(
    len(states)
)

width = 0.35

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    x - width / 2,
    target_distribution,
    width,
    label="Target"
)

plt.bar(
    x + width / 2,
    generated,
    width,
    label="Generated"
)

plt.xticks(
    x,
    states
)

plt.xlabel(
    "Computational Basis State"
)

plt.ylabel(
    "Probability"
)

plt.title(
    "Target vs Generated Quantum Distribution"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# CIRCUIT
# ============================================================

print("\n" + "=" * 65)
print("TRAINED QUANTUM GENERATOR")
print("=" * 65)

print(
    quantum_generator(
        theta_final
    )
)