import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPOCHS = 150

LEARNING_RATE_G = 0.15

LEARNING_RATE_D = 0.5

EPSILON = 1e-8


# ============================================================
# REAL DATA DISTRIBUTION
# ============================================================

# States:
#
# 00 -> 0.10
# 01 -> 0.20
# 10 -> 0.60
# 11 -> 0.10

REAL_PROBABILITIES = np.array([
    0.10,
    0.20,
    0.60,
    0.10
])


STATES = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# ============================================================
# QUANTUM GENERATOR
# ============================================================

def quantum_generator(theta):

    qc = QuantumCircuit(2)

    # Trainable rotations
    qc.ry(
        theta[0],
        0
    )

    qc.ry(
        theta[1],
        1
    )

    # Entanglement
    qc.cx(
        0,
        1
    )

    return qc


# ============================================================
# GENERATOR DISTRIBUTION
# ============================================================

def generator_distribution(theta):

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
# SAMPLE FROM QUANTUM GENERATOR
# ============================================================

def generate_samples(
    theta,
    num_samples,
    rng
):

    probabilities = generator_distribution(
        theta
    )

    indices = rng.choice(
        4,
        size=num_samples,
        p=probabilities
    )

    return STATES[
        indices
    ]


# ============================================================
# LOGISTIC DISCRIMINATOR
# ============================================================

class Discriminator:

    def __init__(self):

        self.weights = np.zeros(
            2
        )

        self.bias = 0.0

    # --------------------------------------------------------
    # Sigmoid
    # --------------------------------------------------------

    def sigmoid(self, z):

        z = np.clip(
            z,
            -50,
            50
        )

        return 1 / (
            1 + np.exp(-z)
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def predict(self, X):

        logits = (
            X @ self.weights
            + self.bias
        )

        return self.sigmoid(
            logits
        )

    # --------------------------------------------------------
    # Binary cross entropy
    # --------------------------------------------------------

    def loss(
        self,
        X,
        y
    ):

        predictions = np.clip(
            self.predict(X),
            EPSILON,
            1 - EPSILON
        )

        return -np.mean(
            y * np.log(predictions)
            +
            (1 - y)
            * np.log(1 - predictions)
        )

    # --------------------------------------------------------
    # Gradient descent
    # --------------------------------------------------------

    def train_step(
        self,
        X,
        y,
        learning_rate
    ):

        predictions = self.predict(
            X
        )

        error = (
            predictions - y
        )

        grad_w = (
            X.T @ error
        ) / len(X)

        grad_b = np.mean(
            error
        )

        self.weights -= (
            learning_rate
            * grad_w
        )

        self.bias -= (
            learning_rate
            * grad_b
        )


# ============================================================
# GENERATOR LOSS
# ============================================================

def generator_loss(
    theta,
    discriminator
):

    probabilities = generator_distribution(
        theta
    )

    discriminator_outputs = (
        discriminator.predict(
            STATES
        )
    )

    # Generator wants D(fake) -> 1
    #
    # Weighted by probability produced
    # by the quantum generator.

    loss = -np.sum(
        probabilities
        * np.log(
            np.clip(
                discriminator_outputs,
                EPSILON,
                1 - EPSILON
            )
        )
    )

    return loss


# ============================================================
# PARAMETER-SHIFT GRADIENT
# ============================================================

def generator_gradient(
    theta,
    discriminator,
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

    loss_plus = generator_loss(
        theta_plus,
        discriminator
    )

    loss_minus = generator_loss(
        theta_minus,
        discriminator
    )

    return (
        loss_plus
        - loss_minus
    ) / 2


# ============================================================
# FULL GENERATOR GRADIENT
# ============================================================

def compute_generator_gradients(
    theta,
    discriminator
):

    gradients = np.zeros_like(
        theta
    )

    for i in range(
        len(theta)
    ):

        gradients[i] = (
            generator_gradient(
                theta,
                discriminator,
                i
            )
        )

    return gradients


# ============================================================
# GENERATOR DISTRIBUTION LOSS
# ============================================================

def distribution_distance(
    generated
):

    return np.mean(
        (
            generated
            - REAL_PROBABILITIES
        ) ** 2
    )


# ============================================================
# TRAIN QGAN
# ============================================================

def train_qgan():

    rng = np.random.default_rng(
        SEED
    )

    # --------------------------------------------------------
    # Initialize quantum generator
    # --------------------------------------------------------

    theta = rng.uniform(
        -0.5,
        0.5,
        2
    )

    # --------------------------------------------------------
    # Initialize discriminator
    # --------------------------------------------------------

    discriminator = Discriminator()

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    generator_losses = []

    discriminator_losses = []

    distribution_losses = []

    print("\n" + "=" * 70)
    print("QUANTUM GENERATIVE ADVERSARIAL NETWORK")
    print("=" * 70)

    for epoch in range(
        EPOCHS
    ):

        # ====================================================
        # STEP 1: GENERATE FAKE DATA
        # ====================================================

        fake_samples = generate_samples(
            theta,
            100,
            rng
        )

        # ====================================================
        # STEP 2: CREATE REAL DATA
        # ====================================================

        real_indices = rng.choice(
            4,
            size=100,
            p=REAL_PROBABILITIES
        )

        real_samples = STATES[
            real_indices
        ]

        # ====================================================
        # STEP 3: TRAIN DISCRIMINATOR
        # ====================================================

        X_disc = np.vstack([
            real_samples,
            fake_samples
        ])

        y_disc = np.concatenate([
            np.ones(len(real_samples)),
            np.zeros(len(fake_samples))
        ])

        discriminator.train_step(
            X_disc,
            y_disc,
            LEARNING_RATE_D
        )

        d_loss = discriminator.loss(
            X_disc,
            y_disc
        )

        # ====================================================
        # STEP 4: TRAIN GENERATOR
        # ====================================================

        gradients = (
            compute_generator_gradients(
                theta,
                discriminator
            )
        )

        theta -= (
            LEARNING_RATE_G
            * gradients
        )

        g_loss = generator_loss(
            theta,
            discriminator
        )

        # ====================================================
        # STEP 5: METRICS
        # ====================================================

        generated_distribution = (
            generator_distribution(theta)
        )

        dist_loss = distribution_distance(
            generated_distribution
        )

        generator_losses.append(
            g_loss
        )

        discriminator_losses.append(
            d_loss
        )

        distribution_losses.append(
            dist_loss
        )

        # ====================================================
        # PRINT
        # ====================================================

        if (
            epoch % 10 == 0
            or epoch == EPOCHS - 1
        ):

            print(
                f"Epoch {epoch + 1:03d} | "
                f"D Loss: {d_loss:.4f} | "
                f"G Loss: {g_loss:.4f} | "
                f"Distribution MSE: {dist_loss:.6f}"
            )

    return (
        theta,
        discriminator,
        generator_losses,
        discriminator_losses,
        distribution_losses
    )


# ============================================================
# TRAIN
# ============================================================

(
    theta_final,
    discriminator,
    generator_losses,
    discriminator_losses,
    distribution_losses
) = train_qgan()


# ============================================================
# FINAL DISTRIBUTION
# ============================================================

generated_distribution = (
    generator_distribution(
        theta_final
    )
)


print("\n" + "=" * 70)
print("FINAL QGAN DISTRIBUTION")
print("=" * 70)

print(
    f"{'State':<10}"
    f"{'Real':<15}"
    f"{'Generated':<15}"
)

print("-" * 40)

for state, real, generated in zip(
    ["00", "01", "10", "11"],
    REAL_PROBABILITIES,
    generated_distribution
):

    print(
        f"{state:<10}"
        f"{real:<15.4f}"
        f"{generated:<15.4f}"
    )


# ============================================================
# FINAL DISTRIBUTION ERROR
# ============================================================

final_distance = distribution_distance(
    generated_distribution
)

print(
    f"\nFinal distribution MSE: "
    f"{final_distance:.8f}"
)


# ============================================================
# GENERATOR PARAMETERS
# ============================================================

print(
    "\nLearned generator parameters:"
)

print(theta_final)


# ============================================================
# DISCRIMINATOR PARAMETERS
# ============================================================

print(
    "\nLearned discriminator weights:"
)

print(discriminator.weights)

print(
    "Discriminator bias:",
    discriminator.bias
)


# ============================================================
# PLOT 1: LOSSES
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    generator_losses,
    label="Generator Loss"
)

plt.plot(
    discriminator_losses,
    label="Discriminator Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "QGAN Adversarial Training"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2: DISTRIBUTION DISTANCE
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    distribution_losses,
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Distribution MSE"
)

plt.title(
    "Generator Distance from Real Distribution"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3: FINAL DISTRIBUTION
# ============================================================

states = [
    "00",
    "01",
    "10",
    "11"
]

x = np.arange(
    len(states)
)

width = 0.35

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    x - width / 2,
    REAL_PROBABILITIES,
    width,
    label="Real"
)

plt.bar(
    x + width / 2,
    generated_distribution,
    width,
    label="Generated"
)

plt.xticks(
    x,
    states
)

plt.xlabel(
    "Quantum State"
)

plt.ylabel(
    "Probability"
)

plt.title(
    "QGAN: Real vs Generated Distribution"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# FINAL QUANTUM CIRCUIT
# ============================================================

print("\n" + "=" * 70)
print("FINAL QUANTUM GENERATOR")
print("=" * 70)

print(
    quantum_generator(
        theta_final
    )
)