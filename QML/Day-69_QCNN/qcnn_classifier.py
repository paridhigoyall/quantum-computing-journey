import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_QUBITS = 4

EPOCHS = 25

LEARNING_RATE = 0.25


# ============================================================
# 1. DATASET
# ============================================================

X, y = make_moons(
    n_samples=120,
    noise=0.12,
    random_state=SEED
)

# Four features are needed because we use four qubits.
#
# make_moons gives us only two features.
# We therefore duplicate the two features:
#
# [x0, x1]
#     ↓
# [x0, x1, x0, x1]

X = np.column_stack([
    X[:, 0],
    X[:, 1],
    X[:, 0],
    X[:, 1]
])

# Scale to rotation angles
scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=SEED,
    stratify=y
)


# ============================================================
# 2. OBSERVABLE
# ============================================================

# Final prediction is made using qubit 0.

observable = SparsePauliOp.from_list([
    ("ZIII", 1.0)
])


# ============================================================
# 3. TWO-QUBIT CONVOLUTION
# ============================================================

def convolution_block(
    qc,
    q1,
    q2,
    theta
):
    """
    Two-qubit parameterized convolution.

    The same theta values can be reused
    across multiple pairs.

    This is analogous to parameter sharing
    in classical convolutional networks.
    """

    qc.ry(theta[0], q1)
    qc.ry(theta[1], q2)

    qc.cx(q1, q2)

    qc.ry(theta[2], q1)
    qc.ry(theta[3], q2)

    qc.cx(q2, q1)


# ============================================================
# 4. POOLING BLOCK
# ============================================================

def pooling_block(
    qc,
    source,
    target
):
    """
    Quantum pooling.

    Information from source is coupled
    into target, after which source is
    no longer used in subsequent layers.
    """

    qc.cx(source, target)

    qc.ry(
        np.pi / 4,
        target
    )


# ============================================================
# 5. QCNN CIRCUIT
# ============================================================

def qcnn_circuit(
    x,
    theta
):

    qc = QuantumCircuit(
        NUM_QUBITS
    )

    # --------------------------------------------------------
    # DATA ENCODING
    # --------------------------------------------------------

    for qubit in range(NUM_QUBITS):

        qc.ry(
            x[qubit],
            qubit
        )

    # --------------------------------------------------------
    # FIRST CONVOLUTION
    #
    # Pairs:
    # (0,1)
    # (2,3)
    #
    # Same parameters are shared.
    # --------------------------------------------------------

    convolution_block(
        qc,
        0,
        1,
        theta[0:4]
    )

    convolution_block(
        qc,
        2,
        3,
        theta[0:4]
    )

    # --------------------------------------------------------
    # FIRST POOLING
    #
    # q1 -> q0
    # q3 -> q2
    # --------------------------------------------------------

    pooling_block(
        qc,
        1,
        0
    )

    pooling_block(
        qc,
        3,
        2
    )

    # --------------------------------------------------------
    # SECOND CONVOLUTION
    #
    # Only q0 and q2 remain active.
    # --------------------------------------------------------

    convolution_block(
        qc,
        0,
        2,
        theta[4:8]
    )

    # --------------------------------------------------------
    # SECOND POOLING
    #
    # q2 -> q0
    # --------------------------------------------------------

    pooling_block(
        qc,
        2,
        0
    )

    return qc


# ============================================================
# 6. FORWARD PASS
# ============================================================

def predict_single(
    x,
    theta
):

    qc = qcnn_circuit(
        x,
        theta
    )

    state = Statevector.from_instruction(
        qc
    )

    expectation = state.expectation_value(
        observable
    )

    expectation = float(
        np.real(expectation)
    )

    # Convert [-1,1] to [0,1]

    probability = (
        expectation + 1
    ) / 2

    return np.clip(
        probability,
        1e-8,
        1 - 1e-8
    )


# ============================================================
# 7. BATCH PREDICTION
# ============================================================

def predict(
    X,
    theta
):

    return np.array([
        predict_single(
            x,
            theta
        )
        for x in X
    ])


# ============================================================
# 8. LOSS
# ============================================================

def binary_cross_entropy(
    y_true,
    y_pred
):

    y_pred = np.clip(
        y_pred,
        1e-8,
        1 - 1e-8
    )

    return -np.mean(
        y_true * np.log(y_pred)
        +
        (1 - y_true)
        * np.log(1 - y_pred)
    )


# ============================================================
# 9. PARAMETER-SHIFT GRADIENT
# ============================================================

def parameter_shift_gradient(
    X,
    y,
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

    loss_plus = binary_cross_entropy(
        y,
        predict(
            X,
            theta_plus
        )
    )

    loss_minus = binary_cross_entropy(
        y,
        predict(
            X,
            theta_minus
        )
    )

    return (
        loss_plus - loss_minus
    ) / 2


# ============================================================
# 10. FULL GRADIENT
# ============================================================

def compute_gradients(
    X,
    y,
    theta
):

    gradients = np.zeros_like(theta)

    for i in range(
        len(theta)
    ):

        gradients[i] = (
            parameter_shift_gradient(
                X,
                y,
                theta,
                i
            )
        )

    return gradients


# ============================================================
# 11. ACCURACY
# ============================================================

def accuracy(
    X,
    y,
    theta
):

    probabilities = predict(
        X,
        theta
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return np.mean(
        predictions == y
    )


# ============================================================
# 12. TRAINING
# ============================================================

def train():

    # --------------------------------------------------------
    # 8 trainable parameters
    #
    # First convolution:
    # theta[0:4]
    #
    # Second convolution:
    # theta[4:8]
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    theta = rng.uniform(
        -0.2,
        0.2,
        8
    )

    loss_history = []

    gradient_history = []

    print("\n" + "=" * 70)
    print("QCNN TRAINING")
    print("=" * 70)

    for epoch in range(
        EPOCHS
    ):

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        predictions = predict(
            X_train,
            theta
        )

        loss = binary_cross_entropy(
            y_train,
            predictions
        )

        # ----------------------------------------------------
        # Gradients
        # ----------------------------------------------------

        gradients = compute_gradients(
            X_train,
            y_train,
            theta
        )

        gradient_norm = np.linalg.norm(
            gradients
        )

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        theta -= (
            LEARNING_RATE
            * gradients
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        train_acc = accuracy(
            X_train,
            y_train,
            theta
        )

        test_acc = accuracy(
            X_test,
            y_test,
            theta
        )

        loss_history.append(
            loss
        )

        gradient_history.append(
            gradient_norm
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {loss:.4f} | "
            f"GradNorm: {gradient_norm:.6f} | "
            f"Train: {train_acc:.3f} | "
            f"Test: {test_acc:.3f}"
        )

    return (
        theta,
        loss_history,
        gradient_history
    )


# ============================================================
# 13. TRAIN
# ============================================================

theta_final, loss_history, gradient_history = train()


# ============================================================
# 14. FINAL RESULTS
# ============================================================

train_accuracy = accuracy(
    X_train,
    y_train,
    theta_final
)

test_accuracy = accuracy(
    X_test,
    y_test,
    theta_final
)

print("\n" + "=" * 70)
print("FINAL QCNN RESULTS")
print("=" * 70)

print(
    f"Train Accuracy : "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy  : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nTrainable parameters:",
    len(theta_final)
)

print(
    "\nLearned parameters:"
)

print(theta_final)


# ============================================================
# 15. LOSS CURVE
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    loss_history,
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Binary Cross Entropy"
)

plt.title(
    "QCNN Training Loss"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 16. GRADIENT CURVE
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    gradient_history,
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Gradient Norm"
)

plt.title(
    "QCNN Gradient Norm"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 17. DISPLAY CIRCUIT
# ============================================================

print("\n" + "=" * 70)
print("QCNN CIRCUIT")
print("=" * 70)

print(
    qcnn_circuit(
        X_train[0],
        theta_final
    )
)