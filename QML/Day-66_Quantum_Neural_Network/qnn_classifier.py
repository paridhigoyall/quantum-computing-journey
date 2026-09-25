import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ============================================================
# 1. DATASET
# ============================================================

X, y = make_moons(
    n_samples=100,
    noise=0.12,
    random_state=42
)

# Scale features to [0, pi]
scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# 2. OBSERVABLE
# ============================================================

observable = SparsePauliOp.from_list([
    ("ZI", 1.0)
])


# ============================================================
# 3. QNN CIRCUIT
# ============================================================

def create_qnn(x, theta, num_layers=2):

    qc = QuantumCircuit(2)

    parameter_index = 0

    # --------------------------------------------------------
    # Data encoding
    # --------------------------------------------------------

    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    # --------------------------------------------------------
    # Trainable quantum layers
    # --------------------------------------------------------

    for layer in range(num_layers):

        qc.ry(
            theta[parameter_index],
            0
        )

        parameter_index += 1

        qc.ry(
            theta[parameter_index],
            1
        )

        parameter_index += 1

        # Entanglement
        qc.cx(0, 1)

    return qc


# ============================================================
# 4. FORWARD PASS
# ============================================================

def predict_single(x, theta):

    qc = create_qnn(
        x,
        theta
    )

    state = Statevector.from_instruction(qc)

    expectation = state.expectation_value(
        observable
    )

    expectation = float(
        np.real(expectation)
    )

    # Convert [-1, 1] to [0, 1]
    probability = (
        expectation + 1
    ) / 2

    return np.clip(
        probability,
        1e-8,
        1 - 1e-8
    )


# ============================================================
# 5. PREDICTIONS
# ============================================================

def predict(X, theta):

    return np.array([
        predict_single(x, theta)
        for x in X
    ])


# ============================================================
# 6. BINARY CROSS ENTROPY
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

    loss = -np.mean(
        y_true * np.log(y_pred)
        +
        (1 - y_true)
        * np.log(1 - y_pred)
    )

    return loss


# ============================================================
# 7. PARAMETER-SHIFT GRADIENT
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

    predictions_plus = predict(
        X,
        theta_plus
    )

    predictions_minus = predict(
        X,
        theta_minus
    )

    loss_plus = binary_cross_entropy(
        y,
        predictions_plus
    )

    loss_minus = binary_cross_entropy(
        y,
        predictions_minus
    )

    gradient = (
        loss_plus - loss_minus
    ) / 2

    return gradient


# ============================================================
# 8. FULL GRADIENT
# ============================================================

def compute_gradients(
    X,
    y,
    theta
):

    gradients = np.zeros_like(theta)

    for i in range(len(theta)):

        gradients[i] = parameter_shift_gradient(
            X,
            y,
            theta,
            i
        )

    return gradients


# ============================================================
# 9. ACCURACY
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
# 10. TRAINING
# ============================================================

def train():

    num_layers = 2

    num_parameters = (
        2 * num_layers
    )

    # Random initialization
    rng = np.random.default_rng(42)

    theta = rng.uniform(
        -0.2,
        0.2,
        num_parameters
    )

    learning_rate = 0.3

    epochs = 30

    loss_history = []

    print("\n" + "=" * 60)
    print("QUANTUM NEURAL NETWORK TRAINING")
    print("=" * 60)

    for epoch in range(epochs):

        # -----------------------------------------------
        # Forward pass
        # -----------------------------------------------

        predictions = predict(
            X_train,
            theta
        )

        loss = binary_cross_entropy(
            y_train,
            predictions
        )

        # -----------------------------------------------
        # Gradients
        # -----------------------------------------------

        gradients = compute_gradients(
            X_train,
            y_train,
            theta
        )

        # -----------------------------------------------
        # Parameter update
        # -----------------------------------------------

        theta -= (
            learning_rate
            * gradients
        )

        # -----------------------------------------------
        # Metrics
        # -----------------------------------------------

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

        loss_history.append(loss)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Test Acc: {test_acc:.3f}"
        )

    return theta, loss_history


# ============================================================
# 11. TRAIN MODEL
# ============================================================

theta_final, loss_history = train()


# ============================================================
# 12. FINAL RESULTS
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

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Training Accuracy : "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Testing Accuracy  : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nLearned parameters:"
)

print(theta_final)


# ============================================================
# 13. LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    loss_history,
    marker="o"
)

plt.xlabel("Epoch")

plt.ylabel("Binary Cross Entropy")

plt.title(
    "QNN Training Loss"
)

plt.grid(True)

plt.tight_layout()

plt.show()