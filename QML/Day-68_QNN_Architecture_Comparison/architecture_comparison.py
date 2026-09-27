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

NUM_QUBITS = 2

EPOCHS = 25

LEARNING_RATE = 0.3

NUM_LAYERS = 3


# ============================================================
# 1. DATASET
# ============================================================

X, y = make_moons(
    n_samples=120,
    noise=0.12,
    random_state=SEED
)

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

observable = SparsePauliOp.from_list([
    ("ZI", 1.0)
])


# ============================================================
# 3. ARCHITECTURE A
#    SHALLOW QNN
# ============================================================

def shallow_qnn(x, theta):

    qc = QuantumCircuit(NUM_QUBITS)

    # Data encoding
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    # One trainable layer
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)

    # Entanglement
    qc.cx(0, 1)

    return qc


# ============================================================
# 4. ARCHITECTURE B
#    DEEP QNN
# ============================================================

def deep_qnn(x, theta):

    qc = QuantumCircuit(NUM_QUBITS)

    # Data encoding ONCE
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    parameter_index = 0

    for _ in range(NUM_LAYERS):

        # Trainable rotations
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
# 5. ARCHITECTURE C
#    DATA RE-UPLOADING QNN
# ============================================================

def reupload_qnn(x, theta):

    qc = QuantumCircuit(NUM_QUBITS)

    parameter_index = 0

    for _ in range(NUM_LAYERS):

        # -----------------------------------------------
        # Re-upload classical data
        # -----------------------------------------------

        qc.ry(x[0], 0)
        qc.ry(x[1], 1)

        # -----------------------------------------------
        # Trainable layer
        # -----------------------------------------------

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

        # -----------------------------------------------
        # Entanglement
        # -----------------------------------------------

        qc.cx(0, 1)

    return qc


# ============================================================
# 6. CIRCUIT FACTORY
# ============================================================

def build_circuit(
    x,
    theta,
    architecture
):

    if architecture == "shallow":

        return shallow_qnn(
            x,
            theta
        )

    elif architecture == "deep":

        return deep_qnn(
            x,
            theta
        )

    elif architecture == "reupload":

        return reupload_qnn(
            x,
            theta
        )

    else:

        raise ValueError(
            f"Unknown architecture: {architecture}"
        )


# ============================================================
# 7. FORWARD PASS
# ============================================================

def predict_single(
    x,
    theta,
    architecture
):

    qc = build_circuit(
        x,
        theta,
        architecture
    )

    state = Statevector.from_instruction(qc)

    expectation = state.expectation_value(
        observable
    )

    expectation = float(
        np.real(expectation)
    )

    # Map [-1,1] -> [0,1]
    probability = (
        expectation + 1
    ) / 2

    return np.clip(
        probability,
        1e-8,
        1 - 1e-8
    )


# ============================================================
# 8. BATCH PREDICTION
# ============================================================

def predict(
    X,
    theta,
    architecture
):

    return np.array([
        predict_single(
            x,
            theta,
            architecture
        )
        for x in X
    ])


# ============================================================
# 9. LOSS
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
# 10. PARAMETER COUNT
# ============================================================

def parameter_count(
    architecture
):

    if architecture == "shallow":

        return 2

    elif architecture == "deep":

        return 2 * NUM_LAYERS

    elif architecture == "reupload":

        return 2 * NUM_LAYERS

    raise ValueError(
        "Unknown architecture"
    )


# ============================================================
# 11. PARAMETER-SHIFT GRADIENT
# ============================================================

def parameter_shift_gradient(
    X,
    y,
    theta,
    architecture,
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
        theta_plus,
        architecture
    )

    predictions_minus = predict(
        X,
        theta_minus,
        architecture
    )

    loss_plus = binary_cross_entropy(
        y,
        predictions_plus
    )

    loss_minus = binary_cross_entropy(
        y,
        predictions_minus
    )

    return (
        loss_plus - loss_minus
    ) / 2


# ============================================================
# 12. FULL GRADIENT
# ============================================================

def compute_gradients(
    X,
    y,
    theta,
    architecture
):

    gradients = np.zeros_like(theta)

    for i in range(len(theta)):

        gradients[i] = parameter_shift_gradient(
            X,
            y,
            theta,
            architecture,
            i
        )

    return gradients


# ============================================================
# 13. ACCURACY
# ============================================================

def accuracy(
    X,
    y,
    theta,
    architecture
):

    probabilities = predict(
        X,
        theta,
        architecture
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return np.mean(
        predictions == y
    )


# ============================================================
# 14. TRAIN ONE ARCHITECTURE
# ============================================================

def train_architecture(
    architecture
):

    count = parameter_count(
        architecture
    )

    rng = np.random.default_rng(
        SEED
    )

    theta = rng.uniform(
        -0.2,
        0.2,
        count
    )

    loss_history = []

    gradient_norm_history = []

    print("\n" + "=" * 70)

    print(
        f"ARCHITECTURE: "
        f"{architecture.upper()}"
    )

    print("=" * 70)

    print(
        f"Parameters: {count}"
    )

    for epoch in range(EPOCHS):

        # -----------------------------------------------
        # Forward pass
        # -----------------------------------------------

        predictions = predict(
            X_train,
            theta,
            architecture
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
            theta,
            architecture
        )

        # -----------------------------------------------
        # Gradient norm
        # -----------------------------------------------

        gradient_norm = np.linalg.norm(
            gradients
        )

        # -----------------------------------------------
        # Parameter update
        # -----------------------------------------------

        theta -= (
            LEARNING_RATE
            * gradients
        )

        # -----------------------------------------------
        # Metrics
        # -----------------------------------------------

        train_acc = accuracy(
            X_train,
            y_train,
            theta,
            architecture
        )

        test_acc = accuracy(
            X_test,
            y_test,
            theta,
            architecture
        )

        loss_history.append(loss)

        gradient_norm_history.append(
            gradient_norm
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {loss:.4f} | "
            f"GradNorm: {gradient_norm:.6f} | "
            f"Train: {train_acc:.3f} | "
            f"Test: {test_acc:.3f}"
        )

    return {
        "theta": theta,
        "loss": loss_history,
        "gradient_norm": gradient_norm_history,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    }


# ============================================================
# 15. RUN EXPERIMENT
# ============================================================

architectures = [
    "shallow",
    "deep",
    "reupload"
]

results = {}

for architecture in architectures:

    results[architecture] = train_architecture(
        architecture
    )


# ============================================================
# 16. FINAL TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL ARCHITECTURE COMPARISON")
print("=" * 70)

print(
    f"{'Architecture':<15}"
    f"{'Params':<10}"
    f"{'Train Acc':<15}"
    f"{'Test Acc':<15}"
    f"{'Final Grad':<15}"
)

print("-" * 70)

for architecture in architectures:

    result = results[architecture]

    params = parameter_count(
        architecture
    )

    final_gradient = (
        result["gradient_norm"][-1]
    )

    print(
        f"{architecture:<15}"
        f"{params:<10}"
        f"{result['train_accuracy']:<15.3f}"
        f"{result['test_accuracy']:<15.3f}"
        f"{final_gradient:<15.6f}"
    )


# ============================================================
# 17. LOSS COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

for architecture in architectures:

    plt.plot(
        results[architecture]["loss"],
        marker="o",
        label=architecture
    )

plt.xlabel("Epoch")

plt.ylabel("Binary Cross Entropy")

plt.title(
    "QNN Architecture Comparison: Training Loss"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 18. GRADIENT COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

for architecture in architectures:

    plt.plot(
        results[architecture]["gradient_norm"],
        marker="o",
        label=architecture
    )

plt.xlabel("Epoch")

plt.ylabel("Gradient Norm")

plt.title(
    "QNN Architecture Comparison: Gradient Norm"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 19. DISPLAY CIRCUITS
# ============================================================

example_x = X_train[0]

print("\n" + "=" * 70)
print("SHALLOW CIRCUIT")
print("=" * 70)

print(
    shallow_qnn(
        example_x,
        results["shallow"]["theta"]
    )
)


print("\n" + "=" * 70)
print("DEEP CIRCUIT")
print("=" * 70)

print(
    deep_qnn(
        example_x,
        results["deep"]["theta"]
    )
)


print("\n" + "=" * 70)
print("DATA RE-UPLOADING CIRCUIT")
print("=" * 70)

print(
    reupload_qnn(
        example_x,
        results["reupload"]["theta"]
    )
)