"""
============================================================
DAY 63 — QUANTUM MACHINE LEARNING
============================================================

Topic:
Variational Quantum Classifier (VQC)

Quantum Journey:
Day 60 -> Quantum Feature Maps
Day 61 -> Quantum Kernel + QSVC
Day 62 -> Feature Map Comparison
Day 63 -> Variational Quantum Classifier

Core idea:

    Input x
       |
       v
    Feature Map
       |
       v
    Trainable Quantum Circuit U(theta)
       |
       v
    Measurement
       |
       v
    Prediction
       |
       v
    Loss
       |
       v
    Classical Optimizer
       |
       +---------> update theta
                         |
                         └------> repeat

============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ============================================================
# 2. DATASET
# ============================================================

print("=" * 70)
print("1. CREATE DATASET")
print("=" * 70)

X, y = make_moons(
    n_samples=100,
    noise=0.15,
    random_state=RANDOM_STATE
)

print("Dataset shape:", X.shape)

print("Class 0:", np.sum(y == 0))
print("Class 1:", np.sum(y == 1))


# ============================================================
# 3. VISUALIZE DATA
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0"
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("Day 63 — Dataset")

plt.legend()

plt.show()


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("=" * 70)
print("2. TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# 5. SCALE DATA
# ============================================================

print("=" * 70)
print("3. FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ============================================================
# 6. CREATE PARAMETERIZED VQC
# ============================================================

print("=" * 70)
print("4. CREATE PARAMETERIZED QUANTUM CIRCUIT")
print("=" * 70)

num_qubits = 2

# Four trainable parameters
theta = ParameterVector(
    "theta",
    length=4
)

qc = QuantumCircuit(
    num_qubits
)


# ------------------------------------------------------------
# DATA ENCODING
# ------------------------------------------------------------

qc.ry(
    0,
    0
)

qc.ry(
    0,
    1
)


# ------------------------------------------------------------
# TRAINABLE LAYER
# ------------------------------------------------------------

qc.ry(
    theta[0],
    0
)

qc.rz(
    theta[1],
    0
)

qc.ry(
    theta[2],
    1
)

qc.rz(
    theta[3],
    1
)


# ------------------------------------------------------------
# ENTANGLEMENT
# ------------------------------------------------------------

qc.cx(
    0,
    1
)


print(qc.draw())


# ============================================================
# 7. EXPLANATION
# ============================================================

print("=" * 70)
print("5. CIRCUIT STRUCTURE")
print("=" * 70)

print(
    """
The circuit contains:

1. Data encoding
2. Trainable rotations
3. Entanglement

Conceptually:

x
|
v
Encoding
|
v
U(theta)
|
v
CNOT
|
v
Measurement
"""
)


# ============================================================
# 8. OBSERVABLE
# ============================================================

print("=" * 70)
print("6. MEASUREMENT OBSERVABLE")
print("=" * 70)

observable = SparsePauliOp.from_list(
    [
        ("ZI", 1.0)
    ]
)

print(
    "Observable:"
)

print(
    observable
)


# ============================================================
# 9. CREATE A CONCRETE CIRCUIT
# ============================================================

def create_circuit(x, parameters):

    circuit = QuantumCircuit(
        2
    )

    # --------------------------------------------------------
    # Data encoding
    # --------------------------------------------------------

    circuit.ry(
        x[0],
        0
    )

    circuit.ry(
        x[1],
        1
    )

    # --------------------------------------------------------
    # Trainable layer
    # --------------------------------------------------------

    circuit.ry(
        parameters[0],
        0
    )

    circuit.rz(
        parameters[1],
        0
    )

    circuit.ry(
        parameters[2],
        1
    )

    circuit.rz(
        parameters[3],
        1
    )

    # --------------------------------------------------------
    # Entanglement
    # --------------------------------------------------------

    circuit.cx(
        0,
        1
    )

    return circuit


# ============================================================
# 10. QUANTUM FORWARD PASS
# ============================================================

def quantum_forward(
    x,
    parameters
):
    """
    Execute the quantum circuit using
    exact statevector simulation.

    Returns:

        <Z_0>
    """

    circuit = create_circuit(
        x,
        parameters
    )

    state = Statevector.from_instruction(
        circuit
    )

    expectation = state.expectation_value(
        observable
    )

    return float(
        np.real(expectation)
    )


# ============================================================
# 11. CONVERT EXPECTATION TO PROBABILITY
# ============================================================

def expectation_to_probability(
    expectation
):

    return (
        expectation + 1
    ) / 2


# ============================================================
# 12. TEST FORWARD PASS
# ============================================================

print("=" * 70)
print("7. TEST QUANTUM FORWARD PASS")
print("=" * 70)

initial_parameters = np.random.uniform(
    -np.pi,
    np.pi,
    4
)

sample = X_train[0]

expectation = quantum_forward(
    sample,
    initial_parameters
)

probability = expectation_to_probability(
    expectation
)

print("Sample:")
print(sample)

print("\nInitial parameters:")
print(initial_parameters)

print("\nExpectation <Z>:")
print(expectation)

print("\nPredicted probability:")
print(probability)


# ============================================================
# 13. LOSS FUNCTION
# ============================================================

def binary_cross_entropy(
    y_true,
    probability
):

    # Avoid log(0)
    probability = np.clip(
        probability,
        1e-8,
        1 - 1e-8
    )

    return -(
        y_true * np.log(probability)
        +
        (1 - y_true)
        * np.log(1 - probability)
    )


# ============================================================
# 14. DATASET LOSS
# ============================================================

def compute_loss(
    parameters,
    X_data,
    y_data
):

    losses = []

    for x, label in zip(
        X_data,
        y_data
    ):

        expectation = quantum_forward(
            x,
            parameters
        )

        probability = expectation_to_probability(
            expectation
        )

        loss = binary_cross_entropy(
            label,
            probability
        )

        losses.append(
            loss
        )

    return np.mean(
        losses
    )


# ============================================================
# 15. INITIAL LOSS
# ============================================================

print("=" * 70)
print("8. INITIAL MODEL LOSS")
print("=" * 70)

initial_loss = compute_loss(
    initial_parameters,
    X_train,
    y_train
)

print(
    "Initial loss:",
    initial_loss
)


# ============================================================
# 16. PARAMETER-SHIFT GRADIENT
# ============================================================

def parameter_shift_gradient(
    parameters,
    X_data,
    y_data
):

    gradients = np.zeros_like(
        parameters
    )

    shift = np.pi / 2

    for i in range(
        len(parameters)
    ):

        plus = parameters.copy()

        minus = parameters.copy()

        plus[i] += shift

        minus[i] -= shift

        loss_plus = compute_loss(
            plus,
            X_data,
            y_data
        )

        loss_minus = compute_loss(
            minus,
            X_data,
            y_data
        )

        gradients[i] = (
            loss_plus - loss_minus
        ) / 2

    return gradients


# ============================================================
# 17. TRAINING LOOP
# ============================================================

print("=" * 70)
print("9. TRAINING")
print("=" * 70)

parameters = initial_parameters.copy()

learning_rate = 0.2

epochs = 30

loss_history = []


for epoch in range(
    epochs
):

    loss = compute_loss(
        parameters,
        X_train,
        y_train
    )

    gradients = parameter_shift_gradient(
        parameters,
        X_train,
        y_train
    )

    parameters -= (
        learning_rate
        * gradients
    )

    loss_history.append(
        loss
    )

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss = {loss:.6f}"
    )


# ============================================================
# 18. FINAL PARAMETERS
# ============================================================

print("=" * 70)
print("10. TRAINED PARAMETERS")
print("=" * 70)

print(
    parameters
)


# ============================================================
# 19. LOSS CURVE
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    loss_history,
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "VQC Training Loss"
)

plt.grid(
    True
)

plt.show()


# ============================================================
# 20. PREDICTION FUNCTION
# ============================================================

def predict(
    X_data,
    parameters
):

    predictions = []

    probabilities = []

    for x in X_data:

        expectation = quantum_forward(
            x,
            parameters
        )

        probability = expectation_to_probability(
            expectation
        )

        probabilities.append(
            probability
        )

        predictions.append(
            int(probability >= 0.5)
        )

    return (
        np.array(predictions),
        np.array(probabilities)
    )


# ============================================================
# 21. TRAINING PREDICTIONS
# ============================================================

train_predictions, train_probabilities = predict(
    X_train,
    parameters
)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print("=" * 70)
print("11. TRAINING PERFORMANCE")
print("=" * 70)

print(
    "Training accuracy:",
    train_accuracy
)


# ============================================================
# 22. TEST PREDICTIONS
# ============================================================

test_predictions, test_probabilities = predict(
    X_test,
    parameters
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("=" * 70)
print("12. TEST PERFORMANCE")
print("=" * 70)

print(
    "Test accuracy:",
    test_accuracy
)


# ============================================================
# 23. CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("13. CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        test_predictions
    )
)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("DAY 63 COMPLETE")
print("=" * 70)

print(
    f"""
Initial Loss : {initial_loss:.6f}

Final Loss   : {loss_history[-1]:.6f}

Train Accuracy:
{train_accuracy:.4f}

Test Accuracy:
{test_accuracy:.4f}


VQC PIPELINE:

             Input x
                |
                v
        Quantum Encoding
                |
                v
       Trainable U(theta)
                |
                v
           Entanglement
                |
                v
          Measurement
                |
                v
          Expectation <Z>
                |
                v
          Probability
                |
                v
             Loss
                |
                v
      Parameter-Shift Gradient
                |
                v
        Update theta
                |
                +-----------> repeat


CORE EQUATION:

    |psi(x, theta)>
        =
    U(theta) U_phi(x) |0...0>


TODAY'S MAIN IDEA:

Unlike the quantum-kernel approach, the quantum
circuit now contains trainable parameters.

The model learns theta from the training data.
"""
)

print("=" * 70)