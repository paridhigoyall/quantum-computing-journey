"""
============================================================
DAY 62 — QUANTUM MACHINE LEARNING
============================================================

Topic:
Quantum Feature Map Comparison

Previous:
Day 60 -> Quantum Feature Maps
Day 61 -> Quantum Kernel + QSVC

Today's Question:

    Does the choice of quantum feature map
    change the useful geometry of the data?

We compare:

1. Custom Angle Encoding
2. Z Feature Map
3. ZZ Feature Map

Pipeline:

Classical Data
      ↓
Feature Scaling
      ↓
Quantum Feature Map
      ↓
Quantum Kernel
      ↓
Kernel Matrix
      ↓
QSVC
      ↓
Classification

Core equation:

    |phi(x)> = U_phi(x)|0...0>

    K(x_i, x_j)
        = |<phi(x_i)|phi(x_j)>|^2

============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit.circuit.library import (
    z_feature_map,
    zz_feature_map,
)

from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel
)

from qiskit_machine_learning.algorithms import (
    QSVC
)

from sklearn.datasets import make_moons

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 3. DATASET
# ============================================================

section("1. CREATE DATASET")

X, y = make_moons(
    n_samples=120,
    noise=0.15,
    random_state=RANDOM_STATE
)

print("Dataset shape:", X.shape)

print("\nClass distribution:")
print("Class 0:", np.sum(y == 0))
print("Class 1:", np.sum(y == 1))


# ============================================================
# 4. VISUALIZE DATA
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

plt.title(
    "Day 62 — Original Dataset"
)

plt.legend()

plt.show()


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

section("2. TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.25,

    random_state=RANDOM_STATE,

    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 6. SCALE FEATURES
# ============================================================

section("3. FEATURE SCALING")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print("Scaled training data:")
print(X_train_scaled[:5])


# ============================================================
# 7. CUSTOM ANGLE FEATURE MAP
# ============================================================

section("4. CUSTOM ANGLE FEATURE MAP")


def angle_feature_map():

    qc = QuantumCircuit(
        2
    )

    qc.ry(
        0,
        0
    )

    qc.ry(
        0,
        1
    )

    return qc


angle_map = angle_feature_map()

print(
    """
Custom Angle Feature Map:

q0 ── Ry(x0)

q1 ── Ry(x1)
"""
)


# ============================================================
# 8. IMPORTANT:
#    THE ABOVE CIRCUIT NEEDS PARAMETERS
# ============================================================

"""
For the quantum-kernel API, we want a parameterized
feature map.

So instead of manually binding parameters here,
we use Qiskit's feature-map functions for the
main comparison.

The custom angle map from Day 60 is kept conceptually
as our baseline.
"""


# ============================================================
# 9. Z FEATURE MAP
# ============================================================

section("5. Z FEATURE MAP")

z_map = z_feature_map(

    feature_dimension=2,

    reps=2,

    entanglement="linear"
)

print(
    z_map.draw()
)


# ============================================================
# 10. ZZ FEATURE MAP
# ============================================================

section("6. ZZ FEATURE MAP")

zz_map = zz_feature_map(

    feature_dimension=2,

    reps=2,

    entanglement="linear"
)

print(
    zz_map.draw()
)


# ============================================================
# 11. FUNCTION TO ANALYZE A CIRCUIT
# ============================================================

def analyze_circuit(
    circuit,
    name
):

    print("\n" + "-" * 60)

    print(
        f"{name}"
    )

    print("-" * 60)

    print(
        "Number of qubits:",
        circuit.num_qubits
    )

    print(
        "Number of parameters:",
        circuit.num_parameters
    )

    print(
        "Circuit depth:",
        circuit.depth()
    )

    print(
        "Gate count:",
        circuit.size()
    )


# ============================================================
# 12. CIRCUIT COMPARISON
# ============================================================

section("7. CIRCUIT COMPARISON")

analyze_circuit(
    z_map,
    "Z Feature Map"
)

analyze_circuit(
    zz_map,
    "ZZ Feature Map"
)


# ============================================================
# 13. CREATE QUANTUM KERNELS
# ============================================================

section("8. CREATE QUANTUM KERNELS")


z_kernel = FidelityQuantumKernel(

    feature_map=z_map
)


zz_kernel = FidelityQuantumKernel(

    feature_map=zz_map
)


print(
    "Z kernel created."
)

print(
    "ZZ kernel created."
)


# ============================================================
# 14. COMPUTE Z KERNEL MATRIX
# ============================================================

section("9. Z KERNEL MATRIX")

K_z = z_kernel.evaluate(

    X_train_scaled
)

print(
    "Shape:",
    K_z.shape
)

print(
    "\nFirst 5 × 5:"
)

print(
    K_z[:5, :5]
)


# ============================================================
# 15. COMPUTE ZZ KERNEL MATRIX
# ============================================================

section("10. ZZ KERNEL MATRIX")

K_zz = zz_kernel.evaluate(

    X_train_scaled
)

print(
    "Shape:",
    K_zz.shape
)

print(
    "\nFirst 5 × 5:"
)

print(
    K_zz[:5, :5]
)


# ============================================================
# 16. KERNEL MATRIX PROPERTIES
# ============================================================

section("11. KERNEL MATRIX PROPERTIES")


print(
    "Z kernel symmetric:",
    np.allclose(
        K_z,
        K_z.T
    )
)


print(
    "ZZ kernel symmetric:",
    np.allclose(
        K_zz,
        K_zz.T
    )
)


print(
    "\nZ diagonal approximately 1:",
    np.allclose(
        np.diag(K_z),
        1.0,
        atol=1e-6
    )
)


print(
    "ZZ diagonal approximately 1:",
    np.allclose(
        np.diag(K_zz),
        1.0,
        atol=1e-6
    )
)


# ============================================================
# 17. VISUALIZE Z KERNEL
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    K_z,
    interpolation="nearest"
)

plt.colorbar(
    label="Quantum Similarity"
)

plt.xlabel(
    "Training Sample"
)

plt.ylabel(
    "Training Sample"
)

plt.title(
    "Z Feature Map — Quantum Kernel"
)

plt.show()


# ============================================================
# 18. VISUALIZE ZZ KERNEL
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    K_zz,
    interpolation="nearest"
)

plt.colorbar(
    label="Quantum Similarity"
)

plt.xlabel(
    "Training Sample"
)

plt.ylabel(
    "Training Sample"
)

plt.title(
    "ZZ Feature Map — Quantum Kernel"
)

plt.show()


# ============================================================
# 19. QUANTUM KERNEL CLASSIFIER — Z
# ============================================================

section("12. QSVC WITH Z FEATURE MAP")

qsvc_z = QSVC(

    quantum_kernel=z_kernel
)

qsvc_z.fit(

    X_train_scaled,
    y_train
)

pred_z = qsvc_z.predict(

    X_test_scaled
)

accuracy_z = accuracy_score(

    y_test,
    pred_z
)

print(
    "Z Feature Map Accuracy:",
    accuracy_z
)


# ============================================================
# 20. QUANTUM KERNEL CLASSIFIER — ZZ
# ============================================================

section("13. QSVC WITH ZZ FEATURE MAP")

qsvc_zz = QSVC(

    quantum_kernel=zz_kernel
)

qsvc_zz.fit(

    X_train_scaled,
    y_train
)

pred_zz = qsvc_zz.predict(

    X_test_scaled
)

accuracy_zz = accuracy_score(

    y_test,
    pred_zz
)

print(
    "ZZ Feature Map Accuracy:",
    accuracy_zz
)


# ============================================================
# 21. CLASSICAL RBF BASELINE
# ============================================================

section("14. CLASSICAL RBF BASELINE")

classical_svm = SVC(

    kernel="rbf",

    C=1.0,

    gamma="scale"
)

classical_svm.fit(

    X_train_scaled,
    y_train
)

pred_classical = classical_svm.predict(

    X_test_scaled
)

accuracy_classical = accuracy_score(

    y_test,
    pred_classical
)

print(
    "Classical RBF SVM Accuracy:",
    accuracy_classical
)


# ============================================================
# 22. FINAL COMPARISON
# ============================================================

section("15. FINAL ACCURACY COMPARISON")

print(
    f"Classical RBF SVM : {accuracy_classical:.4f}"
)

print(
    f"Quantum Z Kernel  : {accuracy_z:.4f}"
)

print(
    f"Quantum ZZ Kernel : {accuracy_zz:.4f}"
)


# ============================================================
# 23. BAR CHART
# ============================================================

models = [

    "Classical\nRBF",

    "Quantum\nZ",

    "Quantum\nZZ"
]

accuracies = [

    accuracy_classical,

    accuracy_z,

    accuracy_zz
]


plt.figure(figsize=(8, 5))

plt.bar(
    models,
    accuracies
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)

plt.title(
    "Day 62 — Feature Map Comparison"
)

plt.show()


# ============================================================
# 24. CLASSIFICATION REPORTS
# ============================================================

section("16. CLASSIFICATION REPORT — Z")

print(
    classification_report(
        y_test,
        pred_z
    )
)


section("17. CLASSIFICATION REPORT — ZZ")

print(
    classification_report(
        y_test,
        pred_zz
    )
)


# ============================================================
# 25. CONFUSION MATRICES
# ============================================================

cm_z = confusion_matrix(

    y_test,
    pred_z
)

cm_zz = confusion_matrix(

    y_test,
    pred_zz
)


plt.figure(figsize=(6, 5))

plt.imshow(
    cm_z
)

plt.colorbar()

plt.title(
    "Z Feature Map — Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()


plt.figure(figsize=(6, 5))

plt.imshow(
    cm_zz
)

plt.colorbar()

plt.title(
    "ZZ Feature Map — Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()


# ============================================================
# 26. FEATURE MAP DEPTH COMPARISON
# ============================================================

section("18. EFFECT OF REPS")

zz_reps = {}

for reps in [1, 2, 3]:

    fmap = zz_feature_map(

        feature_dimension=2,

        reps=reps,

        entanglement="linear"
    )

    kernel = FidelityQuantumKernel(

        feature_map=fmap
    )

    model = QSVC(

        quantum_kernel=kernel
    )

    model.fit(

        X_train_scaled,
        y_train
    )

    prediction = model.predict(

        X_test_scaled
    )

    accuracy = accuracy_score(

        y_test,
        prediction
    )

    zz_reps[reps] = accuracy

    print(
        f"ZZFeatureMap reps={reps}: "
        f"{accuracy:.4f}"
    )


# ============================================================
# 27. REPS VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(

    list(zz_reps.keys()),

    list(zz_reps.values()),

    marker="o"
)

plt.xlabel(
    "Feature Map Repetitions"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Effect of Feature Map Repetitions"
)

plt.xticks(
    [1, 2, 3]
)

plt.ylim(
    0,
    1
)

plt.grid(
    True
)

plt.show()


# ============================================================
# 28. IMPORTANT INTERPRETATION
# ============================================================

section("19. DAY 62 INTERPRETATION")

print(
    """
The most important lesson from today's experiment:

Different quantum feature maps generate different
quantum representations of the SAME classical data.

Therefore:

    Different U_phi(x)
            |
            v
    Different |phi(x)>
            |
            v
    Different K(x_i, x_j)
            |
            v
    Different classifier behavior


We therefore CANNOT simply say:

    "Quantum ML is better."

Instead we ask:

    "Which feature map produces a useful geometry
     for this particular dataset?"


We also compare against strong classical baselines.

Today's experiment compares:

    Classical RBF SVM
    Quantum Z Kernel
    Quantum ZZ Kernel


IMPORTANT:

Higher accuracy on one train/test split does NOT
demonstrate quantum advantage.

A serious experiment requires:

    - multiple datasets
    - repeated trials
    - hyperparameter tuning
    - classical baselines
    - computational cost
    - circuit depth
    - noise
    - statistical analysis
"""
)


# ============================================================
# 29. FINAL PIPELINE
# ============================================================

section("20. DAY 62 COMPLETE")

print(
    """
                    DAY 62

                  Classical Data
                        |
                        v
                  Feature Scaling
                        |
             +----------+----------+
             |          |           |
             v          v           v
           Z Map      ZZ Map    Classical RBF
             |          |           |
             v          v           v
          Quantum    Quantum       SVM
           State      State
             |          |
             v          v
          Quantum    Quantum
          Kernel     Kernel
             |          |
             v          v
           QSVC       QSVC
             |          |
             +----------+
                    |
                    v
              Compare Results


CORE RESEARCH QUESTION:

Does the choice of quantum feature map
create a more useful geometry for the
classification problem?


END OF DAY 62
"""
)

print("=" * 70)