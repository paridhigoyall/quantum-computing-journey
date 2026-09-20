"""
============================================================
DAY 61 — QUANTUM MACHINE LEARNING
============================================================

Topic:
Quantum Kernel Machine Learning

Previous:
Day 60 -> Quantum Feature Maps

Today:
Classical Data
      ↓
Quantum Feature Map
      ↓
Quantum Kernel
      ↓
QSVC
      ↓
Classification

Core equation:

K(x, y) = | <phi(x) | phi(y)> |^2

where:

|phi(x)> = U_phi(x)|0...0>

We will compare:

1. Classical RBF SVM
2. Quantum Kernel SVM (QSVC)

============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ============================================================
# 2. CREATE DATASET
# ============================================================

print("=" * 70)
print("1. DATASET")
print("=" * 70)

X, y = make_moons(
    n_samples=100,
    noise=0.15,
    random_state=RANDOM_STATE
)

print("Dataset shape:", X.shape)
print("Labels shape :", y.shape)

print("\nFirst five samples:")
print(X[:5])

print("\nFirst five labels:")
print(y[:5])


# ============================================================
# 3. VISUALIZE ORIGINAL DATA
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

plt.title("Day 61 — Original Dataset")

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

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. FEATURE SCALING
# ============================================================

print("=" * 70)
print("3. FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Training mean:")
print(X_train_scaled.mean(axis=0))

print("\nTraining standard deviation:")
print(X_train_scaled.std(axis=0))


# ============================================================
# 6. CLASSICAL RBF SVM
# ============================================================

print("=" * 70)
print("4. CLASSICAL RBF SVM")
print("=" * 70)

classical_svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

classical_svm.fit(
    X_train_scaled,
    y_train
)

classical_predictions = classical_svm.predict(
    X_test_scaled
)

classical_accuracy = accuracy_score(
    y_test,
    classical_predictions
)

print("Classical RBF SVM accuracy:")
print(classical_accuracy)


# ============================================================
# 7. QUANTUM FEATURE MAP
# ============================================================

print("=" * 70)
print("5. QUANTUM FEATURE MAP")
print("=" * 70)

feature_map = zz_feature_map(
    feature_dimension=2,
    reps=2,
    entanglement="linear"
)

print(feature_map.draw())


# ============================================================
# 8. CREATE QUANTUM KERNEL
# ============================================================

print("=" * 70)
print("6. QUANTUM KERNEL")
print("=" * 70)

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print("Quantum kernel created successfully.")


# ============================================================
# 9. COMPUTE TRAINING KERNEL MATRIX
# ============================================================

print("=" * 70)
print("7. TRAINING KERNEL MATRIX")
print("=" * 70)

K_train = quantum_kernel.evaluate(
    X_train_scaled
)

print("Kernel matrix shape:")
print(K_train.shape)

print("\nFirst 5x5 section:")
print(K_train[:5, :5])


# ============================================================
# 10. KERNEL MATRIX PROPERTIES
# ============================================================

print("=" * 70)
print("8. KERNEL MATRIX ANALYSIS")
print("=" * 70)

print("Symmetric:")
print(np.allclose(K_train, K_train.T))

print("\nDiagonal:")
print(np.diag(K_train)[:10])

print("\nK(x,x) approximately 1:")
print(
    np.allclose(
        np.diag(K_train),
        1.0,
        atol=1e-6
    )
)


# ============================================================
# 11. VISUALIZE QUANTUM KERNEL
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    K_train,
    interpolation="nearest"
)

plt.colorbar(
    label="Quantum Similarity"
)

plt.xlabel("Training Sample")
plt.ylabel("Training Sample")

plt.title("Quantum Kernel Matrix")

plt.show()


# ============================================================
# 12. QUANTUM KERNEL TEST MATRIX
# ============================================================

print("=" * 70)
print("9. TEST KERNEL MATRIX")
print("=" * 70)

K_test = quantum_kernel.evaluate(
    X_test_scaled,
    X_train_scaled
)

print("Test kernel matrix shape:")
print(K_test.shape)


# ============================================================
# 13. QSVC
# ============================================================

print("=" * 70)
print("10. QUANTUM SUPPORT VECTOR CLASSIFIER")
print("=" * 70)

qsvc = QSVC(
    quantum_kernel=quantum_kernel
)

qsvc.fit(
    X_train_scaled,
    y_train
)

quantum_predictions = qsvc.predict(
    X_test_scaled
)

quantum_accuracy = accuracy_score(
    y_test,
    quantum_predictions
)

print("Quantum Kernel SVM accuracy:")
print(quantum_accuracy)


# ============================================================
# 14. QUANTUM CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("11. QUANTUM CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        quantum_predictions
    )
)


# ============================================================
# 15. CLASSICAL CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("12. CLASSICAL CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        classical_predictions
    )
)


# ============================================================
# 16. CONFUSION MATRICES
# ============================================================

classical_cm = confusion_matrix(
    y_test,
    classical_predictions
)

quantum_cm = confusion_matrix(
    y_test,
    quantum_predictions
)


plt.figure(figsize=(6, 5))

plt.imshow(
    classical_cm,
    interpolation="nearest"
)

plt.title("Classical RBF SVM Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.colorbar()

plt.show()


plt.figure(figsize=(6, 5))

plt.imshow(
    quantum_cm,
    interpolation="nearest"
)

plt.title("Quantum Kernel SVM Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.colorbar()

plt.show()


# ============================================================
# 17. FINAL COMPARISON
# ============================================================

print("=" * 70)
print("13. FINAL COMPARISON")
print("=" * 70)

print(
    f"Classical RBF SVM Accuracy : {classical_accuracy:.4f}"
)

print(
    f"Quantum Kernel SVM Accuracy: {quantum_accuracy:.4f}"
)

print("\nAccuracy difference:")

difference = quantum_accuracy - classical_accuracy

print(f"{difference:+.4f}")


# ============================================================
# 18. INTERPRETATION
# ============================================================

print("=" * 70)
print("14. INTERPRETATION")
print("=" * 70)

if quantum_accuracy > classical_accuracy:

    print(
        """
The quantum kernel achieved higher test accuracy
on this particular train/test split.

IMPORTANT:
This does NOT prove quantum advantage.

A proper experiment would require:
- multiple datasets
- repeated trials
- strong classical baselines
- hyperparameter tuning
- computational cost analysis
- noise analysis
- statistical testing
        """
    )

elif quantum_accuracy < classical_accuracy:

    print(
        """
The classical RBF SVM achieved higher test accuracy
on this particular train/test split.

This is completely valid.

Quantum ML is NOT automatically better simply because
the feature map is quantum.

The important question is whether the quantum feature
map provides useful structure for a particular problem.
        """
    )

else:

    print(
        """
Both models achieved the same test accuracy.

This shows that the quantum feature map and classical
baseline produced comparable classification performance
for this experiment.
        """
    )


# ============================================================
# 19. COMPLETE PIPELINE
# ============================================================

print("=" * 70)
print("DAY 61 COMPLETE")
print("=" * 70)

print(
    """
                QML PIPELINE

              Classical Dataset
                     |
                     v
              Feature Scaling
                     |
          +----------+----------+
          |                     |
          v                     v
     Classical RBF          Quantum Feature
         Kernel                  Map
          |                      |
          v                      v
       SVM                 Quantum States
                                 |
                                 v
                         Quantum Fidelity
                                 |
                                 v
                         Quantum Kernel
                                 |
                                 v
                                QSVC
                                 |
                                 v
                            Prediction


CORE IDEA:

K(x, y) = |<phi(x) | phi(y)>|^2

where:

|phi(x)> = U_phi(x)|0...0>


TODAY'S LESSON:

The quantum computer is being used to construct
a feature-space similarity function.

The final classification algorithm can still be
an SVM.

============================================================
"""
)

print("=" * 70)