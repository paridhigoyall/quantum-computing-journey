"""
===========================================================
DAY 60 — QUANTUM MACHINE LEARNING
===========================================================

Topic:
Quantum Feature Maps

Quantum Journey:
Day 1  → Day 59 : Quantum Computing
Day 60          : Beginning of Quantum Machine Learning

Today's Goals:
1. Understand classical → quantum data encoding
2. Implement angle encoding using Ry gates
3. Inspect the resulting quantum state
4. Verify the state mathematically
5. Add entanglement to create feature interactions
6. Compare unentangled and entangled feature maps
7. Calculate quantum-state similarity
8. Construct a quantum kernel
9. Build a quantum kernel matrix

Core idea:

        Classical Data
              |
              v
      Quantum Feature Map
              |
              v
        Quantum State
              |
              v
       Quantum Similarity
              |
              v
        Quantum Kernel

Mathematical definition:

    |phi(x)> = U_phi(x) |0...0>

Quantum kernel:

    K(x_i, x_j)
        = | <phi(x_i) | phi(x_j)> |^2
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ==========================================================
# 1. HELPER FUNCTIONS
# ==========================================================

def print_section(title):
    """Print a clean section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# 2. BASIC ANGLE-ENCODING FEATURE MAP
# ==========================================================

def angle_feature_map(x):
    """
    Encode classical features using Ry rotations.

    For x = [x1, x2]:

        |phi(x)> =
        Ry(x1) ⊗ Ry(x2) |00>

    Parameters
    ----------
    x : array-like
        Classical feature vector of length 2.

    Returns
    -------
    QuantumCircuit
        Qiskit quantum circuit.
    """

    if len(x) != 2:
        raise ValueError("This Day-60 implementation expects exactly 2 features.")

    qc = QuantumCircuit(2)

    # Encode feature x1 into qubit 0
    qc.ry(x[0], 0)

    # Encode feature x2 into qubit 1
    qc.ry(x[1], 1)

    return qc


# ==========================================================
# 3. ENTANGLED FEATURE MAP
# ==========================================================

def entangled_feature_map(x):
    """
    Encode classical features and introduce entanglement.

    Circuit:

        q0 ── Ry(x1) ──●────
                       |
        q1 ── Ry(x2) ──X────

    The CNOT introduces an interaction between
    the two encoded features.
    """

    if len(x) != 2:
        raise ValueError("This Day-60 implementation expects exactly 2 features.")

    qc = QuantumCircuit(2)

    # Feature encoding
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    # Feature interaction
    qc.cx(0, 1)

    return qc


# ==========================================================
# 4. CLASSICAL INPUT
# ==========================================================

print_section("1. CLASSICAL INPUT")

x = np.array([0.7, 1.2])

print("Classical feature vector:")
print(x)

print("\nFeature 1 =", x[0])
print("Feature 2 =", x[1])


# ==========================================================
# 5. CREATE BASIC FEATURE MAP
# ==========================================================

print_section("2. BASIC ANGLE FEATURE MAP")

qc = angle_feature_map(x)

print(qc.draw())


# ==========================================================
# 6. GET STATEVECTOR
# ==========================================================

print_section("3. QUANTUM STATEVECTOR")

state = Statevector.from_instruction(qc)

print("Quantum state:")
print(state)

print("\nStatevector amplitudes:")
print(state.data)


# ==========================================================
# 7. MATHEMATICAL VERIFICATION
# ==========================================================

print_section("4. MATHEMATICAL VERIFICATION")

x1 = x[0]
x2 = x[1]

# Expected amplitudes from:
#
# Ry(theta)|0>
# =
# cos(theta/2)|0>
# +
# sin(theta/2)|1>

expected = np.array([
    np.cos(x1 / 2) * np.cos(x2 / 2),
    np.cos(x1 / 2) * np.sin(x2 / 2),
    np.sin(x1 / 2) * np.cos(x2 / 2),
    np.sin(x1 / 2) * np.sin(x2 / 2)
])

print("Expected amplitudes:")
print(expected)

print("\nQiskit amplitudes:")
print(state.data)

matches = np.allclose(expected, state.data)

print("\nDoes mathematical calculation match Qiskit?")
print(matches)


# ==========================================================
# 8. PROBABILITY DISTRIBUTION
# ==========================================================

print_section("5. MEASUREMENT PROBABILITIES")

probabilities = np.abs(state.data) ** 2

basis_states = ["00", "01", "10", "11"]

for basis, probability in zip(basis_states, probabilities):
    print(f"|{basis}> : {probability:.6f}")

print("\nProbability sum:")
print(probabilities.sum())


# ==========================================================
# 9. VISUALIZE PROBABILITIES
# ==========================================================

plt.figure(figsize=(7, 5))

plt.bar(basis_states, probabilities)

plt.xlabel("Computational Basis State")
plt.ylabel("Probability")
plt.title("Day 60 — Angle Encoding Probability Distribution")

plt.ylim(0, 1)

plt.show()


# ==========================================================
# 10. CREATE ENTANGLED FEATURE MAP
# ==========================================================

print_section("6. ENTANGLED FEATURE MAP")

qc_entangled = entangled_feature_map(x)

print(qc_entangled.draw())


# ==========================================================
# 11. ENTANGLED STATEVECTOR
# ==========================================================

state_entangled = Statevector.from_instruction(qc_entangled)

print("\nEntangled state:")
print(state_entangled)

print("\nEntangled amplitudes:")
print(state_entangled.data)


# ==========================================================
# 12. COMPARE STATES
# ==========================================================

print_section("7. BASIC VS ENTANGLED FEATURE MAP")

print("Basic feature map state:")
print(state.data)

print("\nEntangled feature map state:")
print(state_entangled.data)

print("\nDifference:")
print(state_entangled.data - state.data)


# ==========================================================
# 13. ENTANGLED PROBABILITIES
# ==========================================================

probabilities_entangled = np.abs(state_entangled.data) ** 2

print("\nEntangled probabilities:")

for basis, probability in zip(basis_states, probabilities_entangled):
    print(f"|{basis}> : {probability:.6f}")

print("\nProbability sum:")
print(probabilities_entangled.sum())


# ==========================================================
# 14. COMPARE PROBABILITY DISTRIBUTIONS
# ==========================================================

x_axis = np.arange(len(basis_states))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x_axis - width / 2,
    probabilities,
    width,
    label="Without Entanglement"
)

plt.bar(
    x_axis + width / 2,
    probabilities_entangled,
    width,
    label="With Entanglement"
)

plt.xticks(x_axis, basis_states)

plt.xlabel("Computational Basis State")
plt.ylabel("Probability")

plt.title("Feature Map Comparison")

plt.legend()

plt.ylim(0, 1)

plt.show()


# ==========================================================
# 15. QUANTUM FEATURE MAP FUNCTION
# ==========================================================

def get_state(x, entangled=True):
    """
    Convert classical data into a quantum state.

    Parameters
    ----------
    x : array-like
        Classical feature vector.

    entangled : bool
        Whether to use entanglement.

    Returns
    -------
    Statevector
        Quantum feature representation.
    """

    if entangled:
        circuit = entangled_feature_map(x)
    else:
        circuit = angle_feature_map(x)

    return Statevector.from_instruction(circuit)


# ==========================================================
# 16. QUANTUM KERNEL
# ==========================================================

def quantum_kernel(x1, x2, entangled=True):
    """
    Calculate quantum kernel similarity.

    K(x1, x2)
        = |<phi(x1)|phi(x2)>|^2
    """

    state1 = get_state(x1, entangled)
    state2 = get_state(x2, entangled)

    # Inner product:
    # <phi(x1)|phi(x2)>
    overlap = np.vdot(state1.data, state2.data)

    # Fidelity / quantum kernel
    kernel_value = abs(overlap) ** 2

    return kernel_value


# ==========================================================
# 17. TEST QUANTUM SIMILARITY
# ==========================================================

print_section("8. QUANTUM STATE SIMILARITY")

sample_1 = np.array([0.7, 1.2])
sample_2 = np.array([0.9, 1.0])

kernel_value = quantum_kernel(
    sample_1,
    sample_2,
    entangled=True
)

print("Sample 1:")
print(sample_1)

print("\nSample 2:")
print(sample_2)

print("\nQuantum kernel:")
print(kernel_value)

print("\nInterpretation:")

if kernel_value > 0.8:
    print("The quantum representations are highly similar.")

elif kernel_value > 0.5:
    print("The quantum representations have moderate similarity.")

else:
    print("The quantum representations are relatively different.")


# ==========================================================
# 18. SELF-SIMILARITY
# ==========================================================

print_section("9. SELF-SIMILARITY")

self_similarity = quantum_kernel(
    sample_1,
    sample_1,
    entangled=True
)

print("K(x, x) =", self_similarity)

print(
    "\nExpected value is approximately 1 because "
    "a quantum state is maximally similar to itself."
)


# ==========================================================
# 19. DATASET
# ==========================================================

print_section("10. SMALL QML DATASET")

X = np.array([
    [0.2, 0.3],
    [0.4, 0.5],
    [0.6, 0.4],
    [0.8, 0.7],
    [2.4, 2.6],
    [2.6, 2.8],
    [2.8, 2.7],
    [3.0, 2.9]
])

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])

print("Dataset:")
print(X)

print("\nLabels:")
print(y)


# ==========================================================
# 20. CONSTRUCT QUANTUM KERNEL MATRIX
# ==========================================================

print_section("11. QUANTUM KERNEL MATRIX")

n_samples = len(X)

K = np.zeros((n_samples, n_samples))

for i in range(n_samples):

    for j in range(n_samples):

        K[i, j] = quantum_kernel(
            X[i],
            X[j],
            entangled=True
        )

np.set_printoptions(precision=4, suppress=True)

print(K)


# ==========================================================
# 21. CHECK KERNEL PROPERTIES
# ==========================================================

print_section("12. KERNEL MATRIX CHECK")

print("Shape:")
print(K.shape)

print("\nDiagonal:")
print(np.diag(K))

print("\nIs the matrix symmetric?")

print(
    np.allclose(K, K.T)
)

print("\nIs K(x,x) approximately 1?")

print(
    np.allclose(np.diag(K), 1)
)


# ==========================================================
# 22. VISUALIZE KERNEL MATRIX
# ==========================================================

plt.figure(figsize=(7, 6))

plt.imshow(K)

plt.colorbar(label="Quantum Similarity")

plt.xlabel("Data Point")
plt.ylabel("Data Point")

plt.title("Quantum Kernel Matrix")

plt.xticks(range(n_samples))
plt.yticks(range(n_samples))

plt.show()


# ==========================================================
# 23. COMPARE WITH NON-ENTANGLED KERNEL
# ==========================================================

print_section("13. ENTANGLED VS NON-ENTANGLED KERNEL")

K_basic = np.zeros((n_samples, n_samples))

K_entangled = np.zeros((n_samples, n_samples))

for i in range(n_samples):

    for j in range(n_samples):

        K_basic[i, j] = quantum_kernel(
            X[i],
            X[j],
            entangled=False
        )

        K_entangled[i, j] = quantum_kernel(
            X[i],
            X[j],
            entangled=True
        )


print("Basic feature-map kernel:")
print(K_basic)

print("\nEntangled feature-map kernel:")
print(K_entangled)


# ==========================================================
# 24. VISUAL COMPARISON
# ==========================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(K_basic)

plt.colorbar()

plt.title("Without Entanglement")

plt.xlabel("Data Point")
plt.ylabel("Data Point")

plt.xticks(range(n_samples))
plt.yticks(range(n_samples))


plt.subplot(1, 2, 2)

plt.imshow(K_entangled)

plt.colorbar()

plt.title("With Entanglement")

plt.xlabel("Data Point")
plt.ylabel("Data Point")

plt.xticks(range(n_samples))
plt.yticks(range(n_samples))


plt.tight_layout()

plt.show()


# ==========================================================
# 25. FINAL SUMMARY
# ==========================================================

print_section("DAY 60 COMPLETE")

print(
    """
Today's pipeline:

Classical Data
      |
      v
Angle Encoding
      |
      v
Quantum State
      |
      +------> Statevector
      |
      +------> Measurement Probabilities
      |
      v
Entanglement
      |
      v
Quantum Feature Representation
      |
      v
State Similarity
      |
      v
Quantum Kernel
      |
      v
Kernel Matrix

Core equation:

    |phi(x)> = U_phi(x)|0...0>

Quantum kernel:

    K(x_i, x_j)
        = |<phi(x_i)|phi(x_j)>|^2


DAY 60 CONCEPT:

Quantum feature maps transform classical data into
quantum states. The resulting quantum geometry can
then be used to construct similarity measures and
eventually quantum-kernel-based machine learning models.
"""
)

print("=" * 70)
print("END OF DAY 60")
print("=" * 70)