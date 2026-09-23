"""
============================================================
DAY 64 — QUANTUM MACHINE LEARNING
============================================================

Topic:
Quantum Gradients & Parameter-Shift Rule

Previous:
Day 63 -> Variational Quantum Classifier

Today:
Understand how quantum circuit parameters are differentiated.

We compare:

1. Analytical derivative
2. Parameter-shift derivative
3. Finite-difference approximation

Core equation:

df/dtheta =
1/2 [
    f(theta + pi/2)
    -
    f(theta - pi/2)
]

============================================================
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp


# ============================================================
# 1. OBSERVABLE
# ============================================================

Z = SparsePauliOp.from_list([
    ("Z", 1.0)
])


# ============================================================
# 2. QUANTUM FUNCTION
# ============================================================

def quantum_expectation(theta):

    qc = QuantumCircuit(1)

    qc.ry(
        theta,
        0
    )

    state = Statevector.from_instruction(qc)

    expectation = state.expectation_value(Z)

    return float(
        np.real(expectation)
    )


# ============================================================
# 3. ANALYTICAL FUNCTION
# ============================================================

def analytical_function(theta):

    return np.cos(theta)


# ============================================================
# 4. ANALYTICAL DERIVATIVE
# ============================================================

def analytical_derivative(theta):

    return -np.sin(theta)


# ============================================================
# 5. PARAMETER-SHIFT GRADIENT
# ============================================================

def parameter_shift_gradient(theta):

    shift = np.pi / 2

    forward = quantum_expectation(
        theta + shift
    )

    backward = quantum_expectation(
        theta - shift
    )

    gradient = (
        forward - backward
    ) / 2

    return gradient


# ============================================================
# 6. FINITE DIFFERENCE
# ============================================================

def finite_difference_gradient(
    theta,
    epsilon=1e-5
):

    forward = quantum_expectation(
        theta + epsilon
    )

    backward = quantum_expectation(
        theta - epsilon
    )

    gradient = (
        forward - backward
    ) / (
        2 * epsilon
    )

    return gradient


# ============================================================
# 7. TEST
# ============================================================

theta = 0.7


print("=" * 70)
print("DAY 64 — QUANTUM GRADIENT")
print("=" * 70)

print(
    "Theta:",
    theta
)


# ============================================================
# 8. FUNCTION VALUES
# ============================================================

print("\nFunction values:")

print(
    "Qiskit:",
    quantum_expectation(theta)
)

print(
    "Analytical:",
    analytical_function(theta)
)


# ============================================================
# 9. GRADIENT COMPARISON
# ============================================================

analytical = analytical_derivative(
    theta
)

parameter_shift = parameter_shift_gradient(
    theta
)

finite_difference = finite_difference_gradient(
    theta
)


print("\nGradient comparison:")

print(
    "Analytical:",
    analytical
)

print(
    "Parameter shift:",
    parameter_shift
)

print(
    "Finite difference:",
    finite_difference
)


# ============================================================
# 10. ERRORS
# ============================================================

parameter_shift_error = abs(
    analytical - parameter_shift
)

finite_difference_error = abs(
    analytical - finite_difference
)


print("\nErrors:")

print(
    "Parameter-shift error:",
    parameter_shift_error
)

print(
    "Finite-difference error:",
    finite_difference_error
)


# ============================================================
# 11. PARAMETER SHIFT STEP-BY-STEP
# ============================================================

print("\n" + "=" * 70)
print("PARAMETER-SHIFT STEP-BY-STEP")
print("=" * 70)

shift = np.pi / 2

theta_plus = theta + shift

theta_minus = theta - shift

f_plus = quantum_expectation(
    theta_plus
)

f_minus = quantum_expectation(
    theta_minus
)

print(
    "theta + pi/2:",
    theta_plus
)

print(
    "f(theta + pi/2):",
    f_plus
)

print(
    "\ntheta - pi/2:",
    theta_minus
)

print(
    "f(theta - pi/2):",
    f_minus
)

print(
    "\nGradient:"
)

print(
    "(f_plus - f_minus) / 2 =",
    (f_plus - f_minus) / 2
)


# ============================================================
# 12. TEST MULTIPLE PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("MULTIPLE PARAMETER TEST")
print("=" * 70)

test_values = np.linspace(
    0,
    2 * np.pi,
    8
)

for theta_value in test_values:

    exact = analytical_derivative(
        theta_value
    )

    shifted = parameter_shift_gradient(
        theta_value
    )

    error = abs(
        exact - shifted
    )

    print(
        f"theta={theta_value:.4f} | "
        f"exact={exact:.6f} | "
        f"shift={shifted:.6f} | "
        f"error={error:.2e}"
    )


# ============================================================
# 13. SIMPLE TRAINING EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SIMPLE PARAMETER OPTIMIZATION")
print("=" * 70)

# Goal:
# Minimize f(theta) = cos(theta)
#
# The minimum is:
#
# theta = pi
#
# because:
#
# cos(pi) = -1


theta = 0.5

learning_rate = 0.2

epochs = 20

print(
    "\nInitial theta:",
    theta
)

for epoch in range(
    epochs
):

    value = quantum_expectation(
        theta
    )

    gradient = parameter_shift_gradient(
        theta
    )

    theta = (
        theta
        -
        learning_rate
        * gradient
    )

    print(
        f"Epoch {epoch + 1:02d} | "
        f"theta={theta:.6f} | "
        f"value={value:.6f} | "
        f"gradient={gradient:.6f}"
    )


print(
    "\nFinal theta:",
    theta
)

print(
    "Final function value:",
    quantum_expectation(theta)
)

print(
    "\nExpected optimum:"
)

print(
    "theta ≈ pi"
)

print(
    "f(theta) ≈ -1"
)


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DAY 64 COMPLETE")
print("=" * 70)

print(
    """
You have now implemented:

    Quantum circuit
          ↓
    Expectation value
          ↓
    Parameter shift
          ↓
    Gradient
          ↓
    Parameter update
          ↓
    Repeat


Core equation:

df/dtheta =
1/2[
    f(theta + pi/2)
    -
    f(theta - pi/2)
]


This is the fundamental mechanism that allows
parameterized quantum circuits to participate in
gradient-based machine learning.
"""
)

print("=" * 70)