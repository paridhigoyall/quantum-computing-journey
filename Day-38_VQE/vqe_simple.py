import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


# ============================================================
# Hamiltonian
#
# H = 0.5 Z + 0.8 X
#
# Exact ground-state energy:
#
# E_min = -sqrt(0.5^2 + 0.8^2)
#       ≈ -0.9434
# ============================================================

Z_COEFF = 0.5
X_COEFF = 0.8


# ============================================================
# Measure <Z>
# ============================================================

def measure_z(theta, shots=1024):

    qc = QuantumCircuit(1, 1)

    qc.ry(theta, 0)
    qc.measure(0, 0)

    simulator = AerSimulator()

    compiled = transpile(qc, simulator)

    result = simulator.run(
        compiled,
        shots=shots
    ).result()

    counts = result.get_counts()

    zeros = counts.get("0", 0)
    ones = counts.get("1", 0)

    expectation_z = (zeros - ones) / shots

    return expectation_z


# ============================================================
# Measure <X>
#
# To measure X, rotate the X basis into the Z basis
# using H before measurement.
# ============================================================

def measure_x(theta, shots=1024):

    qc = QuantumCircuit(1, 1)

    qc.ry(theta, 0)

    # Change from X basis to Z basis
    qc.h(0)

    qc.measure(0, 0)

    simulator = AerSimulator()

    compiled = transpile(qc, simulator)

    result = simulator.run(
        compiled,
        shots=shots
    ).result()

    counts = result.get_counts()

    zeros = counts.get("0", 0)
    ones = counts.get("1", 0)

    expectation_x = (zeros - ones) / shots

    return expectation_x


# ============================================================
# Calculate Hamiltonian expectation value
# ============================================================

def evaluate_energy(theta, shots=1024):

    z = measure_z(
        theta,
        shots
    )

    x = measure_x(
        theta,
        shots
    )

    energy = (
        Z_COEFF * z
        +
        X_COEFF * x
    )

    return energy, z, x


# ============================================================
# Classical objective function
# ============================================================

def objective(params):

    theta = params[0]

    energy, z, x = evaluate_energy(
        theta,
        shots=1024
    )

    print(
        f"theta={theta:.6f} | "
        f"<Z>={z:.4f} | "
        f"<X>={x:.4f} | "
        f"E={energy:.6f}"
    )

    return energy


# ============================================================
# Initial parameter
# ============================================================

initial_theta = np.array([
    0.5
])


print("VQE: H = 0.5 Z + 0.8 X")
print("========================")

print(
    f"Initial theta = "
    f"{initial_theta[0]:.6f}"
)


# ============================================================
# Exact ground-state energy
# ============================================================

exact_energy = -np.sqrt(
    Z_COEFF**2 + X_COEFF**2
)

print(
    f"Exact ground-state energy = "
    f"{exact_energy:.6f}"
)


# ============================================================
# COBYLA optimization
# ============================================================

print("\nCOBYLA Optimization:")

result = minimize(
    objective,
    initial_theta,
    method="COBYLA",
    options={
        "maxiter": 40,
        "rhobeg": 0.5
    }
)


# ============================================================
# Optimization result
# ============================================================

best_theta = result.x[0]

print("\nOptimization Complete:")

print(
    f"Best theta = "
    f"{best_theta:.6f}"
)

print(
    f"Theoretical energy at theta = "
    f"{0.5 * np.cos(best_theta) + 0.8 * np.sin(best_theta):.6f}"
)


# ============================================================
# Final high-shot verification
# ============================================================

final_energy, final_z, final_x = evaluate_energy(
    best_theta,
    shots=4096
)

print("\nFinal Verification:")

print(
    f"<Z> = {final_z:.6f}"
)

print(
    f"<X> = {final_x:.6f}"
)

print(
    f"Measured energy = "
    f"{final_energy:.6f}"
)

print(
    f"Exact ground-state energy = "
    f"{exact_energy:.6f}"
)