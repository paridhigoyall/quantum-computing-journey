import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


# ============================================================
# Two-Qubit VQE
#
# H = 0.5 Z0 + 0.5 Z1 + 0.8 Z0 Z1
#
# Exact ground-state energy = -0.8
#
# Ground states:
# |01> and |10>
# ============================================================

Z0_COEFF = 0.5
Z1_COEFF = 0.5
ZZ_COEFF = 0.8

SHOTS = 1024


# ============================================================
# Ansatz
# ============================================================

def build_ansatz(theta0, theta1):

    qc = QuantumCircuit(2, 2)

    qc.ry(theta0, 0)
    qc.ry(theta1, 1)

    return qc


# ============================================================
# Measure all required Pauli terms
# ============================================================

def evaluate_energy(theta0, theta1, shots=1024):

    qc = build_ansatz(theta0, theta1)

    qc.measure(0, 0)
    qc.measure(1, 1)

    simulator = AerSimulator()

    compiled = transpile(
        qc,
        simulator
    )

    result = simulator.run(
        compiled,
        shots=shots
    ).result()

    counts = result.get_counts()

    z0_total = 0
    z1_total = 0
    zz_total = 0

    for bitstring, count in counts.items():

        # Qiskit displays classical bits as c1 c0.
        q1 = int(bitstring[0])
        q0 = int(bitstring[1])

        # Z eigenvalues:
        # |0> -> +1
        # |1> -> -1

        z0 = 1 if q0 == 0 else -1
        z1 = 1 if q1 == 0 else -1

        zz = z0 * z1

        z0_total += z0 * count
        z1_total += z1 * count
        zz_total += zz * count

    z0 = z0_total / shots
    z1 = z1_total / shots
    zz = zz_total / shots

    energy = (
        Z0_COEFF * z0
        + Z1_COEFF * z1
        + ZZ_COEFF * zz
    )

    return energy, z0, z1, zz, counts


# ============================================================
# Objective function
# ============================================================

def objective(params):

    theta0 = params[0]
    theta1 = params[1]

    energy, z0, z1, zz, _ = evaluate_energy(
        theta0,
        theta1,
        shots=SHOTS
    )

    print(
        f"theta0={theta0:.4f}, "
        f"theta1={theta1:.4f} | "
        f"<Z0>={z0:.4f}, "
        f"<Z1>={z1:.4f}, "
        f"<ZZ>={zz:.4f} | "
        f"E={energy:.4f}"
    )

    return energy


# ============================================================
# Initial parameters
# ============================================================

initial_parameters = np.array([
    0.5,
    0.5
])


# ============================================================
# Header
# ============================================================

print("Two-Qubit VQE")
print("=============")

print(
    "Hamiltonian:"
)

print(
    "H = 0.5 Z0 + 0.5 Z1 + 0.8 Z0Z1"
)

print(
    "\nExact ground-state energy = -0.8"
)

print(
    "Ground states = |01>, |10>"
)

print(
    f"\nInitial theta0 = "
    f"{initial_parameters[0]:.4f}"
)

print(
    f"Initial theta1 = "
    f"{initial_parameters[1]:.4f}"
)


# ============================================================
# Initial measurement
# ============================================================

(
    initial_energy,
    initial_z0,
    initial_z1,
    initial_zz,
    initial_counts
) = evaluate_energy(
    initial_parameters[0],
    initial_parameters[1],
    shots=2048
)

print("\nInitial Measurements:")

print(
    f"<Z0> = {initial_z0:.4f}"
)

print(
    f"<Z1> = {initial_z1:.4f}"
)

print(
    f"<Z0Z1> = {initial_zz:.4f}"
)

print(
    f"Initial Energy = "
    f"{initial_energy:.4f}"
)

print(
    f"Counts = {initial_counts}"
)


# ============================================================
# COBYLA optimization
# ============================================================

print("\nCOBYLA Optimization:")

result = minimize(
    objective,
    initial_parameters,
    method="COBYLA",
    options={
        "maxiter": 40,
        "rhobeg": 0.5
    }
)


# ============================================================
# Optimization result
# ============================================================

best_theta0 = result.x[0]
best_theta1 = result.x[1]

print("\nOptimization Complete:")

print(
    f"Best theta0 = "
    f"{best_theta0:.6f}"
)

print(
    f"Best theta1 = "
    f"{best_theta1:.6f}"
)

print(
    f"Optimizer energy = "
    f"{result.fun:.6f}"
)

print(
    "Exact ground-state energy = -0.800000"
)


# ============================================================
# Final verification
# ============================================================

(
    final_energy,
    final_z0,
    final_z1,
    final_zz,
    final_counts
) = evaluate_energy(
    best_theta0,
    best_theta1,
    shots=4096
)


print("\nFinal Verification:")

print(
    f"<Z0> = "
    f"{final_z0:.6f}"
)

print(
    f"<Z1> = "
    f"{final_z1:.6f}"
)

print(
    f"<Z0Z1> = "
    f"{final_zz:.6f}"
)

print(
    f"Final Energy = "
    f"{final_energy:.6f}"
)

print(
    f"Final Counts = "
    f"{final_counts}"
)

print(
    "\nExact Ground-State Energy = -0.800000"
)