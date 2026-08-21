import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


# ============================================================
# Two-Qubit VQE with X0X1
#
# H = 0.5 Z0 + 0.5 Z1 + 0.8 Z0Z1 + 0.2 X0X1
#
# Exact ground-state energy = -1.0
#
# 4-parameter entangling ansatz:
# RY(theta0) --●-- RY(theta2)
#              |
# RY(theta1) --X-- RY(theta3)
# ============================================================

Z0_COEFF = 0.5
Z1_COEFF = 0.5
ZZ_COEFF = 0.8
XX_COEFF = 0.2

SHOTS = 8192

simulator = AerSimulator()


def build_ansatz(theta0, theta1, theta2, theta3):
    qc = QuantumCircuit(2)

    qc.ry(theta0, 0)
    qc.ry(theta1, 1)

    qc.cx(0, 1)

    qc.ry(theta2, 0)
    qc.ry(theta3, 1)

    return qc


def measure_z_terms(theta0, theta1, theta2, theta3, shots):
    qc = build_ansatz(theta0, theta1, theta2, theta3)
    qc.measure_all()

    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    z0_total = 0
    z1_total = 0
    zz_total = 0

    for bitstring, count in counts.items():
        q1 = int(bitstring[0])
        q0 = int(bitstring[1])

        z0 = 1 if q0 == 0 else -1
        z1 = 1 if q1 == 0 else -1
        zz = z0 * z1

        z0_total += z0 * count
        z1_total += z1 * count
        zz_total += zz * count

    return (
        z0_total / shots,
        z1_total / shots,
        zz_total / shots,
        counts
    )


def measure_xx(theta0, theta1, theta2, theta3, shots):
    qc = build_ansatz(theta0, theta1, theta2, theta3)

    qc.h(0)
    qc.h(1)
    qc.measure_all()

    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    xx_total = 0

    for bitstring, count in counts.items():
        q1 = int(bitstring[0])
        q0 = int(bitstring[1])

        xx_total += count if q0 == q1 else -count

    return xx_total / shots, counts


def evaluate_energy(theta0, theta1, theta2, theta3, shots=SHOTS):
    z0, z1, zz, _ = measure_z_terms(
        theta0, theta1, theta2, theta3, shots
    )

    xx, xx_counts = measure_xx(
        theta0, theta1, theta2, theta3, shots
    )

    energy = (
        Z0_COEFF * z0
        + Z1_COEFF * z1
        + ZZ_COEFF * zz
        + XX_COEFF * xx
    )

    return energy, z0, z1, zz, xx, xx_counts


def objective(params):
    theta0, theta1, theta2, theta3 = params

    energy, z0, z1, zz, xx, _ = evaluate_energy(
        theta0, theta1, theta2, theta3, shots=SHOTS
    )

    print(
        f"theta0={theta0:.4f}, "
        f"theta1={theta1:.4f}, "
        f"theta2={theta2:.4f}, "
        f"theta3={theta3:.4f} | "
        f"<Z0>={z0:.4f}, "
        f"<Z1>={z1:.4f}, "
        f"<ZZ>={zz:.4f}, "
        f"<XX>={xx:.4f} | "
        f"E={energy:.4f}"
    )

    return energy


initial_parameters = np.array([
    0.5,
    0.5,
    0.0,
    0.0
])


print("Two-Qubit VQE with X0X1")
print("=======================")

print("\nHamiltonian:")
print("H = 0.5 Z0 + 0.5 Z1 + 0.8 Z0Z1 + 0.2 X0X1")

print("\nExact ground-state energy = -1.000000")

print(f"Initial theta0 = {initial_parameters[0]:.4f}")
print(f"Initial theta1 = {initial_parameters[1]:.4f}")
print(f"Initial theta2 = {initial_parameters[2]:.4f}")
print(f"Initial theta3 = {initial_parameters[3]:.4f}")


(
    initial_energy,
    initial_z0,
    initial_z1,
    initial_zz,
    initial_xx,
    initial_xx_counts
) = evaluate_energy(
    initial_parameters[0],
    initial_parameters[1],
    initial_parameters[2],
    initial_parameters[3],
    shots=SHOTS
)

print("\nInitial Measurements:")
print(f"<Z0> = {initial_z0:.4f}")
print(f"<Z1> = {initial_z1:.4f}")
print(f"<Z0Z1> = {initial_zz:.4f}")
print(f"<X0X1> = {initial_xx:.4f}")
print(f"Initial Energy = {initial_energy:.4f}")
print(f"XX Measurement Counts = {initial_xx_counts}")


print("\nCOBYLA Optimization:")

result = minimize(
    objective,
    initial_parameters,
    method="COBYLA",
    options={
        "maxiter": 60,
        "rhobeg": 0.5
    }
)


best_theta0 = result.x[0]
best_theta1 = result.x[1]
best_theta2 = result.x[2]
best_theta3 = result.x[3]

print("\nOptimization Complete:")
print(f"Best theta0 = {best_theta0:.6f}")
print(f"Best theta1 = {best_theta1:.6f}")
print(f"Best theta2 = {best_theta2:.6f}")
print(f"Best theta3 = {best_theta3:.6f}")
print(f"Optimizer energy = {result.fun:.6f}")
print("Exact ground-state energy = -1.000000")


(
    final_energy,
    final_z0,
    final_z1,
    final_zz,
    final_xx,
    final_xx_counts
) = evaluate_energy(
    best_theta0,
    best_theta1,
    best_theta2,
    best_theta3,
    shots=16384
)

print("\nFinal Verification:")
print(f"<Z0> = {final_z0:.6f}")
print(f"<Z1> = {final_z1:.6f}")
print(f"<Z0Z1> = {final_zz:.6f}")
print(f"<X0X1> = {final_xx:.6f}")
print(f"Final Energy = {final_energy:.6f}")
print(f"XX Counts = {final_xx_counts}")

print("\nExact Ground-State Energy = -1.000000")