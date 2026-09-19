import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


# ============================================================
# Max-Cut problem
# ============================================================

n = 3

edges = [
    (0, 1),
    (0, 2),
    (1, 2)
]


# ============================================================
# Cost unitary
#
# C_ij = (1 - Z_i Z_j) / 2
# ============================================================

def cost_unitary(gamma):

    qc = QuantumCircuit(n)

    for i, j in edges:

        qc.cx(i, j)
        qc.rz(-gamma, j)
        qc.cx(i, j)

    return qc


# ============================================================
# Mixer unitary
# ============================================================

def mixer_unitary(beta):

    qc = QuantumCircuit(n)

    for q in range(n):
        qc.rx(2 * beta, q)

    return qc


# ============================================================
# Max-Cut value
# ============================================================

def cut_value(bitstring):

    value = 0

    for i, j in edges:

        if bitstring[i] != bitstring[j]:
            value += 1

    return value


# ============================================================
# Build QAOA p=2 circuit
# ============================================================

def build_qaoa_p2(
    gamma1,
    beta1,
    gamma2,
    beta2
):

    qc = QuantumCircuit(n, n)

    # --------------------------------------------------------
    # Initial uniform superposition
    # --------------------------------------------------------

    for q in range(n):
        qc.h(q)

    # ========================================================
    # QAOA LAYER 1
    # ========================================================

    qc.compose(
        cost_unitary(gamma1),
        inplace=True
    )

    qc.compose(
        mixer_unitary(beta1),
        inplace=True
    )

    # ========================================================
    # QAOA LAYER 2
    # ========================================================

    qc.compose(
        cost_unitary(gamma2),
        inplace=True
    )

    qc.compose(
        mixer_unitary(beta2),
        inplace=True
    )

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    qc.measure(
        range(n),
        range(n)
    )

    return qc


# ============================================================
# Evaluate QAOA p=2
# ============================================================

def evaluate_qaoa_p2(
    params,
    shots=1024
):

    gamma1, beta1, gamma2, beta2 = params

    qc = build_qaoa_p2(
        gamma1,
        beta1,
        gamma2,
        beta2
    )

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

    # --------------------------------------------------------
    # Calculate expected cut value
    # --------------------------------------------------------

    total_cost = 0

    for bitstring, count in counts.items():

        cost = cut_value(bitstring)

        total_cost += cost * count

    expectation = total_cost / shots

    return expectation, counts


# ============================================================
# Initial p=2 parameters
# ============================================================

initial_parameters = np.array([
    np.pi / 4,   # gamma1
    np.pi / 4,   # beta1
    np.pi / 4,   # gamma2
    np.pi / 4    # beta2
])


# ============================================================
# Build and display p=2 circuit
# ============================================================

qc = build_qaoa_p2(
    *initial_parameters
)

print("QAOA p=2 Max-Cut Circuit:")
print(qc)

print("\nInitial Parameters:")

print(
    "gamma1 =",
    initial_parameters[0]
)

print(
    "beta1  =",
    initial_parameters[1]
)

print(
    "gamma2 =",
    initial_parameters[2]
)

print(
    "beta2  =",
    initial_parameters[3]
)


# ============================================================
# Initial p=2 experiment
# ============================================================

expectation, counts = evaluate_qaoa_p2(
    initial_parameters,
    shots=2048
)

print("\nInitial Measurement Results:")

for state, count in sorted(
    counts.items(),
    key=lambda item: item[1],
    reverse=True
):

    print(
        f"{state}: {count} "
        f"| cut = {cut_value(state)}"
    )

print(
    "\nInitial Expected Cut:",
    expectation
)

print(
    "Theoretical Maximum:",
    2
)


# ============================================================
# Objective function for COBYLA
# ============================================================

def objective(params):

    expectation, _ = evaluate_qaoa_p2(
        params,
        shots=1024
    )

    print(
        f"gamma1={params[0]:.4f}, "
        f"beta1={params[1]:.4f}, "
        f"gamma2={params[2]:.4f}, "
        f"beta2={params[3]:.4f}, "
        f"cut={expectation:.4f}"
    )

    # COBYLA minimizes.
    # We want to maximize expected cut.
    return -expectation


# ============================================================
# COBYLA optimization
# ============================================================

print("\nCOBYLA Optimization:")

optimization_result = minimize(
    objective,
    initial_parameters,
    method="COBYLA",
    options={
        "maxiter": 50,
        "rhobeg": 0.5
    }
)


# ============================================================
# Optimized parameters
# ============================================================

best_parameters = optimization_result.x

best_gamma1 = best_parameters[0]
best_beta1 = best_parameters[1]
best_gamma2 = best_parameters[2]
best_beta2 = best_parameters[3]

best_expectation = -optimization_result.fun


print("\nOptimization Complete:")

print(
    "Best gamma1 =",
    best_gamma1
)

print(
    "Best beta1  =",
    best_beta1
)

print(
    "Best gamma2 =",
    best_gamma2
)

print(
    "Best beta2  =",
    best_beta2
)

print(
    "Best expected cut =",
    best_expectation
)

print(
    "Maximum possible cut =",
    2
)


# ============================================================
# Final verification
# ============================================================

print("\nFinal Verification:")

final_expectation, final_counts = evaluate_qaoa_p2(
    best_parameters,
    shots=2048
)

optimal_count = 0

for state, count in sorted(
    final_counts.items(),
    key=lambda item: item[1],
    reverse=True
):

    cut = cut_value(state)

    print(
        f"{state}: {count} "
        f"| cut = {cut}"
    )

    if cut == 2:
        optimal_count += count


optimal_probability = (
    optimal_count / 2048
)


print(
    "\nOptimal-solution probability:",
    optimal_probability
)

print(
    "Final expected cut:",
    final_expectation
)

print(
    "Theoretical maximum cut:",
    2
)