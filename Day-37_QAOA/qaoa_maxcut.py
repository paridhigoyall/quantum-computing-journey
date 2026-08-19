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
# QAOA parameters
# ============================================================

gamma = np.pi / 4
beta = np.pi / 4


# ============================================================
# Cost unitary
#
# C_ij = (1 - Z_i Z_j) / 2
# ============================================================
def objective(params):

    best_gamma=optimization_result.x[0]
    best_beta =optimization_result.x[1]

    expectation, _ = evaluate_qaoa(
        gamma,
        beta,
        shots=1024
    )

    value = -expectation

    print(
        f"gamma={gamma:.4f}, "
        f"beta={beta:.4f}, "
        f"cut={expectation:.4f}"
    )

    return value

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
# Calculate Max-Cut value
# ============================================================

def cut_value(bitstring):

    value = 0

    for i, j in edges:

        if bitstring[i] != bitstring[j]:
            value += 1

    return value


# ============================================================
# Build QAOA circuit
# ============================================================

def build_qaoa(gamma, beta):

    qc = QuantumCircuit(n, n)

    # --------------------------------------------------------
    # Initial uniform superposition
    # --------------------------------------------------------

    for q in range(n):
        qc.h(q)

    # --------------------------------------------------------
    # Cost layer
    # --------------------------------------------------------

    qc.compose(
        cost_unitary(gamma),
        inplace=True
    )

    # --------------------------------------------------------
    # Mixer layer
    # --------------------------------------------------------

    qc.compose(
        mixer_unitary(beta),
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
# Evaluate QAOA
# ============================================================

def evaluate_qaoa(gamma, beta, shots=2048):

    qc = build_qaoa(
        gamma,
        beta
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
# Show one QAOA circuit
# ============================================================

qc = build_qaoa(
    gamma,
    beta
)

print("QAOA Max-Cut Circuit:")
print(qc)

print("\nParameters:")
print("gamma =", gamma)
print("beta  =", beta)


# ============================================================
# Run one experiment
# ============================================================

expectation, counts = evaluate_qaoa(
    gamma,
    beta,
    shots=2048
)

print("\nMeasurement Results:")

for state, count in sorted(
    counts.items(),
    key=lambda item: item[1],
    reverse=True
):

    print(
        f"{state}: {count}"
    )

print(
    "\nExpected cut value:",
    expectation
)


# ============================================================
# Parameter experiment
# ============================================================

print("\nParameter Experiment:")

test_parameters = [

    (0.0, 0.0),

    (np.pi / 4, np.pi / 4),

    (np.pi / 2, np.pi / 4),

    (np.pi / 4, np.pi / 2),

    (np.pi / 2, np.pi / 2)

]


for gamma_test, beta_test in test_parameters:

    expectation, counts = evaluate_qaoa(
        gamma_test,
        beta_test,
        shots=1024
    )

    print(
        f"gamma={gamma_test:.3f}, "
        f"beta={beta_test:.3f} "
        f"-> expected cut = {expectation:.4f}"
    )


# ============================================================
# Maximum possible cut
# ============================================================

print("\nTheoretical maximum cut:")
print("Maximum cut value = 2")
# ============================================================
# Grid search for best gamma and beta
# ============================================================

print("\nGrid Search:")

best_expectation = -1
best_gamma = None
best_beta = None

values = np.linspace(
    0,
    np.pi,
    9
)

for gamma_test in values:

    for beta_test in values:

        expectation, _ = evaluate_qaoa(
            gamma_test,
            beta_test,
            shots=512
        )

        if expectation > best_expectation:

            best_expectation = expectation
            best_gamma = gamma_test
            best_beta = beta_test


print("\nBest parameters found:")

print(
    "gamma =",
    best_gamma
)

print(
    "beta =",
    best_beta
)

print(
    "Expected cut =",
    best_expectation
)
# ============================================================
# Classical optimization with COBYLA
# ============================================================

print("\nCOBYLA Optimization:")

initial_parameters = [
    np.pi / 4,
    np.pi / 4
]

optimization_result = minimize(
    objective,
    initial_parameters,
    method="COBYLA",
    options={
        "maxiter": 30,
        "rhobeg": 0.5
    }
)


# ============================================================
# Final result
# ============================================================

best_gamma = optimization_result.x[0]
best_beta = optimization_result.x[1]

best_expectation = -optimization_result.fun


print("\nOptimization Complete:")

print(
    "Best gamma =",
    best_gamma
)

print(
    "Best beta =",
    best_beta
)

print(
    "Best expected cut =",
    best_expectation
)

print(
    "Maximum possible cut = 2"
)

optimal_probability = 0

print("\nFinal Measurement Results:")

for state, count in sorted(
    counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    cut = cut_value(state)

    print(
        f"{state}: {count} "
        f"| cut = {cut}"
    )

    if cut == 2:
        optimal_probability += count

optimal_probability /= 2048

print(
    "\nOptimal-solution probability:",
    optimal_probability
)

print(
    "Expected cut value:",
    expectation
)