import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


# ============================================================
# 1. Observable
# ============================================================

def create_observable(num_qubits, global_cost=True):
    """
    Create either a global Z...Z observable
    or a local Z observable.
    """

    if global_cost:
        pauli_string = "Z" * num_qubits
    else:
        # Qiskit Pauli strings are written q_(n-1)...q_0
        pauli_string = "I" * (num_qubits - 1) + "Z"

    return SparsePauliOp.from_list([
        (pauli_string, 1.0)
    ])


# ============================================================
# 2. Parameterized circuit
# ============================================================

def create_ansatz(num_qubits, parameters, depth):
    """
    Hardware-efficient style ansatz.

    Each layer:
        RY rotations
        followed by nearest-neighbor CNOT entanglement
    """

    qc = QuantumCircuit(num_qubits)

    index = 0

    for layer in range(depth):

        # Parameterized single-qubit rotations
        for qubit in range(num_qubits):
            qc.ry(parameters[index], qubit)
            index += 1

        # Entanglement
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)

    return qc


# ============================================================
# 3. Quantum expectation value
# ============================================================

def quantum_expectation(
    num_qubits,
    parameters,
    depth,
    observable
):
    """

    Calculate:

        <psi(theta)|H|psi(theta)>

    """

    qc = create_ansatz(
        num_qubits,
        parameters,
        depth
    )

    state = Statevector.from_instruction(qc)

    expectation = state.expectation_value(observable)

    return float(np.real(expectation))


# ============================================================
# 4. Parameter-shift gradient
# ============================================================

def parameter_shift_gradient(
    num_qubits,
    parameters,
    depth,
    observable,
    parameter_index
):
    """
    Parameter-shift rule:

    df/dtheta =
        1/2 [
            f(theta + pi/2)
            -
            f(theta - pi/2)
        ]
    """

    shift = np.pi / 2

    params_plus = parameters.copy()
    params_minus = parameters.copy()

    params_plus[parameter_index] += shift
    params_minus[parameter_index] -= shift

    forward = quantum_expectation(
        num_qubits,
        params_plus,
        depth,
        observable
    )

    backward = quantum_expectation(
        num_qubits,
        params_minus,
        depth,
        observable
    )

    gradient = 0.5 * (forward - backward)

    return gradient


# ============================================================
# 5. Measure gradient statistics
# ============================================================

def gradient_statistics(
    num_qubits,
    depth,
    samples,
    global_cost=True
):
    """
    Generate random parameter configurations and
    calculate gradients.

    Returns:
        mean gradient
        gradient variance
        all gradients
    """

    observable = create_observable(
        num_qubits,
        global_cost=global_cost
    )

    num_parameters = num_qubits * depth

    gradients = []

    for _ in range(samples):

        # Random initialization
        parameters = np.random.uniform(
            -np.pi,
            np.pi,
            num_parameters
        )

        # Pick one parameter to study
        parameter_index = 0

        gradient = parameter_shift_gradient(
            num_qubits,
            parameters,
            depth,
            observable,
            parameter_index
        )

        gradients.append(gradient)

    gradients = np.array(gradients)

    mean_gradient = np.mean(gradients)
    variance = np.var(gradients)

    return mean_gradient, variance, gradients


# ============================================================
# 6. Experiment
# ============================================================

def run_experiment():

    # Number of qubits
    qubit_sizes = [2, 3, 4, 5, 6, 7, 8]

    # Circuit depth
    depth = 3

    # Random parameter configurations
    samples = 100

    global_variances = []
    local_variances = []

    global_means = []
    local_means = []

    print("\n" + "=" * 65)
    print("BARREN PLATEAU EXPERIMENT")
    print("=" * 65)

    for num_qubits in qubit_sizes:

        print(f"\nQubits: {num_qubits}")

        # ----------------------------------------------------
        # Global cost
        # ----------------------------------------------------

        global_mean, global_variance, _ = gradient_statistics(
            num_qubits=num_qubits,
            depth=depth,
            samples=samples,
            global_cost=True
        )

        # ----------------------------------------------------
        # Local cost
        # ----------------------------------------------------

        local_mean, local_variance, _ = gradient_statistics(
            num_qubits=num_qubits,
            depth=depth,
            samples=samples,
            global_cost=False
        )

        global_means.append(global_mean)
        global_variances.append(global_variance)

        local_means.append(local_mean)
        local_variances.append(local_variance)

        print(
            f"Global cost -> "
            f"mean = {global_mean:.6e}, "
            f"variance = {global_variance:.6e}"
        )

        print(
            f"Local cost  -> "
            f"mean = {local_mean:.6e}, "
            f"variance = {local_variance:.6e}"
        )

    # ========================================================
    # Results
    # ========================================================

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print("\nQubits | Global Variance | Local Variance")

    for q, gv, lv in zip(
        qubit_sizes,
        global_variances,
        local_variances
    ):

        print(
            f"{q:6d} | "
            f"{gv:.6e} | "
            f"{lv:.6e}"
        )

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.semilogy(
        qubit_sizes,
        global_variances,
        "o-",
        label="Global Cost"
    )

    plt.semilogy(
        qubit_sizes,
        local_variances,
        "s-",
        label="Local Cost"
    )

    plt.xlabel("Number of Qubits")

    plt.ylabel(
        "Gradient Variance (log scale)"
    )

    plt.title(
        "Gradient Variance vs Number of Qubits"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_experiment()