from qiskit import QuantumCircuit


def oracle():
    """
    Oracle marking the state |11>
    """

    qc = QuantumCircuit(2)

    # Flip phase of |11>
    qc.cz(0, 1)

    return qc

